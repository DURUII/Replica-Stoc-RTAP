from src.algos.offline.base_algorithm import BaseAlgorithm
from src.models.bean import Task, Problem
from typing import Dict, Set, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import threading
import time
import copy
import random
from collections import deque, defaultdict
import os
import pickle


class TaskAssignmentEnv(gym.Env):
    """
    Custom Gymnasium environment for task assignment problem.

    This environment models the online task assignment scenario where the agent
    must select vehicles to recruit for each task while learning the execution
    abilities through Gaussian sampling.
    """

    def __init__(self, problem: Problem, task_id: int, task: Task,
                 available_vehicles: List[int], max_recruitment_steps: int = 50):
        """
        Initialize the task assignment environment.

        Args:
            problem: Problem instance with tasks, vehicles, and parameters
            task_id: Current task ID
            task: Current task instance
            available_vehicles: List of available vehicle IDs
            max_recruitment_steps: Maximum number of recruitment actions per episode
        """
        super().__init__()
        self.problem = problem
        self.task_id = task_id
        self.task = task
        self.available_vehicles = available_vehicles
        self.max_recruitment_steps = max_recruitment_steps

        # Track state
        self.recruited_vehicles = set()
        self.observed_executions = {}  # vehicle_id -> list of observed execution samples
        self.step_count = 0

        # Calculate state and action dimensions
        self.n_vehicles = len(available_vehicles)
        self.state_dim = self.n_vehicles * 5 + 3  # cost, p, observed_mean, observed_std, is_recruited + task_params
        self.action_dim = self.n_vehicles

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.n_vehicles)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Reward parameters
        self.cost_penalty_weight = 1.0
        self.constraint_bonus = 100.0
        self.constraint_penalty = 200.0

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.recruited_vehicles = set()
        self.observed_executions = {}
        self.step_count = 0
        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        state = []

        # Vehicle features: cost, arrival probability, observed execution statistics
        for i, vehicle_id in enumerate(self.available_vehicles):
            vehicle = self.problem.vehicles[vehicle_id]

            # Cost for this task
            cost = vehicle.get_expected_cost(self.task_id)
            state.append(cost / 100.0)  # Normalize cost

            # Arrival probability
            arrival_prob = vehicle.get_arrival_probability(self.task_id)
            state.append(arrival_prob)

            # Observed execution statistics
            if vehicle_id in self.observed_executions:
                samples = self.observed_executions[vehicle_id]
                mean_exec = np.mean(samples)
                std_exec = np.std(samples) if len(samples) > 1 else 0.1
            else:
                mean_exec = 0.5  # Prior estimate
                std_exec = 0.3   # Prior uncertainty
            state.append(mean_exec)
            state.append(std_exec)

            # Recruitment status
            state.append(1.0 if vehicle_id in self.recruited_vehicles else 0.0)

        # Task parameters
        state.append(self.problem.l / 10.0)  # Normalize l
        state.append(self.problem.beta)
        state.append(len(self.recruited_vehicles) / 20.0)  # Normalize recruited count

        return np.array(state, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.step_count += 1

        # Get selected vehicle
        vehicle_id = self.available_vehicles[action]

        # Check if vehicle already recruited
        if vehicle_id in self.recruited_vehicles:
            # Invalid action - small penalty
            return self._get_state(), -1.0, False, False, {}

        # Recruit vehicle and observe execution sample
        psi_before = self.problem.compute_psi_oracle(self.task_id, self.recruited_vehicles)
        self.recruited_vehicles.add(vehicle_id)
        execution_sample = self.problem.vehicles[vehicle_id].draw_execution_sample()

        # Update observed executions
        if vehicle_id not in self.observed_executions:
            self.observed_executions[vehicle_id] = []
        self.observed_executions[vehicle_id].append(execution_sample)

        # Calculate reward
        cost = self.problem.vehicles[vehicle_id].get_expected_cost(self.task_id)
        reward = -self.cost_penalty_weight * cost

        # Check constraint satisfaction
        psi_value = self.problem.compute_psi_oracle(self.task_id, self.recruited_vehicles)
        reward += 50.0 * (psi_value - psi_before)

        # Terminal condition: constraint satisfied or max steps reached
        done = (psi_value >= self.problem.beta) or (self.step_count >= self.max_recruitment_steps)

        if done:
            if psi_value >= self.problem.beta:
                reward += self.constraint_bonus
            else:
                reward -= self.constraint_penalty

        info = {
            'recruited_vehicles': len(self.recruited_vehicles),
            'psi_value': psi_value,
            'constraint_satisfied': psi_value >= self.problem.beta,
            'total_cost': sum(self.problem.vehicles[v].get_expected_cost(self.task_id)
                            for v in self.recruited_vehicles)
        }

        return self._get_state(), reward, done, False, info

    def get_available_actions(self) -> List[int]:
        """Get list of available actions (vehicles not yet recruited)."""
        available = []
        for i, vehicle_id in enumerate(self.available_vehicles):
            if vehicle_id not in self.recruited_vehicles:
                available.append(i)
        return available


from stable_baselines3.common.callbacks import BaseCallback


class QNetwork(nn.Module):
    """Simple MLP Q-network mapping state to Q-values over actions."""
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int] = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SimplePolicy:
    """Minimal wrapper to expose optimizer for monitoring callbacks."""
    def __init__(self, optimizer: optim.Optimizer):
        self.optimizer = optimizer


class DoubleDQNAgent:
    """Double DQN agent with online and target networks."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 1e-4,
        target_update_interval: int = 250,
        device: str = None,
    ):
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.policy = _SimplePolicy(optim.Adam(self.q_net.parameters(), lr=learning_rate))
        # Expose dimensions and gym-like spaces for compatibility with downstream code
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(action_dim)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = int(target_update_interval)
        self._update_counter = 0

    def select_action(self, obs: np.ndarray, available_actions: List[int] = None, deterministic: bool = False) -> int:
        """Epsilon-greedy action selection with optional masking of unavailable actions."""
        if (not deterministic) and (np.random.rand() < self.epsilon):
            if available_actions:
                return int(random.choice(available_actions))
            return int(np.random.randint(0, self.q_net.net[-1].out_features))

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_t).squeeze(0).cpu().numpy()

        if available_actions is not None and len(available_actions) > 0:
            mask = np.full_like(q_values, fill_value=-np.inf)
            for a in available_actions:
                mask[a] = q_values[a]
            return int(np.argmax(mask))
        return int(np.argmax(q_values))

    def update(self, batch: Dict[str, Any]) -> float:
        """Perform one Double DQN update step and return loss."""
        states = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.long, device=self.device).unsqueeze(-1)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Current Q values
        current_q = self.q_net(states).gather(1, actions)

        # Double DQN target: action selection via online net, evaluation via target net
        with torch.no_grad():
            next_q_online = self.q_net(next_states)
            B = next_states.shape[0]
            A = self.action_dim
            veh_slice = next_states[:, :A*5]
            rec_flags = veh_slice[:, 4::5]
            costs = veh_slice[:, 0::5]
            arrs = veh_slice[:, 1::5]
            legal_mask = ((rec_flags < 0.5) & ((costs > 0.0) | (arrs > 0.0))).float()
            masked_next_q = next_q_online.clone()
            masked_next_q[legal_mask == 0] = -1e9
            next_actions = torch.argmax(masked_next_q, dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q_target

        loss = F.mse_loss(current_q, target_q)
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        # Target network update
        self._update_counter += 1
        if self._update_counter % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.policy.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'target_update_interval': self.target_update_interval,
        }, path)

    @staticmethod
    def load(path: str, state_dim: int, action_dim: int, learning_rate: float, gamma: float, target_update_interval: int, device: str = None) -> "DoubleDQNAgent":
        agent = DoubleDQNAgent(state_dim, action_dim, learning_rate=learning_rate, gamma=gamma, target_update_interval=target_update_interval, device=device)
        checkpoint = torch.load(path, map_location='cpu')
        agent.q_net.load_state_dict(checkpoint['q_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        agent.policy.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
        agent.gamma = checkpoint.get('gamma', agent.gamma)
        agent.target_update_interval = checkpoint.get('target_update_interval', agent.target_update_interval)
        return agent


class MonitorQValueCallback(BaseCallback):
    """
    Callback to monitor the evolution of the q-value for sampled initial states.
    Compatible with the custom DoubleDQNAgent via q_net and policy.optimizer.
    """
    def __init__(self, env_builder, sample_interval: int = 2500, n_samples: int = 256):
        super().__init__()
        self.timesteps = []
        self.max_q_values = []
        self.sample_interval = sample_interval
        # Sample initial states that will be used to monitor the estimated q-value
        env = env_builder()
        self.start_obs = np.array([env.reset()[0] for _ in range(n_samples)])

    def _on_training_start(self) -> None:
        # Create artificial overestimation for action 0 on the sampled states
        if not hasattr(self.model, 'q_net'):
            return
        obs = torch.tensor(self.start_obs, device=self.model.device).float()
        target_q_values = torch.ones((len(obs), 1), device=self.model.device).float() * 100.0

        for _ in range(50):
            current_q_values = self.model.q_net(obs)
            current_q_values = torch.gather(current_q_values, dim=1, index=torch.zeros((len(obs), 1), device=self.model.device).long())
            loss = F.mse_loss(current_q_values, target_q_values)
            self.model.policy.optimizer.zero_grad()
            loss.backward()
            self.model.policy.optimizer.step()

    def _on_step(self) -> bool:
        # Periodically record max q-values over the sampled states
        if self.n_calls % self.sample_interval == 0:
            obs = torch.tensor(self.start_obs, device=self.model.device).float()
            with torch.no_grad():
                q_values = self.model.q_net(obs).cpu().numpy()
            self.logger.record("train/max_q_value", float(q_values.max()))
            self.timesteps.append(self.num_timesteps)
            self.max_q_values.append(float(q_values.max()))
        return True


class TrainingRewardCallback(BaseCallback):
    """Collect episode rewards during SB3 training."""
    def __init__(self, algo_ref, verbose: int = 0):
        super().__init__(verbose)
        self.algo_ref = algo_ref
        self._episode_returns = None

    def _on_training_start(self) -> None:
        # Initialize return trackers per env
        n_envs = self.training_env.num_envs if hasattr(self.training_env, 'num_envs') else 1
        self._episode_returns = np.zeros(n_envs, dtype=np.float32)

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')
        if rewards is None or dones is None:
            return True
        # Vectorized handling
        if isinstance(rewards, (list, tuple, np.ndarray)):
            for i in range(len(rewards)):
                self._episode_returns[i] += float(rewards[i])
                if bool(dones[i]):
                    self.algo_ref.training_rewards.append(float(self._episode_returns[i]))
                    self._episode_returns[i] = 0.0
        else:
            # Single env
            self._episode_returns += float(rewards)
            if bool(dones):
                self.algo_ref.training_rewards.append(float(self._episode_returns))
                self._episode_returns = 0.0
        return True


class RandomizedTaskAssignmentEnv(TaskAssignmentEnv):
    """
    Training env that randomizes task_id on reset while keeping action/obs spaces
    fixed to all vehicles in the problem for SB3 compatibility.
    """
    def __init__(self, problem: Problem, max_recruitment_steps: int = 50):
        # Use all vehicles to keep spaces constant
        all_vehicles = list(problem.vehicles.keys())
        # Initial random task
        init_task_id = random.choice(list(problem.tasks.keys()))
        super().__init__(problem, init_task_id, problem.tasks[init_task_id], all_vehicles, max_recruitment_steps)

    def reset(self, seed=None, options=None):
        # Randomize task each episode
        self.task_id = random.choice(list(self.problem.tasks.keys()))
        self.task = self.problem.tasks[self.task_id]
        return super().reset(seed=seed, options=options)


class StochasticReinforce(BaseAlgorithm):
    """
    Double DQN-based RL algorithm for online task assignment.

    Kept the original class name for compatibility, but internally uses
    a custom Double DQN to learn recruitment policies and reduce Q-value
    overestimation (Van Hasselt et al., AAAI 2016).
    """

    def __init__(self, problem, num_workers: int = 1, max_episodes_per_worker: int = 100,
                 learning_rate: float = 2.5e-4, gamma: float = 0.99, tau: float = 1.0,
                 max_steps_per_episode: int = 50, random_seed: int = 42, dataset_name: str = 'unknown'):
        super().__init__(problem)
        # Hyperparameters (tau kept for API compatibility, unused by DQN)
        self.num_workers = max(1, int(num_workers))
        self.max_episodes_per_worker = int(max_episodes_per_worker)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.random_seed = int(random_seed)
        self.dataset_name = dataset_name

        # Seeds
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # DDQN agent placeholder
        self.model: DoubleDQNAgent = None
        # Preserve original attribute name for potential external references
        self.global_model = None  # not used; kept for compatibility

        # Metrics
        self.training_rewards: List[float] = []
        self.training_losses: List[float] = []
        self.is_trained = False

        # Model save path
        n_vehicles = len(self.problem.vehicles)
        self.model_path = f'logs/ddqn_model_{self.dataset_name}_{n_vehicles}.pth'
        from src.utils.rl_utils import ReplayBuffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.learning_starts = 100
        self.batch_size = 64
        self.train_freq = 1
        self._task_counter = 0
        self._save_every = 200
        self._global_step = 0
        self.epsilon_decay = 5e-3

    def _make_training_env(self) -> gym.Env:
        return RandomizedTaskAssignmentEnv(self.problem, self.max_steps_per_episode)

    def train(self):
        if len(self.problem.vehicles) == 0:
            return
        env = self._make_training_env()
        if os.path.exists(self.model_path):
            dummy_obs, _ = env.reset()
            state_dim = dummy_obs.shape[0]
            action_dim = env.action_space.n
            self.model = DoubleDQNAgent.load(
                self.model_path,
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                target_update_interval=250,
                device=None,
            )
            self.is_trained = True
            return
        dummy_obs, _ = env.reset()
        state_dim = dummy_obs.shape[0]
        action_dim = env.action_space.n
        self.model = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            target_update_interval=250,
            epsilon_decay=self.epsilon_decay,
        )
        self.is_trained = True

    def run(self, task_id: int, task: Task, remained: List[int]) -> None:
        """Run trained DQN to recruit vehicles for a single task."""
        if not remained:
            self.recruited_vehicles[task_id] = set()
            return

        if self.model is None:
            env_full = self._make_training_env()
            dummy_obs, _ = env_full.reset()
            state_dim = dummy_obs.shape[0]
            action_dim = env_full.action_space.n
            if os.path.exists(self.model_path):
                self.model = DoubleDQNAgent.load(
                    self.model_path,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    target_update_interval=250,
                    device=None,
                )
            else:
                self.model = DoubleDQNAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    target_update_interval=250,
                    epsilon_decay=self.epsilon_decay,
                )
            self.is_trained = True

        env = TaskAssignmentEnv(self.problem, task_id, task, remained, self.max_steps_per_episode)
        obs, _ = env.reset()

        # Pad observation to match the full vehicle space if needed
        if self.model and self.model.observation_space.shape[0] != obs.shape[0]:
            padded_obs = np.zeros(self.model.observation_space.shape, dtype=np.float32)
            # Create a map from remained vehicle_id to its index in the full list
            all_vehicles_list = list(self.problem.vehicles.keys())
            remained_map = {vehicle_id: i for i, vehicle_id in enumerate(all_vehicles_list)}
            
            # Fill vehicle-specific features
            for i, vehicle_id in enumerate(remained):
                if vehicle_id in remained_map:
                    full_idx = remained_map[vehicle_id]
                    padded_obs[full_idx*5:(full_idx+1)*5] = obs[i*5:(i+1)*5]
            
            # Fill task-specific features
            padded_obs[-3:] = obs[-3:]
            obs = padded_obs

        recruited = set()
        done = False
        max_inference_steps = min(20, len(remained))

        for _ in range(max_inference_steps):
            if done:
                break

            if self.model is not None:
                all_vehicles_list = list(self.problem.vehicles.keys())
                remained_map = {vehicle_id: i for i, vehicle_id in enumerate(all_vehicles_list)}
                available_full_actions = [remained_map[v] for v in remained if v not in recruited]
                deterministic_infer = os.path.exists(self.model_path) or (self.replay_buffer.size() > max(self.learning_starts * 2, self.batch_size * 4))
                full_action = self.model.select_action(obs, available_actions=available_full_actions, deterministic=deterministic_infer)
                predicted_vehicle_id = all_vehicles_list[full_action]
                if predicted_vehicle_id in remained and predicted_vehicle_id not in recruited:
                    action = remained.index(predicted_vehicle_id)
                    executed_vehicle_id = predicted_vehicle_id
                    executed_full_action = full_action
                else:
                    available_actions = env.get_available_actions()
                    if available_actions:
                        action = random.choice(available_actions)
                        executed_vehicle_id = remained[action]
                        executed_full_action = remained_map[executed_vehicle_id]
                    else:
                        break
            else:
                action = random.choice(env.get_available_actions() or [0])

            # Fallback to a valid action if chosen invalid
            available_actions = env.get_available_actions()
            if not available_actions:
                break
            if action >= len(remained) or remained[action] in recruited:
                action = random.choice(available_actions)

            prev_obs = obs
            obs, reward, done, _truncated, info = env.step(int(action))
            recruited.add(remained[int(action)])

            # Pad the next observation
            if self.model and self.model.observation_space.shape[0] != obs.shape[0]:
                padded_next_obs = np.zeros(self.model.observation_space.shape, dtype=np.float32)
                all_vehicles_list = list(self.problem.vehicles.keys())
                remained_map = {vehicle_id: i for i, vehicle_id in enumerate(all_vehicles_list)}
                for i, vehicle_id in enumerate(remained):
                    if vehicle_id in remained_map:
                        full_idx = remained_map[vehicle_id]
                        padded_next_obs[full_idx*5:(full_idx+1)*5] = obs[i*5:(i+1)*5]
                padded_next_obs[-3:] = obs[-3:]
                obs = padded_next_obs
            if self.model:
                s_dim = self.model.observation_space.shape[0]
                if prev_obs.shape[0] != s_dim:
                    padded_prev = np.zeros((s_dim,), dtype=np.float32)
                    all_vehicles_list = list(self.problem.vehicles.keys())
                    remained_map = {vehicle_id: i for i, vehicle_id in enumerate(all_vehicles_list)}
                    for i, vehicle_id in enumerate(remained):
                        if vehicle_id in remained_map:
                            full_idx = remained_map[vehicle_id]
                            padded_prev[full_idx*5:(full_idx+1)*5] = prev_obs[i*5:(i+1)*5]
                    padded_prev[-3:] = prev_obs[-3:]
                    prev_obs = padded_prev
                if 'executed_full_action' in locals():
                    self.replay_buffer.add(prev_obs, int(executed_full_action), float(reward), obs, bool(done))
                else:
                    all_vehicles_list = list(self.problem.vehicles.keys())
                    remained_map = {vehicle_id: i for i, vehicle_id in enumerate(all_vehicles_list)}
                    self.replay_buffer.add(prev_obs, int(remained_map[remained[int(action)]]), float(reward), obs, bool(done))
                self._global_step += 1
                if self.replay_buffer.size() > self.learning_starts and (self._global_step % self.train_freq == 0):
                    b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.batch_size)
                    batch = {'states': b_s, 'actions': b_a, 'rewards': b_r, 'next_states': b_ns, 'dones': b_d}
                    loss = self.model.update(batch)
                    self.training_losses.append(loss)

        # Ensure constraint satisfaction by adding more vehicles if needed
        if self.problem.compute_psi_oracle(task_id, recruited) < self.problem.beta:
            # Sort remaining vehicles by cost to add the cheapest ones first
            sorted_remained = sorted(
                [v for v in remained if v not in recruited],
                key=lambda v_id: self.problem.vehicles[v_id].get_expected_cost(task_id)
            )
            for vehicle_id in sorted_remained:
                recruited.add(vehicle_id)
                if self.problem.compute_psi_oracle(task_id, recruited) >= self.problem.beta:
                    break

        self.recruited_vehicles[task_id] = recruited
        self._task_counter += 1
        if (self._task_counter % self._save_every == 0) and self.model is not None:
            try:
                self.model.save(self.model_path)
            except Exception:
                pass

    def get_training_metrics(self) -> Dict[str, List[float]]:
        return {
            'rewards': self.training_rewards,
            'losses': self.training_losses
        }
