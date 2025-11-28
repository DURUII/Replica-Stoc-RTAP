from src.algos.offline.base_algorithm import BaseAlgorithm
from typing import Dict, Set, Tuple, List
import numpy as np
from src.models.bean import Task, Problem
from src.utils.probability import poisson_binomial_pmf, poisson_binomial_cdf
from src.algos.offline.proposed import SubmodularPolicy


class SqrtFirst(SubmodularPolicy):
    """SqrtFirst algorithm for task allocation.

    This algorithm uses a submodular policy to allocate tasks to vehicles,
    with an initial exploration phase followed by an exploitation phase.

    Attributes:
        counts (Dict[int, int]): Execution counts for each vehicle.
        avg_rewards (Dict[int, float]): Average rewards for each vehicle.
        num_tasks (int): Total number of tasks.
        exploration_rounds (float): Number of rounds in the exploration phase.
    """

    def __init__(self, problem, theta: float = 1.0, logs: bool = True):
        """Initializes the SqrtFirst algorithm.

        Args:
            problem (Problem): The problem instance containing tasks and vehicles.
            theta (float): A parameter to control the threshold for recruitment.
        """
        super().__init__(problem, theta)

        ### this is for distribution lens
        self.logs = [] if logs else None

        # Track execution counts and rewards for each vehicle
        self.counts = {v_id: 0 for v_id in self.problem.vehicles}
        self.avg_rewards = {v_id: 0.0 for v_id in self.problem.vehicles}

        # Determine the explsoration phase length
        self.num_tasks = len(self.problem.tasks)
        self.exploration_rounds = np.ceil(np.sqrt(self.num_tasks))

    def update_history(self, vehicles):
        """Updates the execution history for the given vehicles.

        Args:
            vehicles (Set[int]): Set of vehicle IDs to update.
        """
        for v_id in vehicles:
            self.counts[v_id] += 1
            reward = self.problem.vehicles[v_id].draw_execution_sample()
            self.avg_rewards[v_id] = (
                self.avg_rewards[v_id] * (self.counts[v_id] - 1) + reward
            ) / self.counts[v_id]

    def run(self, task_id: int, task: Task, remained: List[int]):
        """Runs the SqrtFirst algorithm for a given task.

        Args:
            task_id (int): The ID of the task to be executed.
            task (Task): The task instance to be executed.
            remained (List[int]): List of remaining vehicle IDs available for task allocation.
        """
        epsilon_r_j, epsilon_p_j = 100, 100
        recruited = set()

        # exploration phase
        if task_id <= self.exploration_rounds:
            recruited = remained
            self.recruited_vehicles[task_id] = recruited
            self.update_history(recruited)
            if self.logs is not None:
                self.logs.append(self.avg_rewards.copy())
            return

        if self.logs is not None:
            self.logs.append(self.avg_rewards.copy())
        epsilon_r_j, epsilon_p_j = self.compute_epsilon(task_id, remained)
        recruited = set()

        # first `l - 1` initialization is added to improve effiency
        vehicle_scores = [(v, self.problem.vehicles[v].get_arrival_probability(
            task_id) * self.avg_rewards[v] / self.problem.vehicles[v].get_expected_cost(task_id)) for v in remained]
        vehicle_scores.sort(key=lambda x: x[1], reverse=True)
        recruited = {v for v, score in vehicle_scores[:self.problem.l - 1]}

        # in each run, only one observed value is fetched, so a dict can store the result.
        observed_value = {v: self.problem.vehicles[v].draw_execution_sample() for v, score in vehicle_scores[:self.problem.l - 1]}

        # updated and recorded value, the core is psi value
        # and the time complexity is reduced for psi, using DP idea
        h_val = (self.problem.l-1) * (epsilon_r_j + epsilon_p_j)
        psi_val = 0
        h_threshold = self.theta * ((self.problem.l * epsilon_r_j +
                                    self.problem.l * epsilon_p_j +
                                    np.log(self.problem.beta)))

        # every time, the threshold is not met (and the satisfaction is stochatic)
        while h_val < h_threshold:
            # recruit one worker into the pool, and get the observed value (for later iterations)
            best_ratio, best_vehicle, best_h_val_updated = -1, None, .0
            best_psi_val_updated = .0

            for v in remained:
                if v in recruited or self.avg_rewards[v] <= 0:
                    continue

                temp_set = recruited.union({v})

                # improvement for Pr[A >= l]
                psi_val_updated = poisson_binomial_pmf([
                    self.problem.vehicles[v].get_arrival_probability(task_id) * observed_value[v]
                    for v in recruited
                ], self.problem.l - 1) * self.problem.vehicles[v].get_arrival_probability(task_id) * self.avg_rewards[v] + psi_val

                assert psi_val_updated > 0, (psi_val_updated, len(recruited), self.problem.l, f"""
                                             {poisson_binomial_pmf([
                    self.problem.vehicles[v].get_arrival_probability(task_id) * observed_value[v]
                    for v in recruited
                ], self.problem.l - 1)}, 
                {self.problem.vehicles[v].get_arrival_probability(task_id)}, 
                {self.avg_rewards[v]}""")

                # the oracle know the hidden distribiton exactly, while the online framework use
                # a constructed emprical distribution
                expected_h_val_updated = self.problem.l * (epsilon_r_j + epsilon_p_j) + np.log(psi_val_updated)
                expected_delta_h = expected_h_val_updated - h_val
                # assert expected_delta_h > 0, f"log psi_val_updated {np.log(psi_val_updated)}"

                # the best ratio (expected, the online frame should consider the explore and exploit tradeoff)
                ratio = expected_delta_h / self.problem.vehicles[v].get_expected_cost(task_id)
                if ratio > best_ratio and expected_delta_h > 0:
                    best_ratio = ratio
                    best_vehicle = v
                    best_h_val_updated = expected_h_val_updated
                    best_psi_val_updated = psi_val_updated

            if best_vehicle is None:
                break

            # the constraint is dependent on a random variable (realize here)
            observed_value[best_vehicle] = self.problem.vehicles[best_vehicle].draw_execution_sample()
            # print(observed_value)
            psi_val = poisson_binomial_pmf([
                self.problem.vehicles[v].get_arrival_probability(task_id) * observed_value[v]
                for v in recruited
            ], self.problem.l - 1) * self.problem.vehicles[best_vehicle].get_arrival_probability(task_id) * observed_value[best_vehicle] + psi_val
            h_val = self.problem.l * (epsilon_r_j + epsilon_p_j) + np.log(psi_val)
            recruited.add(best_vehicle)
        self.recruited_vehicles[task_id] = recruited
