from src.algos.offline.base_algorithm import BaseAlgorithm
from typing import Dict, Set, Tuple, List
import numpy as np
from src.models.bean import Task, Problem
from src.utils.probability import poisson_binomial_pmf, poisson_binomial_cdf
from src.algos.offline.proposed import SubmodularPolicy


class StochasticGreedyMAB(SubmodularPolicy):
    def __init__(self, problem, k=100, theta: float = 1.0, logs: bool = True):
        super().__init__(problem, theta)

        ### this is for distribution lens
        self.logs = [] if logs else None

        # the finite support of the reward distribution
        # discrete from 0-1, and the number of bins is k
        self.k = k
        self.delta = 2 / (len(self.problem.vehicles) * len(self.problem.tasks))**3

        # observed distribution
        self.observed = {v_id: np.zeros(self.k + 1, dtype=int) for v_id in self.problem.vehicles}
        # optimistic empirical distribution
        self.constructed = {v_id: np.zeros(self.k + 1, dtype=float) for v_id in self.problem.vehicles}

    def update_history(self, vehicles, observed_value=None):
        """ update observed distribution"""
        if observed_value is None:
            for v_id in vehicles:
                reward = self.problem.vehicles[v_id].draw_execution_sample()
                idx = min(int(reward * self.k), self.k)
                self.observed[v_id][idx] += 1
            return
        for v_id, reward in observed_value.items():
            idx = min(int(reward * self.k), self.k)
            self.observed[v_id][idx] += 1

    def earth_mover(self):
        """ update optimistic empirical distribution """
        for v_id in self.observed:
            m = self.observed[v_id].sum()
            if m == 0:
                # forced exploration
                self.constructed[v_id][-1] = 1
                continue
            epsilon = np.sqrt(np.log(2 * self.k / self.delta) / (2 * m))
            r_bar = self.observed[v_id] / m
            self.constructed[v_id][-1] = min(r_bar[-1] + epsilon, 1)
            if self.constructed[v_id][-1] == 1:
                self.constructed[v_id][:-1] = 0
            else:
                prefix = np.cumsum(r_bar)
                y = np.searchsorted(prefix, epsilon)
                for j in range(1, self.k):
                    if j == y:
                        self.constructed[v_id][j] = prefix[j] - epsilon
                    elif j > y:
                        self.constructed[v_id][j] = r_bar[j]

    def run(self, task_id: int, task: Task, remained: List[int]):
        epsilon_r_j, epsilon_p_j = 100, 100

        # Cache static vehicle values to avoid redundant calls.
        arrival_cache = {v: self.problem.vehicles[v].get_arrival_probability(task_id) for v in remained}
        cost_cache = {v: self.problem.vehicles[v].get_expected_cost(task_id) for v in remained}

        recruited = set()
        # for early task, recruit all
        if task_id <= 1:
            recruited = remained
            self.recruited_vehicles[task_id] = recruited
            self.update_history(recruited)
            self.earth_mover()
            if self.logs is not None:
                self.logs.append({k: v.copy() for k, v in self.constructed.items()})
            return

        # initialization for first l-1 vehicles
        self.earth_mover()
        if self.logs is not None:
            self.logs.append({k: v.copy() for k, v in self.constructed.items()})
        scores = []
        for v in remained:
            arrival = arrival_cache[v]
            cost = cost_cache[v]
            cons_quality = np.mean(self.constructed[v])
            scores.append((v, arrival * cons_quality / cost))
        scores.sort(key=lambda x: x[1], reverse=True)
        recruited = {v for v, _ in scores[:self.problem.l - 1]}

        # Draw observed samples for the initially recruited.
        observed_value = {v: self.problem.vehicles[v].draw_execution_sample() for v in recruited}
        h_val = (self.problem.l-1) * (epsilon_r_j + epsilon_p_j)
        psi_val = 0.0
        h_threshold = self.theta * (self.problem.l * (epsilon_r_j + epsilon_p_j) + np.log(self.problem.beta))

        # Vectorized candidate evaluation in the main loop.
        while h_val < h_threshold:
            # recruit one worker into the pool, and get the observed value (for later iterations)
            best_ratio, best_vehicle = -1, None
            for v in remained:
                if v in recruited:
                    continue

                expected_quality = np.mean(self.constructed[v])
                # the best (do not consider cost, arrival)
                # ratio = expected_quality 
                # the best ratio (expected, the online frame should consider the explore and exploit tradeoff)
                ratio = expected_quality / self.problem.vehicles[v].get_expected_cost(task_id)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_vehicle = v

            if best_vehicle is None:
                break

            # the constraint is dependent on a random variable (realize here)
            observed_value[best_vehicle] = self.problem.vehicles[best_vehicle].draw_execution_sample()
            # print(observed_value)
            psi_val = poisson_binomial_pmf([
                self.problem.vehicles[v].get_arrival_probability(task_id) * observed_value[v]
                for v in recruited
            ], self.problem.l - 1) * self.problem.vehicles[best_vehicle].get_arrival_probability(task_id) * observed_value[best_vehicle] + psi_val
            # print(psi_val)
            h_val = self.problem.l * (epsilon_r_j + epsilon_p_j) + np.log(psi_val)
            # print(h_val)
            recruited.add(best_vehicle)
            # update history
            self.update_history([], observed_value)

        self.recruited_vehicles[task_id] = recruited
