from src.algos.offline.base_algorithm import BaseAlgorithm
from typing import Dict, Set, Tuple, List
import numpy as np
from src.models.bean import Task, Problem
from src.utils.probability import poisson_binomial_pmf, poisson_binomial_cdf


class SubmodularPolicy(BaseAlgorithm):
    def __init__(self, problem: Problem, theta: float = 1.0):
        super().__init__(problem)
        # arbitrary value, since it is devised for theoretical analysis
        self.theta = theta

    def compute_epsilon(self, task_id: int, remained: List[int]) -> Tuple[float, float]:
        pp = np.array([self.problem.vehicles[v].get_arrival_probability(task_id) for v in remained])
        rr = np.array([self.problem.vehicles[v].get_execution_ability() for v in remained])
        epsilon_r_j = -2 * np.sum(np.log(rr[np.argsort(rr)[:self.problem.l]]))
        epsilon_p_j = -2 * np.sum(np.log(pp[np.argsort(pp)[:self.problem.l]]))
        return epsilon_r_j, epsilon_p_j

    def compute_h(self, task_id: int, W_Phi_j: Set[int], remained: List[int]) -> float:
        epsilon_r_j, epsilon_p_j = self.compute_epsilon(task_id, remained)
        if len(W_Phi_j) < self.problem.l:
            return len(W_Phi_j) * (epsilon_r_j + epsilon_p_j)
        psi_val = self.problem.compute_psi_oracle(task_id, W_Phi_j)
        return self.problem.l * (epsilon_r_j + epsilon_p_j) + np.log(psi_val)

    def run(self, task_id: int, task: Task, remained: List[int]) -> None:
        epsilon_r_j, epsilon_p_j = self.compute_epsilon(task_id, remained)
        recruited = set()
        h_val = self.compute_h(task_id, recruited, remained)
        h_threshold = self.theta * ((self.problem.l * epsilon_r_j +
                                     self.problem.l * epsilon_p_j +
                                     np.log(self.problem.beta)))

        while h_val < h_threshold:
            best_ratio, best_vehicle = -1, None

            for v in remained:
                if v in recruited:
                    continue

                temp_set = recruited.union({v})
                delta_h = self.compute_h(task_id, temp_set, remained) - h_val
                assert delta_h > 0, delta_h

                ratio = delta_h / self.problem.vehicles[v].get_expected_cost(task_id)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_vehicle = v

            # all vehicles have been recruited
            if best_vehicle is None:
                break

            recruited.add(best_vehicle)
            h_val = self.compute_h(task_id, recruited, remained)

        self.recruited_vehicles[task_id] = recruited
