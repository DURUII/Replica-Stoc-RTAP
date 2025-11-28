from src.algos.offline.base_algorithm import BaseAlgorithm
from typing import Dict, Set, Tuple, List
import numpy as np
from src.models.bean import Task, Problem
from src.utils.probability import poisson_binomial_pmf, poisson_binomial_cdf
from src.algos.offline.proposed import SubmodularPolicy


class SubmodularPolicyImproved(SubmodularPolicy):
    def __init__(self, problem, theta: float = 1.0):
        super().__init__(problem, theta)

    def run(self, task_id: int, task: Task, remained: List[int]):
        epsilon_r_j, epsilon_p_j = self.compute_epsilon(task_id, remained)
        recruited = set()

        # initialization is added to improve effiency
        vehicle_scores = [(v, self.problem.vehicles[v].get_arrival_probability(
            task_id) * self.problem.vehicles[v].get_execution_ability() / self.problem.vehicles[v].get_expected_cost(task_id)) for v in remained]
        vehicle_scores.sort(key=lambda x: x[1], reverse=True)
        recruited = {v for v, score in vehicle_scores[:self.problem.l]}

        # if the best l vehicles do not meet the requirement then
        h_val = self.compute_h(task_id, recruited, remained)
        h_threshold = self.theta * ((self.problem.l * epsilon_r_j +
                                    self.problem.l * epsilon_p_j +
                                    np.log(self.problem.beta)))

        psi_val = poisson_binomial_cdf([
            self.problem.vehicles[v].get_arrival_probability(task_id) * self.problem.vehicles[v].get_execution_ability()
            for v in recruited], self.problem.l)

        while h_val < h_threshold:
            best_ratio, best_vehicle, best_h_val_updated = -1, None, .0
            best_psi_val_updated = .0

            for v in remained:
                if v in recruited:
                    continue

                temp_set = recruited.union({v})

                # improvement for Pr[A >= l]
                psi_val_updated = poisson_binomial_pmf([
                    self.problem.vehicles[v].get_arrival_probability(task_id) * self.problem.vehicles[v].get_execution_ability()
                    for v in recruited
                ], self.problem.l - 1) * self.problem.vehicles[v].get_arrival_probability(task_id) * self.problem.vehicles[v].get_execution_ability() + psi_val

                h_val_updated = self.problem.l * (epsilon_r_j + epsilon_p_j) + np.log(psi_val_updated)
                delta_h = h_val_updated - h_val
                assert delta_h > 0, delta_h

                ratio = delta_h / self.problem.vehicles[v].get_expected_cost(task_id)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_vehicle = v

                    best_h_val_updated = h_val_updated
                    best_psi_val_updated = psi_val_updated

            if best_vehicle is None:
                break

            recruited.add(best_vehicle)
            h_val = best_h_val_updated
            psi_val = best_psi_val_updated
        self.recruited_vehicles[task_id] = recruited

    def get_total_penalty(self) -> float:
        """
        Calculate and return the penalty of violation of the constraint.
        """
        ans = 0.0
        for task_id, recruited in self.recruited_vehicles.items():
            remained = [v_id for v_id in self.problem.vehicles.keys()
                        if self.problem.vehicles[v_id].get_expected_cost(task_id) != float('inf')
                        and self.problem.vehicles[v_id].get_arrival_probability(task_id) > 1e-5]

            epsilon_r_j, epsilon_p_j = self.compute_epsilon(task_id, remained)
            h_threshold = self.theta * ((self.problem.l * epsilon_r_j +
                                         self.problem.l * epsilon_p_j +
                                         np.log(self.problem.beta)))
            h_val = self.compute_h(task_id, recruited, remained)
            ans += max(0, (h_threshold - h_val))
        return ans
