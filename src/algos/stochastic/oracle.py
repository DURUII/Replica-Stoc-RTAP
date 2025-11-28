from src.algos.offline.base_algorithm import BaseAlgorithm
from typing import Dict, Set, Tuple, List
import numpy as np
from src.models.bean import Task, Problem
from src.utils.probability import poisson_binomial_pmf, poisson_binomial_cdf
from src.algos.offline.proposed import SubmodularPolicy


class Oracle(SubmodularPolicy):
    """Oracle class that extends SubmodularPolicy to implement a specific algorithm for task assignment.

    Note: The main difference between the offline and the online setting is that
    we do not use `get_execution_ability` but use `get_execution_sample`
    to simulate the stochastic feedback from the user.
    """

    def __init__(self, problem: Problem, theta: float = 1.0):
        """
        Initialize the Oracle with a problem instance and a threshold value.

        Args:
            problem (Problem): The problem instance containing tasks and vehicles.
            theta (float): The threshold value for the algorithm.
        """
        super().__init__(problem, theta)

    def run(self, task_id: int, task: Task, remained: List[int]):
        """
        Run the Oracle algorithm for a specific task.

        Args:
            task_id (int): The ID of the task to be executed.
            task (Task): The task instance.
            remained (List[int]): List of remaining vehicle IDs.
        """
        epsilon_r_j, epsilon_p_j = 100, 100
        recruited = set()

        # initialization is added from SubmodularPolicy to MCTA_PI to improve efficiency
        vehicle_scores = [
            (v, self.problem.vehicles[v].get_arrival_probability(task_id) *
             self.problem.vehicles[v].get_execution_ability() /
             self.problem.vehicles[v].get_expected_cost(task_id))
            for v in remained
        ]
        vehicle_scores.sort(key=lambda x: x[1], reverse=True)
        recruited = {v for v, score in vehicle_scores[:self.problem.l - 1]}

        # in each run, only one observed value is fetched, so a dict can store the result.
        observed_value = {
            v: self.problem.vehicles[v].draw_execution_sample()
            for v, score in vehicle_scores[:self.problem.l - 1]
        }

        # updated and recorded value, the core is psi value
        h_val = (self.problem.l - 1) * (epsilon_r_j + epsilon_p_j)
        psi_val = 0
        h_threshold = self.theta * (
            (self.problem.l * epsilon_r_j +
             self.problem.l * epsilon_p_j +
             np.log(self.problem.beta))
        )

        # every time, the threshold is not met (and the satisfaction is stochastic)
        while h_val < h_threshold:
            best_ratio, best_vehicle, best_h_val_updated = -1, None, .0
            best_psi_val_updated = .0

            for v in remained:
                if v in recruited:
                    continue

                temp_set = recruited.union({v})

                # improvement for Pr[A >= l]
                psi_val_updated = poisson_binomial_pmf([
                    self.problem.vehicles[v].get_arrival_probability(task_id) * observed_value[v]
                    for v in recruited
                ], self.problem.l - 1) * self.problem.vehicles[v].get_arrival_probability(task_id) * self.problem.vehicles[v].get_execution_ability() + psi_val

                # the oracle know the hidden distribution exactly, while the online framework use
                # a constructed empirical distribution
                expected_h_val_updated = self.problem.l * (epsilon_r_j + epsilon_p_j) + np.log(psi_val_updated)
                expected_delta_h = expected_h_val_updated - h_val
                # assert expected_delta_h > 0, expected_delta_h

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
            psi_val = poisson_binomial_pmf([
                self.problem.vehicles[v].get_arrival_probability(task_id) * observed_value[v]
                for v in recruited
            ], self.problem.l - 1) * self.problem.vehicles[best_vehicle].get_arrival_probability(task_id) * observed_value[best_vehicle] + psi_val
            h_val = self.problem.l * (epsilon_r_j + epsilon_p_j) + np.log(psi_val)
            recruited.add(best_vehicle)

        self.recruited_vehicles[task_id] = recruited
