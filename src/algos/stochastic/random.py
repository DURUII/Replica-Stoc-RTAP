import random
from src.algos.offline.base_algorithm import BaseAlgorithm
from typing import Dict, Set, Tuple, List
import numpy as np
from src.models.bean import Task, Problem
from src.utils.probability import poisson_binomial_pmf, poisson_binomial_cdf
from src.algos.offline.proposed import SubmodularPolicy


class StochasticRandomized(SubmodularPolicy):
    """Random algorithm for task assignment.

    Inherits from SubmodularPolicy and assigns tasks to vehicles randomly while
    considering certain constraints and thresholds.

    Attributes:
        problem (Problem): The problem instance containing vehicles and tasks.
        theta (float): A threshold parameter.
    """

    def __init__(self, problem, theta: float = 1.0):
        super().__init__(problem, theta)

    def run(self, task_id: int, task: Task, remained: List[int]):
        """Runs the random algorithm for a given task.

        Args:
            task_id (int): The ID of the task to be assigned.
            task (Task): The task to be assigned.
            remained (List[int]): List of remaining vehicle IDs.

        Raises:
            ValueError: If psi_val is zero or negative.
        """
        # initialization
        recruited = set()
        observed_value = {}

        def compute_psi(observed_value: Dict[int, float]) -> float:
            mul_subset = [
                self.problem.vehicles[v_id].get_arrival_probability(task_id) * observed_value[v_id]
                for v_id in recruited
            ]
            return poisson_binomial_cdf(mul_subset, self.problem.l)

        # randomly select a vehicle until the solution is feasible
        while compute_psi(observed_value) < self.problem.beta and len(remained) > 0:
            v_id = np.random.choice(remained)
            recruited.add(v_id)
            observed_value[v_id] = self.problem.vehicles[v_id].draw_execution_sample()
            remained.remove(v_id)

        # update solution
        self.recruited_vehicles[task_id] = recruited
