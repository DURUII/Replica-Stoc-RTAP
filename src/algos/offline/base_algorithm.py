from abc import ABC, abstractmethod
from typing import Dict, Set, List
import numpy as np
from src.models.bean import Problem, Task, Vehicle


class BaseAlgorithm(ABC):
    def __init__(self, problem: Problem):
        """
        Initialize the base algorithm with a problem instance.

        Args:
            problem (Problem): The problem instance containing tasks and vehicles.
        """
        self.problem = problem
        self.recruited_vehicles: Dict[int, Set[int]] = {task_id: set() for task_id in self.problem.tasks}
        self.total_expected_cost: float = 0.0

    def run_framework(self) -> Dict[int, Set[int]]:
        """
        Execute the algorithm for all tasks in the problem.

        Returns:
            Dict[int, Set[int]]: A dictionary mapping task IDs to sets of recruited vehicle IDs.
        """
        for task_id, task in self.problem.tasks.items():
            # check if there is a feasible solution
            remained = [v_id for v_id in self.problem.vehicles.keys()
                        if self.problem.vehicles[v_id].get_expected_cost(task_id) != float('inf')
                        and self.problem.vehicles[v_id].get_arrival_probability(task_id) > 1e-5
                        and self.problem.vehicles[v_id].get_execution_ability() > 1e-5]

            self.run(task_id, task, remained)
        return self.recruited_vehicles

    @abstractmethod
    def run(self, task_id: int, task: Task, remained: List[int]) -> None:
        """
        Abstract method to execute the algorithm for a single task.

        Args:
            task_id (int): The ID of the task.
            task (Task): The task instance.
        """
        pass

    def get_total_expected_cost(self) -> float:
        """
        Calculate and return the total expected cost of recruited vehicles.

        Returns:
            float: The total expected recruitment cost.
        """
        self.total_expected_cost = self.problem.compute_total_expected_cost(self.recruited_vehicles)
        return self.total_expected_cost
