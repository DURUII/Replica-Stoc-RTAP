from src.algos.offline.base_algorithm import BaseAlgorithm
from src.models.bean import Task
from typing import Dict, Set, List
import numpy as np


class MinimumCostGreedy(BaseAlgorithm):
    def __init__(self, problem):
        super().__init__(problem)

    def run(self, task_id: int, task: Task, remained: List[int]) -> None:
        recruited = set()

        # sort vehicles by ascending order of expected cost
        remained.sort(key=lambda v_id: self.problem.vehicles[v_id].get_expected_cost(task_id))

        # add vehicles one by one with the minimum cost until the solution is feasible
        for v in remained:
            if self.problem.compute_psi_oracle(task_id, recruited) >= self.problem.beta:
                break
            recruited.add(v)

        # update solution
        self.recruited_vehicles[task_id] = recruited
