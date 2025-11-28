from src.algos.offline.base_algorithm import BaseAlgorithm
from src.models.bean import Task
from typing import Dict, Set, List
import numpy as np


class Randomized(BaseAlgorithm):
    def __init__(self, problem):
        super().__init__(problem)

    def run(self, task_id: int, task: Task, remained: List[int]) -> None:
        recruited = set()

        # randomly select a vehicle until the solution is feasible
        while self.problem.compute_psi_oracle(task_id, recruited) < self.problem.beta and len(remained) > 0:
            v = np.random.choice(remained)
            recruited.add(v)
            remained.remove(v)

        # update solution
        self.recruited_vehicles[task_id] = recruited
