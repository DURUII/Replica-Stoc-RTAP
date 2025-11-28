from src.utils.probability import poisson_binomial_cdf
import numpy as np
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Set, Tuple
from tqdm.auto import tqdm


@dataclass
class Task:
    """
    Represents a task in the system.

    Attributes:
        idx (int): The index of the task.

        idx (int): int: The index of the task.
        The index starts from 0.
    """
    idx: int


@dataclass
class Vehicle:
    """
    Represents a vehicle in the system.

    Attributes:
        idx (int): The index of the vehicle. The index starts from 0.
        p (Dict[int, float]): The probability of vehicle v_i can arrive in PoI of task s_j at time t in [t_{s_j}, t_{e_j}].
        r (float): The execution ability of vehicle v_i to perform tasks or the feedback from the user, reflecting the probability of the vehicle to perform well.
        c (Dict[int, float]): The cost of the vehicle v_i to perform the task s_j.
    """
    idx: int

    # p_{i,j}: The probability of vehicle v_i can arrive in PoI
    # of task s_j at time t in [t_{s_j}, t_{e_j}]
    p: object

    # r_i: The execution ability of vehicle $v_i$ to perform tasks
    # or the feedback from the user, reflecting the probability of
    # the vehicle to perform well.
    r: float

    # c_{i,j}: The cost of the vehicle $v_i$ to perform the task $s_j$
    c: object

    def __post_init__(self):
        assert len(self.p) == len(self.c), f"RuntimeError: shape mismatch: shape [{len(self.p)}] and [{len(self.c)}]"
        # used for a gaussian arm in the online setting (inspired by 'AUCB')
        self.sigma = random.uniform(0, min(self.r / 3, (1 - self.r) / 3))

    def get_arrival_probability(self, task_id: int) -> float:
        """
        Calculate the probability that the vehicle can arrive at the Point of Interest (PoI) of the given task.

        Args:
            task_id (int): The ID of the task.

        Returns:
            float: The probability that the vehicle can arrive at the PoI of the task. 
                   If the task ID is not found, returns a very small probability (1e-7).
        """
        idx = task_id % len(self.c)
        if isinstance(self.p, dict):
            return self.p.get(idx, 1e-7)
        try:
            return float(self.p[idx])
        except Exception:
            return 1e-7

    def draw_execution_sample(self) -> float:
        """ execution feedback may be unknown to the platform in the online setting, 
            and manually truncate the output to 0-1. """
        return min(max(1e-3, random.gauss(self.r, self.sigma)), 1.0)

    def get_execution_ability(self) -> float:
        """ execution ability is known to the offline oracle """
        return self.r

    def get_cost(self, task_id: int) -> float:
        """
        Get the cost of the vehicle to perform the given task.

        Args:
            task_id (int): The ID of the task.

        Returns:
            float: The cost of the vehicle to perform the task. 
                   If the task ID is not found, returns infinity.
        """
        idx = task_id % len(self.c)
        if isinstance(self.c, dict):
            return self.c.get(idx, float('inf'))
        try:
            return float(self.c[idx])
        except Exception:
            return float('inf')

    def get_expected_cost(self, task_id: int) -> float:
        # you can also try this, they are equivalent:
        # > return self.get_cost(task_id) * self.get_arrival_probability(task_id)
        # return self.get_cost(task_id) * self.get_arrival_probability(task_id)
        return self.get_cost(task_id)


class Problem:
    """
    Represents the problem to be solved, including tasks and vehicles.

    Attributes:
        tasks (Dict[int, Task]): A dictionary of tasks indexed by their IDs.
        vehicles (Dict[int, Vehicle]): A dictionary of vehicles indexed by their IDs.
        l (int): The threshold of the number of recruiting vehicles.
        beta (float): The robustness threshold.
    """

    def __init__(self, tasks: List[Task], vehicles: List[Vehicle],
                 l: int, beta: float):
        # the dict of tasks and vehicles
        self.vehicles = {vehicle.idx: vehicle for vehicle in vehicles}
        self.tasks = {task.idx: task for task in tasks}

        # threshold of the number of recruiting vehicle
        self.l = l

        # robustness threshold
        self.beta = beta

    def check_global(self) -> int:
        """
        Check how many tasks can be done theoretically under the current parameter setting.

        Returns:
            int: The number of tasks that can be done.
        """
        return sum([self.check_constraint(j) for j in self.tasks.keys()])

    def check_constraint(self, task_id: int, W_Phi_j: Set[int] = {}, l: float = -1, beta: float = -1) -> bool:
        """
        Check if the jth constraint is satisfied.

        Args:
            task_id (int): The ID of the task.
            W_Phi_j (Set[int], optional): A set of vehicle IDs that are considered for the task. Defaults to an empty set.
            l (float, optional): The threshold of the number of recruiting vehicles. Defaults to -1, which means using the class attribute `self.l`.
            beta (float, optional): The robustness threshold. Defaults to -1, which means using the class attribute `self.beta`.

        Returns:
            bool: True if the constraint is satisfied, False otherwise.
        """
        if l == -1:
            l = self.l

        if beta == -1:
            beta = self.beta

        if not W_Phi_j:
            # filter out the vehicles that cannot satisfy the constraint
            remained = [v_id for v_id in self.vehicles.keys()
                        if self.vehicles[v_id].get_expected_cost(task_id) != float('inf')
                        and self.vehicles[v_id].get_arrival_probability(task_id) > 1e-5]
            W_Phi_j = set(remained)

        # check if the constraint is satisfied under the current setting
        return self.compute_psi_oracle(task_id, W_Phi_j) >= beta

    def compute_psi_oracle(self, task_id, W_Phi_j: Set[int], l: float = None) -> float:
        """
        The jth constraint ψ(W_{Φ}^j, l) = Pr(A(W_{Φ}^j) >= l), 
        in the offline setting, the execution ability is known as a priori.

        Args:
            task_id (int): The ID of the task.
            W_Phi_j (Set[int]): A set of vehicle IDs.
            l (float, optional): The threshold of the number of recruiting vehicles. Defaults to None.

        Returns:
            float: The Psi value.
        """
        l = self.l if l is None else l

        # definitely cannot satisfy the condition
        if len(W_Phi_j) < l:
            return 0.0

        # Pr(A >= l)
        mul_subset = [
            self.vehicles[v_id].get_arrival_probability(task_id) * self.vehicles[v_id].get_execution_ability()
            for v_id in W_Phi_j
        ]
        Pr_A_geq_l = poisson_binomial_cdf(mul_subset, l)

        return Pr_A_geq_l

    def compute_total_expected_cost(self, W_Phi: Dict[int, Set[int]]) -> float:
        """
        Compute the total expected cost (objective) of a solution.

        Args:
            W_Phi (Dict[int, Set[int]]): A dictionary where keys are task IDs and values are sets of vehicle IDs.

        Returns:
            float: The total expected cost.
        """
        total_cost = 0.0
        for j, W_Phi_j in W_Phi.items():
            # a small trick here to re-use the parameter: MOD operation
            total_cost += sum([self.vehicles[v_id].get_expected_cost(j) for v_id in W_Phi_j])
        return total_cost

    def find_optimal_solution(self, task_id: int) -> Tuple[float, Set[int]]:
        """
        Brute-force method to find the optimal solution.
        Returns the minimum cost and the corresponding set of vehicles.

        Args:
            task_id (int): The ID of the task.

        Returns:
            Tuple[float, Set[int]]: The minimum cost and the corresponding set of vehicles.
        """
        W_j = set(self.vehicles.keys())
        # This step is time-inefficient
        search_space = []
        for k in range(self.l, len(W_j)+1):
            search_space.extend([set(combo) for combo in combinations(W_j, k)])

        # Traversal search space, a solution corresponds to a set of vehicles, W_{Phi}^j
        cost_star, solution_star = float('inf'), set()
        for solution in tqdm(search_space):
            psi_val = self.compute_psi_oracle(task_id, solution)
            if psi_val >= self.beta:
                # Compute expected recruitment cost
                # cost = sum([self.vehicles[v_id].get_cost(task_id) * self.vehicles[v_id].get_arrival_probability(task_id)
                #             for v_id in solution])
                cost = sum([self.vehicles[v_id].get_expected_cost(task_id) for v_id in solution])
                if cost < cost_star:
                    cost_star, solution_star = cost, solution
        return cost_star, solution_star
