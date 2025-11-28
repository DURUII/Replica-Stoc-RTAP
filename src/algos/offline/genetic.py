"""
Version: 2.0
Change summary: 
This release fixes premature convergence and reduces cost inflation. 
It removes the incorrect “perfect solution” early stop (`best_fitness < constraint_penalty`) 
and replaces “stop at first feasible” with patience-based early stopping on the **best feasible cost**. 
It adds **elitism** to preserve top individuals each generation and introduces a **cost-aware repair** 
that greedily maximizes marginal robustness gain (Δψ) per unit cost (fallback to cheapest if Δψ=0), 
instead of random additions. Mutation now uses an adaptive flip probability based on genome length; 
vehicle costs are cached per task to avoid recomputation; and the initial population mixes sparse/dense 
genomes to improve diversity. Public API and DEAP wiring remain unchanged. 
In practice, this version yields cheaper feasible solutions 
than the previous GA and the “random until feasible” baseline, 
especially when penalties previously caused early, non-feasible minima to be repaired expensively.

Version: 2.1
Change summary: 
This release adds dynamic penalty (Adaptive Penalty) to the genetic algorithm.
"""

from src.algos.offline.base_algorithm import BaseAlgorithm
from src.models.bean import Task
from typing import Dict, Set, List
import numpy as np
import random
from deap import base, creator, tools
from tqdm.auto import tqdm


class GeneticAlgorithm(BaseAlgorithm):
    """
    Genetic Algorithm for robust task assignment problem.

    Uses evolutionary optimization to find near-optimal vehicle sets that minimize
    cost while satisfying robustness constraints. Inherits from BaseAlgorithm
    for compatibility with existing framework.
    """

    def __init__(self, problem, population_size=100 # 100 for offline.py SYN N<=1000
                , generations=200,
                 cx_prob=0.7, mut_prob=0.1, tournament_size=3,
                 initial_penalty=100,          # 惩罚系数初始值
                 penalty_adapt_factor=1.2,       # 惩罚系数调整（乘/除）因子
                 feasible_ratio_low=0.2,         # 可行解比例下限
                 feasible_ratio_high=0.5,        # 可行解比例上限
                 penalty_min_bound=1.0,          # 惩罚系数最小约束
                 penalty_max_bound=1e9,          # 惩罚系数最大约束
                 random_seed=42,
                 patience=30, elites=2):
        """
        Initialize the Genetic Algorithm.

        Args:
            problem: Problem instance with tasks, vehicles, and parameters
            population_size: Size of the population (default: 100)
            generations: Number of generations to evolve (default: 200)
            cx_prob: Probability of crossover (default: 0.7)
            mut_prob: Probability of mutation (default: 0.1)
            tournament_size: Size of tournament for selection (default: 3)
            constraint_penalty: Penalty for constraint violations (default: 1000.0)
            random_seed: Random seed for reproducibility (default: 42)
            patience: Early stopping patience (generations without feasible best improvement)
            elites: Number of elites to carry over each generation
        """
        super().__init__(problem)
        self.population_size = population_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.tournament_size = tournament_size
        # self.constraint_penalty = constraint_penalty
        # Store adaptive penalty parameters
        self.initial_penalty = initial_penalty
        self.penalty_adapt_factor = penalty_adapt_factor
        self.feasible_ratio_low = feasible_ratio_low
        self.feasible_ratio_high = feasible_ratio_high
        self.penalty_min_bound = penalty_min_bound
        self.penalty_max_bound = penalty_max_bound
        
        # This will be reset in run()
        self.current_penalty_factor = self.initial_penalty

        self.patience = max(0, patience)
        self.elites = max(0, min(elites, population_size))

        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Store current task context for fitness evaluation
        self.current_task_id = None
        self.current_task = None
        self.remained_vehicles: List[int] = None

        # Per-run caches
        self._cost_cache: Dict[int, float] = {}

        # Setup DEAP framework
        self._setup_deap()

    def _setup_deap(self):
        """Setup DEAP framework for genetic operations."""
        # Clean up existing creator classes to avoid warnings
        self.cleanup()

        # Create fitness class (minimization)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # Create individual class
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Toolbox setup
        self.toolbox = base.Toolbox()

        # Attribute generator: binary gene
        self.toolbox.register("attr_bool", random.randint, 0, 1)

        # Structure initializers: individual and population
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_bool)
        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("select", tools.selTournament,
                              tournsize=self.tournament_size)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

        # Register fitness evaluation
        self.toolbox.register("evaluate", self._evaluate_fitness)

    # --------- Helpers ---------

    def _cost(self, vehicle_id: int) -> float:
        """Cached expected cost for current task."""
        if vehicle_id not in self._cost_cache:
            self._cost_cache[vehicle_id] = self.problem.vehicles[vehicle_id].get_expected_cost(self.current_task_id)
        return self._cost_cache[vehicle_id]

    def _ind_to_set(self, individual: List[int]) -> Set[int]:
        """Convert binary genome to set of recruited vehicle IDs."""
        recruited = set()
        for i, gene in enumerate(individual):
            if gene == 1 and i < len(self.remained_vehicles):
                recruited.add(self.remained_vehicles[i])
        return recruited

    # --------- Fitness / Repair ---------

    def _evaluate_fitness(self, individual):
        """
        Evaluate fitness of an individual (vehicle selection).

        Returns:
            tuple: (total_cost + penalty)
        """
        recruited = self._ind_to_set(individual)

        # Calculate total expected cost
        total_cost = 0.0
        for vehicle_id in recruited:
            total_cost += self._cost(vehicle_id)

        # Calculate robustness constraint satisfaction
        psi_value = self.problem.compute_psi_oracle(self.current_task_id, recruited)

        # Apply penalty if constraint not satisfied
        if psi_value < self.problem.beta:
            # 使用动态的惩罚系数
            penalty = self.current_penalty_factor * (self.problem.beta - psi_value)
            total_cost += penalty

        return (total_cost,)

    def _repair_individual(self, individual):
        """
        Cost-aware repair to push an individual to feasibility with minimal extra cost.

        Strategy:
          - While psi < beta:
              * For each available vehicle j:
                   Δpsi_j = psi(S ∪ {j}) - psi(S)
                   score_j = Δpsi_j / (cost_j + 1e-12)
              * Pick j with max score_j (tie -> cheaper).
              * If all Δpsi_j == 0: pick the cheapest j.
        """
        recruited = self._ind_to_set(individual)

        available_indices = [i for i, gene in enumerate(individual)
                             if gene == 0 and i < len(self.remained_vehicles)]

        # Quick exit
        if self.problem.compute_psi_oracle(self.current_task_id, recruited) >= self.problem.beta:
            return

        # Iteratively add vehicles until feasible or none left
        while available_indices:
            current_psi = self.problem.compute_psi_oracle(self.current_task_id, recruited)
            if current_psi >= self.problem.beta:
                break

            # Compute marginal gains
            best_idx = None
            best_score = -1.0
            best_cost = float('inf')
            best_delta = 0.0

            for idx in available_indices:
                vid = self.remained_vehicles[idx]
                c = self._cost(vid)

                # marginal psi
                new_psi = self.problem.compute_psi_oracle(self.current_task_id, recruited | {vid})
                delta = new_psi - current_psi

                # score: delta per cost
                score = (delta / (c + 1e-12)) if delta > 0 else -1.0

                if delta > 0:
                    if score > best_score or (abs(score - best_score) < 1e-12 and c < best_cost):
                        best_score = score
                        best_idx = idx
                        best_cost = c
                        best_delta = delta

            if best_idx is None:
                # No positive gain: fall back to cheapest available
                cheapest_idx = min(available_indices,
                                   key=lambda k: self._cost(self.remained_vehicles[k]))
                chosen_idx = cheapest_idx
            else:
                chosen_idx = best_idx

            # Add the chosen vehicle
            individual[chosen_idx] = 1
            recruited.add(self.remained_vehicles[chosen_idx])
            available_indices.remove(chosen_idx)

        # end while; if不可行也到头了，交给兜底

    # --------- Main Run ---------

    def run(self, task_id: int, task: Task, remained: List[int]) -> None:
        """
        Run genetic algorithm for a single task.

        Args:
            task_id: ID of the current task
            task: Task instance
            remained: List of available vehicle IDs
        """

        # 重置当前任务的惩罚系数
        self.current_penalty_factor = self.initial_penalty
        
        # Set current task context for fitness evaluation
        self.current_task_id = task_id
        self.current_task = task
        self.remained_vehicles = list(remained)  # ensure we don't alias external list
        self._cost_cache = {}

        if not self.remained_vehicles:
            self.recruited_vehicles[task_id] = set()
            return

        genome_len = len(self.remained_vehicles)
        # Slightly adaptive mutation prob (if希望): keep user-provided if not None
        if genome_len > 0:
            indpb = min(0.5, max(1.0 / genome_len, 0.01))
            # re-register mutate with adaptive indpb
            self.toolbox.unregister("mutate")
            self.toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)

        # Create initial population with dynamic individual size
        population = []
        for _ in tqdm(range(self.population_size), desc=f'GA[{task_id}] init', leave=False):
            # Initialize with a mix of sparse/dense patterns to improve diversity
            if random.random() < 0.5:
                # sparse
                individual = [1 if random.random() < 0.2 else 0 for _ in range(genome_len)]
            else:
                # random 0/1
                individual = [random.randint(0, 1) for _ in range(genome_len)]
            population.append(creator.Individual(individual))

        # Evaluate initial population
        fitnesses = [self.toolbox.evaluate(ind) for ind in tqdm(population, desc=f'GA[{task_id}] eval init', leave=False)]
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Track global best feasible (by pure cost)
        best_feasible_ind = None
        best_feasible_cost = float('inf')
        best_feasible_gen = -1

        def is_feasible(ind) -> bool:
            S = self._ind_to_set(ind)
            return self.problem.compute_psi_oracle(self.current_task_id, S) >= self.problem.beta

        def cost_of(ind) -> float:
            return sum(self._cost(v) for v in self._ind_to_set(ind))

        # Initialize best feasible from initial pop
        for ind in population:
            if is_feasible(ind):
                c = cost_of(ind)
                if c < best_feasible_cost:
                    best_feasible_ind = self.toolbox.clone(ind)
                    best_feasible_cost = c
                    best_feasible_gen = 0

        no_improve = 0

        # Run evolution
        for gen in tqdm(range(1, self.generations + 1), total=self.generations, desc=f'GA[{task_id}] evolve', leave=False):
            # ---- Elitism: keep top elites by fitness ----
            elites = tools.selBest(population, self.elites) if self.elites > 0 else []

            # Select and clone
            offspring = self.toolbox.select(population, len(population) - len(elites))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob and len(child1) > 1 and len(child2) > 1:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Next generation = elites + offspring
            population = elites + offspring

            # 在评估新一代后，更新下一次评估要用的惩罚系数
            num_feasible = 0
            for ind in population:
                if is_feasible(ind):
                    num_feasible += 1
            
            feasible_ratio = num_feasible / len(population)

            if feasible_ratio < self.feasible_ratio_low:
                # 可行解太少，惩罚过高，降低惩罚
                self.current_penalty_factor /= self.penalty_adapt_factor
            elif feasible_ratio > self.feasible_ratio_high:
                # 可行解（可能）太多，惩罚不够，增加惩罚
                self.current_penalty_factor *= self.penalty_adapt_factor
            
            # 约束惩罚系数在合理范围内
            self.current_penalty_factor = max(self.penalty_min_bound, 
                                               min(self.current_penalty_factor, self.penalty_max_bound))

            # Update best feasible
            improved = False
            for ind in population:
                # lightweight screen: only check feasibility if fitness looks small enough
                if is_feasible(ind):
                    c = cost_of(ind)
                    if c < best_feasible_cost - 1e-12:
                        best_feasible_cost = c
                        best_feasible_ind = self.toolbox.clone(ind)
                        best_feasible_gen = gen
                        improved = True

            if improved:
                no_improve = 0
            else:
                no_improve += 1

            # Early stopping based on patience (only when we already have a feasible)
            if best_feasible_ind is not None and self.patience > 0 and no_improve >= self.patience:
                break

        # Prefer best feasible individual if found; otherwise repair global best by fitness
        final_ind = None
        if best_feasible_ind is not None:
            final_ind = best_feasible_ind
        else:
            # Use best by fitness and try to repair
            final_ind = tools.selBest(population, 1)[0]
            # Try cost-aware repair to reach feasibility
            self._repair_individual(final_ind)
            if not is_feasible(final_ind):
                # Fallback: greedy
                recruited = self._fallback_greedy(task_id, self.remained_vehicles)
                self.recruited_vehicles[task_id] = recruited
                return

        # Convert final individual to vehicle set
        recruited = self._ind_to_set(final_ind)

        # Update solution
        self.recruited_vehicles[task_id] = recruited

    # --------- Fallback Greedy ---------

    def _fallback_greedy(self, task_id: int, remained: List[int]) -> Set[int]:
        """
        Fallback greedy method if genetic algorithm fails.

        Args:
            task_id: ID of the current task
            remained: List of available vehicle IDs

        Returns:
            Set of recruited vehicle IDs
        """
        recruited = set()
        # Sort by cost and add until constraint satisfied
        remained_sorted = sorted(remained, key=lambda v: self.problem.vehicles[v].get_expected_cost(task_id))

        for vehicle_id in remained_sorted:
            if self.problem.compute_psi_oracle(task_id, recruited) >= self.problem.beta:
                break
            recruited.add(vehicle_id)

        return recruited

    # --------- Cleanup ---------

    def cleanup(self):
        """Clean up DEAP creator classes to avoid conflicts."""
        # Check if creator classes exist in the module and remove them
        if hasattr(creator, 'FitnessMin'):
            del creator.FitnessMin
        if hasattr(creator, 'Individual'):
            del creator.Individual

        # Also clear from creator's internal registry if it exists
        if hasattr(creator, '_creator'):
            if 'FitnessMin' in creator._creator:
                del creator._creator['FitnessMin']
            if 'Individual' in creator._creator:
                del creator._creator['Individual']