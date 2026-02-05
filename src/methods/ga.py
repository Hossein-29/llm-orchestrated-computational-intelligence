"""
Genetic Algorithm (GA) implementation.

Supports:
- permutation encoding (TSP, combinatorial)
- continuous encoding (function optimization)

Parameters:
- population_size: int (default: 100)
- generations: int (default: 500)
- crossover_rate: float (default: 0.8)
- mutation_rate: float (default: 0.1)
- elitism: int (default: 2)
- encoding: str (default: "permutation", options: "permutation", "continuous")
"""

import time
from typing import Any

import numpy as np

from src.methods.base import BaseMethod


class GeneticAlgorithm(BaseMethod):
    """Genetic Algorithm for optimization problems."""
    
    def run(self, problem: Any, parameters: dict) -> dict:
        """
        Solve optimization problem using Genetic Algorithm.
        
        Args:
            problem: Problem object with:
                - distance_matrix (for permutation encoding)
                - fitness_function and bounds (for continuous encoding)
            parameters: GA parameters including 'encoding' type.
            
        Returns:
            Result dictionary with best solution and fitness.
        """
        start_time = time.time()
        
        # Get parameters
        defaults = self.get_default_parameters()
        params = {**defaults, **parameters}
        
        pop_size = params["population_size"]
        generations = params["generations"]
        crossover_rate = params["crossover_rate"]
        mutation_rate = params["mutation_rate"]
        elitism = params["elitism"]
        encoding = params["encoding"]
        
        # Set up operators based on encoding parameter
        if encoding == "permutation":
            distances = np.array(problem.distance_matrix)
            n = len(distances)
            
            init_fn = lambda: list(np.random.permutation(n))
            fitness_fn = lambda ind: self._tour_length(ind, distances)
            crossover_fn = self._order_crossover
            mutate_fn = self._swap_mutation
            
        elif encoding == "continuous":
            bounds = np.array(problem.bounds)
            
            init_fn = lambda: list(np.random.uniform(bounds[:, 0], bounds[:, 1]))
            fitness_fn = problem.fitness_function
            crossover_fn = lambda p1, p2: self._blend_crossover(p1, p2, bounds)
            mutate_fn = lambda ind: self._gaussian_mutation(ind, bounds)
        else:
            raise ValueError(f"Unknown encoding: {encoding}. Use 'permutation' or 'continuous'")
        
        # 1. Initialize population
        population = [init_fn() for _ in range(pop_size)]
        fitness = [fitness_fn(ind) for ind in population]
        
        # Track best (minimization)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy() if isinstance(population[best_idx], list) else list(population[best_idx])
        best_fitness = fitness[best_idx]
        self.convergence_history = []
        
        # Main GA loop
        for generation in range(generations):
            new_population = []
            
            # Elitism
            sorted_indices = np.argsort(fitness)
            for i in range(elitism):
                ind = population[sorted_indices[i]]
                new_population.append(ind.copy() if isinstance(ind, list) else list(ind))
            
            # Generate rest of population
            while len(new_population) < pop_size:
                parent1 = self._tournament_select(population, fitness)
                parent2 = self._tournament_select(population, fitness)
                
                if np.random.random() < crossover_rate:
                    child1, child2 = crossover_fn(parent1, parent2)
                else:
                    child1 = parent1.copy() if isinstance(parent1, list) else list(parent1)
                    child2 = parent2.copy() if isinstance(parent2, list) else list(parent2)
                
                if np.random.random() < mutation_rate:
                    mutate_fn(child1)
                if np.random.random() < mutation_rate:
                    mutate_fn(child2)
                
                new_population.append(child1)
                if len(new_population) < pop_size:
                    new_population.append(child2)
            
            population = new_population
            fitness = [fitness_fn(ind) for ind in population]
            
            # Update best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy() if isinstance(population[gen_best_idx], list) else list(population[gen_best_idx])
            
            self.convergence_history.append(best_fitness)
            self._report_progress(generation, best_fitness)
        
        return {
            "best_solution": best_solution,
            "best_fitness": best_fitness,
            "convergence_history": self.convergence_history,
            "computation_time": time.time() - start_time,
            "iterations_completed": generations,
        }
    
    # === Common ===
    def _tournament_select(self, population: list, fitness: list, k: int = 3) -> list:
        """Tournament selection (minimization)."""
        indices = np.random.choice(len(population), k, replace=False)
        best = min(indices, key=lambda i: fitness[i])
        return population[best].copy() if isinstance(population[best], list) else list(population[best])
    
    # === Permutation operators (TSP) ===
    def _tour_length(self, tour: list, distances: np.ndarray) -> float:
        """Calculate TSP tour length."""
        return sum(distances[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    
    def _order_crossover(self, p1: list, p2: list) -> tuple:
        """Order crossover (OX) for permutations."""
        n = len(p1)
        pt1, pt2 = sorted(np.random.choice(n, 2, replace=False))
        
        c1, c2 = [None] * n, [None] * n
        c1[pt1:pt2], c2[pt1:pt2] = p1[pt1:pt2], p2[pt1:pt2]
        
        self._fill_child(c1, p2, pt2)
        self._fill_child(c2, p1, pt2)
        return c1, c2
    
    def _fill_child(self, child: list, parent: list, start: int):
        """Fill remaining positions in OX."""
        n = len(child)
        pos = start
        for i in range(n):
            gene = parent[(start + i) % n]
            if gene not in child:
                while child[pos % n] is not None:
                    pos += 1
                child[pos % n] = gene
    
    def _swap_mutation(self, ind: list):
        """Swap mutation for permutations."""
        i, j = np.random.choice(len(ind), 2, replace=False)
        ind[i], ind[j] = ind[j], ind[i]
    
    # === Continuous operators ===
    def _blend_crossover(self, p1: list, p2: list, bounds: np.ndarray, alpha: float = 0.5) -> tuple:
        """BLX-alpha crossover for continuous."""
        c1, c2 = [], []
        for i in range(len(p1)):
            lo, hi = min(p1[i], p2[i]), max(p1[i], p2[i])
            d = hi - lo
            new_lo, new_hi = lo - alpha * d, hi + alpha * d
            new_lo = max(new_lo, bounds[i, 0])
            new_hi = min(new_hi, bounds[i, 1])
            c1.append(np.random.uniform(new_lo, new_hi))
            c2.append(np.random.uniform(new_lo, new_hi))
        return c1, c2
    
    def _gaussian_mutation(self, ind: list, bounds: np.ndarray, sigma: float = 0.1):
        """Gaussian mutation for continuous."""
        for i in range(len(ind)):
            if np.random.random() < 0.2:  # per-gene probability
                range_i = bounds[i, 1] - bounds[i, 0]
                ind[i] += np.random.normal(0, sigma * range_i)
                ind[i] = np.clip(ind[i], bounds[i, 0], bounds[i, 1])
    
    def get_default_parameters(self) -> dict:
        """Get default GA parameters."""
        return {
            "population_size": 100,
            "generations": 500,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elitism": 2,
            "encoding": "permutation",  # "permutation" or "continuous"
        }
    
    def validate_parameters(self, parameters: dict) -> bool:
        """Validate GA parameters."""
        try:
            pop_size = parameters.get("population_size", 100)
            generations = parameters.get("generations", 500)
            crossover_rate = parameters.get("crossover_rate", 0.8)
            mutation_rate = parameters.get("mutation_rate", 0.1)
            elitism = parameters.get("elitism", 2)
            encoding = parameters.get("encoding", "permutation")
            
            if not isinstance(pop_size, int) or pop_size < 2:
                return False
            if not isinstance(generations, int) or generations < 1:
                return False
            if not (0 <= crossover_rate <= 1):
                return False
            if not (0 <= mutation_rate <= 1):
                return False
            if not isinstance(elitism, int) or elitism < 0 or elitism >= pop_size:
                return False
            if encoding not in ["permutation", "continuous"]:
                return False
            
            return True
        except Exception:
            return False
