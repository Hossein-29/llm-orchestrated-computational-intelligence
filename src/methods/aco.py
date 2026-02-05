"""
Ant Colony Optimization (ACO) implementation.

Parameters:
- n_ants: int (default: 30)
- alpha: float (default: 1.0, pheromone importance)
- beta: float (default: 2.0, heuristic importance)
- evaporation_rate: float (default: 0.5)
- iterations: int (default: 500)
- q: float (default: 100, pheromone deposit factor)
"""

import time
from typing import Any

import numpy as np

from src.methods.base import BaseMethod


class ACO(BaseMethod):
    """Ant Colony Optimization for combinatorial optimization (e.g., TSP)."""
    
    def run(self, problem: Any, parameters: dict) -> dict:
        """
        Solve optimization problem using ACO.
        
        Args:
            problem: Combinatorial optimization problem (e.g., TSP).
            parameters: ACO parameters.
            
        Returns:
            Result dictionary with best tour and length.
        """
        start_time = time.time()
        
        # Get parameters with defaults
        defaults = self.get_default_parameters()
        params = {**defaults, **parameters}
        
        n_ants = params["n_ants"]
        alpha = params["alpha"]
        beta = params["beta"]
        evaporation_rate = params["evaporation_rate"]
        iterations = params["iterations"]
        q = params.get("q", 100)
        
        # Get distance matrix from problem
        if hasattr(problem, 'distance_matrix') and problem.distance_matrix is not None:
            distances = np.array(problem.distance_matrix)
        else:
            raise ValueError("Problem must have a distance_matrix attribute")
        
        n_cities = len(distances)
        
        # Compute heuristic information (1/distance)
        with np.errstate(divide='ignore'):
            heuristic = 1.0 / distances
            heuristic[distances == 0] = 0
        
        # 1. Initialize pheromone trails
        pheromones = np.ones((n_cities, n_cities))
        
        # Track best solution
        best_tour = None
        best_length = float('inf')
        self.convergence_history = []
        
        # Main ACO loop
        for iteration in range(iterations):
            all_tours = []
            all_lengths = []
            
            # 2. Construct solutions for each ant
            for _ in range(n_ants):
                tour = self._construct_tour(n_cities, pheromones, heuristic, alpha, beta)
                length = self._calculate_tour_length(tour, distances)
                all_tours.append(tour)
                all_lengths.append(length)
                
                if length < best_length:
                    best_length = length
                    best_tour = tour.copy()
            
            # 3. Update pheromones (evaporation + deposit)
            pheromones *= (1 - evaporation_rate)
            
            # All ants deposit pheromone
            for tour, length in zip(all_tours, all_lengths):
                deposit = q / length
                for i in range(n_cities):
                    from_city = tour[i]
                    to_city = tour[(i + 1) % n_cities]
                    pheromones[from_city, to_city] += deposit
                    pheromones[to_city, from_city] += deposit
            
            # 4. Track best solution
            self.convergence_history.append(best_length)
            self._report_progress(iteration, best_length)
        
        computation_time = time.time() - start_time
        
        return {
            "best_solution": best_tour,
            "best_fitness": best_length,
            "convergence_history": self.convergence_history,
            "computation_time": computation_time,
            "iterations_completed": iterations,
        }
    
    def _construct_tour(
        self, n_cities: int, pheromones: np.ndarray, 
        heuristic: np.ndarray, alpha: float, beta: float
    ) -> list:
        """Construct a tour for one ant using probabilistic selection."""
        tour = []
        visited = set()
        
        # Start from a random city
        current = np.random.randint(n_cities)
        tour.append(current)
        visited.add(current)
        
        # Visit all other cities
        while len(tour) < n_cities:
            # Calculate probabilities for unvisited cities
            probs = np.zeros(n_cities)
            for j in range(n_cities):
                if j not in visited:
                    probs[j] = (pheromones[current, j] ** alpha) * (heuristic[current, j] ** beta)
            
            # Normalize probabilities
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs /= prob_sum
            else:
                # If all probs are zero, choose uniformly from unvisited
                for j in range(n_cities):
                    if j not in visited:
                        probs[j] = 1.0
                probs /= probs.sum()
            
            # Select next city
            next_city = np.random.choice(n_cities, p=probs)
            tour.append(next_city)
            visited.add(next_city)
            current = next_city
        
        return tour
    
    def _calculate_tour_length(self, tour: list, distances: np.ndarray) -> float:
        """Calculate total length of a tour."""
        length = 0.0
        n = len(tour)
        for i in range(n):
            length += distances[tour[i], tour[(i + 1) % n]]
        return length
    
    def get_default_parameters(self) -> dict:
        """Get default ACO parameters."""
        return {
            "n_ants": 30,
            "alpha": 1.0,
            "beta": 2.0,
            "evaporation_rate": 0.5,
            "iterations": 500,
            "q": 100,
        }
    
    def validate_parameters(self, parameters: dict) -> bool:
        """Validate ACO parameters."""
        try:
            n_ants = parameters.get("n_ants", 30)
            alpha = parameters.get("alpha", 1.0)
            beta = parameters.get("beta", 2.0)
            evaporation_rate = parameters.get("evaporation_rate", 0.5)
            iterations = parameters.get("iterations", 500)
            q = parameters.get("q", 100)
            
            if not isinstance(n_ants, int) or n_ants < 1:
                return False
            if not isinstance(alpha, (int, float)) or alpha < 0:
                return False
            if not isinstance(beta, (int, float)) or beta < 0:
                return False
            if not isinstance(evaporation_rate, (int, float)) or not (0 <= evaporation_rate <= 1):
                return False
            if not isinstance(iterations, int) or iterations < 1:
                return False
            if not isinstance(q, (int, float)) or q <= 0:
                return False
            
            return True
        except Exception:
            return False
