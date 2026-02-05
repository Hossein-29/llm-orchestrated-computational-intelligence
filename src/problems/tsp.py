"""
Traveling Salesman Problem (TSP) implementation.

Test instances: n = 10, 20, 30, 50, 100 cities
"""

import numpy as np


class TSPProblem:
    """Traveling Salesman Problem."""
    
    def __init__(self, problem_info: dict):
        """
        Initialize TSP problem.
        
        Args:
            problem_info: Dictionary containing:
                - n_cities: Number of cities
                - distance_matrix: 2D array of distances (optional)
                - coordinates: City coordinates for generating distances (optional)
        """
        self.problem_info = problem_info
        self.n_cities = problem_info.get("n_cities", 10)
        self.distance_matrix = problem_info.get("distance_matrix")
        self.known_optimal = problem_info.get("known_optimal")
        
    def generate_random_instance(self, n_cities: int, seed: int = None) -> np.ndarray:
        """
        Generate a random TSP instance.
        
        Args:
            n_cities: Number of cities.
            seed: Random seed for reproducibility.
            
        Returns:
            Distance matrix.
        """
        # TODO: Implement random instance generation
        # 1. Generate random city coordinates
        # 2. Compute Euclidean distance matrix
        raise NotImplementedError("generate_random_instance not yet implemented")
    
    def evaluate(self, tour: list) -> float:
        """
        Evaluate the total distance of a tour.
        
        Args:
            tour: List of city indices representing the tour.
            
        Returns:
            Total tour distance.
        """
        # TODO: Implement tour evaluation
        raise NotImplementedError("evaluate not yet implemented")
    
    def get_distance_matrix(self) -> np.ndarray:
        """Get the distance matrix for this instance."""
        # TODO: Return or generate distance matrix
        raise NotImplementedError("get_distance_matrix not yet implemented")
