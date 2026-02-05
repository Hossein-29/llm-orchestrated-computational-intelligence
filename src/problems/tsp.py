"""
Traveling Salesman Problem (TSP) implementation.

Test instances: n = 10, 20, 30, 50, 100 cities
"""


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

