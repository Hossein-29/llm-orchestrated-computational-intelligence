"""
Continuous Optimization Problems.

Benchmark functions:
- Rastrigin
- Ackley
- Rosenbrock
"""

import numpy as np


class OptimizationProblem:
    """Continuous optimization problem with benchmark functions."""
    
    FUNCTIONS = ["rastrigin", "ackley", "rosenbrock"]
    
    def __init__(self, problem_info: dict):
        """
        Initialize optimization problem.
        
        Args:
            problem_info: Dictionary containing:
                - function_name: Name of benchmark function
                - dimensions: Number of dimensions (10, 20, 30)
                - bounds: Search space bounds
        """
        self.problem_info = problem_info
        self.function_name = problem_info.get("function_name", "rastrigin")
        self.dimensions = problem_info.get("dimensions", 10)
        self.bounds = problem_info.get("bounds", (-5.12, 5.12))
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the objective function at point x.
        
        Args:
            x: Solution vector.
            
        Returns:
            Function value at x.
        """
        if self.function_name == "rastrigin":
            return self._rastrigin(x)
        elif self.function_name == "ackley":
            return self._ackley(x)
        elif self.function_name == "rosenbrock":
            return self._rosenbrock(x)
        else:
            raise ValueError(f"Unknown function: {self.function_name}")
    
    def _rastrigin(self, x: np.ndarray) -> float:
        """
        Rastrigin function.
        Global minimum: f(0,...,0) = 0
        """
        # TODO: Implement Rastrigin function
        # f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
        raise NotImplementedError("_rastrigin not yet implemented")
    
    def _ackley(self, x: np.ndarray) -> float:
        """
        Ackley function.
        Global minimum: f(0,...,0) = 0
        """
        # TODO: Implement Ackley function
        raise NotImplementedError("_ackley not yet implemented")
    
    def _rosenbrock(self, x: np.ndarray) -> float:
        """
        Rosenbrock function.
        Global minimum: f(1,...,1) = 0
        """
        # TODO: Implement Rosenbrock function
        raise NotImplementedError("_rosenbrock not yet implemented")
    
    def get_bounds(self) -> tuple:
        """Get the search space bounds."""
        return self.bounds
    
    def get_known_optimal(self) -> float:
        """Get the known optimal value (0 for all benchmark functions)."""
        return 0.0
