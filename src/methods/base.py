"""
Base interface for all CI methods.

All methods must implement this standardized interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable


class BaseMethod(ABC):
    """Abstract base class for all CI methods."""
    
    def __init__(self):
        """Initialize the method."""
        self.name = self.__class__.__name__
        self.convergence_history = []
        
    @abstractmethod
    def run(self, problem: Any, parameters: dict) -> dict:
        """
        Execute the method on the given problem.
        
        Args:
            problem: Problem instance to solve.
            parameters: Method-specific parameters.
            
        Returns:
            Result dictionary containing:
                - best_solution: The best solution found
                - best_fitness: Fitness of the best solution
                - convergence_history: List of fitness values over iterations
                - computation_time: Time taken in seconds
                - iterations_completed: Number of iterations run
        """
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> dict:
        """
        Get default parameters for this method.
        
        Returns:
            Dictionary of parameter names and default values.
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: dict) -> bool:
        """
        Validate the given parameters.
        
        Args:
            parameters: Parameters to validate.
            
        Returns:
            True if parameters are valid, False otherwise.
        """
        pass
    
    def set_progress_callback(self, callback: Callable[[int, float], None]) -> None:
        """
        Set a callback function for progress reporting.
        
        Args:
            callback: Function that takes (iteration, fitness) as arguments.
        """
        self._progress_callback = callback
        
    def _report_progress(self, iteration: int, fitness: float) -> None:
        """Report progress if callback is set."""
        if hasattr(self, '_progress_callback') and self._progress_callback:
            self._progress_callback(iteration, fitness)
