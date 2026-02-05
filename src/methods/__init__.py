"""CI Methods module - implementations of 9 computational intelligence methods."""

from src.methods.base import BaseMethod
from src.methods.perceptron import Perceptron
from src.methods.mlp import MLP
from src.methods.som import SOM
from src.methods.hopfield import HopfieldNetwork
from src.methods.ga import GeneticAlgorithm
from src.methods.pso import PSO
from src.methods.aco import ACO
from src.methods.de import DifferentialEvolution

# Registry of available methods
METHODS = {
    "perceptron": Perceptron,
    "mlp": MLP,
    "som": SOM,
    "hopfield": HopfieldNetwork,
    "ga": GeneticAlgorithm,
    "pso": PSO,
    "aco": ACO,
    "de": DifferentialEvolution,
}


def get_method(method_name: str) -> BaseMethod:
    """
    Get a method instance by name.
    
    Args:
        method_name: Name of the method (lowercase).
        
    Returns:
        Instance of the requested method.
        
    Raises:
        ValueError: If method name is not recognized.
    """
    method_name = method_name.lower()
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHODS.keys())}")
    return METHODS[method_name]()


__all__ = [
    "BaseMethod",
    "Perceptron",
    "MLP",
    "SOM",
    "HopfieldNetwork",
    "GeneticAlgorithm",
    "PSO",
    "ACO",
    "DifferentialEvolution",
    "get_method",
    "METHODS",
]
