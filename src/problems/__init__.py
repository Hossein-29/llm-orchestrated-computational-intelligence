"""Problems module - benchmark problem definitions."""

from src.problems.tsp import TSPProblem
from src.problems.optimization import OptimizationProblem
from src.problems.classification import ClassificationProblem
from src.problems.clustering import ClusteringProblem


def load_problem(problem_info: dict):
    """
    Load and initialize a problem based on problem info.
    
    Args:
        problem_info: Parsed problem information containing problem_type and data.
        
    Returns:
        Initialized problem instance.
    """
    problem_type = problem_info.get("problem_type", "").lower()
    
    if problem_type == "tsp":
        return TSPProblem(problem_info)
    elif problem_type == "optimization":
        return OptimizationProblem(problem_info)
    elif problem_type == "classification":
        return ClassificationProblem(problem_info)
    elif problem_type == "clustering":
        return ClusteringProblem(problem_info)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


__all__ = [
    "TSPProblem",
    "OptimizationProblem",
    "ClassificationProblem",
    "ClusteringProblem",
    "load_problem",
]
