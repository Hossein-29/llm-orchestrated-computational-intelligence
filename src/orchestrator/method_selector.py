"""
Method Selector module.

Uses LLM to choose appropriate CI method based on problem analysis.
"""

from pydantic import BaseModel, Field

from src.utils.llm_service import get_chat_completion_structured, Model
from src.methods import METHODS


# Available methods for selection
AVAILABLE_METHODS = list(METHODS.keys())

# Problem type classifications
PROBLEM_TYPES = [
    "combinatorial_optimization",
    "continuous_optimization",
    "classification",
    "clustering",
]

# Method parameter specifications
METHOD_PARAMETERS = {
    "perceptron": {
        "learning_rate": "float (default: 0.01, range: 0.001-0.1)",
        "max_epochs": "int (default: 100, range: 50-1000)",
        "bias": "bool (default: True)",
    },
    "mlp": {
        "hidden_layers": "list[int] (default: [64, 32])",
        "activation": "str (default: 'relu', options: relu, sigmoid, tanh)",
        "learning_rate": "float (default: 0.001, range: 0.0001-0.01)",
        "max_epochs": "int (default: 500, range: 100-2000)",
        "batch_size": "int (default: 32, range: 16-128)",
        "optimizer": "str (default: 'adam', options: adam, sgd, rmsprop)",
    },
    "som": {
        "map_size": "tuple (default: (10, 10), range: (5,5) to (50,50))",
        "learning_rate_initial": "float (default: 0.5, range: 0.1-1.0)",
        "learning_rate_final": "float (default: 0.01)",
        "neighborhood_initial": "float (default: 5.0)",
        "max_epochs": "int (default: 1000, range: 500-5000)",
        "topology": "str (default: 'rectangular', options: rectangular, hexagonal)",
    },
    "hopfield": {
        "max_iterations": "int (default: 100, range: 50-500)",
        "threshold": "float (default: 0.0)",
        "async_update": "bool (default: True)",
    },
    "ga": {
        "population_size": "int (default: 100)",
        "crossover_rate": "float (default: 0.8)",
        "mutation_rate": "float (default: 0.1)",
        "selection": "str (options: tournament, roulette)",
        "generations": "int (default: 500)",
    },
    "pso": {
        "n_particles": "int (default: 50)",
        "w": "float (inertia weight, default: 0.7)",
        "c1": "float (cognitive, default: 1.5)",
        "c2": "float (social, default: 1.5)",
        "iterations": "int (default: 500)",
    },
    "aco": {
        "n_ants": "int (default: 30)",
        "alpha": "float (pheromone importance, default: 1.0)",
        "beta": "float (heuristic importance, default: 2.0)",
        "evaporation_rate": "float (default: 0.5)",
        "iterations": "int (default: 500)",
    },
    "de": {
        "population_size": "int (default: 50)",
        "F": "float (scaling factor, default: 0.8)",
        "CR": "float (crossover rate, default: 0.9)",
        "strategy": "str (default: 'best/1/bin')",
        "iterations": "int (default: 500)",
    },
}


class MethodSelection(BaseModel):
    """Structured response for method selection from LLM."""
    problem_type: str = Field(description="Classification of the problem type (combinatorial_optimization, continuous_optimization, classification, clustering)")
    selected_method: str = Field(description="Name of the selected CI method")
    reasoning: str = Field(description="Detailed explanation for why this method was selected")
    parameters: dict = Field(description="Suggested initial parameters for the selected method")
    backup_method: str = Field(description="Alternative method if the primary doesn't perform well")
    confidence: float = Field(description="Confidence score between 0 and 1 for this selection")


def _format_method_parameters() -> str:
    """Format method parameters for the prompt."""
    lines = []
    for method, params in METHOD_PARAMETERS.items():
        lines.append(f"\n**{method}**:")
        for param_name, param_spec in params.items():
            lines.append(f"  - {param_name}: {param_spec}")
    return "\n".join(lines)


def _build_selection_prompt(problem_info: dict) -> list[dict]:
    """Build the prompt messages for method selection."""
    system_prompt = f"""You are an expert in Computational Intelligence methods. Your task is to analyze a problem and select the most appropriate CI method to solve it.

Available methods and their parameters:

1. **perceptron** - Simple linear classifier, good for linearly separable binary classification
2. **mlp** - Multi-Layer Perceptron, powerful for complex classification tasks with non-linear boundaries
3. **som** - Self-Organizing Map (Kohonen), excellent for clustering and dimensionality reduction
4. **hopfield** - Hopfield Network, used for associative memory and small-scale optimization (TSP)
5. **ga** - Genetic Algorithm, versatile for combinatorial and continuous optimization
6. **pso** - Particle Swarm Optimization, effective for continuous optimization problems
7. **aco** - Ant Colony Optimization, ideal for graph-based routing/TSP problems
8. **de** - Differential Evolution, robust for continuous optimization with multimodal landscapes

Method Parameters:
{_format_method_parameters()}

Method Selection Guidelines:
- For TSP/routing problems: ACO (primary), GA or Hopfield (backup)
- For continuous optimization (Rastrigin, Ackley, Rosenbrock): PSO or DE (primary), GA (backup)
- For binary/multi-class classification: MLP (primary), Perceptron for simple cases
- For clustering/segmentation: SOM (primary)

Consider problem size, constraints, and user preferences (speed vs quality) when selecting parameters.

You must select a method from this exact list: {AVAILABLE_METHODS}
You must provide parameters using the exact parameter names listed above."""

    user_prompt = f"""Analyze the following problem and select the best CI method:

Problem Information:
{problem_info}

Provide your selection with detailed reasoning and suggested parameters."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _validate_selection(selection: MethodSelection) -> MethodSelection:
    """Validate and normalize the method selection."""
    # Normalize method names to lowercase
    selection.selected_method = selection.selected_method.lower()
    selection.backup_method = selection.backup_method.lower()
    
    # Validate selected method exists
    if selection.selected_method not in METHODS:
        raise ValueError(
            f"LLM selected unknown method: {selection.selected_method}. "
            f"Available methods: {AVAILABLE_METHODS}"
        )
    
    # Validate backup method exists
    if selection.backup_method not in METHODS:
        # If backup is invalid, default to GA as it's versatile
        selection.backup_method = "ga"
    
    # Clamp confidence to valid range
    selection.confidence = max(0.0, min(1.0, selection.confidence))
    
    return selection


def select_method(problem_info: dict, model: str = Model.GLM_4_7.value) -> dict:
    """
    Select the most appropriate CI method for the given problem.
    
    Args:
        problem_info: Parsed problem information from problem_parser.
        model: LLM model to use for selection.
        
    Returns:
        Selection dictionary containing:
            - problem_type: classified problem type
            - selected_method: name of chosen method
            - reasoning: explanation for selection
            - parameters: suggested parameters for the method
            - backup_method: alternative method suggestion
            - confidence: confidence score (0-1)
    """
    # 1. Build prompt and send to LLM
    messages = _build_selection_prompt(problem_info)
    
    # 2. Get structured response from LLM
    selection = get_chat_completion_structured(
        model=model,
        messages=messages,
        response_format=MethodSelection,
        max_tokens=1000,
    )
    
    # 3. Validate selected method exists
    selection = _validate_selection(selection)
    
    # 4. Return structured selection result as dict
    return selection.model_dump()
