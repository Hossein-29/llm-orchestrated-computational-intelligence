"""
Main orchestration module for the MetaMind framework.

This module implements the 7-step pipeline:
1. Problem Input
2. LLM Analysis
3. Method Selection
4. Method Execution
5. Evaluation
6. Feedback to LLM
7. Recommendations
"""

from src.orchestrator.problem_parser import parse_problem
from src.orchestrator.method_selector import select_method
from src.orchestrator.parameter_configurator import configure_parameters
from src.orchestrator.result_interpreter import interpret_results
from src.orchestrator.recommendation_engine import generate_recommendations

from src.methods import get_method
from src.problems import load_problem
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualization import plot_convergence
from src.datasets.dataset_loader import dataset_loader


def run_pipeline(problem_input: str) -> dict:
    """
    Run the complete MetaMind pipeline.
    
    Args:
        problem_input: Dictionary containing problem description, data, and constraints.
        
    Returns:
        Dictionary containing results, analysis, and recommendations.
    """
    # Step 1: Parse problem input
    problem_info = parse_problem(problem_input)
    print("step 1: ",problem_info)

    # Step 1.5: extract data from the data set and add it to the problem_info
    dataset_loader(problem_info)
    print("step 1.5: ",problem_info)

    # Step 2 & 3: LLM Analysis and Method Selection
    selection = select_method(problem_info)
    print("step 2: ",selection)
    # Step 3: Configure parameters
    # parameters = configure_parameters(selection, problem_info)
    
    # Step 4: Execute method
    method = get_method(selection["selected_method"])
    problem = load_problem(problem_info)
    result = method.run(problem, selection["parameters"])
    print(result)
    # Step 5: Evaluation
    metrics = compute_metrics(result, problem_info)
    plot_convergence(result.get("convergence_history", []))
    print(metrics)
    # Step 6: Feedback to LLM
    interpretation = interpret_results(result, selection, metrics, problem_info)
    print(interpretation)
    # Step 7: Recommendations
    # recommendations = generate_recommendations(interpretation, problem_info)
    
    return {
        "result": result,
        "metrics": metrics,
        "interpretation": interpretation,
        # "recommendations": recommendations,
    }


def main():
    """Entry point for the MetaMind framework."""
    # Example usage - to be replaced with actual problem input
    # example_input = {
    #     "problem_type": "optimization",
    #     "description": "Example problem",
    #     "data": None,
    # }
    problem_info = """Problem: Traveling Salesman Problem
    Cities: 10
    Distance Matrix:
    [[  0,  29,  82,  46,  68,  52,  72,  42,  51,  55],
    [ 29,   0,  55,  46,  42,  43,  43,  23,  23,  31],
    [ 82,  55,   0,  68,  46,  55,  23,  43,  41,  29],
    [ 46,  46,  68,   0,  82,  15,  72,  31,  62,  42],
    [ 68,  42,  46,  82,   0,  74,  23,  52,  21,  46],
    [ 52,  43,  55,  15,  74,   0,  61,  23,  55,  31],
    [ 72,  43,  23,  72,  23,  61,   0,  42,  23,  31],
    [ 42,  23,  43,  31,  52,  23,  42,   0,  33,  15],
    [ 51,  23,  41,  62,  21,  55,  23,  33,   0,  29],
    [ 55,  31,  29,  42,  46,  31,  31,  15,  29,   0]]
    Objective: Minimize total tour distance
    Time Limit: 40 seconds
    Priority: Solution quality over speed
    known optimal: 240
    """
    print(run_pipeline(problem_info))
    print("MetaMind Framework")
    print("==================")
    print("Provide a problem input to run the pipeline.")
    # results = run_pipeline(example_input)


if __name__ == "__main__":
    main()
