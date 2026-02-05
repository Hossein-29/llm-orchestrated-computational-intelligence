"""
Result Interpreter module.

Analyzes execution results and generates insights using LLM.
"""


def interpret_results(result: dict, metrics: dict, problem_info: dict) -> dict:
    """
    Interpret and analyze the execution results.
    
    Args:
        result: Raw execution results from the CI method.
        metrics: Computed evaluation metrics.
        problem_info: Original problem information.
        
    Returns:
        Interpretation dictionary containing:
            - performance_assessment: good/acceptable/poor
            - comparison: comparison with expected performance
            - explanation: natural language explanation
            - convergence_analysis: analysis of convergence behavior
    """
    # TODO: Implement result interpretation
    # 1. Compare results with known optima (if available)
    # 2. Analyze convergence behavior
    # 3. Generate natural language explanation via LLM
    # 4. Assess overall performance
    raise NotImplementedError("interpret_results not yet implemented")
