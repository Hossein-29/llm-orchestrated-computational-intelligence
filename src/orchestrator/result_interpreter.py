"""
Result Interpreter module.

Analyzes execution results and generates insights using LLM.
"""
from pydantic import BaseModel,Field

from src.utils import llm_service


def interpret_results(result: dict,selection:dict, metrics: dict, problem_info: dict) -> dict:
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
    system_prompt = f"""You are the "result analyst" component of the MetaMind CI Framework.
    user gives you the problem details, the method used and the metrics.

    the analysis result should include:
        1. Performance Assessment: Did it meet expectations (accuracy/speed).
            - assessment should include one of the words GOOD/ACCEPTABLE/POOR.
            - the assessment should compare the optimal solution and found solution.
        2. Analysis: Explain the Gap to Optimal and Convergence Speed. Also explain if the results meet the expectations of the user.
        3. Suggestions: Based on the results, recommend changes in parameters to IMPROVE the results.
            - suggestions MUST include parameter tuning.
            - suggestions can include changing the whole method.
                - The replacement method MUST be picked between this list and nothing else:
                    [ACO,PSO,GA,GP,MLP,Fuzzy System,Kohonen self-organizing map, Hopfield network, perceptron]
            - suggestions can include both parameter tuning and method change

        the output should be in markdown format as follows:

        ## result analysis
            [write performance assessment here] 
        ### observations
            [write analysis here]
        ### Recommendations
            write recommendations here 
        ### confidence in solution
    """
    user_prompt=f""" For the problem {problem_info}
    i used the method {selection["selected_method"]}
    using parameters {selection["parameters"]}
    and the confidence of the selection is {selection["confidence"]}
    got this results {result} and
    this metrics {metrics} 
    Assess, analyse and give suggestions for performance improvement.
    """
    resp = llm_service.get_chat_completion(
            llm_service.Model.GPT_OSS.value, 
            [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800
        )
    
    return resp
