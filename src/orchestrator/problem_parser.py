"""
Problem Parser module.

Extracts problem type, size, constraints from user input.
"""
from pydantic import BaseModel,Field

from src.utils import llm_service

PROBLEM_DETAILS={
    "TSP": {
        "n_cities":" int is the number of cities provided by problem",
        "known_optimal":"int is the optimal length of the path optimal path of the problem",
        "distance_matrix":"list the matrix determining the distance of each two cities",
        "objective":"str the objective the user is seeking in the problem and should be solved",
        "time_limit":"float the time limitation that the solution provided shouldn't exceed",
        "priority": "str preferences of the user about accuracy of the solution and speed of the solution"
    },
    "Optimization": {
        "hidden_layers": "list[int] (default: [64, 32])",
        "activation": "str (default: 'relu', options: relu, sigmoid, tanh)",
        "learning_rate": "float (default: 0.001, range: 0.0001-0.01)",
        "max_epochs": "int (default: 500, range: 100-2000)",
        "batch_size": "int (default: 32, range: 16-128)",
        "optimizer": "str (default: 'adam', options: adam, sgd, rmsprop)",
    },
    "Classification": {
        "map_size": "tuple (default: (10, 10), range: (5,5) to (50,50))",
        "learning_rate_initial": "float (default: 0.5, range: 0.1-1.0)",
        "learning_rate_final": "float (default: 0.01)",
        "neighborhood_initial": "float (default: 5.0)",
        "max_epochs": "int (default: 1000, range: 500-5000)",
        "topology": "str (default: 'rectangular', options: rectangular, hexagonal)",
    },
    "Clustering": {
        "max_iterations": "int (default: 100, range: 50-500)",
        "threshold": "float (default: 0.0)",
        "async_update": "bool (default: True)",
    }
}

def _format_problem_details() -> str:
    """Format method parameters for the prompt."""
    lines = []
    for method, params in PROBLEM_DETAILS.items():
        lines.append(f"\n**{method}**:")
        for param_name, param_spec in params.items():
            lines.append(f"  - {param_name}: {param_spec}")
    return "\n".join(lines)

class Answer(BaseModel):
    """
    Structured representation of the parsed problem as required by the 
    LLM-Orchestrated Framework.
    """
    problem_type: str = Field(description="TSP, optimization, classification, or clustering")
    content: dict = Field(description="a dictionary of details of the problem according to the problem_type")


def parse_problem(problem_input: str) -> dict:
    """
    Parse and extract problem information from user input.
    
    Args:
        problem_input: Raw problem input containing:
            - problem_type: optimization, classification, clustering
            - data: problem data (distance matrix, dataset, function)
            - constraints: problem constraints
            - performance_expectations: speed vs accuracy preference
            
    Returns:
        Structured problem information dictionary.
    """
    # TODO: Implement problem parsing logic
    # 1. Identify problem type (combinatorial, continuous, supervised, unsupervised)
    # 2. Extract problem size and complexity
    # 3. Parse constraints and requirements
    # 4. Validate input format

    system_prompt = f"""You are the "Problem Parser" component of the MetaMind CI Framework. 
    Your task is to convert unstructured user problem descriptions into a strict JSON format.

    Available problem types and their parameters:

    1. **TSP** - Traveler sales man problem with cities and distance between them
    2. **Optimization** - Trying to optimize a given function using approximation methods
    3. **Classification** - Classifying the given data by choosing a class between given classes
    4. **Clustering** - clustering given data into n given groups

    Problems Parameters:
    {_format_problem_details()}

    ### RULES:
    1. OUTPUT ONLY VALID JSON. No markdown backticks (```json), no conversational text, and no preamble.
    2. TYPES:
    - **problem_type**: Must be exactly one of "TSP" or "Function optimization" or "classification", or "clustering".
    - **content**: Is a dictionary fields are determined according to the **problem_type**.
    4. BREVITY: Keep descriptions short to avoid token truncation.
    5. DEFAULTS: If information is missing, use "not_specified".
    6. convert all the given times to **seconds** if they are given in other time units.

    ### EXAMPLE INPUT/OUTPUT:
    #### INPUT:
    Problem: Traveling Salesman Problem (TSP) Cities: 52 (Instance: berlin52) optimal path is 420 units Data is Provided as a symmetric distance matrix
    Objective: Minimize the total tour distance Constraints: The solution must be found within a 90-second time limit
    #### OUTPUT STRUCTURE:
    {{
        "problem_type": "TSP",
        content:{{
            - n_cities: 52 
            - known_optimal: 420
            - distance_matrix: Provided as a symmetric distance matrix
            - objective: Minimize the total tour distance
            - time_limit: 90.0
            - priority: not_specified
        }}
    }}
    """
    resp = llm_service.get_chat_completion_structured(
            llm_service.Model.GPT_OSS.value, 
            [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": f"Analyze the following problem and parse it {problem_input}"}
            ], 
            Answer,
            max_tokens=500
        )
    
    return resp.model_dump()
    