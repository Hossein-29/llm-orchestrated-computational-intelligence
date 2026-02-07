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
        "data_source":"three options: 1_ list the exact distance matrix that is provided by user"\
            "2_ str MUST be the word \"generated\" if user has prompted to generate"\
            "3_ str name of the dataset. Options are \"eil51\" or \"berlin52\" or \"kroa100\"",
        "objective":"str the objective the user is seeking in the problem and should be solved",
        "time_limit":"float the time limitation that the solution provided shouldn't exceed",
        "priority": "str preferences of the user about accuracy of the solution and speed of the solution",
    },
    "Optimization": {
        "dataset_name":"str name of the dataset for the problem if user has provided the name of the dataset instead of distance_matrix directly",
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
        "data_source":"three options: 1_ make list of dicts where each dict is one record of user;'s provided data if user has provided the data in prompt"\
            "2_ str MUST be the word \"generated\" if user has prompted to generate"\
            "3_ str name of the dataset. Options are \"iris\" or \"mall customer\"",
        "n_samples": "int number of samples of the dataset if user has provided in input.",
        "features": "list name of the all features if provided by user.",
        "n_features": "int number of the all features.",
        "n_clusters": "int number of clusters if provided by user.",
        "use_case": "str description of user's purpose of doing this task.",
        "use_features": "list name of the features user wants to be treated as clustering arguments. Default value is the name of all features.",
        "cluster_std":"float standard deviation of generated data if provided by user",
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
    #### INPUT for TSP:
    Problem: Traveling Salesman Problem (TSP) Cities: 52 (Instance: berlin52) optimal path is 240 units Data is Provided as a symmetric distance matrix
    Objective: Minimize the total tour distance Constraints: The solution must be found within a 90-second time limit
    #### OUTPUT STRUCTURE:
    {{
        "problem_type": "TSP",
        content:{{
            - n_cities: 52 
            - known_optimal: 240
            - data_source: berlin52
            - objective: Minimize the total tour distance
            - time_limit: 90.0
            - priority: not_specified
        }}
    }}
    #### INPUT for Clustering:
    dataset:iris
    Samples: 150
    Features: 4 (sepal length/width, petal length/width)
    True clusters: 3 (species: setosa, versicolor, virginica)
    Use: Validate clustering against known labels
    #### OUTPUT STRUCTURE:
    {{
        "problem_type": "Clustering",
        content:{{
            - data_source :iris,
            - n_samples: 150,
            - features: ["sepal_length","sepal_width","petal_length","petal_width"],
            - n_features: 4,
            - n_clusters: 3,
            - use_case: Validate clustering against known labels,
            - use_features: all,
            - "cluster_std":not_specified,
        }}
    }}
    #### INPUT for Clustering:
    Source: Kaggle Mall Customer Segmentation
    Samples: 200
    Features: 5 (CustomerID, Gender, Age, Annual Income, Spending Score)
    Use features: Age, Annual Income, Spending Score
    Expected clusters: 5 customer segments
    #### OUTPUT STRUCTURE:
    {{
        "problem_type": "Clustering",
        content:{{
            - data_source :mall customer,
            - n_samples: 200,
            - features: ["CustomerID", "Gender", "Age", "Annual_Income", "Spending_Score"],
            - n_features: 5,
            - n_clusters: 5,
            - use_case: not_specified,
            - use_features: ["age","annual_income","spending_score"],
            - "cluster_std":not_specified,
        }}
    }}
    #### INPUT for Clustering:
    Source: generate using blobs in scikit learn
    Samples: 200
    Features: 11
    Expected clusters: 6
    standard deviation for clusters is 3
    #### OUTPUT STRUCTURE:
    {{
        "problem_type": "Clustering",
        content:{{
            - data_source :generated,
            - n_samples: 200,
            - features: not_specified,
            - n_features: 11,
            - n_clusters: 6,
            - use_case: not_specified,
            - use_features: not_specified,
            - "cluster_std": 3,
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
            max_tokens=1000,
        )
    
    return resp.model_dump()
    