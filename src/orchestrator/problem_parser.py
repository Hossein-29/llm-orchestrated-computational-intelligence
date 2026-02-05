"""
Problem Parser module.

Extracts problem type, size, constraints from user input.
"""
from pydantic import BaseModel,Field
from typing import Dict, Any
from src.utils import llm_service

class Answer(BaseModel):
    """
    Structured representation of the parsed problem as required by the 
    LLM-Orchestrated Framework.
    """
    problem_type: str = Field(description="TSP, optimization, classification, or clustering")
    category: str = Field(description="combinatorial, continuous, supervised, or unsupervised")
    size_complexity: Dict[str, Any] = Field(description="Must be a dictionary of size metrics of the given problem")
    constraints: Dict[str, Any] = Field(description="Must be a dictionary of constraints of given problem")
    performance_expectations: str = Field(description="User preference regarding speed/accuracy")
    data_summary: str = Field(description="Summary of data structure")


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

    system_prompt = """You are the "Problem Parser" component of the MetaMind CI Framework. 
    Your task is to convert unstructured user problem descriptions into a strict JSON format.

    ### RULES:
    1. OUTPUT ONLY VALID JSON. No markdown backticks (```json), no conversational text, and no preamble.
    2. TYPES: 
    - problem_type: Must be exactly "TSP", "Function optimization", "classification", or "clustering".
    - category: Must be exactly "combinatorial", "continuous", "supervised", or "unsupervised".
    3. DICTIONARY ENFORCEMENT: 'size_complexity' and 'constraints' MUST be key-value pairs (objects). Never return them as strings or lists.
    4. BREVITY: Keep descriptions short to avoid token truncation.
    5. DEFAULTS: If information is missing, use "not_specified".
    ### EXAMPLE INPUT/OUTPUT STRUCTURE:
    #### INPUT:
    Problem: Traveling Salesman Problem (TSP) Cities: 52 (Instance: berlin52) Data: Provided as a symmetric distance matrix Objective: Minimize the total tour distance Constraints: The solution must be found within a 90-second time limit
    #### OUTPUT:
    {
    "problem_type": "Traveling Salesman Problem (TSP)",
    "category": "combinatorial",
    "size_complexity": {"n_cities": 30, "domain": "discrete"},
    "constraints": {"time_limit_s": 60, "objective": "minimize_distance"},
    "performance_expectations": "quality_over_speed",
    "data_summary": "Symmetric distance matrix for 30 nodes"
    }
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
    