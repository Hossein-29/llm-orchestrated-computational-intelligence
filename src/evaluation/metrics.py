"""
Evaluation Metrics module.

Metrics for different problem types:
- Optimization: best fitness, mean, std, error, success rate
- Classification: accuracy, precision, recall, F1, AUC-ROC
- Clustering: silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI
"""

import numpy as np


def compute_metrics(result: dict, problem_info: dict) -> dict:
    """
    Compute problem-specific evaluation metrics.
    
    Args:
        result: Execution result from CI method.
        problem_info: Problem information containing type and known optimal.
        
    Returns:
        Dictionary of computed metrics.
    """
    problem_type = problem_info.get("problem_type", "").lower()
    
    if problem_type == "tsp":
        return compute_tsp_metrics(result, problem_info)
    if problem_type == "optimization":
        return compute_optimization_metrics(result, problem_info)
    elif problem_type == "classification":
        return compute_classification_metrics(result, problem_info)
    elif problem_type == "clustering":
        return compute_clustering_metrics(result, problem_info)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

def compute_tsp_metrics(result:dict, problem_info:dict)->dict:

    optimal_length=float(problem_info["content"]["known_optimal"])
    best_found_length=float(result["best_fitness"])
    history=result["convergence_history"]
    
    metrics=dict()

    metrics["tour_length"]=best_found_length

    metrics["gap_to_optimal"]=(best_found_length-optimal_length)*100/optimal_length

    metrics["computation_time"]=result["computation_time"]

    metrics["success_rate"]=100 #temporary


    initial_error = history[0]
    final_error = history[-1]
    
    target_error = initial_error - 0.9 * (initial_error - final_error)
    
    convergence_indices = np.where(history <= target_error)[0]
    if len(convergence_indices) > 0:
        metrics["convergence_speed"] = int(convergence_indices[0])
    else:
        metrics["convergence_speed"] = len(history)


    return metrics

def compute_optimization_metrics(result: dict, problem_info: dict) -> dict:
    """
    Compute optimization metrics.
    
    Metrics:
    - best_fitness: Best value found
    - mean_fitness: Average across runs
    - std_fitness: Standard deviation
    - error: Absolute error from optimal
    - success_rate: % of runs with error < threshold
    - function_evaluations: Total objective function calls
    """
    # TODO: Implement optimization metrics
    raise NotImplementedError("compute_optimization_metrics not yet implemented")


def compute_classification_metrics(result: dict, problem_info: dict) -> dict:
    """
    Compute classification metrics.
    
    Metrics:
    - accuracy: Overall correctness
    - precision: Positive predictive value
    - recall: Sensitivity
    - f1_score: Balanced measure
    - auc_roc: Area under ROC curve
    - confusion_matrix: [[TN, FP], [FN, TP]]
    - cross_val_score: Mean accuracy across k folds
    """
    # TODO: Implement classification metrics
    raise NotImplementedError("compute_classification_metrics not yet implemented")


def compute_clustering_metrics(result: dict, problem_info: dict) -> dict:
    """
    Compute clustering metrics.
    
    Metrics:
    - silhouette_score: Cohesion vs separation (-1 to 1)
    - davies_bouldin_index: Cluster similarity (lower is better)
    - calinski_harabasz_index: Variance ratio (higher is better)
    - adjusted_rand_index: Agreement with true labels (if available)
    - normalized_mutual_info: Information shared with true labels (if available)
    - inertia: Within-cluster sum of squares
    """
    # TODO: Implement clustering metrics
    raise NotImplementedError("compute_clustering_metrics not yet implemented")


def statistical_analysis(results_list: list) -> dict:
    """
    Perform statistical analysis across multiple runs.
    
    Args:
        results_list: List of results from multiple runs.
        
    Returns:
        Statistical summary including mean, std, confidence intervals.
    """
    # TODO: Implement statistical analysis
    # Including Wilcoxon test for comparing methods
    raise NotImplementedError("statistical_analysis not yet implemented")
