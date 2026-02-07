"""
Clustering Problems.

Datasets:
- Synthetic data (controlled environment)
"""

import numpy as np


class ClusteringProblem:
    """Clustering problem with multiple datasets."""
    
    DATASETS = ["synthetic"]
    
    def __init__(self, problem_info: dict):
        """
        Initialize clustering problem.
        
        Args:
            problem_info: Dictionary containing:
                - dataset_name: Name of dataset
                - n_clusters: Expected number of clusters
                - data_path: Path to data (optional)
                - X: Optional feature matrix (numpy array or array-like)
                - true_labels: Optional ground-truth labels
        """
        self.problem_info = problem_info
        self.dataset_name = problem_info.get("dataset_name", "iris")
        self.n_clusters = problem_info.get("n_clusters")
        self.data_path = problem_info.get("data_path")
        
        X = problem_info.get("X")
        self.X = np.asarray(X, dtype=float) if X is not None else None
        self.true_labels = problem_info.get("true_labels")  # Available for Iris
        
    def load_data(self) -> None:
        """Load the specified dataset."""
        if self.X is not None:
            return
        if self.dataset_name == "synthetic":
            self._generate_synthetic()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _generate_synthetic(self) -> None:
        """
        Generate synthetic clustering data.
        
        Uses sklearn.datasets.make_blobs with configurable:
        - n_samples: 500
        - n_features: 2, 5, or 10
        - n_clusters: 5
        - cluster_std: 1.0
        """
        # TODO: Implement synthetic data generation
        raise NotImplementedError("_generate_synthetic not yet implemented")
    
    def get_data(self) -> np.ndarray:
        """Get the data matrix."""
        if self.X is None:
            self.load_data()
        return self.X
    
    def get_true_labels(self) -> np.ndarray:
        """Get true labels (if available)."""
        return self.true_labels
    
    def has_true_labels(self) -> bool:
        """Check if true labels are available."""
        return self.true_labels is not None
