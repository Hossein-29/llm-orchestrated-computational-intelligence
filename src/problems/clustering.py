"""
Clustering Problems.

Datasets:
- Iris (for validation with known labels)
- Mall Customers (real-world segmentation)
- Synthetic data (controlled environment)
"""

import numpy as np


class ClusteringProblem:
    """Clustering problem with multiple datasets."""
    
    DATASETS = ["iris", "mall", "synthetic"]
    
    def __init__(self, problem_info: dict):
        """
        Initialize clustering problem.
        
        Args:
            problem_info: Dictionary containing:
                - dataset_name: Name of dataset
                - n_clusters: Expected number of clusters
                - data_path: Path to data (for mall dataset)
        """
        self.problem_info = problem_info
        self.dataset_name = problem_info.get("dataset_name", "iris")
        self.n_clusters = problem_info.get("n_clusters")
        self.data_path = problem_info.get("data_path")
        
        self.X = None
        self.true_labels = None  # Available for Iris
        
    def load_data(self) -> None:
        """Load the specified dataset."""
        if self.dataset_name == "iris":
            self._load_iris()
        elif self.dataset_name == "mall":
            self._load_mall()
        elif self.dataset_name == "synthetic":
            self._generate_synthetic()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_iris(self) -> None:
        """
        Load Iris dataset.
        
        150 samples, 4 features, 3 true clusters.
        """
        # TODO: Implement Iris data loading
        raise NotImplementedError("_load_iris not yet implemented")
    
    def _load_mall(self) -> None:
        """
        Load Mall Customers dataset.
        
        200 samples, features: Age, Annual Income, Spending Score.
        """
        # TODO: Implement Mall data loading
        raise NotImplementedError("_load_mall not yet implemented")
    
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
        # TODO: Return data
        raise NotImplementedError("get_data not yet implemented")
    
    def get_true_labels(self) -> np.ndarray:
        """Get true labels (if available)."""
        return self.true_labels
    
    def has_true_labels(self) -> bool:
        """Check if true labels are available."""
        return self.true_labels is not None
