"""
Kohonen Self-Organizing Map (SOM) implementation.

Parameters:
- map_size: tuple (default: (10, 10), range: (5,5) to (50,50))
- learning_rate_initial: float (default: 0.5, range: 0.1-1.0)
- learning_rate_final: float (default: 0.01)
- neighborhood_initial: float (default: 5.0)
- max_epochs: int (default: 1000, range: 500-5000)
- topology: str (default: "rectangular", options: "rectangular", "hexagonal")
"""

import time
from typing import Any, Tuple

import numpy as np

from src.methods.base import BaseMethod


class SOM(BaseMethod):
    """Kohonen Self-Organizing Map for clustering."""
    
    def run(self, problem: Any, parameters: dict) -> dict:
        """
        Train the SOM on the given data.
        
        Args:
            problem: Clustering problem with data points.
            parameters: SOM parameters.
            
        Returns:
            Result dictionary with cluster assignments and map.
        """
        params = {**self.get_default_parameters(), **(parameters or {})}
        if parameters is None or "map_size" not in parameters:
            n_clusters = getattr(problem, "n_clusters", None)
            if isinstance(n_clusters, int) and n_clusters > 0:
                params["map_size"] = (n_clusters, 1)
        if not self.validate_parameters(params):
            raise ValueError("Invalid SOM parameters")

        data = problem.get_data()
        if data is None or len(data) == 0:
            raise ValueError("No data provided to SOM")

        map_size: Tuple[int, int] = tuple(params["map_size"])
        lr_initial = float(params["learning_rate_initial"])
        lr_final = float(params["learning_rate_final"])
        radius_initial = float(params["neighborhood_initial"])
        max_epochs = int(params["max_epochs"])

        n_rows, n_cols = map_size
        n_samples, n_features = data.shape
        rng = np.random.default_rng()

        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        weights = rng.uniform(data_min, data_max, size=(n_rows, n_cols, n_features))

        grid_r, grid_c = np.indices((n_rows, n_cols))
        grid_positions = np.stack([grid_r, grid_c], axis=-1)

        start_time = time.time()
        self.convergence_history = []

        for epoch in range(max_epochs):
            lr = lr_initial + (lr_final - lr_initial) * (epoch / max_epochs)
            radius = radius_initial * (1.0 - (epoch / max_epochs))
            radius = max(radius, 1e-3)

            indices = rng.permutation(n_samples)
            total_error = 0.0

            for idx in indices:
                x = data[idx]
                dists = np.linalg.norm(weights - x, axis=2)
                bmu_index = np.unravel_index(np.argmin(dists), dists.shape)
                bmu_pos = np.array(bmu_index)

                grid_dist = np.linalg.norm(grid_positions - bmu_pos, axis=2)
                influence = np.exp(-(grid_dist ** 2) / (2.0 * radius ** 2))
                weights += lr * influence[..., None] * (x - weights)

                total_error += dists[bmu_index]

            mean_error = total_error / n_samples
            self.convergence_history.append(mean_error)
            self._report_progress(epoch + 1, mean_error)

        # Assign each sample to its BMU
        flat_weights = weights.reshape(-1, n_features)
        bmu_indices = []
        for x in data:
            dists = np.linalg.norm(flat_weights - x, axis=1)
            bmu_indices.append(int(np.argmin(dists)))
        bmu_indices = np.array(bmu_indices)

        computation_time = time.time() - start_time
        return {
            "cluster_assignments": bmu_indices,
            "data": data,
            "labels": bmu_indices,
            "true_labels": getattr(problem, "get_true_labels", lambda: None)(),
            "map_weights": weights,
            "convergence_history": self.convergence_history,
            "computation_time": computation_time,
            "iterations_completed": max_epochs,
        }
    
    def get_default_parameters(self) -> dict:
        """Get default SOM parameters."""
        return {
            "map_size": (10, 10),
            "learning_rate_initial": 0.5,
            "learning_rate_final": 0.01,
            "neighborhood_initial": 5.0,
            "max_epochs": 1000,
            "topology": "rectangular",
        }
    
    def validate_parameters(self, parameters: dict) -> bool:
        """Validate SOM parameters."""
        try:
            map_size = parameters.get("map_size")
            if not map_size or len(map_size) != 2:
                return False
            if int(map_size[0]) <= 0 or int(map_size[1]) <= 0:
                return False
            if float(parameters.get("learning_rate_initial", 0)) <= 0:
                return False
            if float(parameters.get("learning_rate_final", 0)) < 0:
                return False
            if float(parameters.get("neighborhood_initial", 0)) <= 0:
                return False
            if int(parameters.get("max_epochs", 0)) <= 0:
                return False
            topology = parameters.get("topology", "rectangular")
            if topology not in {"rectangular", "hexagonal"}:
                return False
        except (TypeError, ValueError):
            return False
        return True
