"""
Classification Problem - Titanic Dataset.

Features:
- Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
"""

import numpy as np


class ClassificationProblem:
    """Classification problem using Titanic dataset."""
    
    def __init__(self, problem_info: dict):
        """
        Initialize classification problem.
        
        Args:
            problem_info: Dictionary containing:
                - data_path: Path to dataset (optional)
                - test_size: Fraction for test split
                - random_state: Random seed
        """
        self.problem_info = problem_info
        self.data_path = problem_info.get("data_path")
        self.test_size = problem_info.get("test_size", 0.15)
        self.random_state = problem_info.get("random_state", 42)
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_data(self) -> None:
        """
        Load and preprocess the Titanic dataset.
        
        Preprocessing steps:
        1. Handle missing values (Age, Cabin, Embarked)
        2. Encode categorical variables
        3. Normalize numerical features
        4. Split into train/val/test (70/15/15)
        """
        # TODO: Implement data loading and preprocessing
        raise NotImplementedError("load_data not yet implemented")
    
    def get_train_data(self) -> tuple:
        """Get training data (X_train, y_train)."""
        # TODO: Return training data
        raise NotImplementedError("get_train_data not yet implemented")
    
    def get_val_data(self) -> tuple:
        """Get validation data (X_val, y_val)."""
        # TODO: Return validation data
        raise NotImplementedError("get_val_data not yet implemented")
    
    def get_test_data(self) -> tuple:
        """Get test data (X_test, y_test)."""
        # TODO: Return test data
        raise NotImplementedError("get_test_data not yet implemented")
    
    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
