"""
Base classes for preprocessing components.

This module provides the base classes and interfaces for all preprocessing components
in the TimeFusion system, including the BasePreprocessor abstract class and Pipeline.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple


class BasePreprocessor(ABC):
    """
    Base class for all preprocessing components.
    
    This class defines the common interface for all preprocessing components
    in the TimeFusion system. All preprocessor implementations must inherit from this class
    and implement the required methods.
    
    Attributes:
        name (str): Name of the preprocessor
        is_fitted (bool): Whether the preprocessor has been fitted
        params (Dict[str, Any]): Preprocessor parameters
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the preprocessor.
        
        Args:
            name: Name of the preprocessor
            **kwargs: Additional parameters for the preprocessor
        """
        self.name = name
        self.is_fitted = False
        self.params = kwargs
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BasePreprocessor':
        """
        Fit the preprocessor to the data.
        
        Args:
            data: Input data
            **kwargs: Additional parameters for fitting
            
        Returns:
            self: The fitted preprocessor
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply the preprocessor to the data.
        
        Args:
            data: Input data
            **kwargs: Additional parameters for transformation
            
        Returns:
            pd.DataFrame: Transformed data
        """
        pass
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Fit the preprocessor to the data and then transform it.
        
        Args:
            data: Input data
            **kwargs: Additional parameters for fitting and transformation
            
        Returns:
            pd.DataFrame: Transformed data
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)
    
    def __repr__(self) -> str:
        """
        String representation of the preprocessor.
        
        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__}(name='{self.name}', is_fitted={self.is_fitted})"


class Pipeline:
    """
    Pipeline for chaining multiple preprocessors.
    
    This class allows multiple preprocessing steps to be chained together
    and applied sequentially to data.
    
    Attributes:
        steps (List[Tuple[str, BasePreprocessor]]): List of (name, preprocessor) tuples
        preprocessors (Dict[str, BasePreprocessor]): Dictionary of preprocessors by name
    """
    
    def __init__(self, steps: List[Tuple[str, BasePreprocessor]]):
        """
        Initialize the pipeline.
        
        Args:
            steps: List of (name, preprocessor) tuples defining the pipeline
        """
        self.steps = steps
        self.preprocessors = {name: preprocessor for name, preprocessor in steps}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> 'Pipeline':
        """
        Fit all preprocessors in the pipeline.
        
        Args:
            data: Input data
            **kwargs: Additional parameters for fitting
            
        Returns:
            self: The fitted pipeline
        """
        transformed_data = data.copy()
        for name, preprocessor in self.steps:
            transformed_data = preprocessor.fit_transform(transformed_data, **kwargs)
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply all preprocessors in the pipeline.
        
        Args:
            data: Input data
            **kwargs: Additional parameters for transformation
            
        Returns:
            pd.DataFrame: Transformed data
        """
        transformed_data = data.copy()
        for name, preprocessor in self.steps:
            transformed_data = preprocessor.transform(transformed_data, **kwargs)
        return transformed_data
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Fit all preprocessors in the pipeline and then transform the data.
        
        Args:
            data: Input data
            **kwargs: Additional parameters for fitting and transformation
            
        Returns:
            pd.DataFrame: Transformed data
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)
    
    def __repr__(self) -> str:
        """
        String representation of the pipeline.
        
        Returns:
            str: String representation
        """
        return f"Pipeline(steps={[name for name, _ in self.steps]})"
