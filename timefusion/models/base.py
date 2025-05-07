"""
Base classes for forecasting models.

This module provides the base classes and interfaces for all forecasting models
in the TimeFusion system, including the BaseModel abstract class and ModelRegistry.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple


class BaseModel(ABC):
    """
    Base class for all forecasting models.
    
    This class defines the common interface for all forecasting models
    in the TimeFusion system. All model implementations must inherit from this class
    and implement the required methods.
    
    Attributes:
        name (str): Name of the model
        is_fitted (bool): Whether the model has been fitted
        params (Dict[str, Any]): Model parameters
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the model.
        
        Args:
            name: Name of the model
            **kwargs: Additional parameters for the model
        """
        self.name = name
        self.is_fitted = False
        self.params = kwargs
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, target_column: str, **kwargs) -> 'BaseModel':
        """
        Fit the model to the data.
        
        Args:
            data: Input data
            target_column: Name of the target column
            **kwargs: Additional parameters for fitting
            
        Returns:
            self: The fitted model
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, horizon: int, **kwargs) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            data: Input data
            horizon: Forecast horizon
            **kwargs: Additional parameters for prediction
            
        Returns:
            pd.DataFrame: Forecasts
        """
        pass
    
    def __repr__(self) -> str:
        """
        String representation of the model.
        
        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__}(name='{self.name}', is_fitted={self.is_fitted})"


class ModelRegistry:
    """
    Registry for forecasting models.
    
    This class implements a registry pattern, providing a central
    registry for all forecasting models in the system.
    
    Attributes:
        models (Dict[str, BaseModel]): Dictionary of registered models
    """
    
    def __init__(self):
        """
        Initialize the model registry.
        """
        self.models = {}
        
    def register(self, model: BaseModel) -> None:
        """
        Register a model.
        
        Args:
            model: Model to register
        """
        self.models[model.name] = model
        
    def get(self, name: str) -> BaseModel:
        """
        Get a model by name.
        
        Args:
            name: Name of the model
            
        Returns:
            BaseModel: The model
            
        Raises:
            KeyError: If the model is not found
        """
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self.models[name]
    
    def list(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List[str]: List of model names
        """
        return list(self.models.keys())
    
    def __repr__(self) -> str:
        """
        String representation of the model registry.
        
        Returns:
            str: String representation
        """
        return f"ModelRegistry(models={self.list()})"
