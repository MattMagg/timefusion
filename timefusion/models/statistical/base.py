"""
Base class for statistical forecasting models.

This module provides the StatisticalModel base class, which extends the BaseModel
class and provides common functionality for statistical forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple, Type
from abc import abstractmethod

from ...models.base import BaseModel
from ...utils.time_series import create_forecast_index, validate_fitted


class StatisticalModel(BaseModel):
    """
    Base class for all statistical forecasting models.
    
    This class extends the BaseModel class and provides common functionality
    for statistical forecasting models, including support for confidence intervals
    and integration with statsmodels.
    
    Attributes:
        name (str): Name of the model
        is_fitted (bool): Whether the model has been fitted
        params (Dict[str, Any]): Model parameters
        model: The underlying statistical model
        result: The result of fitting the model
        target_column (str): Name of the target column
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the statistical model.
        
        Args:
            name: Name of the model
            **kwargs: Additional parameters for the model
        """
        super().__init__(name, **kwargs)
        self.model = None
        self.result = None
        self.target_column = None
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, target_column: str, **kwargs) -> 'StatisticalModel':
        """
        Fit the statistical model to the data.
        
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
        Generate forecasts using the statistical model.
        
        Args:
            data: Input data
            horizon: Forecast horizon
            **kwargs: Additional parameters for prediction
            
        Returns:
            pd.DataFrame: Forecasts
        """
        pass
    
    def predict_with_confidence(
        self, 
        data: pd.DataFrame, 
        horizon: int, 
        confidence_level: float = 0.95,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate forecasts with confidence intervals.
        
        Args:
            data: Input data
            horizon: Forecast horizon
            confidence_level: Confidence level (default: 0.95)
            **kwargs: Additional parameters for prediction
            
        Returns:
            pd.DataFrame: Forecasts with confidence intervals
        """
        validate_fitted(self.is_fitted, "predict_with_confidence")
        
        # This is a default implementation that should be overridden by subclasses
        # that have native support for confidence intervals
        forecast = self.predict(data, horizon, **kwargs)
        
        # Add placeholder confidence intervals
        lower_col = f"{self.target_column}_lower_{int(confidence_level*100)}"
        upper_col = f"{self.target_column}_upper_{int(confidence_level*100)}"
        forecast[lower_col] = forecast[self.target_column]
        forecast[upper_col] = forecast[self.target_column]
        
        return forecast
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters
        """
        validate_fitted(self.is_fitted, "get_params")
        
        # This is a default implementation that should be overridden by subclasses
        return self.params
