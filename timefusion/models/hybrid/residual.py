"""
Residual model for time series forecasting.

This module provides the ResidualModel class, which implements a sequential
hybrid approach where a statistical model is first fitted to the data, and
then a deep learning model is fitted to the residuals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

from .base import HybridModel
from ..base import BaseModel
from ..statistical import StatisticalModel
from ..deep_learning import DeepLearningModel


class ResidualModel(HybridModel):
    """
    Residual model for time series forecasting.
    
    This class implements a sequential hybrid approach where a statistical model
    is first fitted to the data, and then a deep learning model is fitted to the
    residuals. The final forecast is the sum of the statistical model's forecast
    and the deep learning model's forecast of the residuals.
    
    Attributes:
        name (str): Name of the model
        is_fitted (bool): Whether the model has been fitted
        params (Dict[str, Any]): Model parameters
        statistical_model (StatisticalModel): Statistical model for trend/seasonality
        deep_learning_model (DeepLearningModel): Deep learning model for residuals
        target_column (str): Name of the target column
        residuals (pd.DataFrame): Residuals from the statistical model
    """
    
    def __init__(
        self,
        name: str = "residual",
        statistical_model: StatisticalModel = None,
        deep_learning_model: DeepLearningModel = None,
        **kwargs
    ):
        """
        Initialize the residual model.
        
        Args:
            name: Name of the model
            statistical_model: Statistical model for trend/seasonality
            deep_learning_model: Deep learning model for residuals
            **kwargs: Additional parameters for the model
        """
        super().__init__(name, **kwargs)
        self.statistical_model = statistical_model
        self.deep_learning_model = deep_learning_model
        self.residuals = None
        
        # Add models to the list
        if statistical_model:
            self.models.append(statistical_model)
        if deep_learning_model:
            self.models.append(deep_learning_model)
    
    def fit(self, data: pd.DataFrame, target_column: str, **kwargs) -> 'ResidualModel':
        """
        Fit the residual model to the data.
        
        This method fits the statistical model to the original data and
        the deep learning model to the residuals.
        
        Args:
            data: Input data
            target_column: Name of the target column
            **kwargs: Additional parameters for fitting
            
        Returns:
            self: The fitted model
        """
        self.target_column = target_column
        
        # Validate models
        if self.statistical_model is None:
            raise ValueError("Statistical model is required")
        if self.deep_learning_model is None:
            raise ValueError("Deep learning model is required")
        
        # Fit statistical model to original data
        self.statistical_model.fit(data, target_column, **kwargs)
        
        # Generate in-sample predictions from statistical model
        in_sample_pred = self.statistical_model.predict(
            data, horizon=0, in_sample=True, **kwargs
        )
        
        # Calculate residuals
        self.residuals = data.copy()
        self.residuals[target_column] = data[target_column] - in_sample_pred[target_column]
        
        # Fit deep learning model to residuals
        self.deep_learning_model.fit(self.residuals, target_column, **kwargs)
        
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame, horizon: int, **kwargs) -> pd.DataFrame:
        """
        Generate forecasts using the residual model.
        
        This method generates forecasts from the statistical model and
        the deep learning model, and combines them.
        
        Args:
            data: Input data
            horizon: Forecast horizon
            **kwargs: Additional parameters for prediction
            
        Returns:
            pd.DataFrame: Forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        # Generate forecasts from statistical model
        stat_forecast = self.statistical_model.predict(data, horizon, **kwargs)
        
        # Generate residual forecasts from deep learning model
        residual_forecast = self.deep_learning_model.predict(data, horizon, **kwargs)
        
        # Combine forecasts
        combined_forecast = stat_forecast.copy()
        combined_forecast[self.target_column] += residual_forecast[self.target_column]
        
        return combined_forecast
    
    def get_component_forecasts(
        self, data: pd.DataFrame, horizon: int, **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Get forecasts from each component model.
        
        Args:
            data: Input data
            horizon: Forecast horizon
            **kwargs: Additional parameters for prediction
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping component names to forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        # Generate forecasts from statistical model
        stat_forecast = self.statistical_model.predict(data, horizon, **kwargs)
        
        # Generate residual forecasts from deep learning model
        residual_forecast = self.deep_learning_model.predict(data, horizon, **kwargs)
        
        # Combine forecasts
        combined_forecast = stat_forecast.copy()
        combined_forecast[self.target_column] += residual_forecast[self.target_column]
        
        return {
            "statistical": stat_forecast,
            "residual": residual_forecast,
            "combined": combined_forecast
        }
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters
        """
        params = super().get_params()
        params.update({
            "statistical_model": self.statistical_model.name if self.statistical_model else None,
            "deep_learning_model": self.deep_learning_model.name if self.deep_learning_model else None
        })
        return params
