"""
Ensemble model for time series forecasting.

This module provides the EnsembleModel class, which implements ensemble
forecasting models that combine multiple base models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

from .base import HybridModel
from ..base import BaseModel


class EnsembleModel(HybridModel):
    """
    Ensemble model for time series forecasting.
    
    This class implements an ensemble forecasting model that combines
    multiple base models using weighted averaging.
    
    Attributes:
        name (str): Name of the model
        is_fitted (bool): Whether the model has been fitted
        params (Dict[str, Any]): Model parameters
        models (List[BaseModel]): List of component models
        weights (List[float]): List of model weights
        ensemble_method (str): Method for combining forecasts
        target_column (str): Name of the target column
    """
    
    def __init__(
        self,
        name: str = "ensemble",
        models: List[BaseModel] = None,
        weights: List[float] = None,
        ensemble_method: str = "weighted_average",
        **kwargs
    ):
        """
        Initialize the ensemble model.
        
        Args:
            name: Name of the model
            models: List of component models
            weights: List of model weights (must match length of models)
            ensemble_method: Method for combining forecasts
                ("weighted_average", "mean", "median")
            **kwargs: Additional parameters for the model
        """
        super().__init__(name, models, **kwargs)
        
        # Validate weights
        if weights is not None and models is not None:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
        
        self.weights = weights or []
        self.ensemble_method = ensemble_method
        
        # Initialize weights if not provided
        if models and not weights:
            self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize model weights.
        
        If weights are not provided, initialize with equal weights.
        """
        n_models = len(self.models)
        if n_models > 0:
            self.weights = [1.0 / n_models] * n_models
    
    def add_model(self, model: BaseModel, weight: float = None) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: Model to add
            weight: Weight for the model (if None, weights will be rebalanced)
        """
        super().add_model(model)
        
        # Update weights
        if weight is not None:
            # Add the new weight and rescale to sum to 1.0
            current_sum = sum(self.weights)
            if current_sum > 0:
                # Rescale existing weights
                self.weights = [w * (1.0 - weight) / current_sum for w in self.weights]
            self.weights.append(weight)
        else:
            # Rebalance with equal weights
            self._initialize_weights()
    
    def fit(self, data: pd.DataFrame, target_column: str, **kwargs) -> 'EnsembleModel':
        """
        Fit the ensemble model to the data.
        
        This method fits each component model to the data.
        
        Args:
            data: Input data
            target_column: Name of the target column
            **kwargs: Additional parameters for fitting
            
        Returns:
            self: The fitted model
        """
        self.target_column = target_column
        
        # Fit each model
        for model in self.models:
            model.fit(data, target_column, **kwargs)
        
        # Initialize weights if not already set
        if not self.weights:
            self._initialize_weights()
        
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame, horizon: int, **kwargs) -> pd.DataFrame:
        """
        Generate forecasts using the ensemble model.
        
        This method combines the forecasts from each component model
        according to the specified ensemble method.
        
        Args:
            data: Input data
            horizon: Forecast horizon
            **kwargs: Additional parameters for prediction
            
        Returns:
            pd.DataFrame: Forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get forecasts from each model
        forecasts = []
        for model in self.models:
            forecast = model.predict(data, horizon, **kwargs)
            forecasts.append(forecast)
        
        # Combine forecasts
        if self.ensemble_method == "weighted_average":
            combined_forecast = self._weighted_average(forecasts)
        elif self.ensemble_method == "mean":
            combined_forecast = self._mean(forecasts)
        elif self.ensemble_method == "median":
            combined_forecast = self._median(forecasts)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return combined_forecast
    
    def _weighted_average(self, forecasts: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine forecasts using weighted average.
        
        Args:
            forecasts: List of forecasts from component models
            
        Returns:
            pd.DataFrame: Combined forecast
        """
        # Initialize with the first forecast's index
        result = pd.DataFrame(index=forecasts[0].index)
        
        # Combine forecasts with weights
        result[self.target_column] = 0.0
        for i, forecast in enumerate(forecasts):
            result[self.target_column] += forecast[self.target_column] * self.weights[i]
        
        return result
    
    def _mean(self, forecasts: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine forecasts using simple average.
        
        Args:
            forecasts: List of forecasts from component models
            
        Returns:
            pd.DataFrame: Combined forecast
        """
        # Initialize with the first forecast's index
        result = pd.DataFrame(index=forecasts[0].index)
        
        # Stack forecasts and take mean
        stacked = pd.concat([forecast[self.target_column] for forecast in forecasts], axis=1)
        result[self.target_column] = stacked.mean(axis=1)
        
        return result
    
    def _median(self, forecasts: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine forecasts using median.
        
        Args:
            forecasts: List of forecasts from component models
            
        Returns:
            pd.DataFrame: Combined forecast
        """
        # Initialize with the first forecast's index
        result = pd.DataFrame(index=forecasts[0].index)
        
        # Stack forecasts and take median
        stacked = pd.concat([forecast[self.target_column] for forecast in forecasts], axis=1)
        result[self.target_column] = stacked.median(axis=1)
        
        return result
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get the model weights.
        
        Returns:
            Dict[str, float]: Dictionary mapping model names to weights
        """
        return {model.name: weight for model, weight in zip(self.models, self.weights)}
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters
        """
        params = super().get_params()
        params.update({
            "weights": self.weights,
            "ensemble_method": self.ensemble_method
        })
        return params
