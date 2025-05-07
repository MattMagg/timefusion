"""
Exponential Smoothing model for time series forecasting.

This module provides the ExponentialSmoothingModel class, which implements
exponential smoothing models for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

from .base import StatisticalModel


class ExponentialSmoothingModel(StatisticalModel):
    """
    Exponential Smoothing model for time series forecasting.
    
    This class provides an implementation of exponential smoothing models
    for time series forecasting using statsmodels.
    
    Attributes:
        name (str): Name of the model
        is_fitted (bool): Whether the model has been fitted
        params (Dict[str, Any]): Model parameters
        trend (Optional[str]): Trend component ('add', 'mul', None)
        seasonal (Optional[str]): Seasonal component ('add', 'mul', None)
        seasonal_periods (Optional[int]): Number of periods in a seasonal cycle
        damped_trend (bool): Whether to use damped trend
        model: The underlying exponential smoothing model
        result: The result of fitting the model
        target_column (str): Name of the target column
    """
    
    def __init__(
        self,
        name: str = "ets",
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        damped_trend: bool = False,
        auto_params: bool = False,
        **kwargs
    ):
        """
        Initialize the Exponential Smoothing model.
        
        Args:
            name: Name of the model
            trend: Trend component ('add', 'mul', None)
            seasonal: Seasonal component ('add', 'mul', None)
            seasonal_periods: Number of periods in a seasonal cycle
            damped_trend: Whether to use damped trend
            auto_params: Whether to automatically select the parameters
            **kwargs: Additional parameters for the model
        """
        super().__init__(name, **kwargs)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.auto_params = auto_params
    
    def fit(self, data: pd.DataFrame, target_column: str, **kwargs) -> 'ExponentialSmoothingModel':
        """
        Fit the Exponential Smoothing model to the data.
        
        Args:
            data: Input data
            target_column: Name of the target column
            **kwargs: Additional parameters for fitting
            
        Returns:
            self: The fitted model
        """
        self.target_column = target_column
        
        # Automatically select parameters if requested
        if self.auto_params:
            self._auto_select_params(data, target_column)
        
        # Create Exponential Smoothing model
        self.model = ExponentialSmoothing(
            data[target_column],
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend,
            **kwargs
        )
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.result = self.model.fit()
        
        self.is_fitted = True
        
        return self
    
    def predict(self, data: pd.DataFrame, horizon: int, **kwargs) -> pd.DataFrame:
        """
        Generate forecasts using the Exponential Smoothing model.
        
        Args:
            data: Input data
            horizon: Forecast horizon
            **kwargs: Additional parameters for prediction
            
        Returns:
            pd.DataFrame: Forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        # Generate forecasts
        forecast = self.result.forecast(steps=horizon)
        
        # Create forecast DataFrame
        forecast_index = self._create_forecast_index(data, horizon)
        forecast_df = pd.DataFrame({self.target_column: forecast}, index=forecast_index)
        
        return forecast_df
    
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
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        # Generate forecasts
        forecast = self.result.forecast(steps=horizon)
        
        # Create forecast DataFrame
        forecast_index = self._create_forecast_index(data, horizon)
        
        # Calculate confidence intervals
        # Note: statsmodels ExponentialSmoothing doesn't provide built-in confidence intervals,
        # so we'll estimate them based on the residuals
        residuals = self.result.resid
        residual_std = residuals.std()
        z_value = abs(np.percentile(np.random.normal(0, 1, 10000), (1 - confidence_level) / 2 * 100))
        
        lower_bound = forecast - z_value * residual_std
        upper_bound = forecast + z_value * residual_std
        
        forecast_df = pd.DataFrame({
            self.target_column: forecast,
            f"{self.target_column}_lower_{int(confidence_level*100)}": lower_bound,
            f"{self.target_column}_upper_{int(confidence_level*100)}": upper_bound
        }, index=forecast_index)
        
        return forecast_df
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        return {
            "trend": self.trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods,
            "damped_trend": self.damped_trend,
            "smoothing_level": getattr(self.result, 'params', {}).get('smoothing_level', None),
            "smoothing_trend": getattr(self.result, 'params', {}).get('smoothing_trend', None),
            "smoothing_seasonal": getattr(self.result, 'params', {}).get('smoothing_seasonal', None),
            "damping_trend": getattr(self.result, 'params', {}).get('damping_trend', None),
            "aic": getattr(self.result, 'aic', None),
            "bic": getattr(self.result, 'bic', None),
            "mse": getattr(self.result, 'mse', None)
        }
    
    def _auto_select_params(self, data: pd.DataFrame, target_column: str) -> None:
        """
        Automatically select the Exponential Smoothing parameters.
        
        Args:
            data: Input data
            target_column: Name of the target column
        """
        # This is a simple implementation that tries different combinations
        # and selects the one with the lowest AIC
        best_aic = float('inf')
        best_params = {
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'damped_trend': self.damped_trend
        }
        
        # Try different combinations
        for trend in [None, 'add', 'mul']:
            for seasonal in [None, 'add', 'mul']:
                for damped_trend in [False, True]:
                    # Skip invalid combinations
                    if seasonal and not self.seasonal_periods:
                        continue
                    if damped_trend and not trend:
                        continue
                    
                    try:
                        model = ExponentialSmoothing(
                            data[target_column],
                            trend=trend,
                            seasonal=seasonal,
                            seasonal_periods=self.seasonal_periods,
                            damped_trend=damped_trend
                        )
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            result = model.fit()
                        aic = result.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_params = {
                                'trend': trend,
                                'seasonal': seasonal,
                                'seasonal_periods': self.seasonal_periods,
                                'damped_trend': damped_trend
                            }
                    except:
                        continue
        
        # Update parameters
        self.trend = best_params['trend']
        self.seasonal = best_params['seasonal']
        self.seasonal_periods = best_params['seasonal_periods']
        self.damped_trend = best_params['damped_trend']
