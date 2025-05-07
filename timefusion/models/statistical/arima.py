"""
ARIMA model for time series forecasting.

This module provides the ARIMAModel class, which implements the ARIMA
(AutoRegressive Integrated Moving Average) model for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

from .base import StatisticalModel


class ARIMAModel(StatisticalModel):
    """
    ARIMA model for time series forecasting.
    
    This class provides an implementation of the ARIMA (AutoRegressive Integrated
    Moving Average) model for time series forecasting using statsmodels.
    
    Attributes:
        name (str): Name of the model
        is_fitted (bool): Whether the model has been fitted
        params (Dict[str, Any]): Model parameters
        order (Tuple[int, int, int]): ARIMA order (p, d, q)
        seasonal_order (Optional[Tuple[int, int, int, int]]): Seasonal order (P, D, Q, s)
        trend (Optional[str]): Trend component ('n', 'c', 't', 'ct')
        model: The underlying ARIMA model
        result: The result of fitting the model
        target_column (str): Name of the target column
    """
    
    def __init__(
        self,
        name: str = "arima",
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        trend: Optional[str] = None,
        auto_order: bool = False,
        **kwargs
    ):
        """
        Initialize the ARIMA model.
        
        Args:
            name: Name of the model
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            trend: Trend component ('n', 'c', 't', 'ct')
            auto_order: Whether to automatically select the order
            **kwargs: Additional parameters for the model
        """
        super().__init__(name, **kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.auto_order = auto_order
    
    def fit(self, data: pd.DataFrame, target_column: str, **kwargs) -> 'ARIMAModel':
        """
        Fit the ARIMA model to the data.
        
        Args:
            data: Input data
            target_column: Name of the target column
            **kwargs: Additional parameters for fitting
            
        Returns:
            self: The fitted model
        """
        self.target_column = target_column
        
        # Automatically select order if requested
        if self.auto_order:
            self._auto_select_order(data, target_column)
        
        # Create ARIMA model
        if self.seasonal_order:
            self.model = SARIMAX(
                data[target_column],
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                **kwargs
            )
        else:
            self.model = ARIMA(
                data[target_column],
                order=self.order,
                trend=self.trend,
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
        Generate forecasts using the ARIMA model.
        
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
        
        # Generate forecasts with confidence intervals
        alpha = 1 - confidence_level
        forecast_result = self.result.get_forecast(steps=horizon)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int(alpha=alpha)
        
        # Create forecast DataFrame
        forecast_index = self._create_forecast_index(data, horizon)
        forecast_df = pd.DataFrame({
            self.target_column: forecast_mean,
            f"{self.target_column}_lower_{int(confidence_level*100)}": forecast_ci.iloc[:, 0],
            f"{self.target_column}_upper_{int(confidence_level*100)}": forecast_ci.iloc[:, 1]
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
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "aic": self.result.aic,
            "bic": self.result.bic,
            "coefficients": self.result.params.to_dict()
        }
    
    def _auto_select_order(self, data: pd.DataFrame, target_column: str) -> None:
        """
        Automatically select the ARIMA order.
        
        Args:
            data: Input data
            target_column: Name of the target column
        """
        # This is a simple implementation that uses AIC to select the best order
        # A more sophisticated implementation would use auto_arima from pmdarima
        best_aic = float('inf')
        best_order = self.order
        
        # Try different orders
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(
                            data[target_column],
                            order=(p, d, q),
                            trend=self.trend
                        )
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            result = model.fit()
                        aic = result.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        self.order = best_order
