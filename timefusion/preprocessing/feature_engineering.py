"""
Feature engineering for time series data.

This module provides the FeatureEngineering class for creating derived features
from time series data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

from .base import BasePreprocessor


class FeatureEngineering(BasePreprocessor):
    """
    Feature engineering for time series data.
    
    This class provides methods for creating derived features from time series data,
    including lag features, window features, date-based features, and trend features.
    
    Attributes:
        name (str): Name of the feature engineering component
        is_fitted (bool): Whether the component has been fitted
        params (Dict[str, Any]): Component parameters
        lag_features (Optional[List[int]]): Lag periods for lag features
        window_features (Optional[Dict[str, List[int]]]): Window sizes and functions for window features
        date_features (Optional[List[str]]): Date-based features to create
        fourier_features (Optional[Dict[str, int]]): Fourier features to create
        target_column (Optional[str]): Target column for feature engineering
    """
    
    def __init__(
        self,
        name: str = "feature_engineering",
        lag_features: Optional[List[int]] = None,
        window_features: Optional[Dict[str, List[int]]] = None,
        date_features: Optional[List[str]] = None,
        fourier_features: Optional[Dict[str, int]] = None,
        target_column: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the feature engineering component.
        
        Args:
            name: Name of the feature engineering component
            lag_features: List of lag periods for lag features
            window_features: Dictionary of window functions and sizes for window features
            date_features: List of date-based features to create
            fourier_features: Dictionary of seasonal periods and orders for Fourier features
            target_column: Target column for feature engineering
            **kwargs: Additional parameters for the component
        """
        super().__init__(name=name, **kwargs)
        self.lag_features = lag_features or []
        self.window_features = window_features or {}
        self.date_features = date_features or []
        self.fourier_features = fourier_features or {}
        self.target_column = target_column
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'FeatureEngineering':
        """
        Fit the feature engineering component to the data.
        
        Args:
            data: Input data
            **kwargs: Additional parameters for fitting
            
        Returns:
            self: The fitted component
        """
        # Set target column if not specified
        if self.target_column is None and 'target_column' in kwargs:
            self.target_column = kwargs['target_column']
        elif self.target_column is None and len(data.columns) > 0:
            self.target_column = data.columns[0]
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply feature engineering to the data.
        
        Args:
            data: Input data
            **kwargs: Additional parameters for transformation
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        if not self.is_fitted:
            raise ValueError("Feature engineering component not fitted")
        
        # Create a copy of the data
        result = data.copy()
        
        # Check if the data has a datetime index
        has_datetime_index = isinstance(result.index, pd.DatetimeIndex)
        
        # Create lag features
        if self.lag_features and self.target_column in result.columns:
            result = self._create_lag_features(result)
        
        # Create window features
        if self.window_features and self.target_column in result.columns:
            result = self._create_window_features(result)
        
        # Create date-based features
        if self.date_features and has_datetime_index:
            result = self._create_date_features(result)
        
        # Create Fourier features
        if self.fourier_features and has_datetime_index:
            result = self._create_fourier_features(result)
        
        return result
    
    def _create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features.
        
        Args:
            data: Input data
            
        Returns:
            pd.DataFrame: Data with lag features
        """
        result = data.copy()
        
        for lag in self.lag_features:
            result[f"{self.target_column}_lag_{lag}"] = result[self.target_column].shift(lag)
        
        return result
    
    def _create_window_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create window features.
        
        Args:
            data: Input data
            
        Returns:
            pd.DataFrame: Data with window features
        """
        result = data.copy()
        
        for func_name, window_sizes in self.window_features.items():
            for window_size in window_sizes:
                # Get the window function
                if func_name == 'mean':
                    window_func = lambda x: x.rolling(window=window_size, min_periods=1).mean()
                elif func_name == 'std':
                    window_func = lambda x: x.rolling(window=window_size, min_periods=1).std()
                elif func_name == 'min':
                    window_func = lambda x: x.rolling(window=window_size, min_periods=1).min()
                elif func_name == 'max':
                    window_func = lambda x: x.rolling(window=window_size, min_periods=1).max()
                else:
                    raise ValueError(f"Unknown window function: {func_name}")
                
                # Apply the window function
                result[f"{self.target_column}_{func_name}_{window_size}"] = window_func(result[self.target_column])
        
        return result
    
    def _create_date_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create date-based features.
        
        Args:
            data: Input data
            
        Returns:
            pd.DataFrame: Data with date-based features
        """
        result = data.copy()
        
        for feature in self.date_features:
            if feature == 'hour':
                result['hour'] = result.index.hour
            elif feature == 'day':
                result['day'] = result.index.day
            elif feature == 'day_of_week':
                result['day_of_week'] = result.index.dayofweek
            elif feature == 'day_of_year':
                result['day_of_year'] = result.index.dayofyear
            elif feature == 'week_of_year':
                result['week_of_year'] = result.index.isocalendar().week
            elif feature == 'month':
                result['month'] = result.index.month
            elif feature == 'quarter':
                result['quarter'] = result.index.quarter
            elif feature == 'year':
                result['year'] = result.index.year
            elif feature == 'is_weekend':
                result['is_weekend'] = result.index.dayofweek.isin([5, 6]).astype(int)
            else:
                raise ValueError(f"Unknown date feature: {feature}")
        
        return result
    
    def _create_fourier_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create Fourier features for capturing seasonality.
        
        Args:
            data: Input data
            
        Returns:
            pd.DataFrame: Data with Fourier features
        """
        result = data.copy()
        
        # Create a time index (days since the start)
        time_idx = (result.index - result.index[0]).total_seconds() / (24 * 60 * 60)
        
        for period, order in self.fourier_features.items():
            for n in range(1, order + 1):
                # Create sine and cosine features
                result[f"fourier_sin_{period}_{n}"] = np.sin(2 * np.pi * n * time_idx / float(period))
                result[f"fourier_cos_{period}_{n}"] = np.cos(2 * np.pi * n * time_idx / float(period))
        
        return result
