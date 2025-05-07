"""
Cleaner for time series data.

This module provides the Cleaner class for detecting and handling outliers
and anomalies in time series data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

from .base import BasePreprocessor


class Cleaner(BasePreprocessor):
    """
    Cleaner for time series data.
    
    This class provides methods for detecting and handling outliers and anomalies
    in time series data. It supports multiple outlier detection methods and
    handling strategies.
    
    Attributes:
        name (str): Name of the cleaner
        is_fitted (bool): Whether the cleaner has been fitted
        params (Dict[str, Any]): Cleaner parameters
        method (str): Outlier detection method
        threshold (float): Threshold for outlier detection
        strategy (str): Strategy for handling outliers
        columns (Optional[List[str]]): Columns to clean
    """
    
    def __init__(
        self,
        name: str = "cleaner",
        method: str = "z-score",
        threshold: float = 3.0,
        strategy: str = "clip",
        columns: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the cleaner.
        
        Args:
            name: Name of the cleaner
            method: Outlier detection method ('z-score', 'iqr', 'percentile')
            threshold: Threshold for outlier detection
            strategy: Strategy for handling outliers ('clip', 'remove', 'replace')
            columns: Columns to clean (if None, all numeric columns are cleaned)
            **kwargs: Additional parameters for the cleaner
        """
        super().__init__(name=name, **kwargs)
        self.method = method
        self.threshold = threshold
        self.strategy = strategy
        self.columns = columns
        self.stats = {}
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'Cleaner':
        """
        Fit the cleaner to the data.
        
        Args:
            data: Input data
            **kwargs: Additional parameters for fitting
            
        Returns:
            self: The fitted cleaner
        """
        # Determine columns to clean
        if self.columns is None:
            self.columns = data.select_dtypes(include=np.number).columns.tolist()
        
        # Calculate statistics for each column
        for column in self.columns:
            if column not in data.columns:
                continue
            
            column_data = data[column].dropna()
            
            if self.method == 'z-score':
                self.stats[column] = {
                    'mean': column_data.mean(),
                    'std': column_data.std()
                }
            elif self.method == 'iqr':
                q1 = column_data.quantile(0.25)
                q3 = column_data.quantile(0.75)
                iqr = q3 - q1
                self.stats[column] = {
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr
                }
            elif self.method == 'percentile':
                self.stats[column] = {
                    'lower': column_data.quantile(0.01),
                    'upper': column_data.quantile(0.99)
                }
            else:
                raise ValueError(f"Unknown outlier detection method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply the cleaner to the data.
        
        Args:
            data: Input data
            **kwargs: Additional parameters for transformation
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        if not self.is_fitted:
            raise ValueError("Cleaner not fitted")
        
        # Create a copy of the data
        cleaned_data = data.copy()
        
        # Clean each column
        for column in self.columns:
            if column not in cleaned_data.columns:
                continue
            
            # Detect outliers
            outliers = self._detect_outliers(cleaned_data[column], column)
            
            # Handle outliers
            if self.strategy == 'clip':
                cleaned_data[column] = self._clip_outliers(cleaned_data[column], column, outliers)
            elif self.strategy == 'remove':
                cleaned_data = cleaned_data[~outliers]
            elif self.strategy == 'replace':
                cleaned_data[column] = self._replace_outliers(cleaned_data[column], outliers)
            else:
                raise ValueError(f"Unknown outlier handling strategy: {self.strategy}")
        
        return cleaned_data
    
    def _detect_outliers(self, data: pd.Series, column: str) -> pd.Series:
        """
        Detect outliers in a column.
        
        Args:
            data: Column data
            column: Column name
            
        Returns:
            pd.Series: Boolean mask of outliers
        """
        if self.method == 'z-score':
            mean = self.stats[column]['mean']
            std = self.stats[column]['std']
            z_scores = np.abs((data - mean) / std)
            return z_scores > self.threshold
        
        elif self.method == 'iqr':
            q1 = self.stats[column]['q1']
            q3 = self.stats[column]['q3']
            iqr = self.stats[column]['iqr']
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            return (data < lower_bound) | (data > upper_bound)
        
        elif self.method == 'percentile':
            lower = self.stats[column]['lower']
            upper = self.stats[column]['upper']
            return (data < lower) | (data > upper)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")
    
    def _clip_outliers(self, data: pd.Series, column: str, outliers: pd.Series) -> pd.Series:
        """
        Clip outliers to the threshold values.
        
        Args:
            data: Column data
            column: Column name
            outliers: Boolean mask of outliers
            
        Returns:
            pd.Series: Data with outliers clipped
        """
        if self.method == 'z-score':
            mean = self.stats[column]['mean']
            std = self.stats[column]['std']
            lower_bound = mean - self.threshold * std
            upper_bound = mean + self.threshold * std
            return data.clip(lower=lower_bound, upper=upper_bound)
        
        elif self.method == 'iqr':
            q1 = self.stats[column]['q1']
            q3 = self.stats[column]['q3']
            iqr = self.stats[column]['iqr']
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            return data.clip(lower=lower_bound, upper=upper_bound)
        
        elif self.method == 'percentile':
            lower = self.stats[column]['lower']
            upper = self.stats[column]['upper']
            return data.clip(lower=lower, upper=upper)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")
    
    def _replace_outliers(self, data: pd.Series, outliers: pd.Series) -> pd.Series:
        """
        Replace outliers with NaN.
        
        Args:
            data: Column data
            outliers: Boolean mask of outliers
            
        Returns:
            pd.Series: Data with outliers replaced
        """
        result = data.copy()
        result[outliers] = np.nan
        return result
