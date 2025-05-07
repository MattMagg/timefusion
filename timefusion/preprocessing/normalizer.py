"""
Normalizer for time series data.

This module provides the Normalizer class for scaling and normalizing
time series data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

from .base import BasePreprocessor


class Normalizer(BasePreprocessor):
    """
    Normalizer for time series data.
    
    This class provides methods for scaling and normalizing time series data.
    It supports multiple normalization methods and can handle both univariate
    and multivariate time series.
    
    Attributes:
        name (str): Name of the normalizer
        is_fitted (bool): Whether the normalizer has been fitted
        params (Dict[str, Any]): Normalizer parameters
        method (str): Normalization method
        columns (Optional[List[str]]): Columns to normalize
        stats (Dict[str, Dict[str, float]]): Statistics for each column
    """
    
    def __init__(
        self,
        name: str = "normalizer",
        method: str = "min-max",
        columns: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the normalizer.
        
        Args:
            name: Name of the normalizer
            method: Normalization method ('min-max', 'z-score', 'robust')
            columns: Columns to normalize (if None, all numeric columns are normalized)
            **kwargs: Additional parameters for the normalizer
        """
        super().__init__(name=name, **kwargs)
        self.method = method
        self.columns = columns
        self.stats = {}
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'Normalizer':
        """
        Fit the normalizer to the data.
        
        Args:
            data: Input data
            **kwargs: Additional parameters for fitting
            
        Returns:
            self: The fitted normalizer
        """
        # Determine columns to normalize
        if self.columns is None:
            self.columns = data.select_dtypes(include=np.number).columns.tolist()
        
        # Calculate statistics for each column
        for column in self.columns:
            if column not in data.columns:
                continue
            
            column_data = data[column].dropna()
            
            if self.method == 'min-max':
                self.stats[column] = {
                    'min': column_data.min(),
                    'max': column_data.max()
                }
            elif self.method == 'z-score':
                self.stats[column] = {
                    'mean': column_data.mean(),
                    'std': column_data.std()
                }
            elif self.method == 'robust':
                q1 = column_data.quantile(0.25)
                q3 = column_data.quantile(0.75)
                iqr = q3 - q1
                self.stats[column] = {
                    'median': column_data.median(),
                    'iqr': iqr
                }
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply the normalizer to the data.
        
        Args:
            data: Input data
            **kwargs: Additional parameters for transformation
            
        Returns:
            pd.DataFrame: Normalized data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted")
        
        # Create a copy of the data
        normalized_data = data.copy()
        
        # Normalize each column
        for column in self.columns:
            if column not in normalized_data.columns:
                continue
            
            if self.method == 'min-max':
                normalized_data[column] = self._min_max_scale(normalized_data[column], column)
            elif self.method == 'z-score':
                normalized_data[column] = self._z_score_scale(normalized_data[column], column)
            elif self.method == 'robust':
                normalized_data[column] = self._robust_scale(normalized_data[column], column)
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
        
        return normalized_data
    
    def inverse_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Inverse transform the normalized data.
        
        Args:
            data: Normalized data
            **kwargs: Additional parameters for inverse transformation
            
        Returns:
            pd.DataFrame: Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted")
        
        # Create a copy of the data
        original_data = data.copy()
        
        # Inverse transform each column
        for column in self.columns:
            if column not in original_data.columns:
                continue
            
            if self.method == 'min-max':
                original_data[column] = self._inverse_min_max_scale(original_data[column], column)
            elif self.method == 'z-score':
                original_data[column] = self._inverse_z_score_scale(original_data[column], column)
            elif self.method == 'robust':
                original_data[column] = self._inverse_robust_scale(original_data[column], column)
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
        
        return original_data
    
    def _min_max_scale(self, data: pd.Series, column: str) -> pd.Series:
        """
        Apply min-max scaling to a column.
        
        Args:
            data: Column data
            column: Column name
            
        Returns:
            pd.Series: Scaled data
        """
        min_val = self.stats[column]['min']
        max_val = self.stats[column]['max']
        
        # Handle the case where min == max
        if min_val == max_val:
            return pd.Series(np.zeros(len(data)), index=data.index)
        
        return (data - min_val) / (max_val - min_val)
    
    def _inverse_min_max_scale(self, data: pd.Series, column: str) -> pd.Series:
        """
        Inverse min-max scaling.
        
        Args:
            data: Scaled data
            column: Column name
            
        Returns:
            pd.Series: Original scale data
        """
        min_val = self.stats[column]['min']
        max_val = self.stats[column]['max']
        
        return data * (max_val - min_val) + min_val
    
    def _z_score_scale(self, data: pd.Series, column: str) -> pd.Series:
        """
        Apply z-score scaling to a column.
        
        Args:
            data: Column data
            column: Column name
            
        Returns:
            pd.Series: Scaled data
        """
        mean = self.stats[column]['mean']
        std = self.stats[column]['std']
        
        # Handle the case where std == 0
        if std == 0:
            return pd.Series(np.zeros(len(data)), index=data.index)
        
        return (data - mean) / std
    
    def _inverse_z_score_scale(self, data: pd.Series, column: str) -> pd.Series:
        """
        Inverse z-score scaling.
        
        Args:
            data: Scaled data
            column: Column name
            
        Returns:
            pd.Series: Original scale data
        """
        mean = self.stats[column]['mean']
        std = self.stats[column]['std']
        
        return data * std + mean
    
    def _robust_scale(self, data: pd.Series, column: str) -> pd.Series:
        """
        Apply robust scaling to a column.
        
        Args:
            data: Column data
            column: Column name
            
        Returns:
            pd.Series: Scaled data
        """
        median = self.stats[column]['median']
        iqr = self.stats[column]['iqr']
        
        # Handle the case where iqr == 0
        if iqr == 0:
            return pd.Series(np.zeros(len(data)), index=data.index)
        
        return (data - median) / iqr
    
    def _inverse_robust_scale(self, data: pd.Series, column: str) -> pd.Series:
        """
        Inverse robust scaling.
        
        Args:
            data: Scaled data
            column: Column name
            
        Returns:
            pd.Series: Original scale data
        """
        median = self.stats[column]['median']
        iqr = self.stats[column]['iqr']
        
        return data * iqr + median
