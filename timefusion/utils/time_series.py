"""
Time series utility functions.

This module provides utility functions for time series data manipulation,
index creation, and other common operations used across the library.
"""

import pandas as pd
from typing import Optional, Union


def create_forecast_index(data: pd.DataFrame, horizon: int) -> pd.Index:
    """
    Create an index for forecast DataFrames.
    
    This function is used to create appropriate index values for forecasts,
    handling both DatetimeIndex and regular numeric indices.
    
    Args:
        data: Input data used as reference for creating the forecast index
        horizon: Forecast horizon (number of periods to forecast)
        
    Returns:
        pd.Index: Index for the forecast DataFrame
    """
    if isinstance(data.index, pd.DatetimeIndex):
        # For time series data with DatetimeIndex
        last_date = data.index[-1]
        freq = pd.infer_freq(data.index)
        if freq is None:
            # If frequency cannot be inferred, assume daily
            freq = 'D'
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(1, unit=freq), 
            periods=horizon, 
            freq=freq
        )
    else:
        # For data without DatetimeIndex
        last_idx = data.index[-1]
        forecast_index = range(last_idx + 1, last_idx + horizon + 1)
    
    return forecast_index


def validate_fitted(is_fitted: bool, method_name: Optional[str] = None) -> None:
    """
    Validate that a model is fitted before performing operations.
    
    Args:
        is_fitted: Boolean indicating if the model is fitted
        method_name: Optional method name to include in the error message
        
    Raises:
        ValueError: If the model is not fitted
    """
    if not is_fitted:
        msg = "Model is not fitted. Call fit() first."
        if method_name:
            msg = f"Cannot call {method_name}(). {msg}"
        raise ValueError(msg)


def validate_data(
    data: pd.DataFrame, 
    required_columns: Optional[list] = None, 
    min_rows: Optional[int] = None
) -> None:
    """
    Validate input data for forecasting models.
    
    Args:
        data: Input DataFrame to validate
        required_columns: List of column names that must be present in the data
        min_rows: Minimum number of rows required
        
    Raises:
        ValueError: If validation fails
    """
    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Required columns missing: {missing_columns}")
    
    # Check minimum rows
    if min_rows and len(data) < min_rows:
        raise ValueError(f"Data must have at least {min_rows} rows (found {len(data)})")


def prepare_target_and_features(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[list] = None
) -> tuple:
    """
    Extract target and feature columns from data.
    
    Args:
        data: Input DataFrame
        target_column: Name of the target column
        feature_columns: List of feature column names (if None, use all columns except target)
        
    Returns:
        tuple: (feature_columns, X_data, y_data) where X_data and y_data are numpy arrays
    """
    # Validate target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Determine feature columns if not provided
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != target_column]
    
    # Extract features and target
    X_data = data[feature_columns].values
    y_data = data[target_column].values
    
    return feature_columns, X_data, y_data