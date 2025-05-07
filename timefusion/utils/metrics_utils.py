"""
Utility functions for metrics calculations.

This module provides helper functions for safe metric calculations,
handling edge cases, and standardized input preprocessing.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Callable
import warnings


def preprocess_inputs(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: Union[np.ndarray, pd.Series],
    remove_nan: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess inputs for metric calculations.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        remove_nan: Whether to remove NaN values
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed true and predicted values
    """
    # Convert inputs to numpy arrays
    y_true_array = np.asarray(y_true).astype(float)
    y_pred_array = np.asarray(y_pred).astype(float)
    
    # Check if arrays have the same shape
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Got {y_true_array.shape} and {y_pred_array.shape}")
    
    # Handle NaN values if requested
    if remove_nan:
        mask = ~(np.isnan(y_true_array) | np.isnan(y_pred_array))
        if not np.all(mask):
            warnings.warn(f"Removed {np.sum(~mask)} NaN values from inputs")
            y_true_array = y_true_array[mask]
            y_pred_array = y_pred_array[mask]
    
    return y_true_array, y_pred_array


def safe_divide(
    numerator: np.ndarray, 
    denominator: np.ndarray, 
    default_value: float = np.nan
) -> np.ndarray:
    """
    Safely divide arrays, handling division by zero.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        default_value: Value to use when denominator is zero
        
    Returns:
        np.ndarray: Result of division with zeros replaced by default_value
    """
    result = np.zeros_like(numerator, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    result[~mask] = default_value
    return result


def safe_metric(
    metric_fn: Callable, 
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: Union[np.ndarray, pd.Series],
    **kwargs
) -> float:
    """
    Safely compute a metric, handling edge cases.
    
    Args:
        metric_fn: Metric function to call
        y_true: True values
        y_pred: Predicted values
        **kwargs: Additional arguments for the metric function
        
    Returns:
        float: Computed metric value
    """
    try:
        y_true_array, y_pred_array = preprocess_inputs(y_true, y_pred)
        
        # Check if arrays are empty after preprocessing
        if len(y_true_array) == 0:
            warnings.warn("Empty arrays after preprocessing, returning NaN")
            return np.nan
        
        return metric_fn(y_true_array, y_pred_array, **kwargs)
    except Exception as e:
        warnings.warn(f"Error computing metric: {e}")
        return np.nan