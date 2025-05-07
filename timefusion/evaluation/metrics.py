"""
Evaluation metrics for time series forecasting.

This module provides metrics for evaluating forecasting models,
including common error metrics and accuracy measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings

from ..utils.metrics_utils import preprocess_inputs, safe_divide, safe_metric


class Metrics:
    """
    Evaluation metrics for time series forecasting.

    This class provides static methods for calculating common evaluation metrics
    for time series forecasting models.
    """

    @staticmethod
    def mse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate Mean Squared Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: Mean Squared Error
        """
        y_true, y_pred = preprocess_inputs(y_true, y_pred)
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def rmse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate Root Mean Squared Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: Root Mean Squared Error
        """
        return np.sqrt(Metrics.mse(y_true, y_pred))

    @staticmethod
    def mae(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate Mean Absolute Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: Mean Absolute Error
        """
        y_true, y_pred = preprocess_inputs(y_true, y_pred)
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate Mean Absolute Percentage Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: Mean Absolute Percentage Error
        """
        y_true, y_pred = preprocess_inputs(y_true, y_pred)
        
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            warnings.warn("All true values are zero, cannot calculate MAPE")
            return np.nan
            
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def smape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: Symmetric Mean Absolute Percentage Error
        """
        y_true, y_pred = preprocess_inputs(y_true, y_pred)
        
        # Calculate denominator
        denominator = np.abs(y_true) + np.abs(y_pred)
        mask = denominator != 0
        
        if not np.any(mask):
            warnings.warn("All denominators are zero, cannot calculate SMAPE")
            return np.nan
            
        return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100

    @staticmethod
    def mase(
        y_true: Union[np.ndarray, pd.Series], 
        y_pred: Union[np.ndarray, pd.Series],
        y_train: Optional[Union[np.ndarray, pd.Series]] = None, 
        seasonality: int = 1
    ) -> float:
        """
        Calculate Mean Absolute Scaled Error.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_train: Training data used to compute the scale (if None, y_true is used)
            seasonality: Seasonality period (default: 1 for non-seasonal data)

        Returns:
            float: Mean Absolute Scaled Error
        """
        y_true, y_pred = preprocess_inputs(y_true, y_pred)

        # If y_train is not provided, use y_true
        if y_train is None:
            y_train = y_true
        else:
            y_train = np.asarray(y_train).astype(float)

        # Calculate MAE
        mae = np.mean(np.abs(y_true - y_pred))

        # Check if training data is sufficient
        if len(y_train) <= seasonality:
            warnings.warn(f"Training data length ({len(y_train)}) must be greater than seasonality ({seasonality})")
            return np.nan

        # Calculate naive forecast errors
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])

        # Calculate MAE of naive forecast
        mae_naive = np.mean(naive_errors)

        # Avoid division by zero
        if mae_naive == 0:
            warnings.warn("Naive forecast error is zero, cannot calculate MASE")
            return np.nan

        return mae / mae_naive

    @staticmethod
    def r2(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate R-squared (coefficient of determination).

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: R-squared value
        """
        y_true, y_pred = preprocess_inputs(y_true, y_pred)

        # Calculate total sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        # Calculate residual sum of squares
        ss_res = np.sum((y_true - y_pred) ** 2)

        # Check for division by zero
        if ss_tot == 0:
            warnings.warn("Total sum of squares is zero, R² is undefined")
            return 0  

        return 1 - (ss_res / ss_tot)

    @staticmethod
    def wape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate Weighted Absolute Percentage Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            float: Weighted Absolute Percentage Error
        """
        y_true, y_pred = preprocess_inputs(y_true, y_pred)

        # Check for division by zero
        sum_abs_true = np.sum(np.abs(y_true))
        if sum_abs_true == 0:
            warnings.warn("Sum of absolute true values is zero, cannot calculate WAPE")
            return np.nan

        return np.sum(np.abs(y_true - y_pred)) / sum_abs_true * 100

    @staticmethod
    def calculate_metrics(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_train: Optional[Union[np.ndarray, pd.Series]] = None,
        seasonality: int = 1,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate multiple metrics at once.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_train: Training data used to compute the scale for MASE
            seasonality: Seasonality period for MASE
            metrics: List of metric names to calculate (default: ['mse', 'rmse', 'mae', 'mape', 'smape', 'mase', 'r2', 'wape'])

        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        if metrics is None:
            metrics = ['mse', 'rmse', 'mae', 'mape', 'smape', 'mase', 'r2', 'wape']

        # Define metric function mapping
        metric_funcs = {
            'mse': lambda y_t, y_p: Metrics.mse(y_t, y_p),
            'rmse': lambda y_t, y_p: Metrics.rmse(y_t, y_p),
            'mae': lambda y_t, y_p: Metrics.mae(y_t, y_p),
            'mape': lambda y_t, y_p: Metrics.mape(y_t, y_p),
            'smape': lambda y_t, y_p: Metrics.smape(y_t, y_p),
            'mase': lambda y_t, y_p: Metrics.mase(y_t, y_p, y_train, seasonality),
            'r2': lambda y_t, y_p: Metrics.r2(y_t, y_p),
            'wape': lambda y_t, y_p: Metrics.wape(y_t, y_p)
        }

        # Calculate requested metrics
        result = {}
        for metric in metrics:
            metric = metric.lower()
            if metric in metric_funcs:
                result[metric] = safe_metric(metric_funcs[metric], y_true, y_pred)
            else:
                warnings.warn(f"Unknown metric: {metric}")
                
        return result
