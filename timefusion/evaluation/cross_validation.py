"""
Cross-validation utilities for time series forecasting.

This module provides functions for time series cross-validation,
including time series split and blocked time series split.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Iterator, Dict, Any
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def time_series_split(
    data: pd.DataFrame,
    n_splits: int = 5,
    test_size: Optional[int] = None,
    gap: int = 0
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split time series data into training and testing sets.
    
    Args:
        data: Input data
        n_splits: Number of splits
        test_size: Size of each test set (if None, calculated based on n_splits)
        gap: Gap between train and test sets
        
    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame]]: List of (train, test) pairs
    """
    if test_size is None:
        # Calculate test_size based on n_splits
        test_size = len(data) // (n_splits + 1)
    
    # Initialize results
    splits = []
    
    # Calculate split points
    data_size = len(data)
    for i in range(n_splits):
        # Calculate test start and end
        test_end = data_size - i * test_size
        test_start = test_end - test_size
        
        # Ensure test_start is valid
        if test_start < 0:
            break
        
        # Calculate train end (considering gap)
        train_end = test_start - gap
        
        # Ensure train_end is valid
        if train_end <= 0:
            break
        
        # Split data
        train = data.iloc[:train_end]
        test = data.iloc[test_start:test_end]
        
        # Store split
        splits.append((train, test))
    
    return splits


def blocked_time_series_split(
    data: pd.DataFrame,
    n_splits: int = 5,
    test_size: Optional[int] = None,
    train_size: Optional[int] = None,
    gap: int = 0
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split time series data into training and testing sets using blocked cross-validation.
    
    Args:
        data: Input data
        n_splits: Number of splits
        test_size: Size of each test set (if None, calculated based on n_splits)
        train_size: Size of each training set (if None, uses all available data)
        gap: Gap between train and test sets
        
    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame]]: List of (train, test) pairs
    """
    if test_size is None:
        # Calculate test_size based on n_splits
        test_size = len(data) // (n_splits + 1)
    
    # Initialize results
    splits = []
    
    # Calculate split points
    data_size = len(data)
    for i in range(n_splits):
        # Calculate test start and end
        test_end = data_size - i * test_size
        test_start = test_end - test_size
        
        # Ensure test_start is valid
        if test_start < 0:
            break
        
        # Calculate train end (considering gap)
        train_end = test_start - gap
        
        # Calculate train start
        if train_size is None:
            train_start = 0
        else:
            train_start = max(0, train_end - train_size)
        
        # Ensure train_end is valid
        if train_end <= train_start:
            break
        
        # Split data
        train = data.iloc[train_start:train_end]
        test = data.iloc[test_start:test_end]
        
        # Store split
        splits.append((train, test))
    
    return splits


def rolling_window_split(
    data: pd.DataFrame,
    window_size: int,
    step_size: int = 1,
    horizon: int = 1,
    min_train_size: Optional[int] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split time series data using a rolling window approach.
    
    Args:
        data: Input data
        window_size: Size of the rolling window
        step_size: Step size for moving the window
        horizon: Forecast horizon
        min_train_size: Minimum training set size
        
    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame]]: List of (train, test) pairs
    """
    if min_train_size is None:
        min_train_size = window_size
    
    # Initialize results
    splits = []
    
    # Calculate split points
    data_size = len(data)
    for i in range(min_train_size, data_size - horizon + 1, step_size):
        # Calculate train start and end
        train_start = max(0, i - window_size)
        train_end = i
        
        # Calculate test start and end
        test_start = train_end
        test_end = min(test_start + horizon, data_size)
        
        # Split data
        train = data.iloc[train_start:train_end]
        test = data.iloc[test_start:test_end]
        
        # Store split
        splits.append((train, test))
    
    return splits


def plot_time_series_split(
    data: pd.DataFrame,
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
    target_column: str,
    title: str = "Time Series Cross-Validation",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot time series cross-validation splits.
    
    Args:
        data: Input data
        splits: List of (train, test) pairs
        target_column: Name of the target column
        title: Plot title
        figsize: Figure size
    """
    if plt is None:
        raise ImportError("Matplotlib is required for plotting")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot full data
    ax.plot(data.index, data[target_column], label="Data", color="black", alpha=0.3)
    
    # Plot splits
    colors = plt.cm.tab10.colors
    for i, (train, test) in enumerate(splits):
        color = colors[i % len(colors)]
        
        # Plot train data
        ax.plot(train.index, train[target_column], label=f"Split {i+1} Train", color=color, alpha=0.6)
        
        # Plot test data
        ax.plot(test.index, test[target_column], label=f"Split {i+1} Test", color=color, linestyle="--", alpha=0.6)
    
    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel(target_column)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True)
    
    # Show plot
    plt.tight_layout()
    plt.show()
