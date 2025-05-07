"""
Evaluation module for time series forecasting.

This module provides components for evaluating forecasting models,
including metrics, backtesting, and cross-validation.
"""

from .metrics import Metrics
from .backtesting import Backtesting, BacktestingStrategy
from .cross_validation import (
    time_series_split,
    blocked_time_series_split,
    rolling_window_split,
    plot_time_series_split
)

__all__ = [
    'Metrics',
    'Backtesting',
    'BacktestingStrategy',
    'time_series_split',
    'blocked_time_series_split',
    'rolling_window_split',
    'plot_time_series_split'
]
