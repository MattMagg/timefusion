"""
Utilities module for time series forecasting.

This module provides cross-cutting functionality used by all other modules,
including configuration management, logging, visualization, time series
operations, metrics utilities, and hyperparameter optimization.
"""

from .config import Config, get_default_config
from .logging import setup_logger
from .visualization import plot_forecast, plot_comparison
from .time_series import (
    create_forecast_index,
    validate_fitted,
    validate_data,
    prepare_target_and_features
)
from .metrics_utils import (
    preprocess_inputs,
    safe_divide,
    safe_metric
)
# These will be implemented in Phase 8
# from .hpo import GridSearch, RandomSearch

__all__ = [
    'Config',
    'get_default_config',
    'setup_logger',
    'plot_forecast',
    'plot_comparison',
    'create_forecast_index',
    'validate_fitted',
    'validate_data',
    'prepare_target_and_features',
    'preprocess_inputs',
    'safe_divide',
    'safe_metric',
    # 'GridSearch',
    # 'RandomSearch',
]
