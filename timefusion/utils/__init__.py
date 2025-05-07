"""
Utilities module for time series forecasting.

This module provides cross-cutting functionality used by all other modules,
including configuration management, logging, visualization, and hyperparameter
optimization.
"""

from .config import Config, get_default_config
from .logging import setup_logger
from .visualization import plot_forecast, plot_comparison
# These will be implemented in Phase 8
# from .hpo import GridSearch, RandomSearch

__all__ = [
    'Config',
    'get_default_config',
    'setup_logger',
    'plot_forecast',
    'plot_comparison',
    # 'GridSearch',
    # 'RandomSearch',
]
