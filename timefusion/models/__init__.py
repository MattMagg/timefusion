"""
Models module for time series forecasting.

This module provides components for statistical, deep learning, and hybrid
forecasting models.
"""

# Import base classes
from .base import BaseModel, ModelRegistry

# Import model implementations
# Phase 4: Statistical Models
from .statistical import ARIMAModel, ExponentialSmoothingModel
# Phase 5: Deep Learning Models
from .deep_learning import LSTMModel, SimpleRNNModel
# Phase 6: Hybrid Models
from .hybrid import HybridModel, EnsembleModel, ResidualModel

__all__ = [
    'BaseModel',
    'ModelRegistry',
    'ARIMAModel',
    'ExponentialSmoothingModel',
    'LSTMModel',
    'SimpleRNNModel',
    'HybridModel',
    'EnsembleModel',
    'ResidualModel',
]
