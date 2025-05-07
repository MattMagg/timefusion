"""
Tests for the hybrid model base class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from timefusion.models.hybrid import HybridModel
from timefusion.models.base import BaseModel


# Create a concrete implementation of HybridModel for testing
class DummyHybridModel(HybridModel):
    def fit(self, data, target_column, **kwargs):
        self.target_column = target_column
        self.is_fitted = True
        return self
    
    def predict(self, data, horizon, **kwargs):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        forecast_index = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq=pd.infer_freq(data.index)
        )
        return pd.DataFrame({self.target_column: np.ones(horizon)}, index=forecast_index)


# Create a dummy model for testing
class DummyModel(BaseModel):
    def __init__(self, name, constant_value=1.0, **kwargs):
        super().__init__(name, **kwargs)
        self.constant_value = constant_value
    
    def fit(self, data, target_column, **kwargs):
        self.target_column = target_column
        self.is_fitted = True
        return self
    
    def predict(self, data, horizon, **kwargs):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        forecast_index = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq=pd.infer_freq(data.index)
        )
        return pd.DataFrame({self.target_column: np.ones(horizon) * self.constant_value}, index=forecast_index)


def generate_sample_data(n_samples=50, freq='D', seed=42):
    """
    Generate sample time series data with trend, seasonality, and noise.
    
    Args:
        n_samples: Number of samples to generate
        freq: Frequency of the time series
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Sample time series data
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq=freq)
    
    # Create trend component
    trend = np.linspace(0, 10, n_samples)
    
    # Create seasonal component (weekly seasonality)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
    
    # Create noise component
    noise = np.random.normal(0, 1, n_samples)
    
    # Combine components
    values = trend + seasonal + noise
    
    # Create DataFrame
    data = pd.DataFrame({'value': values}, index=dates)
    
    return data


def test_hybrid_model_init():
    """Test HybridModel initialization."""
    model1 = DummyModel(name="model1")
    model2 = DummyModel(name="model2")
    hybrid_model = DummyHybridModel(name="test_hybrid", models=[model1, model2])
    assert hybrid_model.name == "test_hybrid"
    assert hybrid_model.is_fitted is False
    assert len(hybrid_model.models) == 2
    assert hybrid_model.models[0].name == "model1"
    assert hybrid_model.models[1].name == "model2"
    assert hybrid_model.target_column is None


def test_hybrid_model_add_model():
    """Test HybridModel add_model method."""
    hybrid_model = DummyHybridModel(name="test_hybrid")
    model1 = DummyModel(name="model1")
    model2 = DummyModel(name="model2")
    
    # Add models
    hybrid_model.add_model(model1)
    hybrid_model.add_model(model2)
    
    # Check models
    assert len(hybrid_model.models) == 2
    assert hybrid_model.models[0].name == "model1"
    assert hybrid_model.models[1].name == "model2"


def test_hybrid_model_fit():
    """Test HybridModel fit method."""
    hybrid_model = DummyHybridModel(name="test_hybrid")
    data = generate_sample_data()
    hybrid_model.fit(data, "value")
    assert hybrid_model.is_fitted is True
    assert hybrid_model.target_column == "value"


def test_hybrid_model_predict():
    """Test HybridModel predict method."""
    hybrid_model = DummyHybridModel(name="test_hybrid")
    data = generate_sample_data()
    hybrid_model.fit(data, "value")
    
    # Generate forecasts
    horizon = 10
    forecast = hybrid_model.predict(data, horizon)
    
    # Check forecast
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == horizon
    assert "value" in forecast.columns
    assert (forecast["value"] == 1).all()


def test_hybrid_model_get_model_contributions():
    """Test HybridModel get_model_contributions method."""
    model1 = DummyModel(name="model1", constant_value=1.0)
    model2 = DummyModel(name="model2", constant_value=2.0)
    hybrid_model = DummyHybridModel(name="test_hybrid", models=[model1, model2])
    data = generate_sample_data()
    
    # Fit models
    model1.fit(data, "value")
    model2.fit(data, "value")
    hybrid_model.fit(data, "value")
    
    # Get model contributions
    horizon = 10
    contributions = hybrid_model.get_model_contributions(data, horizon)
    
    # Check contributions
    assert isinstance(contributions, dict)
    assert len(contributions) == 2
    assert "model1" in contributions
    assert "model2" in contributions
    assert isinstance(contributions["model1"], pd.DataFrame)
    assert isinstance(contributions["model2"], pd.DataFrame)
    assert len(contributions["model1"]) == horizon
    assert len(contributions["model2"]) == horizon
    assert (contributions["model1"]["value"] == 1.0).all()
    assert (contributions["model2"]["value"] == 2.0).all()


def test_hybrid_model_get_params():
    """Test HybridModel get_params method."""
    model1 = DummyModel(name="model1")
    model2 = DummyModel(name="model2")
    hybrid_model = DummyHybridModel(name="test_hybrid", models=[model1, model2])
    data = generate_sample_data()
    hybrid_model.fit(data, "value")
    
    # Get parameters
    params = hybrid_model.get_params()
    
    # Check parameters
    assert isinstance(params, dict)
    assert "models" in params
    assert params["models"] == ["model1", "model2"]
