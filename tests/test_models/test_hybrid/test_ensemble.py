"""
Tests for the ensemble model.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from timefusion.models.hybrid import EnsembleModel
from timefusion.models.base import BaseModel


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


def test_ensemble_init():
    """Test EnsembleModel initialization."""
    model1 = DummyModel(name="model1", constant_value=1.0)
    model2 = DummyModel(name="model2", constant_value=2.0)
    ensemble = EnsembleModel(
        name="test_ensemble",
        models=[model1, model2],
        weights=[0.3, 0.7],
        ensemble_method="weighted_average"
    )
    assert ensemble.name == "test_ensemble"
    assert ensemble.is_fitted is False
    assert len(ensemble.models) == 2
    assert ensemble.models[0].name == "model1"
    assert ensemble.models[1].name == "model2"
    assert ensemble.weights == [0.3, 0.7]
    assert ensemble.ensemble_method == "weighted_average"
    assert ensemble.target_column is None


def test_ensemble_init_with_invalid_weights():
    """Test EnsembleModel initialization with invalid weights."""
    model1 = DummyModel(name="model1")
    model2 = DummyModel(name="model2")
    
    # Test with weights not summing to 1.0
    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        EnsembleModel(
            name="test_ensemble",
            models=[model1, model2],
            weights=[0.3, 0.3]
        )
    
    # Test with wrong number of weights
    with pytest.raises(ValueError, match="Number of weights must match number of models"):
        EnsembleModel(
            name="test_ensemble",
            models=[model1, model2],
            weights=[0.3, 0.4, 0.3]
        )


def test_ensemble_init_with_auto_weights():
    """Test EnsembleModel initialization with automatic weight initialization."""
    model1 = DummyModel(name="model1")
    model2 = DummyModel(name="model2")
    ensemble = EnsembleModel(
        name="test_ensemble",
        models=[model1, model2]
    )
    assert ensemble.weights == [0.5, 0.5]


def test_ensemble_add_model():
    """Test EnsembleModel add_model method."""
    ensemble = EnsembleModel(name="test_ensemble")
    model1 = DummyModel(name="model1")
    model2 = DummyModel(name="model2")
    
    # Add models
    ensemble.add_model(model1)
    assert len(ensemble.models) == 1
    assert ensemble.weights == [1.0]
    
    # Add another model
    ensemble.add_model(model2)
    assert len(ensemble.models) == 2
    assert ensemble.weights == [0.5, 0.5]
    
    # Add model with weight
    model3 = DummyModel(name="model3")
    ensemble.add_model(model3, weight=0.4)
    assert len(ensemble.models) == 3
    assert np.isclose(sum(ensemble.weights), 1.0)
    assert ensemble.weights[2] == 0.4


def test_ensemble_fit():
    """Test EnsembleModel fit method."""
    model1 = DummyModel(name="model1", constant_value=1.0)
    model2 = DummyModel(name="model2", constant_value=2.0)
    ensemble = EnsembleModel(
        name="test_ensemble",
        models=[model1, model2],
        weights=[0.3, 0.7]
    )
    data = generate_sample_data()
    ensemble.fit(data, "value")
    assert ensemble.is_fitted is True
    assert ensemble.target_column == "value"
    assert model1.is_fitted is True
    assert model2.is_fitted is True


def test_ensemble_predict_weighted_average():
    """Test EnsembleModel predict method with weighted_average ensemble method."""
    model1 = DummyModel(name="model1", constant_value=1.0)
    model2 = DummyModel(name="model2", constant_value=2.0)
    ensemble = EnsembleModel(
        name="test_ensemble",
        models=[model1, model2],
        weights=[0.3, 0.7],
        ensemble_method="weighted_average"
    )
    data = generate_sample_data()
    ensemble.fit(data, "value")
    
    # Generate forecasts
    horizon = 10
    forecast = ensemble.predict(data, horizon)
    
    # Check forecast
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == horizon
    assert "value" in forecast.columns
    
    # Check weighted average: 0.3 * 1.0 + 0.7 * 2.0 = 0.3 + 1.4 = 1.7
    expected_value = 0.3 * 1.0 + 0.7 * 2.0
    assert np.allclose(forecast["value"].values, expected_value)


def test_ensemble_predict_mean():
    """Test EnsembleModel predict method with mean ensemble method."""
    model1 = DummyModel(name="model1", constant_value=1.0)
    model2 = DummyModel(name="model2", constant_value=2.0)
    model3 = DummyModel(name="model3", constant_value=3.0)
    ensemble = EnsembleModel(
        name="test_ensemble",
        models=[model1, model2, model3],
        ensemble_method="mean"
    )
    data = generate_sample_data()
    ensemble.fit(data, "value")
    
    # Generate forecasts
    horizon = 10
    forecast = ensemble.predict(data, horizon)
    
    # Check forecast
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == horizon
    assert "value" in forecast.columns
    
    # Check mean: (1.0 + 2.0 + 3.0) / 3 = 2.0
    expected_value = (1.0 + 2.0 + 3.0) / 3
    assert np.allclose(forecast["value"].values, expected_value)


def test_ensemble_predict_median():
    """Test EnsembleModel predict method with median ensemble method."""
    model1 = DummyModel(name="model1", constant_value=1.0)
    model2 = DummyModel(name="model2", constant_value=2.0)
    model3 = DummyModel(name="model3", constant_value=3.0)
    ensemble = EnsembleModel(
        name="test_ensemble",
        models=[model1, model2, model3],
        ensemble_method="median"
    )
    data = generate_sample_data()
    ensemble.fit(data, "value")
    
    # Generate forecasts
    horizon = 10
    forecast = ensemble.predict(data, horizon)
    
    # Check forecast
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == horizon
    assert "value" in forecast.columns
    
    # Check median: median(1.0, 2.0, 3.0) = 2.0
    expected_value = 2.0
    assert np.allclose(forecast["value"].values, expected_value)


def test_ensemble_predict_invalid_method():
    """Test EnsembleModel predict method with invalid ensemble method."""
    model1 = DummyModel(name="model1")
    model2 = DummyModel(name="model2")
    ensemble = EnsembleModel(
        name="test_ensemble",
        models=[model1, model2],
        ensemble_method="invalid"
    )
    data = generate_sample_data()
    ensemble.fit(data, "value")
    
    # Try to generate forecasts with invalid method
    with pytest.raises(ValueError, match="Unknown ensemble method"):
        ensemble.predict(data, horizon=10)


def test_ensemble_get_weights():
    """Test EnsembleModel get_weights method."""
    model1 = DummyModel(name="model1")
    model2 = DummyModel(name="model2")
    ensemble = EnsembleModel(
        name="test_ensemble",
        models=[model1, model2],
        weights=[0.3, 0.7]
    )
    
    # Get weights
    weights = ensemble.get_weights()
    
    # Check weights
    assert isinstance(weights, dict)
    assert weights == {"model1": 0.3, "model2": 0.7}


def test_ensemble_get_params():
    """Test EnsembleModel get_params method."""
    model1 = DummyModel(name="model1")
    model2 = DummyModel(name="model2")
    ensemble = EnsembleModel(
        name="test_ensemble",
        models=[model1, model2],
        weights=[0.3, 0.7],
        ensemble_method="weighted_average"
    )
    data = generate_sample_data()
    ensemble.fit(data, "value")
    
    # Get parameters
    params = ensemble.get_params()
    
    # Check parameters
    assert isinstance(params, dict)
    assert "models" in params
    assert "weights" in params
    assert "ensemble_method" in params
    assert params["models"] == ["model1", "model2"]
    assert params["weights"] == [0.3, 0.7]
    assert params["ensemble_method"] == "weighted_average"
