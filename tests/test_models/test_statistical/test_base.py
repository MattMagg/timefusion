"""
Tests for the statistical model base class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from timefusion.models.statistical import StatisticalModel


# Create a concrete implementation of StatisticalModel for testing
class DummyStatisticalModel(StatisticalModel):
    def fit(self, data, target_column, **kwargs):
        self.target_column = target_column
        self.is_fitted = True
        return self
    
    def predict(self, data, horizon, **kwargs):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        forecast_index = self._create_forecast_index(data, horizon)
        return pd.DataFrame({self.target_column: np.ones(horizon)}, index=forecast_index)


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


def test_statistical_model_init():
    """Test StatisticalModel initialization."""
    model = DummyStatisticalModel(name="test_model", param1=1, param2="test")
    assert model.name == "test_model"
    assert model.is_fitted is False
    assert model.params == {"param1": 1, "param2": "test"}
    assert model.model is None
    assert model.result is None
    assert model.target_column is None


def test_statistical_model_fit():
    """Test StatisticalModel fit method."""
    model = DummyStatisticalModel(name="test_model")
    data = generate_sample_data()
    model.fit(data, "value")
    assert model.is_fitted is True
    assert model.target_column == "value"


def test_statistical_model_predict():
    """Test StatisticalModel predict method."""
    model = DummyStatisticalModel(name="test_model")
    data = generate_sample_data()
    model.fit(data, "value")
    
    # Generate forecasts
    horizon = 10
    forecast = model.predict(data, horizon)
    
    # Check forecast
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == horizon
    assert "value" in forecast.columns
    assert (forecast["value"] == 1).all()


def test_statistical_model_predict_with_confidence():
    """Test StatisticalModel predict_with_confidence method."""
    model = DummyStatisticalModel(name="test_model")
    data = generate_sample_data()
    model.fit(data, "value")
    
    # Generate forecasts with confidence intervals
    horizon = 10
    forecast = model.predict_with_confidence(data, horizon, confidence_level=0.95)
    
    # Check forecast
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == horizon
    assert "value" in forecast.columns
    assert "value_lower_95" in forecast.columns
    assert "value_upper_95" in forecast.columns
    
    # Check confidence intervals (default implementation returns same value for all)
    assert (forecast["value_lower_95"] == forecast["value"]).all()
    assert (forecast["value_upper_95"] == forecast["value"]).all()


def test_statistical_model_get_params():
    """Test StatisticalModel get_params method."""
    model = DummyStatisticalModel(name="test_model", param1=1, param2="test")
    data = generate_sample_data()
    model.fit(data, "value")
    
    # Get parameters
    params = model.get_params()
    
    # Check parameters
    assert isinstance(params, dict)
    assert params == {"param1": 1, "param2": "test"}


def test_statistical_model_create_forecast_index():
    """Test StatisticalModel _create_forecast_index method."""
    model = DummyStatisticalModel(name="test_model")
    
    # Test with DatetimeIndex
    data = generate_sample_data()
    horizon = 10
    forecast_index = model._create_forecast_index(data, horizon)
    assert isinstance(forecast_index, pd.DatetimeIndex)
    assert len(forecast_index) == horizon
    assert forecast_index[0] > data.index[-1]
    
    # Test with regular index
    data_regular = pd.DataFrame({'value': np.arange(50)})
    forecast_index_regular = model._create_forecast_index(data_regular, horizon)
    assert isinstance(forecast_index_regular, range)
    assert len(list(forecast_index_regular)) == horizon
    assert forecast_index_regular.start == data_regular.index[-1] + 1
