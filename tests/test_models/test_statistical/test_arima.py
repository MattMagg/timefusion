"""
Tests for the ARIMA model.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from timefusion.models.statistical import ARIMAModel


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


def test_arima_init():
    """Test ARIMAModel initialization."""
    model = ARIMAModel(name="test_arima", order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    assert model.name == "test_arima"
    assert model.is_fitted is False
    assert model.order == (1, 1, 1)
    assert model.seasonal_order == (1, 1, 1, 7)


def test_arima_fit():
    """Test ARIMAModel fit method."""
    model = ARIMAModel(name="test_arima", order=(1, 1, 0))
    data = generate_sample_data()
    model.fit(data, "value")
    assert model.is_fitted is True
    assert model.target_column == "value"
    assert model.model is not None
    assert model.result is not None


def test_arima_predict():
    """Test ARIMAModel predict method."""
    model = ARIMAModel(name="test_arima", order=(1, 1, 0))
    data = generate_sample_data()
    model.fit(data, "value")
    
    # Generate forecasts
    horizon = 10
    forecast = model.predict(data, horizon)
    
    # Check forecast
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == horizon
    assert "value" in forecast.columns


def test_arima_predict_with_confidence():
    """Test ARIMAModel predict_with_confidence method."""
    model = ARIMAModel(name="test_arima", order=(1, 1, 0))
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
    
    # Check confidence intervals
    assert (forecast["value_lower_95"] <= forecast["value"]).all()
    assert (forecast["value_upper_95"] >= forecast["value"]).all()


def test_arima_get_params():
    """Test ARIMAModel get_params method."""
    model = ARIMAModel(name="test_arima", order=(1, 1, 0))
    data = generate_sample_data()
    model.fit(data, "value")
    
    # Get parameters
    params = model.get_params()
    
    # Check parameters
    assert isinstance(params, dict)
    assert "order" in params
    assert "seasonal_order" in params
    assert "trend" in params
    assert "aic" in params
    assert "bic" in params
    assert "coefficients" in params


def test_arima_auto_order():
    """Test ARIMAModel auto_order parameter."""
    model = ARIMAModel(name="test_arima", auto_order=True)
    data = generate_sample_data()
    model.fit(data, "value")
    
    # Check that order was selected
    assert model.order is not None
    assert len(model.order) == 3
    assert all(isinstance(x, int) for x in model.order)
