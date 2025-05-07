"""
Tests for the Exponential Smoothing model.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from timefusion.models.statistical import ExponentialSmoothingModel


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


def test_ets_init():
    """Test ExponentialSmoothingModel initialization."""
    model = ExponentialSmoothingModel(
        name="test_ets",
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        damped_trend=True
    )
    assert model.name == "test_ets"
    assert model.is_fitted is False
    assert model.trend == "add"
    assert model.seasonal == "add"
    assert model.seasonal_periods == 7
    assert model.damped_trend is True


def test_ets_fit():
    """Test ExponentialSmoothingModel fit method."""
    model = ExponentialSmoothingModel(
        name="test_ets",
        trend="add",
        seasonal="add",
        seasonal_periods=7
    )
    data = generate_sample_data()
    model.fit(data, "value")
    assert model.is_fitted is True
    assert model.target_column == "value"
    assert model.model is not None
    assert model.result is not None


def test_ets_predict():
    """Test ExponentialSmoothingModel predict method."""
    model = ExponentialSmoothingModel(
        name="test_ets",
        trend="add",
        seasonal="add",
        seasonal_periods=7
    )
    data = generate_sample_data()
    model.fit(data, "value")
    
    # Generate forecasts
    horizon = 10
    forecast = model.predict(data, horizon)
    
    # Check forecast
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == horizon
    assert "value" in forecast.columns


def test_ets_predict_with_confidence():
    """Test ExponentialSmoothingModel predict_with_confidence method."""
    model = ExponentialSmoothingModel(
        name="test_ets",
        trend="add",
        seasonal="add",
        seasonal_periods=7
    )
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


def test_ets_get_params():
    """Test ExponentialSmoothingModel get_params method."""
    model = ExponentialSmoothingModel(
        name="test_ets",
        trend="add",
        seasonal="add",
        seasonal_periods=7
    )
    data = generate_sample_data()
    model.fit(data, "value")
    
    # Get parameters
    params = model.get_params()
    
    # Check parameters
    assert isinstance(params, dict)
    assert "trend" in params
    assert "seasonal" in params
    assert "seasonal_periods" in params
    assert "damped_trend" in params


def test_ets_auto_params():
    """Test ExponentialSmoothingModel auto_params parameter."""
    model = ExponentialSmoothingModel(
        name="test_ets",
        seasonal_periods=7,
        auto_params=True
    )
    data = generate_sample_data()
    model.fit(data, "value")
    
    # Check that parameters were selected
    assert model.trend is not None
    assert model.seasonal is not None
    assert model.seasonal_periods == 7
