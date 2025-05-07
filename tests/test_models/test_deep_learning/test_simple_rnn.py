"""
Tests for the SimpleRNN model.
"""

import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

from timefusion.models.deep_learning import SimpleRNNModel


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
    
    # Add some additional features for multivariate forecasting
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['day_of_year'] = data.index.dayofyear
    
    return data


def test_simple_rnn_init():
    """Test SimpleRNNModel initialization."""
    model = SimpleRNNModel(
        name="test_rnn",
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        sequence_length=5,
        batch_size=16,
        learning_rate=0.01,
        num_epochs=20
    )
    assert model.name == "test_rnn"
    assert model.is_fitted is False
    assert model.hidden_size == 32
    assert model.num_layers == 1
    assert model.dropout == 0.1
    assert model.sequence_length == 5
    assert model.batch_size == 16
    assert model.learning_rate == 0.01
    assert model.num_epochs == 20
    assert model.device == "cpu"  # Using CPU for testing
    assert model.model is None
    assert model.target_column is None
    assert model.feature_columns is None


def test_simple_rnn_create_model():
    """Test SimpleRNNModel _create_model method."""
    model = SimpleRNNModel(
        name="test_rnn",
        hidden_size=32,
        num_layers=1,
        dropout=0.1
    )
    
    # Create model
    rnn_model = model._create_model(input_size=3)
    
    # Check model
    assert rnn_model is not None
    assert isinstance(rnn_model, torch.nn.Module)
    
    # Check forward pass
    batch_size = 5
    seq_len = model.sequence_length
    input_size = 3
    x = torch.randn(batch_size, seq_len, input_size)
    output = rnn_model(x)
    assert output.shape == (batch_size, 1)


def test_simple_rnn_fit():
    """Test SimpleRNNModel fit method."""
    model = SimpleRNNModel(
        name="test_rnn",
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        num_epochs=1  # Use 1 epoch for faster testing
    )
    data = generate_sample_data()
    model.fit(data, "value")
    assert model.is_fitted is True
    assert model.target_column == "value"
    assert model.feature_columns == ['day_of_week', 'month', 'day_of_year']
    assert model.model is not None


def test_simple_rnn_predict():
    """Test SimpleRNNModel predict method."""
    model = SimpleRNNModel(
        name="test_rnn",
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        num_epochs=1  # Use 1 epoch for faster testing
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


def test_simple_rnn_get_params():
    """Test SimpleRNNModel get_params method."""
    model = SimpleRNNModel(
        name="test_rnn",
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        sequence_length=5,
        batch_size=16,
        learning_rate=0.01,
        num_epochs=20
    )
    data = generate_sample_data()
    model.fit(data, "value")
    
    # Get parameters
    params = model.get_params()
    
    # Check parameters
    assert isinstance(params, dict)
    assert "hidden_size" in params
    assert "num_layers" in params
    assert "dropout" in params
    assert "sequence_length" in params
    assert "batch_size" in params
    assert "learning_rate" in params
    assert "num_epochs" in params
    assert params["hidden_size"] == 32
    assert params["num_layers"] == 1
    assert params["dropout"] == 0.1
    assert params["sequence_length"] == 5
    assert params["batch_size"] == 16
    assert params["learning_rate"] == 0.01
    assert params["num_epochs"] == 20
