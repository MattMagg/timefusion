"""
Tests for the deep learning model base class.
"""

import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta

from timefusion.models.deep_learning import DeepLearningModel


# Create a concrete implementation of DeepLearningModel for testing
class DummyNet(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape input: [batch_size, seq_len, features] -> [batch_size, seq_len * features]
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DummyDeepLearningModel(DeepLearningModel):
    def __init__(
        self,
        name="dummy_dl",
        hidden_size=32,
        sequence_length=10,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=10,
        device="cpu",
        **kwargs
    ):
        super().__init__(
            name=name,
            sequence_length=sequence_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=device,
            hidden_size=hidden_size,  # Pass hidden_size to parent class
            **kwargs
        )
        self.hidden_size = hidden_size

    def _create_model(self, input_size, output_size=1):
        return DummyNet(
            input_size=input_size * self.sequence_length,
            hidden_size=self.hidden_size,
            output_size=output_size
        ).to(self.device)

    def get_params(self) -> dict:
        """
        Get the model parameters.

        Returns:
            Dict[str, Any]: Model parameters
        """
        params = super().get_params()
        params.update({"hidden_size": self.hidden_size})
        return params


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


def test_deep_learning_model_init():
    """Test DeepLearningModel initialization."""
    model = DummyDeepLearningModel(
        name="test_dl",
        hidden_size=64,
        sequence_length=5,
        batch_size=16,
        learning_rate=0.01,
        num_epochs=20
    )
    assert model.name == "test_dl"
    assert model.is_fitted is False
    assert model.hidden_size == 64
    assert model.sequence_length == 5
    assert model.batch_size == 16
    assert model.learning_rate == 0.01
    assert model.num_epochs == 20
    assert model.device == "cpu"  # Using CPU for testing
    assert model.model is None
    assert model.target_column is None
    assert model.feature_columns is None


def test_deep_learning_model_fit():
    """Test DeepLearningModel fit method."""
    model = DummyDeepLearningModel(num_epochs=1)  # Use 1 epoch for faster testing
    data = generate_sample_data()
    model.fit(data, "value")
    assert model.is_fitted is True
    assert model.target_column == "value"
    assert model.feature_columns == ['day_of_week', 'month', 'day_of_year']
    assert model.model is not None
    assert isinstance(model.model, nn.Module)


def test_deep_learning_model_fit_with_feature_columns():
    """Test DeepLearningModel fit method with specified feature columns."""
    model = DummyDeepLearningModel(num_epochs=1)  # Use 1 epoch for faster testing
    data = generate_sample_data()
    model.fit(data, "value", feature_columns=["day_of_week"])
    assert model.is_fitted is True
    assert model.target_column == "value"
    assert model.feature_columns == ["day_of_week"]
    assert model.model is not None
    assert isinstance(model.model, nn.Module)


def test_deep_learning_model_predict():
    """Test DeepLearningModel predict method."""
    model = DummyDeepLearningModel(num_epochs=1)  # Use 1 epoch for faster testing
    data = generate_sample_data()
    model.fit(data, "value")

    # Generate forecasts
    horizon = 10
    forecast = model.predict(data, horizon)

    # Check forecast
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == horizon
    assert "value" in forecast.columns


def test_deep_learning_model_get_params():
    """Test DeepLearningModel get_params method."""
    model = DummyDeepLearningModel(
        name="test_dl",
        hidden_size=64,
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
    assert "sequence_length" in params
    assert "batch_size" in params
    assert "learning_rate" in params
    assert "num_epochs" in params
    assert params["hidden_size"] == 64
    assert params["sequence_length"] == 5
    assert params["batch_size"] == 16
    assert params["learning_rate"] == 0.01
    assert params["num_epochs"] == 20


def test_deep_learning_model_predict_not_fitted():
    """Test DeepLearningModel predict method when not fitted."""
    model = DummyDeepLearningModel()
    data = generate_sample_data()

    # Try to generate forecasts without fitting
    with pytest.raises(ValueError, match="Model is not fitted"):
        model.predict(data, horizon=10)
