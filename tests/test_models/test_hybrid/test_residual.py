"""
Tests for the residual model.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from timefusion.models.hybrid import ResidualModel
from timefusion.models.statistical import StatisticalModel
from timefusion.models.deep_learning import DeepLearningModel


# Create dummy models for testing
class DummyStatisticalModel(StatisticalModel):
    def __init__(self, name, constant_value=1.0, **kwargs):
        super().__init__(name, **kwargs)
        self.constant_value = constant_value
    
    def fit(self, data, target_column, **kwargs):
        self.target_column = target_column
        self.is_fitted = True
        return self
    
    def predict(self, data, horizon, in_sample=False, **kwargs):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if in_sample:
            # Return in-sample predictions
            result = data.copy()
            result[self.target_column] = self.constant_value
            return result
        else:
            # Return out-of-sample forecasts
            forecast_index = pd.date_range(
                start=data.index[-1] + pd.Timedelta(days=1),
                periods=horizon,
                freq=pd.infer_freq(data.index)
            )
            return pd.DataFrame({self.target_column: np.ones(horizon) * self.constant_value}, index=forecast_index)


class DummyDeepLearningModel(DeepLearningModel):
    def __init__(self, name, constant_value=0.5, **kwargs):
        super().__init__(name, **kwargs)
        self.constant_value = constant_value
    
    def _create_model(self, input_size, output_size=1):
        return None  # Not needed for testing
    
    def fit(self, data, target_column, feature_columns=None, **kwargs):
        self.target_column = target_column
        self.feature_columns = feature_columns or [col for col in data.columns if col != target_column]
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


def test_residual_init():
    """Test ResidualModel initialization."""
    stat_model = DummyStatisticalModel(name="stat_model", constant_value=1.0)
    dl_model = DummyDeepLearningModel(name="dl_model", constant_value=0.5)
    residual_model = ResidualModel(
        name="test_residual",
        statistical_model=stat_model,
        deep_learning_model=dl_model
    )
    assert residual_model.name == "test_residual"
    assert residual_model.is_fitted is False
    assert residual_model.statistical_model.name == "stat_model"
    assert residual_model.deep_learning_model.name == "dl_model"
    assert residual_model.target_column is None
    assert residual_model.residuals is None
    assert len(residual_model.models) == 2
    assert residual_model.models[0].name == "stat_model"
    assert residual_model.models[1].name == "dl_model"


def test_residual_fit():
    """Test ResidualModel fit method."""
    stat_model = DummyStatisticalModel(name="stat_model", constant_value=1.0)
    dl_model = DummyDeepLearningModel(name="dl_model", constant_value=0.5)
    residual_model = ResidualModel(
        name="test_residual",
        statistical_model=stat_model,
        deep_learning_model=dl_model
    )
    data = generate_sample_data()
    residual_model.fit(data, "value")
    assert residual_model.is_fitted is True
    assert residual_model.target_column == "value"
    assert stat_model.is_fitted is True
    assert dl_model.is_fitted is True
    assert residual_model.residuals is not None
    assert len(residual_model.residuals) == len(data)
    
    # Check residuals: original - stat_model prediction (which is constant 1.0)
    expected_residuals = data["value"] - 1.0
    assert np.allclose(residual_model.residuals["value"].values, expected_residuals.values)


def test_residual_fit_missing_models():
    """Test ResidualModel fit method with missing models."""
    # Test with missing statistical model
    residual_model = ResidualModel(
        name="test_residual",
        deep_learning_model=DummyDeepLearningModel(name="dl_model")
    )
    data = generate_sample_data()
    with pytest.raises(ValueError, match="Statistical model is required"):
        residual_model.fit(data, "value")
    
    # Test with missing deep learning model
    residual_model = ResidualModel(
        name="test_residual",
        statistical_model=DummyStatisticalModel(name="stat_model")
    )
    with pytest.raises(ValueError, match="Deep learning model is required"):
        residual_model.fit(data, "value")


def test_residual_predict():
    """Test ResidualModel predict method."""
    stat_model = DummyStatisticalModel(name="stat_model", constant_value=1.0)
    dl_model = DummyDeepLearningModel(name="dl_model", constant_value=0.5)
    residual_model = ResidualModel(
        name="test_residual",
        statistical_model=stat_model,
        deep_learning_model=dl_model
    )
    data = generate_sample_data()
    residual_model.fit(data, "value")
    
    # Generate forecasts
    horizon = 10
    forecast = residual_model.predict(data, horizon)
    
    # Check forecast
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == horizon
    assert "value" in forecast.columns
    
    # Check forecast values: stat_model (1.0) + dl_model (0.5) = 1.5
    expected_value = 1.0 + 0.5
    assert np.allclose(forecast["value"].values, expected_value)


def test_residual_get_component_forecasts():
    """Test ResidualModel get_component_forecasts method."""
    stat_model = DummyStatisticalModel(name="stat_model", constant_value=1.0)
    dl_model = DummyDeepLearningModel(name="dl_model", constant_value=0.5)
    residual_model = ResidualModel(
        name="test_residual",
        statistical_model=stat_model,
        deep_learning_model=dl_model
    )
    data = generate_sample_data()
    residual_model.fit(data, "value")
    
    # Get component forecasts
    horizon = 10
    component_forecasts = residual_model.get_component_forecasts(data, horizon)
    
    # Check component forecasts
    assert isinstance(component_forecasts, dict)
    assert len(component_forecasts) == 3
    assert "statistical" in component_forecasts
    assert "residual" in component_forecasts
    assert "combined" in component_forecasts
    
    # Check statistical forecast
    assert isinstance(component_forecasts["statistical"], pd.DataFrame)
    assert len(component_forecasts["statistical"]) == horizon
    assert "value" in component_forecasts["statistical"].columns
    assert np.allclose(component_forecasts["statistical"]["value"].values, 1.0)
    
    # Check residual forecast
    assert isinstance(component_forecasts["residual"], pd.DataFrame)
    assert len(component_forecasts["residual"]) == horizon
    assert "value" in component_forecasts["residual"].columns
    assert np.allclose(component_forecasts["residual"]["value"].values, 0.5)
    
    # Check combined forecast
    assert isinstance(component_forecasts["combined"], pd.DataFrame)
    assert len(component_forecasts["combined"]) == horizon
    assert "value" in component_forecasts["combined"].columns
    assert np.allclose(component_forecasts["combined"]["value"].values, 1.5)


def test_residual_get_params():
    """Test ResidualModel get_params method."""
    stat_model = DummyStatisticalModel(name="stat_model", constant_value=1.0)
    dl_model = DummyDeepLearningModel(name="dl_model", constant_value=0.5)
    residual_model = ResidualModel(
        name="test_residual",
        statistical_model=stat_model,
        deep_learning_model=dl_model
    )
    data = generate_sample_data()
    residual_model.fit(data, "value")
    
    # Get parameters
    params = residual_model.get_params()
    
    # Check parameters
    assert isinstance(params, dict)
    assert "models" in params
    assert "statistical_model" in params
    assert "deep_learning_model" in params
    assert params["models"] == ["stat_model", "dl_model"]
    assert params["statistical_model"] == "stat_model"
    assert params["deep_learning_model"] == "dl_model"
