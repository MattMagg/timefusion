"""
Tests for the base model classes.
"""

import pytest
import pandas as pd
import numpy as np
from timefusion.models.base import BaseModel, ModelRegistry


# Create a concrete implementation of BaseModel for testing
class DummyModel(BaseModel):
    def fit(self, data, target_column, **kwargs):
        self.is_fitted = True
        return self
    
    def predict(self, data, horizon, **kwargs):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame({
            'forecast': np.ones(horizon)
        })


def test_base_model_init():
    """Test BaseModel initialization."""
    model = DummyModel(name="dummy", param1=1, param2="test")
    assert model.name == "dummy"
    assert model.is_fitted is False
    assert model.params == {"param1": 1, "param2": "test"}


def test_base_model_fit():
    """Test BaseModel fit method."""
    model = DummyModel(name="dummy")
    data = pd.DataFrame({"value": [1, 2, 3]})
    model.fit(data, "value")
    assert model.is_fitted is True


def test_base_model_predict():
    """Test BaseModel predict method."""
    model = DummyModel(name="dummy")
    data = pd.DataFrame({"value": [1, 2, 3]})
    model.fit(data, "value")
    forecast = model.predict(data, horizon=3)
    assert isinstance(forecast, pd.DataFrame)
    assert forecast.shape == (3, 1)
    assert forecast['forecast'].tolist() == [1.0, 1.0, 1.0]


def test_model_registry():
    """Test ModelRegistry."""
    registry = ModelRegistry()
    model1 = DummyModel(name="model1")
    model2 = DummyModel(name="model2")
    
    # Test registration
    registry.register(model1)
    registry.register(model2)
    assert len(registry.models) == 2
    
    # Test retrieval
    assert registry.get("model1") is model1
    assert registry.get("model2") is model2
    
    # Test listing
    assert set(registry.list()) == {"model1", "model2"}
    
    # Test error on unknown model
    with pytest.raises(KeyError):
        registry.get("unknown")
