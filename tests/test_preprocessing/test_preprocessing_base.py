"""
Tests for the base preprocessing classes.
"""

import pytest
import pandas as pd
import numpy as np
from timefusion.preprocessing.base import BasePreprocessor, Pipeline


# Create a concrete implementation of BasePreprocessor for testing
class DummyPreprocessor(BasePreprocessor):
    def fit(self, data, **kwargs):
        self.is_fitted = True
        return self
    
    def transform(self, data, **kwargs):
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted")
        return data * 2


def test_base_preprocessor_init():
    """Test BasePreprocessor initialization."""
    preprocessor = DummyPreprocessor(name="dummy", param1=1, param2="test")
    assert preprocessor.name == "dummy"
    assert preprocessor.is_fitted is False
    assert preprocessor.params == {"param1": 1, "param2": "test"}


def test_base_preprocessor_fit():
    """Test BasePreprocessor fit method."""
    preprocessor = DummyPreprocessor(name="dummy")
    data = pd.DataFrame({"value": [1, 2, 3]})
    preprocessor.fit(data)
    assert preprocessor.is_fitted is True


def test_base_preprocessor_transform():
    """Test BasePreprocessor transform method."""
    preprocessor = DummyPreprocessor(name="dummy")
    data = pd.DataFrame({"value": [1, 2, 3]})
    preprocessor.fit(data)
    transformed = preprocessor.transform(data)
    assert transformed.equals(pd.DataFrame({"value": [2, 4, 6]}))


def test_base_preprocessor_fit_transform():
    """Test BasePreprocessor fit_transform method."""
    preprocessor = DummyPreprocessor(name="dummy")
    data = pd.DataFrame({"value": [1, 2, 3]})
    transformed = preprocessor.fit_transform(data)
    assert preprocessor.is_fitted is True
    assert transformed.equals(pd.DataFrame({"value": [2, 4, 6]}))


def test_pipeline():
    """Test Pipeline."""
    preprocessor1 = DummyPreprocessor(name="preprocessor1")
    preprocessor2 = DummyPreprocessor(name="preprocessor2")
    pipeline = Pipeline([
        ("step1", preprocessor1),
        ("step2", preprocessor2)
    ])
    
    # Test initialization
    assert len(pipeline.steps) == 2
    assert pipeline.preprocessors["step1"] is preprocessor1
    assert pipeline.preprocessors["step2"] is preprocessor2
    
    # Test fit
    data = pd.DataFrame({"value": [1, 2, 3]})
    pipeline.fit(data)
    assert preprocessor1.is_fitted is True
    assert preprocessor2.is_fitted is True
    
    # Test transform
    transformed = pipeline.transform(data)
    # Each preprocessor multiplies by 2, so the result should be multiplied by 4
    assert transformed.equals(pd.DataFrame({"value": [4, 8, 12]}))
    
    # Test fit_transform
    pipeline = Pipeline([
        ("step1", DummyPreprocessor(name="preprocessor1")),
        ("step2", DummyPreprocessor(name="preprocessor2"))
    ])
    transformed = pipeline.fit_transform(data)
    assert transformed.equals(pd.DataFrame({"value": [4, 8, 12]}))
