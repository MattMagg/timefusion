"""
Tests for the Normalizer class.
"""

import pytest
import pandas as pd
import numpy as np
from timefusion.preprocessing.normalizer import Normalizer


def test_normalizer_init():
    """Test Normalizer initialization."""
    normalizer = Normalizer(
        name="test_normalizer",
        method="min-max",
        columns=["value"]
    )
    assert normalizer.name == "test_normalizer"
    assert normalizer.method == "min-max"
    assert normalizer.columns == ["value"]
    assert normalizer.is_fitted is False


def test_normalizer_fit():
    """Test Normalizer fit method."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    })

    # Test min-max method
    normalizer = Normalizer(method="min-max", columns=["value"])
    normalizer.fit(data)
    assert normalizer.is_fitted is True
    assert "value" in normalizer.stats
    assert "min" in normalizer.stats["value"]
    assert "max" in normalizer.stats["value"]

    # Test z-score method
    normalizer = Normalizer(method="z-score", columns=["value"])
    normalizer.fit(data)
    assert normalizer.is_fitted is True
    assert "value" in normalizer.stats
    assert "mean" in normalizer.stats["value"]
    assert "std" in normalizer.stats["value"]

    # Test robust method
    normalizer = Normalizer(method="robust", columns=["value"])
    normalizer.fit(data)
    assert normalizer.is_fitted is True
    assert "value" in normalizer.stats
    assert "median" in normalizer.stats["value"]
    assert "iqr" in normalizer.stats["value"]

    # Test with no columns specified
    normalizer = Normalizer(method="min-max")
    normalizer.fit(data)
    assert normalizer.is_fitted is True
    assert "value" in normalizer.stats

    # Test with invalid method
    normalizer = Normalizer(method="invalid")
    with pytest.raises(ValueError):
        normalizer.fit(data)


def test_normalizer_transform_min_max():
    """Test Normalizer transform method with min-max scaling."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    })

    normalizer = Normalizer(method="min-max", columns=["value"])
    normalizer.fit(data)
    result = normalizer.transform(data)

    # Check that values are scaled to [0, 1]
    assert result["value"].min() == 0.0
    assert result["value"].max() == 1.0
    assert result["value"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]

    # Test with constant values
    data = pd.DataFrame({
        "value": [3, 3, 3, 3, 3]
    })
    normalizer = Normalizer(method="min-max", columns=["value"])
    normalizer.fit(data)
    result = normalizer.transform(data)
    assert result["value"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_normalizer_transform_z_score():
    """Test Normalizer transform method with z-score scaling."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    })

    normalizer = Normalizer(method="z-score", columns=["value"])
    normalizer.fit(data)
    result = normalizer.transform(data)

    # Check that values are standardized (mean=0, std=1)
    assert abs(result["value"].mean()) < 1e-10  # Close to 0
    assert abs(result["value"].std() - 1.0) < 1e-10  # Close to 1

    # Test with constant values
    data = pd.DataFrame({
        "value": [3, 3, 3, 3, 3]
    })
    normalizer = Normalizer(method="z-score", columns=["value"])
    normalizer.fit(data)
    result = normalizer.transform(data)
    assert result["value"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_normalizer_transform_robust():
    """Test Normalizer transform method with robust scaling."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    })

    normalizer = Normalizer(method="robust", columns=["value"])
    normalizer.fit(data)
    result = normalizer.transform(data)

    # Check that values are scaled relative to median and IQR
    # The test data is [1, 2, 3, 4, 5]
    # median = 3.0
    # Q1 = 2.0, Q3 = 4.0, IQR = 2.0
    # The expected values are [-1.0, -0.5, 0.0, 0.5, 1.0]
    expected = [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert np.allclose(result["value"].values, expected)

    # Test with constant values
    data = pd.DataFrame({
        "value": [3, 3, 3, 3, 3]
    })
    normalizer = Normalizer(method="robust", columns=["value"])
    normalizer.fit(data)
    result = normalizer.transform(data)
    assert result["value"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_normalizer_inverse_transform():
    """Test Normalizer inverse_transform method."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    })

    # Test min-max method
    normalizer = Normalizer(method="min-max", columns=["value"])
    normalizer.fit(data)
    transformed = normalizer.transform(data)
    original = normalizer.inverse_transform(transformed)
    assert np.allclose(original["value"].values, data["value"].values)

    # Test z-score method
    normalizer = Normalizer(method="z-score", columns=["value"])
    normalizer.fit(data)
    transformed = normalizer.transform(data)
    original = normalizer.inverse_transform(transformed)
    assert np.allclose(original["value"].values, data["value"].values)

    # Test robust method
    normalizer = Normalizer(method="robust", columns=["value"])
    normalizer.fit(data)
    transformed = normalizer.transform(data)
    original = normalizer.inverse_transform(transformed)
    assert np.allclose(original["value"].values, data["value"].values)

    # Test with invalid method
    normalizer = Normalizer(method="invalid", columns=["value"])
    normalizer.is_fitted = True  # Hack to bypass fit check
    with pytest.raises(ValueError):
        normalizer.inverse_transform(data)

    # Test without fitting
    normalizer = Normalizer(method="min-max", columns=["value"])
    with pytest.raises(ValueError):
        normalizer.inverse_transform(data)
