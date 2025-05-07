"""
Tests for the Imputer class.
"""

import pytest
import pandas as pd
import numpy as np
from timefusion.preprocessing.imputer import Imputer


def test_imputer_init():
    """Test Imputer initialization."""
    imputer = Imputer(
        name="test_imputer",
        method="linear",
        window_size=3,
        columns=["value"]
    )
    assert imputer.name == "test_imputer"
    assert imputer.method == "linear"
    assert imputer.window_size == 3
    assert imputer.columns == ["value"]
    assert imputer.is_fitted is False


def test_imputer_fit():
    """Test Imputer fit method."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, np.nan, 4, 5]
    })
    
    # Test mean method
    imputer = Imputer(method="mean", columns=["value"])
    imputer.fit(data)
    assert imputer.is_fitted is True
    assert "value" in imputer.fill_values
    assert imputer.fill_values["value"] == 3.0  # mean of [1, 2, 4, 5]
    
    # Test median method
    imputer = Imputer(method="median", columns=["value"])
    imputer.fit(data)
    assert imputer.is_fitted is True
    assert "value" in imputer.fill_values
    assert imputer.fill_values["value"] == 3.0  # median of [1, 2, 4, 5]
    
    # Test mode method
    imputer = Imputer(method="mode", columns=["value"])
    imputer.fit(data)
    assert imputer.is_fitted is True
    assert "value" in imputer.fill_values
    assert imputer.fill_values["value"] == 1.0  # mode of [1, 2, 4, 5] (first value)
    
    # Test constant method
    imputer = Imputer(method="constant", columns=["value"])
    imputer.fit(data, value=99)
    assert imputer.is_fitted is True
    assert "value" in imputer.fill_values
    assert imputer.fill_values["value"] == 99
    
    # Test with no columns specified
    imputer = Imputer(method="mean")
    imputer.fit(data)
    assert imputer.is_fitted is True
    assert "value" in imputer.fill_values


def test_imputer_transform_forward():
    """Test Imputer transform method with forward fill."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, np.nan, 4, 5]
    })
    
    # Test without window size
    imputer = Imputer(method="forward", columns=["value"])
    imputer.fit(data)
    result = imputer.transform(data)
    assert result["value"].tolist() == [1, 2, 2, 4, 5]
    
    # Test with window size
    imputer = Imputer(method="forward", window_size=1, columns=["value"])
    imputer.fit(data)
    result = imputer.transform(data)
    assert pd.isna(result["value"].iloc[2])  # NaN remains because window_size=1


def test_imputer_transform_backward():
    """Test Imputer transform method with backward fill."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, np.nan, 4, 5]
    })
    
    # Test without window size
    imputer = Imputer(method="backward", columns=["value"])
    imputer.fit(data)
    result = imputer.transform(data)
    assert result["value"].tolist() == [1, 2, 4, 4, 5]
    
    # Test with window size
    imputer = Imputer(method="backward", window_size=1, columns=["value"])
    imputer.fit(data)
    result = imputer.transform(data)
    assert pd.isna(result["value"].iloc[2])  # NaN remains because window_size=1


def test_imputer_transform_linear():
    """Test Imputer transform method with linear interpolation."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, np.nan, 4, 5]
    })
    
    # Test without window size
    imputer = Imputer(method="linear", columns=["value"])
    imputer.fit(data)
    result = imputer.transform(data)
    assert result["value"].tolist() == [1, 2, 3, 4, 5]  # Linear interpolation
    
    # Test with window size
    imputer = Imputer(method="linear", window_size=1, columns=["value"])
    imputer.fit(data)
    result = imputer.transform(data)
    assert pd.isna(result["value"].iloc[2])  # NaN remains because window_size=1


def test_imputer_transform_constant():
    """Test Imputer transform method with constant values."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, np.nan, 4, 5]
    })
    
    # Test mean
    imputer = Imputer(method="mean", columns=["value"])
    imputer.fit(data)
    result = imputer.transform(data)
    assert result["value"].tolist() == [1, 2, 3, 4, 5]  # Mean = 3
    
    # Test median
    imputer = Imputer(method="median", columns=["value"])
    imputer.fit(data)
    result = imputer.transform(data)
    assert result["value"].tolist() == [1, 2, 3, 4, 5]  # Median = 3
    
    # Test constant
    imputer = Imputer(method="constant", columns=["value"])
    imputer.fit(data, value=99)
    result = imputer.transform(data)
    assert result["value"].tolist() == [1, 2, 99, 4, 5]  # Constant = 99


def test_imputer_transform_invalid():
    """Test Imputer transform method with invalid method."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, np.nan, 4, 5]
    })
    
    # Test with invalid method
    imputer = Imputer(method="invalid", columns=["value"])
    imputer.fit(data)
    with pytest.raises(ValueError):
        imputer.transform(data)
    
    # Test without fitting
    imputer = Imputer(method="mean", columns=["value"])
    with pytest.raises(ValueError):
        imputer.transform(data)
