"""
Tests for the Cleaner class.
"""

import pytest
import pandas as pd
import numpy as np
from timefusion.preprocessing.cleaner import Cleaner


def test_cleaner_init():
    """Test Cleaner initialization."""
    cleaner = Cleaner(
        name="test_cleaner",
        method="z-score",
        threshold=3.0,
        strategy="clip",
        columns=["value"]
    )
    assert cleaner.name == "test_cleaner"
    assert cleaner.method == "z-score"
    assert cleaner.threshold == 3.0
    assert cleaner.strategy == "clip"
    assert cleaner.columns == ["value"]
    assert cleaner.is_fitted is False


def test_cleaner_fit():
    """Test Cleaner fit method."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5, 100]  # 100 is an outlier
    })
    
    # Test z-score method
    cleaner = Cleaner(method="z-score", columns=["value"])
    cleaner.fit(data)
    assert cleaner.is_fitted is True
    assert "value" in cleaner.stats
    assert "mean" in cleaner.stats["value"]
    assert "std" in cleaner.stats["value"]
    
    # Test IQR method
    cleaner = Cleaner(method="iqr", columns=["value"])
    cleaner.fit(data)
    assert cleaner.is_fitted is True
    assert "value" in cleaner.stats
    assert "q1" in cleaner.stats["value"]
    assert "q3" in cleaner.stats["value"]
    assert "iqr" in cleaner.stats["value"]
    
    # Test percentile method
    cleaner = Cleaner(method="percentile", columns=["value"])
    cleaner.fit(data)
    assert cleaner.is_fitted is True
    assert "value" in cleaner.stats
    assert "lower" in cleaner.stats["value"]
    assert "upper" in cleaner.stats["value"]
    
    # Test with no columns specified
    cleaner = Cleaner(method="z-score")
    cleaner.fit(data)
    assert cleaner.is_fitted is True
    assert "value" in cleaner.stats
    
    # Test with invalid method
    cleaner = Cleaner(method="invalid")
    with pytest.raises(ValueError):
        cleaner.fit(data)


def test_cleaner_transform_z_score():
    """Test Cleaner transform method with z-score method."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5, 100]  # 100 is an outlier
    })
    
    # Test clip strategy
    cleaner = Cleaner(method="z-score", strategy="clip", threshold=2.0)
    cleaner.fit(data)
    result = cleaner.transform(data)
    assert result["value"].max() < 100
    
    # Test remove strategy
    cleaner = Cleaner(method="z-score", strategy="remove", threshold=2.0)
    cleaner.fit(data)
    result = cleaner.transform(data)
    assert len(result) < len(data)
    
    # Test replace strategy
    cleaner = Cleaner(method="z-score", strategy="replace", threshold=2.0)
    cleaner.fit(data)
    result = cleaner.transform(data)
    assert pd.isna(result["value"].iloc[-1])
    
    # Test with invalid strategy
    cleaner = Cleaner(method="z-score", strategy="invalid")
    cleaner.fit(data)
    with pytest.raises(ValueError):
        cleaner.transform(data)


def test_cleaner_transform_iqr():
    """Test Cleaner transform method with IQR method."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5, 100]  # 100 is an outlier
    })
    
    # Test clip strategy
    cleaner = Cleaner(method="iqr", strategy="clip", threshold=1.5)
    cleaner.fit(data)
    result = cleaner.transform(data)
    assert result["value"].max() < 100
    
    # Test remove strategy
    cleaner = Cleaner(method="iqr", strategy="remove", threshold=1.5)
    cleaner.fit(data)
    result = cleaner.transform(data)
    assert len(result) < len(data)
    
    # Test replace strategy
    cleaner = Cleaner(method="iqr", strategy="replace", threshold=1.5)
    cleaner.fit(data)
    result = cleaner.transform(data)
    assert pd.isna(result["value"].iloc[-1])


def test_cleaner_transform_percentile():
    """Test Cleaner transform method with percentile method."""
    # Create sample data with 100 values and one outlier
    np.random.seed(42)
    values = np.random.normal(0, 1, 99).tolist() + [100]
    data = pd.DataFrame({"value": values})
    
    # Test clip strategy
    cleaner = Cleaner(method="percentile", strategy="clip")
    cleaner.fit(data)
    result = cleaner.transform(data)
    assert result["value"].max() < 100
    
    # Test remove strategy
    cleaner = Cleaner(method="percentile", strategy="remove")
    cleaner.fit(data)
    result = cleaner.transform(data)
    assert len(result) < len(data)
    
    # Test replace strategy
    cleaner = Cleaner(method="percentile", strategy="replace")
    cleaner.fit(data)
    result = cleaner.transform(data)
    assert pd.isna(result["value"].iloc[-1])
