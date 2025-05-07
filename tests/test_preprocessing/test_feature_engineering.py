"""
Tests for the FeatureEngineering class.
"""

import pytest
import pandas as pd
import numpy as np
from timefusion.preprocessing.feature_engineering import FeatureEngineering


def test_feature_engineering_init():
    """Test FeatureEngineering initialization."""
    fe = FeatureEngineering(
        name="test_fe",
        lag_features=[1, 2],
        window_features={"mean": [3, 5]},
        date_features=["day_of_week", "month"],
        fourier_features={"365": 2},
        target_column="value"
    )
    assert fe.name == "test_fe"
    assert fe.lag_features == [1, 2]
    assert fe.window_features == {"mean": [3, 5]}
    assert fe.date_features == ["day_of_week", "month"]
    assert fe.fourier_features == {"365": 2}
    assert fe.target_column == "value"
    assert fe.is_fitted is False


def test_feature_engineering_fit():
    """Test FeatureEngineering fit method."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    })

    # Test with target column specified in init
    fe = FeatureEngineering(target_column="value")
    fe.fit(data)
    assert fe.is_fitted is True
    assert fe.target_column == "value"

    # Test with target column specified in fit
    fe = FeatureEngineering()
    fe.fit(data, target_column="value")
    assert fe.is_fitted is True
    assert fe.target_column == "value"

    # Test with no target column specified
    fe = FeatureEngineering()
    fe.fit(data)
    assert fe.is_fitted is True
    assert fe.target_column == "value"  # First column


def test_feature_engineering_transform_lag():
    """Test FeatureEngineering transform method with lag features."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    })

    # Test with lag features
    fe = FeatureEngineering(lag_features=[1, 2], target_column="value")
    fe.fit(data)
    result = fe.transform(data)

    # Check that lag features are created
    assert "value_lag_1" in result.columns
    assert "value_lag_2" in result.columns

    # Check values using pandas isna for the NaN values and direct comparison for the rest
    assert pd.isna(result["value_lag_1"].iloc[0])
    assert result["value_lag_1"].iloc[1:].tolist() == [1.0, 2.0, 3.0, 4.0]

    assert pd.isna(result["value_lag_2"].iloc[0])
    assert pd.isna(result["value_lag_2"].iloc[1])
    assert result["value_lag_2"].iloc[2:].tolist() == [1.0, 2.0, 3.0]


def test_feature_engineering_transform_window():
    """Test FeatureEngineering transform method with window features."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    })

    # Test with window features
    fe = FeatureEngineering(
        window_features={"mean": [2, 3], "std": [2]},
        target_column="value"
    )
    fe.fit(data)
    result = fe.transform(data)

    # Check that window features are created
    assert "value_mean_2" in result.columns
    assert "value_mean_3" in result.columns
    assert "value_std_2" in result.columns

    # Check values (rolling with min_periods=1)
    assert result["value_mean_2"].tolist() == [1.0, 1.5, 2.5, 3.5, 4.5]
    assert result["value_mean_3"].tolist() == [1.0, 1.5, 2.0, 3.0, 4.0]

    # Test with invalid window function
    fe = FeatureEngineering(
        window_features={"invalid": [2]},
        target_column="value"
    )
    fe.fit(data)
    with pytest.raises(ValueError):
        fe.transform(data)


def test_feature_engineering_transform_date():
    """Test FeatureEngineering transform method with date features."""
    # Create sample data with datetime index
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    }, index=dates)

    # Test with date features
    fe = FeatureEngineering(
        date_features=["day", "day_of_week", "month", "is_weekend"],
        target_column="value"
    )
    fe.fit(data)
    result = fe.transform(data)

    # Check that date features are created
    assert "day" in result.columns
    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "is_weekend" in result.columns

    # Check values
    assert result["day"].tolist() == [1, 2, 3, 4, 5]  # Days of month
    assert result["day_of_week"].tolist() == [6, 0, 1, 2, 3]  # 0=Monday, 6=Sunday
    assert result["month"].tolist() == [1, 1, 1, 1, 1]  # January
    assert result["is_weekend"].tolist() == [1, 0, 0, 0, 0]  # Sunday is weekend

    # Test with invalid date feature
    fe = FeatureEngineering(
        date_features=["invalid"],
        target_column="value"
    )
    fe.fit(data)
    with pytest.raises(ValueError):
        fe.transform(data)


def test_feature_engineering_transform_fourier():
    """Test FeatureEngineering transform method with Fourier features."""
    # Create sample data with datetime index
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    }, index=dates)

    # Test with Fourier features
    fe = FeatureEngineering(
        fourier_features={"7": 1},  # Weekly seasonality with order 1
        target_column="value"
    )
    fe.fit(data)
    result = fe.transform(data)

    # Check that Fourier features are created
    assert "fourier_sin_7_1" in result.columns
    assert "fourier_cos_7_1" in result.columns

    # Check that values are in [-1, 1]
    assert all(-1 <= x <= 1 for x in result["fourier_sin_7_1"])
    assert all(-1 <= x <= 1 for x in result["fourier_cos_7_1"])


def test_feature_engineering_transform_all():
    """Test FeatureEngineering transform method with all feature types."""
    # Create sample data with datetime index
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    data = pd.DataFrame({
        "value": range(1, 11)
    }, index=dates)

    # Test with all feature types
    fe = FeatureEngineering(
        lag_features=[1, 2],
        window_features={"mean": [3]},
        date_features=["day_of_week"],
        fourier_features={"7": 1},
        target_column="value"
    )
    fe.fit(data)
    result = fe.transform(data)

    # Check that all feature types are created
    assert "value_lag_1" in result.columns
    assert "value_lag_2" in result.columns
    assert "value_mean_3" in result.columns
    assert "day_of_week" in result.columns
    assert "fourier_sin_7_1" in result.columns
    assert "fourier_cos_7_1" in result.columns

    # Check that original column is preserved
    assert "value" in result.columns
    assert result["value"].equals(data["value"])


def test_feature_engineering_transform_errors():
    """Test FeatureEngineering transform method error handling."""
    # Create sample data
    data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5]
    })

    # Test without fitting
    fe = FeatureEngineering(target_column="value")
    with pytest.raises(ValueError):
        fe.transform(data)
