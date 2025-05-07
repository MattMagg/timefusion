"""
Common test fixtures for TimeFusion.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_data():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Generate trend, seasonality, and noise components
    trend = np.linspace(0, 10, 100)
    seasonality = 5 * np.sin(2 * np.pi * np.arange(100) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 1, 100)
    
    # Combine components
    values = trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({'value': values}, index=dates)
    return df


@pytest.fixture
def sample_data_with_missing():
    """Generate sample time series data with missing values for testing."""
    data = sample_data()
    
    # Introduce missing values
    missing_indices = np.random.choice(len(data), size=10, replace=False)
    data.iloc[missing_indices, 0] = np.nan
    
    return data


@pytest.fixture
def sample_data_with_outliers():
    """Generate sample time series data with outliers for testing."""
    data = sample_data()
    
    # Introduce outliers
    outlier_indices = np.random.choice(len(data), size=5, replace=False)
    data.iloc[outlier_indices, 0] = data.iloc[outlier_indices, 0] + 20
    
    return data
