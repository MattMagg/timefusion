"""
Tests for the cross-validation module.
"""

import pytest
import numpy as np
import pandas as pd
from timefusion.evaluation.cross_validation import (
    time_series_split,
    blocked_time_series_split,
    rolling_window_split
)


# Create sample data for testing
def create_sample_data(n_samples=100):
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
    values = np.sin(np.linspace(0, 4 * np.pi, n_samples)) + np.random.normal(0, 0.1, n_samples)
    return pd.DataFrame({"value": values}, index=dates)


def test_time_series_split():
    """Test time_series_split function."""
    # Create sample data
    data = create_sample_data()
    
    # Perform split
    splits = time_series_split(data, n_splits=5, test_size=10, gap=0)
    
    # Check number of splits
    assert len(splits) == 5
    
    # Check each split
    for i, (train, test) in enumerate(splits):
        # Check types
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        
        # Check sizes
        assert len(test) == 10
        
        # Check train/test relationship
        assert train.index[-1] < test.index[0]
        
        # Check that test sets don't overlap
        if i < len(splits) - 1:
            next_train, next_test = splits[i + 1]
            assert test.index[0] > next_test.index[0]


def test_blocked_time_series_split():
    """Test blocked_time_series_split function."""
    # Create sample data
    data = create_sample_data()
    
    # Perform split with fixed train size
    splits = blocked_time_series_split(data, n_splits=5, test_size=10, train_size=50, gap=0)
    
    # Check number of splits
    assert len(splits) == 5
    
    # Check each split
    for i, (train, test) in enumerate(splits):
        # Check types
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        
        # Check sizes
        assert len(test) == 10
        assert len(train) <= 50  # May be less for early splits
        
        # Check train/test relationship
        assert train.index[-1] < test.index[0]
        
        # Check that test sets don't overlap
        if i < len(splits) - 1:
            next_train, next_test = splits[i + 1]
            assert test.index[0] > next_test.index[0]


def test_rolling_window_split():
    """Test rolling_window_split function."""
    # Create sample data
    data = create_sample_data()
    
    # Perform split
    splits = rolling_window_split(data, window_size=50, step_size=10, horizon=5, min_train_size=50)
    
    # Check number of splits
    expected_splits = (len(data) - 50 - 5 + 1) // 10 + 1
    assert len(splits) == expected_splits
    
    # Check each split
    for train, test in splits:
        # Check types
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        
        # Check sizes
        assert len(train) <= 50  # May be less for early splits
        assert len(test) <= 5  # May be less for the last split
        
        # Check train/test relationship
        assert train.index[-1] < test.index[0]
        assert test.index[0] == train.index[-1] + pd.Timedelta(days=1)


def test_time_series_split_with_gap():
    """Test time_series_split function with gap."""
    # Create sample data
    data = create_sample_data()
    
    # Perform split with gap
    gap = 5
    splits = time_series_split(data, n_splits=5, test_size=10, gap=gap)
    
    # Check each split
    for train, test in splits:
        # Check that there's a gap between train and test
        assert test.index[0] - train.index[-1] > pd.Timedelta(days=1)
        assert (test.index[0] - train.index[-1]).days == gap + 1


def test_blocked_time_series_split_with_gap():
    """Test blocked_time_series_split function with gap."""
    # Create sample data
    data = create_sample_data()
    
    # Perform split with gap
    gap = 5
    splits = blocked_time_series_split(data, n_splits=5, test_size=10, train_size=50, gap=gap)
    
    # Check each split
    for train, test in splits:
        # Check that there's a gap between train and test
        assert test.index[0] - train.index[-1] > pd.Timedelta(days=1)
        assert (test.index[0] - train.index[-1]).days == gap + 1
