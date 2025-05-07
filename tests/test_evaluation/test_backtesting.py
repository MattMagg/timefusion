"""
Tests for the backtesting module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from timefusion.evaluation.backtesting import Backtesting, BacktestingStrategy
from timefusion.models.base import BaseModel


# Create a dummy model for testing
class DummyModel(BaseModel):
    def __init__(self, name="dummy", constant_value=1.0, **kwargs):
        super().__init__(name, **kwargs)
        self.constant_value = constant_value
        self.target_column = None
    
    def fit(self, data, target_column, **kwargs):
        self.target_column = target_column
        self.is_fitted = True
        return self
    
    def predict(self, data, horizon, **kwargs):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Create forecast index
        if isinstance(data.index, pd.DatetimeIndex):
            # Use date index
            last_date = data.index[-1]
            forecast_index = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq=pd.infer_freq(data.index)
            )
        else:
            # Use integer index
            forecast_index = range(len(data), len(data) + horizon)
        
        # Create forecast
        forecast = pd.DataFrame(
            {self.target_column: np.ones(horizon) * self.constant_value},
            index=forecast_index
        )
        
        return forecast
    
    def get_params(self):
        return {"constant_value": self.constant_value}


# Create sample data for testing
def create_sample_data(n_samples=100):
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
    values = np.sin(np.linspace(0, 4 * np.pi, n_samples)) + np.random.normal(0, 0.1, n_samples)
    return pd.DataFrame({"value": values}, index=dates)


def test_backtesting_init():
    """Test Backtesting initialization."""
    # Test with default parameters
    backtesting = Backtesting()
    assert backtesting.strategy == BacktestingStrategy.WALK_FORWARD
    assert backtesting.initial_train_size is None
    assert backtesting.step_size == 1
    assert backtesting.window_size is None
    assert backtesting.verbose is False
    
    # Test with custom parameters
    backtesting = Backtesting(
        strategy=BacktestingStrategy.EXPANDING_WINDOW,
        initial_train_size=50,
        step_size=5,
        window_size=30,
        verbose=True
    )
    assert backtesting.strategy == BacktestingStrategy.EXPANDING_WINDOW
    assert backtesting.initial_train_size == 50
    assert backtesting.step_size == 5
    assert backtesting.window_size == 30
    assert backtesting.verbose is True
    
    # Test with string strategy
    backtesting = Backtesting(strategy="sliding_window")
    assert backtesting.strategy == BacktestingStrategy.SLIDING_WINDOW
    
    # Test with invalid strategy
    with pytest.raises(ValueError):
        Backtesting(strategy="invalid")


def test_backtesting_backtest():
    """Test Backtesting.backtest method."""
    # Create sample data
    data = create_sample_data()
    
    # Create model
    model = DummyModel()
    
    # Create backtesting instance
    backtesting = Backtesting(
        strategy=BacktestingStrategy.EXPANDING_WINDOW,
        initial_train_size=70,
        step_size=10
    )
    
    # Perform backtesting
    results = backtesting.backtest(
        model=model,
        data=data,
        target_column="value",
        horizon=1
    )
    
    # Check results
    assert "strategy" in results
    assert "initial_train_size" in results
    assert "step_size" in results
    assert "window_size" in results
    assert "horizon" in results
    assert "metrics" in results
    assert "forecasts" in results
    assert "actuals" in results
    assert "train_sizes" in results
    assert "test_indices" in results
    
    # Check metrics
    assert "mse" in results["metrics"]
    assert "rmse" in results["metrics"]
    assert "mae" in results["metrics"]
    assert "mape" in results["metrics"]
    assert "smape" in results["metrics"]
    
    # Check dimensions
    assert len(results["forecasts"]) == len(results["actuals"])
    assert len(results["train_sizes"]) == len(results["test_indices"])
    assert len(results["forecasts"]) == len(results["train_sizes"])


def test_backtesting_cross_validation():
    """Test Backtesting.cross_validation method."""
    # Create sample data
    data = create_sample_data()
    
    # Create model
    model = DummyModel()
    
    # Create backtesting instance
    backtesting = Backtesting()
    
    # Perform cross-validation
    results = backtesting.cross_validation(
        model=model,
        data=data,
        target_column="value",
        n_splits=5,
        horizon=1
    )
    
    # Check results
    assert "n_splits" in results
    assert "horizon" in results
    assert "folds" in results
    assert "avg_metrics" in results
    
    # Check folds
    assert len(results["folds"]) == 5
    for fold in results["folds"]:
        assert "fold" in fold
        assert "train_size" in fold
        assert "test_size" in fold
        assert "metrics" in fold
        assert "forecast" in fold
        assert "actuals" in fold
    
    # Check metrics
    assert "mse" in results["avg_metrics"]
    assert "rmse" in results["avg_metrics"]
    assert "mae" in results["avg_metrics"]
    assert "mape" in results["avg_metrics"]
    assert "smape" in results["avg_metrics"]


def test_backtesting_compare_models():
    """Test Backtesting.compare_models method."""
    # Create sample data
    data = create_sample_data()
    
    # Create models
    model1 = DummyModel(name="model1", constant_value=1.0)
    model2 = DummyModel(name="model2", constant_value=0.5)
    
    # Create backtesting instance
    backtesting = Backtesting(
        strategy=BacktestingStrategy.EXPANDING_WINDOW,
        initial_train_size=70,
        step_size=10
    )
    
    # Compare models
    results = backtesting.compare_models(
        models=[model1, model2],
        data=data,
        target_column="value",
        horizon=1
    )
    
    # Check results
    assert "model1" in results
    assert "model2" in results
    
    # Check metrics
    assert "mse" in results["model1"]
    assert "rmse" in results["model1"]
    assert "mae" in results["model1"]
    assert "mape" in results["model1"]
    assert "smape" in results["model1"]
    
    assert "mse" in results["model2"]
    assert "rmse" in results["model2"]
    assert "mae" in results["model2"]
    assert "mape" in results["model2"]
    assert "smape" in results["model2"]
