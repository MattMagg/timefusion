"""
Tests for the hyperparameter optimization utilities.
"""

import pytest
import os
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime, timedelta
from timefusion.utils.hpo import (
    HPO,
    GridSearch,
    RandomSearch,
    BayesianSearch,
    OptimizationMethod
)
from timefusion.models.base import BaseModel


class DummyModel(BaseModel):
    """
    Dummy model for testing HPO.
    
    This model predicts a constant value based on the 'factor' parameter.
    """
    
    def __init__(self, name="dummy", factor=1.0, **kwargs):
        """
        Initialize the model.
        
        Args:
            name: Name of the model
            factor: Factor to multiply the predictions by
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self.factor = factor
        self.target_column = None
    
    def fit(self, data, target_column, **kwargs):
        """
        Fit the model to the data.
        
        Args:
            data: Input data
            target_column: Name of the target column
            **kwargs: Additional parameters for fitting
            
        Returns:
            self: The fitted model
        """
        self.target_column = target_column
        self.is_fitted = True
        return self
    
    def predict(self, data, horizon, **kwargs):
        """
        Generate forecasts.
        
        Args:
            data: Input data
            horizon: Forecast horizon
            **kwargs: Additional parameters for prediction
            
        Returns:
            pd.DataFrame: Forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Get the last value
        last_value = data[self.target_column].iloc[-1]
        
        # Generate predictions
        predictions = np.ones(horizon) * last_value * self.factor
        
        # Create forecast DataFrame
        forecast = pd.DataFrame({self.target_column: predictions})
        
        return forecast


@pytest.fixture
def time_series_data():
    """Create sample time series data."""
    dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
    values = np.ones(30)
    df = pd.DataFrame({"value": values}, index=dates)
    return df


def test_hpo_init():
    """Test HPO initialization."""
    # Test with default parameters
    hpo = HPO(model_class=DummyModel)
    assert hpo.model_class == DummyModel
    assert hpo.method == OptimizationMethod.GRID_SEARCH
    assert hpo.model_kwargs == {}
    assert hpo.cv == 5
    assert hpo.verbose == True
    assert hpo.random_state is None
    assert hpo.results == {}
    
    # Test with custom parameters
    hpo = HPO(
        model_class=DummyModel,
        method=OptimizationMethod.RANDOM_SEARCH,
        model_kwargs={"name": "custom"},
        cv=10,
        n_jobs=2,
        verbose=False,
        random_state=42
    )
    assert hpo.model_class == DummyModel
    assert hpo.method == OptimizationMethod.RANDOM_SEARCH
    assert hpo.model_kwargs == {"name": "custom"}
    assert hpo.cv == 10
    assert hpo.n_jobs == 2
    assert hpo.verbose == False
    assert hpo.random_state == 42
    
    # Test with string method
    hpo = HPO(model_class=DummyModel, method="random_search")
    assert hpo.method == OptimizationMethod.RANDOM_SEARCH


def test_grid_search(time_series_data):
    """Test grid search."""
    # Create HPO instance
    hpo = GridSearch(model_class=DummyModel)
    
    # Define parameter grid
    param_grid = {"factor": [0.5, 1.0, 1.5]}
    
    # Run optimization
    results = hpo.optimize(
        data=time_series_data,
        param_grid=param_grid,
        target_column="value",
        metric="mse"
    )
    
    # Check results
    assert "method" in results
    assert results["method"] == "grid_search"
    assert "best_params" in results
    assert "best_score" in results
    assert "all_results" in results
    assert len(results["all_results"]) == 3
    
    # Check that the best factor is 1.0 (since the data is all ones)
    assert results["best_params"]["factor"] == 1.0


def test_random_search(time_series_data):
    """Test random search."""
    # Create HPO instance
    hpo = RandomSearch(model_class=DummyModel, random_state=42)
    
    # Define parameter grid
    param_grid = {"factor": [0.5, 1.0, 1.5]}
    
    # Run optimization
    results = hpo.optimize(
        data=time_series_data,
        param_grid=param_grid,
        target_column="value",
        metric="mse",
        n_trials=5
    )
    
    # Check results
    assert "method" in results
    assert results["method"] == "random_search"
    assert "best_params" in results
    assert "best_score" in results
    assert "all_results" in results
    assert len(results["all_results"]) == 5


@pytest.mark.skipif(not hasattr(HPO, "_bayesian_search"), reason="Optuna not available")
def test_bayesian_search(time_series_data):
    """Test Bayesian search."""
    try:
        # Create HPO instance
        hpo = BayesianSearch(model_class=DummyModel, random_state=42)
        
        # Define parameter grid
        param_grid = {"factor": [0.5, 1.0, 1.5]}
        
        # Run optimization
        results = hpo.optimize(
            data=time_series_data,
            param_grid=param_grid,
            target_column="value",
            metric="mse",
            n_trials=5
        )
        
        # Check results
        assert "method" in results
        assert results["method"] == "bayesian"
        assert "best_params" in results
        assert "best_score" in results
        assert "study" in results
    except ImportError:
        pytest.skip("Optuna not available")


def test_plot_results(time_series_data):
    """Test plot_results method."""
    # Create HPO instance
    hpo = GridSearch(model_class=DummyModel)
    
    # Define parameter grid
    param_grid = {"factor": [0.5, 1.0, 1.5]}
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
    
    try:
        # Run optimization
        hpo.optimize(
            data=time_series_data,
            param_grid=param_grid,
            target_column="value",
            metric="mse"
        )
        
        # Plot results
        fig = hpo.plot_results(save_path=temp_path)
        
        # Check that the figure was created and saved
        assert fig is not None
        assert os.path.exists(temp_path)
        
        # Test with no results
        hpo = HPO(model_class=DummyModel)
        with pytest.raises(ValueError):
            hpo.plot_results()
    finally:
        # Clean up
        import matplotlib.pyplot as plt
        plt.close("all")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_save_load_results(time_series_data):
    """Test save and load methods."""
    # Create HPO instance
    hpo = GridSearch(model_class=DummyModel)
    
    # Define parameter grid
    param_grid = {"factor": [0.5, 1.0, 1.5]}
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name
    
    try:
        # Run optimization
        hpo.optimize(
            data=time_series_data,
            param_grid=param_grid,
            target_column="value",
            metric="mse"
        )
        
        # Save results
        hpo.save(temp_path)
        
        # Check that the file was created
        assert os.path.exists(temp_path)
        
        # Create a new HPO instance
        new_hpo = HPO(model_class=DummyModel)
        
        # Load results
        new_hpo.load(temp_path)
        
        # Check that the results were loaded
        assert new_hpo.results == hpo.results
        
        # Test with no results
        empty_hpo = HPO(model_class=DummyModel)
        with pytest.raises(ValueError):
            empty_hpo.save(temp_path)
        
        # Test with nonexistent file
        with pytest.raises(FileNotFoundError):
            empty_hpo.load("nonexistent.pkl")
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
