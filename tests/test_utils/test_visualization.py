"""
Tests for the visualization utilities.
"""

import pytest
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from datetime import datetime, timedelta
from timefusion.utils.visualization import (
    set_style,
    plot_time_series,
    plot_forecast,
    plot_residuals,
    plot_metrics,
    plot_comparison,
    plot_feature_importance,
    plot_correlation_matrix
)


@pytest.fixture
def time_series_data():
    """Create sample time series data."""
    dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
    values = np.sin(np.linspace(0, 2 * np.pi, 30)) + np.random.normal(0, 0.1, 30)
    df = pd.DataFrame({"value": values}, index=dates)
    return df


@pytest.fixture
def forecast_data(time_series_data):
    """Create sample forecast data."""
    # Use the last 10 points as the forecast period
    actual = time_series_data.iloc[:-10]
    forecast = pd.DataFrame(
        {"value": np.sin(np.linspace(2 * np.pi * 20/30, 2 * np.pi, 10)) + np.random.normal(0, 0.1, 10)},
        index=time_series_data.index[-10:]
    )
    return actual, forecast


@pytest.fixture
def confidence_intervals(forecast_data):
    """Create sample confidence intervals."""
    _, forecast = forecast_data
    lower = forecast.copy()
    lower["value_lower"] = lower["value"] - 0.2
    lower["value_upper"] = lower["value"] + 0.2
    return {"95%": lower}


@pytest.fixture
def residuals(time_series_data):
    """Create sample residuals."""
    residuals = pd.DataFrame(
        {"value": np.random.normal(0, 0.1, 30)},
        index=time_series_data.index
    )
    return residuals


@pytest.fixture
def metrics():
    """Create sample metrics."""
    return {
        "rmse": 0.1,
        "mae": 0.05,
        "mape": 5.0,
        "r2": 0.95
    }


@pytest.fixture
def comparison_results():
    """Create sample comparison results."""
    return {
        "model1": {
            "rmse": 0.1,
            "mae": 0.05,
            "mape": 5.0,
            "r2": 0.95
        },
        "model2": {
            "rmse": 0.15,
            "mae": 0.08,
            "mape": 8.0,
            "r2": 0.9
        }
    }


@pytest.fixture
def feature_importance():
    """Create sample feature importance."""
    return {
        "feature1": 0.5,
        "feature2": 0.3,
        "feature3": 0.2
    }


@pytest.fixture
def correlation_data():
    """Create sample correlation data."""
    np.random.seed(42)
    data = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 30),
        "feature2": np.random.normal(0, 1, 30),
        "feature3": np.random.normal(0, 1, 30)
    })
    return data


def test_set_style():
    """Test set_style function."""
    # Test with default style
    set_style()
    
    # Test with custom style
    set_style("ggplot")


def test_plot_time_series(time_series_data):
    """Test plot_time_series function."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
    
    try:
        # Test with default parameters
        fig = plot_time_series(time_series_data)
        assert isinstance(fig, plt.Figure)
        
        # Test with custom parameters
        fig = plot_time_series(
            data=time_series_data,
            columns=["value"],
            title="Test Time Series",
            xlabel="Date",
            ylabel="Value",
            figsize=(10, 6),
            color_map="viridis",
            grid=True,
            legend=True,
            date_format="%Y-%m-%d",
            save_path=temp_path
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(temp_path)
        
        # Test with multiple columns
        time_series_data["value2"] = time_series_data["value"] * 2
        fig = plot_time_series(time_series_data)
        assert isinstance(fig, plt.Figure)
    finally:
        # Clean up
        plt.close("all")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_plot_forecast(forecast_data, confidence_intervals):
    """Test plot_forecast function."""
    actual, forecast = forecast_data
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
    
    try:
        # Test with default parameters
        fig = plot_forecast(actual, forecast)
        assert isinstance(fig, plt.Figure)
        
        # Test with custom parameters
        fig = plot_forecast(
            actual=actual,
            forecast=forecast,
            column="value",
            confidence_intervals=confidence_intervals,
            title="Test Forecast",
            xlabel="Date",
            ylabel="Value",
            figsize=(10, 6),
            actual_color="blue",
            forecast_color="red",
            ci_color="gray",
            ci_alpha=0.2,
            grid=True,
            legend=True,
            date_format="%Y-%m-%d",
            save_path=temp_path
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(temp_path)
    finally:
        # Clean up
        plt.close("all")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_plot_residuals(residuals):
    """Test plot_residuals function."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
    
    try:
        # Test with default parameters
        fig = plot_residuals(residuals)
        assert isinstance(fig, plt.Figure)
        
        # Test with custom parameters
        fig = plot_residuals(
            residuals=residuals,
            column="value",
            title="Test Residuals",
            figsize=(12, 8),
            color="blue",
            grid=True,
            save_path=temp_path
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(temp_path)
    finally:
        # Clean up
        plt.close("all")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_plot_metrics(metrics):
    """Test plot_metrics function."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
    
    try:
        # Test with default parameters
        fig = plot_metrics(metrics)
        assert isinstance(fig, plt.Figure)
        
        # Test with custom parameters
        fig = plot_metrics(
            metrics=metrics,
            title="Test Metrics",
            figsize=(10, 6),
            color="blue",
            grid=True,
            save_path=temp_path
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(temp_path)
    finally:
        # Clean up
        plt.close("all")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_plot_comparison(comparison_results):
    """Test plot_comparison function."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
    
    try:
        # Test with default parameters
        fig = plot_comparison(comparison_results)
        assert isinstance(fig, plt.Figure)
        
        # Test with custom parameters
        fig = plot_comparison(
            results=comparison_results,
            metrics=["rmse", "mae"],
            title="Test Comparison",
            figsize=(12, 8),
            color_map="viridis",
            grid=True,
            save_path=temp_path
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(temp_path)
    finally:
        # Clean up
        plt.close("all")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_plot_feature_importance(feature_importance):
    """Test plot_feature_importance function."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
    
    try:
        # Test with default parameters
        fig = plot_feature_importance(feature_importance)
        assert isinstance(fig, plt.Figure)
        
        # Test with custom parameters
        fig = plot_feature_importance(
            feature_importance=feature_importance,
            title="Test Feature Importance",
            figsize=(10, 6),
            color="blue",
            grid=True,
            save_path=temp_path
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(temp_path)
    finally:
        # Clean up
        plt.close("all")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_plot_correlation_matrix(correlation_data):
    """Test plot_correlation_matrix function."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
    
    try:
        # Test with default parameters
        fig = plot_correlation_matrix(correlation_data)
        assert isinstance(fig, plt.Figure)
        
        # Test with custom parameters
        fig = plot_correlation_matrix(
            data=correlation_data,
            title="Test Correlation Matrix",
            figsize=(10, 8),
            cmap="coolwarm",
            annot=True,
            save_path=temp_path
        )
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(temp_path)
    finally:
        # Clean up
        plt.close("all")
        if os.path.exists(temp_path):
            os.remove(temp_path)
