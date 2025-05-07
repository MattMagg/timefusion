"""
Backtesting framework for time series forecasting.

This module provides tools for backtesting forecasting models
using different strategies such as walk-forward validation.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from ..models.base import BaseModel
from .metrics import Metrics


class BacktestingStrategy(Enum):
    """
    Backtesting strategies for time series forecasting.

    Attributes:
        WALK_FORWARD: Retrain model at each step
        EXPANDING_WINDOW: Expand training window at each step
        SLIDING_WINDOW: Fixed-size sliding window
    """
    WALK_FORWARD = "walk_forward"
    EXPANDING_WINDOW = "expanding_window"
    SLIDING_WINDOW = "sliding_window"


class Backtesting:
    """
    Backtesting framework for time series forecasting.

    This class provides methods for backtesting forecasting models
    using different strategies such as walk-forward validation.

    Attributes:
        strategy (BacktestingStrategy): Backtesting strategy to use
        initial_train_size (Optional[Union[int, float]]): Initial training set size
        step_size (int): Step size for moving forward
        window_size (Optional[int]): Window size for sliding window strategy
        refit_frequency (int): Frequency of model refitting
        verbose (bool): Whether to print progress information
    """

    def __init__(
        self,
        strategy: Union[str, BacktestingStrategy] = BacktestingStrategy.WALK_FORWARD,
        initial_train_size: Optional[Union[int, float]] = None,
        step_size: int = 1,
        window_size: Optional[int] = None,
        refit_frequency: int = 1,
        verbose: bool = False
    ):
        """
        Initialize the backtesting framework.

        Args:
            strategy: Backtesting strategy to use
            initial_train_size: Initial training set size (int for absolute size, float for proportion)
            step_size: Step size for moving forward
            window_size: Window size for sliding window strategy
            refit_frequency: Frequency of model refitting (every n steps)
            verbose: Whether to print progress information
        """
        if isinstance(strategy, str):
            try:
                self.strategy = BacktestingStrategy(strategy)
            except ValueError:
                raise ValueError(f"Unknown backtesting strategy: {strategy}")
        else:
            self.strategy = strategy

        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.window_size = window_size
        self.refit_frequency = refit_frequency
        self.verbose = verbose

    def backtest(
        self,
        model: BaseModel,
        data: pd.DataFrame,
        target_column: str,
        horizon: int = 1,
        feature_columns: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        custom_metrics: Optional[Dict[str, Callable]] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform backtesting.

        Args:
            model: Forecasting model
            data: Input data
            target_column: Name of the target column
            horizon: Forecast horizon
            feature_columns: List of feature columns
            metrics: List of metric names to compute
            custom_metrics: Dictionary of custom metric functions
            train_kwargs: Additional parameters for model fitting
            predict_kwargs: Additional parameters for model prediction

        Returns:
            Dict[str, Any]: Backtesting results
        """
        if metrics is None:
            metrics = ['mse', 'rmse', 'mae', 'mape', 'smape']

        if train_kwargs is None:
            train_kwargs = {}

        if predict_kwargs is None:
            predict_kwargs = {}

        # Determine initial training set size
        if self.initial_train_size is None:
            # Default to 70% of data
            initial_train_size = int(len(data) * 0.7)
        elif isinstance(self.initial_train_size, float) and 0 < self.initial_train_size < 1:
            # Interpret as a fraction of the data
            initial_train_size = int(len(data) * self.initial_train_size)
        else:
            # Use as absolute size
            initial_train_size = self.initial_train_size

        # Ensure initial_train_size is valid
        if initial_train_size <= 0 or initial_train_size >= len(data):
            raise ValueError(f"Invalid initial_train_size: {initial_train_size}")

        # Determine window size for sliding window strategy
        if self.strategy == BacktestingStrategy.SLIDING_WINDOW:
            if self.window_size is None:
                # Default to initial_train_size
                window_size = initial_train_size
            else:
                window_size = self.window_size

            # Ensure window_size is valid
            if window_size <= 0 or window_size >= len(data):
                raise ValueError(f"Invalid window_size: {window_size}")

        # Initialize results
        forecasts = []
        actuals = []
        train_sizes = []
        test_indices = []
        model_states = []

        # Determine test start and end indices
        test_start = initial_train_size
        test_end = len(data) - horizon

        # Iterate over test points
        for i in range(test_start, test_end, self.step_size):
            if self.verbose:
                print(f"Backtesting at index {i}/{test_end} ({i/test_end:.1%})")

            # Determine training data based on strategy
            if self.strategy == BacktestingStrategy.EXPANDING_WINDOW:
                # Use all data up to current point
                train_data = data.iloc[:i].copy()
            elif self.strategy == BacktestingStrategy.SLIDING_WINDOW:
                # Use fixed-size window ending at current point
                train_data = data.iloc[max(0, i - window_size):i].copy()
            else:  # WALK_FORWARD
                # Use all data up to current point (same as expanding window)
                train_data = data.iloc[:i].copy()

            # Determine test data
            test_data = data.iloc[i:i+horizon].copy()

            # Refit model if needed
            if (i - test_start) % self.refit_frequency == 0:
                # Create a new instance of the model to avoid state leakage
                model_copy = type(model)(name=model.name, **model.params)
                model_copy.fit(train_data, target_column, **train_kwargs)
                current_model = model_copy
            elif self.strategy == BacktestingStrategy.WALK_FORWARD:
                # For walk-forward, always refit
                model_copy = type(model)(name=model.name, **model.params)
                model_copy.fit(train_data, target_column, **train_kwargs)
                current_model = model_copy
            else:
                # Use the previously fitted model
                current_model = model_copy

            # Generate forecast
            forecast = current_model.predict(train_data, horizon, **predict_kwargs)

            # Store results
            forecasts.append(forecast[target_column].values)
            actuals.append(test_data[target_column].values)
            train_sizes.append(len(train_data))
            test_indices.append(i)
            model_states.append(current_model.get_params() if hasattr(current_model, 'get_params') else {})

        # Combine results
        all_forecasts = np.concatenate(forecasts)
        all_actuals = np.concatenate(actuals)

        # Compute metrics
        results = {}
        for metric in metrics:
            if hasattr(Metrics, metric.lower()):
                metric_func = getattr(Metrics, metric.lower())
                results[metric] = metric_func(all_actuals, all_forecasts)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Compute custom metrics
        if custom_metrics:
            for metric_name, metric_func in custom_metrics.items():
                results[metric_name] = metric_func(all_actuals, all_forecasts)

        # Return results
        return {
            "strategy": self.strategy.value,
            "initial_train_size": initial_train_size,
            "step_size": self.step_size,
            "window_size": self.window_size if self.strategy == BacktestingStrategy.SLIDING_WINDOW else None,
            "horizon": horizon,
            "metrics": results,
            "forecasts": forecasts,
            "actuals": actuals,
            "train_sizes": train_sizes,
            "test_indices": test_indices,
            "model_states": model_states
        }

    def cross_validation(
        self,
        model: BaseModel,
        data: pd.DataFrame,
        target_column: str,
        n_splits: int = 5,
        horizon: int = 1,
        feature_columns: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        custom_metrics: Optional[Dict[str, Callable]] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation.

        Args:
            model: Forecasting model
            data: Input data
            target_column: Name of the target column
            n_splits: Number of cross-validation splits
            horizon: Forecast horizon
            feature_columns: List of feature columns
            metrics: List of metric names to compute
            custom_metrics: Dictionary of custom metric functions
            train_kwargs: Additional parameters for model fitting
            predict_kwargs: Additional parameters for model prediction

        Returns:
            Dict[str, Any]: Cross-validation results
        """
        if metrics is None:
            metrics = ['mse', 'rmse', 'mae', 'mape', 'smape']

        if train_kwargs is None:
            train_kwargs = {}

        if predict_kwargs is None:
            predict_kwargs = {}

        # Calculate split sizes
        data_size = len(data)
        fold_size = data_size // n_splits

        # Initialize results
        fold_results = []

        # Perform cross-validation
        for i in range(n_splits):
            if self.verbose:
                print(f"Cross-validation fold {i+1}/{n_splits}")

            # Determine test indices
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, data_size)

            # Split data
            train_data = pd.concat([data.iloc[:test_start], data.iloc[test_end:]])
            test_data = data.iloc[test_start:test_end]

            # Create a new instance of the model
            model_copy = type(model)(name=model.name, **model.params)

            # Fit the model
            model_copy.fit(train_data, target_column, **train_kwargs)

            # Generate forecast
            forecast = model_copy.predict(train_data, len(test_data), **predict_kwargs)

            # Compute metrics
            fold_metrics = {}
            for metric in metrics:
                if hasattr(Metrics, metric.lower()):
                    metric_func = getattr(Metrics, metric.lower())
                    fold_metrics[metric] = metric_func(test_data[target_column].values, forecast[target_column].values)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

            # Compute custom metrics
            if custom_metrics:
                for metric_name, metric_func in custom_metrics.items():
                    fold_metrics[metric_name] = metric_func(test_data[target_column].values, forecast[target_column].values)

            # Store results
            fold_results.append({
                "fold": i,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "metrics": fold_metrics,
                "forecast": forecast,
                "actuals": test_data[target_column]
            })

        # Compute average metrics
        avg_metrics = {}
        for metric in metrics:
            avg_metrics[metric] = np.mean([fold["metrics"][metric] for fold in fold_results])

        # Compute average custom metrics
        if custom_metrics:
            for metric_name in custom_metrics:
                avg_metrics[metric_name] = np.mean([fold["metrics"][metric_name] for fold in fold_results])

        # Return results
        return {
            "n_splits": n_splits,
            "horizon": horizon,
            "folds": fold_results,
            "avg_metrics": avg_metrics
        }

    def compare_models(
        self,
        models: List[BaseModel],
        data: pd.DataFrame,
        target_column: str,
        horizon: int = 1,
        metrics: Optional[List[str]] = None,
        custom_metrics: Optional[Dict[str, Callable]] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models using backtesting.

        Args:
            models: List of forecasting models
            data: Input data
            target_column: Name of the target column
            horizon: Forecast horizon
            metrics: List of metric names to compute
            custom_metrics: Dictionary of custom metric functions
            train_kwargs: Additional parameters for model fitting
            predict_kwargs: Additional parameters for model prediction

        Returns:
            Dict[str, Dict[str, float]]: Comparison results
        """
        if metrics is None:
            metrics = ['mse', 'rmse', 'mae', 'mape', 'smape']

        if train_kwargs is None:
            train_kwargs = {}

        if predict_kwargs is None:
            predict_kwargs = {}

        # Initialize results
        results = {}

        # Backtest each model
        for model in models:
            if self.verbose:
                print(f"Backtesting model: {model.name}")

            # Perform backtesting
            model_results = self.backtest(
                model=model,
                data=data,
                target_column=target_column,
                horizon=horizon,
                feature_columns=None,
                metrics=metrics,
                custom_metrics=custom_metrics,
                train_kwargs=train_kwargs,
                predict_kwargs=predict_kwargs
            )

            # Store results
            results[model.name] = model_results["metrics"]

        return results

    def plot_results(
        self,
        results: Dict[str, Any],
        title: str = "Backtesting Results",
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot backtesting results.

        Args:
            results: Backtesting results from backtest method
            title: Plot title
            figsize: Figure size
        """
        if plt is None:
            raise ImportError("Matplotlib is required for plotting")

        # Extract data
        forecasts = results["forecasts"]
        actuals = results["actuals"]
        test_indices = results["test_indices"]

        # Flatten data
        flat_forecasts = np.concatenate(forecasts)
        flat_actuals = np.concatenate(actuals)

        # Create indices for plotting
        indices = []
        for i, idx in enumerate(test_indices):
            indices.extend([idx + j for j in range(len(forecasts[i]))])

        # Create figure
        plt.figure(figsize=figsize)

        # Plot actuals
        plt.plot(indices, flat_actuals, label="Actual", color="blue")

        # Plot forecasts
        plt.plot(indices, flat_forecasts, label="Forecast", color="red", linestyle="--")

        # Add labels and title
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend()
        plt.grid(True)

        # Show plot
        plt.tight_layout()
        plt.show()

    def plot_metrics_by_horizon(
        self,
        results: Dict[str, Any],
        metric: str = "rmse",
        title: str = "Metric by Horizon",
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot metrics by forecast horizon.

        Args:
            results: Backtesting results from backtest method
            metric: Metric to plot
            title: Plot title
            figsize: Figure size
        """
        if plt is None:
            raise ImportError("Matplotlib is required for plotting")

        # Extract data
        forecasts = results["forecasts"]
        actuals = results["actuals"]

        # Compute metrics for each horizon
        horizon_metrics = []
        max_horizon = max(len(f) for f in forecasts)

        for h in range(1, max_horizon + 1):
            h_forecasts = []
            h_actuals = []

            for f, a in zip(forecasts, actuals):
                if len(f) >= h and len(a) >= h:
                    h_forecasts.append(f[h-1])
                    h_actuals.append(a[h-1])

            if h_forecasts and h_actuals:
                if hasattr(Metrics, metric.lower()):
                    metric_func = getattr(Metrics, metric.lower())
                    horizon_metrics.append(metric_func(np.array(h_actuals), np.array(h_forecasts)))
                else:
                    raise ValueError(f"Metric {metric} not found in Metrics class")

        # Create figure
        plt.figure(figsize=figsize)

        # Plot metrics by horizon
        plt.plot(range(1, len(horizon_metrics) + 1), horizon_metrics, marker="o")

        # Add labels and title
        plt.xlabel("Forecast Horizon")
        plt.ylabel(metric.upper())
        plt.title(title)
        plt.grid(True)

        # Show plot
        plt.tight_layout()
        plt.show()

    def plot_model_comparison(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        title: str = "Model Comparison",
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Plot model comparison results.

        Args:
            comparison_results: Results from compare_models method
            metrics: List of metrics to plot
            title: Plot title
            figsize: Figure size
        """
        if plt is None:
            raise ImportError("Matplotlib is required for plotting")

        if metrics is None:
            # Use all available metrics
            metrics = list(next(iter(comparison_results.values())).keys())

        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]

        # Plot each metric
        for i, metric in enumerate(metrics):
            # Extract values
            model_names = list(comparison_results.keys())
            metric_values = [results[metric] for results in comparison_results.values()]

            # Plot bar chart
            axes[i].bar(model_names, metric_values)

            # Add labels and title
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f"{metric.upper()} by Model")
            axes[i].grid(True, axis="y")

            # Add value labels
            for j, value in enumerate(metric_values):
                axes[i].text(j, value, f"{value:.4f}", ha="center", va="bottom")

        # Add overall title
        fig.suptitle(title)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Show plot
        plt.show()
