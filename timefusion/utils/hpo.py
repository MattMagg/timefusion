"""
Hyperparameter optimization for TimeFusion.

This module provides utilities for hyperparameter optimization
for forecasting models in the TimeFusion system.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple, Callable, Type
from enum import Enum
import itertools
import random
import multiprocessing
import matplotlib.pyplot as plt

from ..models.base import BaseModel
from ..evaluation.metrics import Metrics

# Check if optuna is available for Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class OptimizationMethod(Enum):
    """Optimization method enum."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"


class HPO:
    """
    Hyperparameter optimization for forecasting models.

    This class provides methods for hyperparameter optimization
    for forecasting models in the TimeFusion system.

    Attributes:
        method (OptimizationMethod): Optimization method
        model_class (Type[BaseModel]): Model class
        model_kwargs (Dict[str, Any]): Model initialization parameters
        cv (int): Number of cross-validation folds
        n_jobs (int): Number of parallel jobs
        verbose (bool): Whether to print progress
        random_state (Optional[int]): Random state for reproducibility
        results (Dict[str, Any]): Optimization results
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        method: Union[str, OptimizationMethod] = OptimizationMethod.GRID_SEARCH,
        model_kwargs: Optional[Dict[str, Any]] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize the HPO.

        Args:
            model_class: Model class
            method: Optimization method
            model_kwargs: Model initialization parameters
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Whether to print progress
            random_state: Random state for reproducibility
        """
        self.model_class = model_class

        if isinstance(method, str):
            self.method = OptimizationMethod(method)
        else:
            self.method = method

        self.model_kwargs = model_kwargs or {}
        self.cv = cv
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self.verbose = verbose
        self.random_state = random_state
        self.results = {}

        # Set random seed if provided
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            if OPTUNA_AVAILABLE:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optimize(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        target_column: str,
        metric: Union[str, Callable] = "rmse",
        feature_columns: Optional[List[str]] = None,
        train_size: float = 0.8,
        n_trials: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters.

        Args:
            data: Input data
            param_grid: Parameter grid
            target_column: Name of the target column
            metric: Metric to optimize
            feature_columns: List of feature columns
            train_size: Train size for validation
            n_trials: Number of trials for random search and Bayesian optimization
            **kwargs: Additional parameters for model fitting and prediction

        Returns:
            Dict[str, Any]: Optimization results

        Raises:
            ValueError: If the optimization method is not supported
        """
        # Split data into train and validation sets
        train_size = int(len(data) * train_size)
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:]

        # Get metric function
        if isinstance(metric, str):
            metric_name = metric.lower()
            if hasattr(Metrics, metric_name):
                metric_fn = getattr(Metrics, metric_name)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        else:
            metric_fn = metric
            metric_name = metric_fn.__name__

        # Run optimization
        if self.method == OptimizationMethod.GRID_SEARCH:
            self._grid_search(
                train_data=train_data,
                val_data=val_data,
                param_grid=param_grid,
                target_column=target_column,
                metric_fn=metric_fn,
                feature_columns=feature_columns,
                **kwargs
            )
        elif self.method == OptimizationMethod.RANDOM_SEARCH:
            self._random_search(
                train_data=train_data,
                val_data=val_data,
                param_grid=param_grid,
                target_column=target_column,
                metric_fn=metric_fn,
                feature_columns=feature_columns,
                n_trials=n_trials,
                **kwargs
            )
        elif self.method == OptimizationMethod.BAYESIAN:
            if not OPTUNA_AVAILABLE:
                raise ImportError("Optuna is required for Bayesian optimization")

            self._bayesian_search(
                train_data=train_data,
                val_data=val_data,
                param_grid=param_grid,
                target_column=target_column,
                metric_fn=metric_fn,
                feature_columns=feature_columns,
                n_trials=n_trials,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

        # Return best parameters
        return self.results

    def _evaluate_model(
        self,
        params: Dict[str, Any],
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        target_column: str,
        metric_fn: Callable,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> float:
        """
        Evaluate a model with the given parameters.

        Args:
            params: Model parameters
            train_data: Training data
            val_data: Validation data
            target_column: Name of the target column
            metric_fn: Metric function
            feature_columns: List of feature columns
            **kwargs: Additional parameters for model fitting and prediction

        Returns:
            float: Metric value
        """
        # Create model
        model_kwargs = {**self.model_kwargs, **params}
        model = self.model_class(**model_kwargs)

        # Fit model
        fit_kwargs = kwargs.get("fit_kwargs", {})
        model.fit(
            data=train_data,
            target_column=target_column,
            feature_columns=feature_columns,
            **fit_kwargs
        )

        # Generate predictions
        predict_kwargs = kwargs.get("predict_kwargs", {})
        horizon = len(val_data)
        predictions = model.predict(
            data=train_data,
            horizon=horizon,
            **predict_kwargs
        )

        # Evaluate predictions
        y_true = val_data[target_column].values
        y_pred = predictions[target_column].values

        # Calculate metric
        score = metric_fn(y_true, y_pred)

        return score

    def _grid_search(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        target_column: str,
        metric_fn: Callable,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Perform grid search.

        Args:
            train_data: Training data
            val_data: Validation data
            param_grid: Parameter grid
            target_column: Name of the target column
            metric_fn: Metric function
            feature_columns: List of feature columns
            **kwargs: Additional parameters for model fitting and prediction
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        # Initialize results
        all_results = []

        # Evaluate all parameter combinations
        for i, param_combination in enumerate(param_combinations):
            if self.verbose:
                print(f"Evaluating parameter combination {i+1}/{len(param_combinations)}")

            # Create parameter dictionary
            params = {name: value for name, value in zip(param_names, param_combination)}

            # Evaluate model
            score = self._evaluate_model(
                params=params,
                train_data=train_data,
                val_data=val_data,
                target_column=target_column,
                metric_fn=metric_fn,
                feature_columns=feature_columns,
                **kwargs
            )

            # Store results
            all_results.append({
                "params": params,
                "score": score
            })

        # Sort results by score
        all_results.sort(key=lambda x: x["score"])

        # Store results
        self.results = {
            "method": self.method.value,
            "best_params": all_results[0]["params"],
            "best_score": all_results[0]["score"],
            "all_results": all_results
        }

    def _random_search(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        target_column: str,
        metric_fn: Callable,
        feature_columns: Optional[List[str]] = None,
        n_trials: int = 10,
        **kwargs
    ) -> None:
        """
        Perform random search.

        Args:
            train_data: Training data
            val_data: Validation data
            param_grid: Parameter grid
            target_column: Name of the target column
            metric_fn: Metric function
            feature_columns: List of feature columns
            n_trials: Number of trials
            **kwargs: Additional parameters for model fitting and prediction
        """
        # Initialize results
        all_results = []

        # Evaluate random parameter combinations
        for i in range(n_trials):
            if self.verbose:
                print(f"Evaluating parameter combination {i+1}/{n_trials}")

            # Create random parameter dictionary
            params = {name: random.choice(values) for name, values in param_grid.items()}

            # Evaluate model
            score = self._evaluate_model(
                params=params,
                train_data=train_data,
                val_data=val_data,
                target_column=target_column,
                metric_fn=metric_fn,
                feature_columns=feature_columns,
                **kwargs
            )

            # Store results
            all_results.append({
                "params": params,
                "score": score
            })

        # Sort results by score
        all_results.sort(key=lambda x: x["score"])

        # Store results
        self.results = {
            "method": self.method.value,
            "best_params": all_results[0]["params"],
            "best_score": all_results[0]["score"],
            "all_results": all_results
        }

    def _bayesian_search(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        target_column: str,
        metric_fn: Callable,
        feature_columns: Optional[List[str]] = None,
        n_trials: int = 10,
        **kwargs
    ) -> None:
        """
        Perform Bayesian optimization.

        Args:
            train_data: Training data
            val_data: Validation data
            param_grid: Parameter grid
            target_column: Name of the target column
            metric_fn: Metric function
            feature_columns: List of feature columns
            n_trials: Number of trials
            **kwargs: Additional parameters for model fitting and prediction
        """
        # Define objective function
        def objective(trial):
            # Create parameter dictionary
            params = {}
            for name, values in param_grid.items():
                if isinstance(values[0], int):
                    params[name] = trial.suggest_int(name, min(values), max(values))
                elif isinstance(values[0], float):
                    params[name] = trial.suggest_float(name, min(values), max(values))
                else:
                    params[name] = trial.suggest_categorical(name, values)

            # Evaluate model
            score = self._evaluate_model(
                params=params,
                train_data=train_data,
                val_data=val_data,
                target_column=target_column,
                metric_fn=metric_fn,
                feature_columns=feature_columns,
                **kwargs
            )

            return score

        # Create study
        study = optuna.create_study(direction="minimize")

        # Optimize
        study.optimize(objective, n_trials=n_trials)

        # Store results
        self.results = {
            "method": self.method.value,
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study
        }

    def plot_results(
        self,
        title: str = 'Hyperparameter Optimization Results',
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot optimization results.

        Args:
            title: Plot title
            figsize: Figure size
            save_path: Path to save the figure (if None, the figure is not saved)

        Returns:
            plt.Figure: Figure object

        Raises:
            ValueError: If no results are available
        """
        if not self.results:
            raise ValueError("No results available. Run optimize() first.")

        if self.method == OptimizationMethod.BAYESIAN and OPTUNA_AVAILABLE:
            # Plot Bayesian optimization results
            fig = optuna.visualization.plot_optimization_history(self.results["study"])

            if save_path:
                fig.write_image(save_path)

            return fig
        else:
            # Plot grid search or random search results
            all_results = self.results["all_results"]
            scores = [result["score"] for result in all_results]

            fig, ax = plt.subplots(figsize=figsize)

            ax.plot(range(1, len(scores) + 1), scores, marker='o')

            ax.set_title(title)
            ax.set_xlabel('Trial')
            ax.set_ylabel('Score')
            ax.grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)

            return fig

    def save(self, path: str) -> None:
        """
        Save optimization results.

        Args:
            path: Path to save the results

        Raises:
            ValueError: If no results are available
        """
        if not self.results:
            raise ValueError("No results available. Run optimize() first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save results
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)

    def load(self, path: str) -> None:
        """
        Load optimization results.

        Args:
            path: Path to load the results from

        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Results file not found: {path}")

        # Load results
        with open(path, 'rb') as f:
            self.results = pickle.load(f)


class GridSearch(HPO):
    """
    Grid search for hyperparameter optimization.

    This class is a convenience wrapper around the HPO class
    with the grid search method.
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        model_kwargs: Optional[Dict[str, Any]] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize the grid search.

        Args:
            model_class: Model class
            model_kwargs: Model initialization parameters
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Whether to print progress
            random_state: Random state for reproducibility
        """
        super().__init__(
            model_class=model_class,
            method=OptimizationMethod.GRID_SEARCH,
            model_kwargs=model_kwargs,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )


class RandomSearch(HPO):
    """
    Random search for hyperparameter optimization.

    This class is a convenience wrapper around the HPO class
    with the random search method.
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        model_kwargs: Optional[Dict[str, Any]] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize the random search.

        Args:
            model_class: Model class
            model_kwargs: Model initialization parameters
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Whether to print progress
            random_state: Random state for reproducibility
        """
        super().__init__(
            model_class=model_class,
            method=OptimizationMethod.RANDOM_SEARCH,
            model_kwargs=model_kwargs,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )


class BayesianSearch(HPO):
    """
    Bayesian search for hyperparameter optimization.

    This class is a convenience wrapper around the HPO class
    with the Bayesian search method.
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        model_kwargs: Optional[Dict[str, Any]] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize the Bayesian search.

        Args:
            model_class: Model class
            model_kwargs: Model initialization parameters
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Whether to print progress
            random_state: Random state for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")

        super().__init__(
            model_class=model_class,
            method=OptimizationMethod.BAYESIAN,
            model_kwargs=model_kwargs,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )
