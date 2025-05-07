"""
Example usage of utility modules in TimeFusion.

This example demonstrates how to use the utility modules in TimeFusion,
including configuration, logging, visualization, and hyperparameter optimization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from timefusion.utils.config import Config, get_default_config
from timefusion.utils.logging import Logger, log_time
from timefusion.utils.visualization import (
    plot_time_series,
    plot_forecast,
    plot_residuals,
    plot_metrics,
    plot_comparison,
    plot_feature_importance
)
from timefusion.utils.hpo import GridSearch

from timefusion.models.statistical import ARIMAModel
from timefusion.models.deep_learning import LSTMModel
from timefusion.models.hybrid import EnsembleModel
from timefusion.evaluation.metrics import Metrics


def generate_sample_data(n_samples=100):
    """
    Generate sample time series data.
    
    Args:
        n_samples: Number of samples
        
    Returns:
        pd.DataFrame: Sample data
    """
    # Generate dates
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")
    
    # Generate values with trend, seasonality, and noise
    trend = np.linspace(0, 2, n_samples)
    seasonality = np.sin(np.linspace(0, 8 * np.pi, n_samples))
    noise = np.random.normal(0, 0.2, n_samples)
    
    values = trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({"value": values}, index=dates)
    
    # Add some features
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    
    return df


def main():
    """Main function."""
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Set up logging
    logger = Logger(
        name="timefusion",
        level="INFO",
        log_file="output/utilities_example.log",
        console=True
    )
    
    logger.info("Starting utilities example")
    
    # Load configuration
    with log_time(logger, "Loading configuration"):
        config = get_default_config()
        logger.info(f"Loaded default configuration with {len(config.config)} sections")
        
        # Update configuration
        config.set("models.arima.order", [5, 1, 0])
        config.set("models.lstm.hidden_size", 64)
        
        # Save configuration
        config.save_to_file("output/config.json")
        logger.info(f"Saved configuration to output/config.json")
    
    # Generate sample data
    with log_time(logger, "Generating sample data"):
        data = generate_sample_data(n_samples=100)
        logger.info(f"Generated sample data with shape {data.shape}")
        
        # Split data into train and test sets
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        logger.info(f"Split data into train ({len(train_data)}) and test ({len(test_data)}) sets")
    
    # Visualize data
    with log_time(logger, "Visualizing data"):
        # Plot time series
        fig = plot_time_series(
            data=data,
            columns=["value"],
            title="Sample Time Series",
            save_path="output/time_series.png"
        )
        logger.info("Saved time series plot to output/time_series.png")
    
    # Train models
    models = {}
    
    with log_time(logger, "Training ARIMA model"):
        # Create and train ARIMA model
        arima_model = ARIMAModel(
            name="arima",
            order=config.get("models.arima.order")
        )
        
        arima_model.fit(train_data, target_column="value")
        models["ARIMA"] = arima_model
        logger.info(f"Trained ARIMA model with order {config.get('models.arima.order')}")
    
    with log_time(logger, "Training LSTM model"):
        # Create and train LSTM model
        lstm_model = LSTMModel(
            name="lstm",
            hidden_size=config.get("models.lstm.hidden_size"),
            num_layers=config.get("models.lstm.num_layers"),
            dropout=config.get("models.lstm.dropout"),
            batch_size=config.get("models.lstm.batch_size"),
            learning_rate=config.get("models.lstm.learning_rate"),
            num_epochs=config.get("models.lstm.num_epochs")
        )
        
        lstm_model.fit(
            train_data,
            target_column="value",
            feature_columns=["day_of_week", "month", "day_of_year"]
        )
        models["LSTM"] = lstm_model
        logger.info(f"Trained LSTM model with hidden_size {config.get('models.lstm.hidden_size')}")
    
    with log_time(logger, "Training Ensemble model"):
        # Create and train Ensemble model
        ensemble_model = EnsembleModel(
            name="ensemble",
            models=[arima_model, lstm_model]
        )
        
        ensemble_model.fit(train_data, target_column="value")
        models["Ensemble"] = ensemble_model
        logger.info("Trained Ensemble model")
    
    # Generate forecasts
    forecasts = {}
    metrics_results = {}
    
    with log_time(logger, "Generating forecasts"):
        for name, model in models.items():
            # Generate forecast
            forecast = model.predict(
                data=train_data,
                horizon=len(test_data)
            )
            forecasts[name] = forecast
            
            # Calculate metrics
            y_true = test_data["value"].values
            y_pred = forecast["value"].values
            
            metrics = {
                "mse": Metrics.mse(y_true, y_pred),
                "rmse": Metrics.rmse(y_true, y_pred),
                "mae": Metrics.mae(y_true, y_pred),
                "mape": Metrics.mape(y_true, y_pred)
            }
            metrics_results[name] = metrics
            
            logger.info(f"Generated forecast for {name} model with metrics: {metrics}")
            
            # Plot forecast
            fig = plot_forecast(
                actual=test_data,
                forecast=forecast,
                title=f"{name} Forecast",
                save_path=f"output/{name.lower()}_forecast.png"
            )
            logger.info(f"Saved {name} forecast plot to output/{name.lower()}_forecast.png")
            
            # Plot residuals
            residuals = pd.DataFrame(
                {"value": y_true - y_pred},
                index=test_data.index
            )
            
            fig = plot_residuals(
                residuals=residuals,
                title=f"{name} Residuals",
                save_path=f"output/{name.lower()}_residuals.png"
            )
            logger.info(f"Saved {name} residuals plot to output/{name.lower()}_residuals.png")
    
    # Plot metrics
    with log_time(logger, "Plotting metrics"):
        # Plot metrics for each model
        for name, metrics in metrics_results.items():
            fig = plot_metrics(
                metrics=metrics,
                title=f"{name} Metrics",
                save_path=f"output/{name.lower()}_metrics.png"
            )
            logger.info(f"Saved {name} metrics plot to output/{name.lower()}_metrics.png")
        
        # Plot model comparison
        fig = plot_comparison(
            results=metrics_results,
            title="Model Comparison",
            save_path="output/model_comparison.png"
        )
        logger.info("Saved model comparison plot to output/model_comparison.png")
    
    # Hyperparameter optimization
    with log_time(logger, "Hyperparameter optimization"):
        # Create HPO instance
        hpo = GridSearch(model_class=ARIMAModel)
        
        # Define parameter grid
        param_grid = {
            "order": [
                [1, 1, 0],
                [5, 1, 0],
                [5, 1, 1]
            ]
        }
        
        # Run optimization
        results = hpo.optimize(
            data=data,
            param_grid=param_grid,
            target_column="value",
            metric="rmse"
        )
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best score: {results['best_score']}")
        
        # Plot results
        fig = hpo.plot_results(
            title="ARIMA Hyperparameter Optimization",
            save_path="output/hpo_results.png"
        )
        logger.info("Saved HPO results plot to output/hpo_results.png")
        
        # Save results
        hpo.save("output/hpo_results.pkl")
        logger.info("Saved HPO results to output/hpo_results.pkl")
    
    logger.info("Utilities example completed")


if __name__ == "__main__":
    main()
