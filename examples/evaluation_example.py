"""
Example usage of the evaluation module in TimeFusion.

This example demonstrates how to use the evaluation module in TimeFusion,
including metrics, backtesting, and cross-validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from timefusion.models.statistical import ARIMAModel, ExponentialSmoothingModel
from timefusion.models.deep_learning import LSTMModel
from timefusion.models.hybrid import EnsembleModel
from timefusion.preprocessing.base import Pipeline
from timefusion.preprocessing.normalizer import Normalizer
from timefusion.evaluation.metrics import Metrics
from timefusion.evaluation.backtesting import Backtesting, BacktestingStrategy
from timefusion.evaluation.cross_validation import (
    time_series_split,
    blocked_time_series_split,
    plot_time_series_split
)


def create_sample_data(n_samples=365, freq='D'):
    """Create sample time series data."""
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq=freq)
    
    # Create trend component
    trend = np.linspace(0, 0.5, n_samples)
    
    # Create seasonal component (weekly and yearly)
    weekly_seasonality = 0.5 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
    yearly_seasonality = 1.0 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
    
    # Create noise component
    np.random.seed(42)
    noise = 0.2 * np.random.normal(0, 1, n_samples)
    
    # Combine components
    values = 10 + trend + weekly_seasonality + yearly_seasonality + noise
    
    # Create DataFrame
    data = pd.DataFrame({'value': values}, index=dates)
    
    return data


def main():
    """Run the example."""
    # Create sample data
    print("Creating sample data...")
    data = create_sample_data()
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'])
    plt.title('Sample Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Split data into train and test
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Create models
    print("\nCreating models...")
    
    # Statistical models
    arima_model = ARIMAModel(
        name="arima",
        order=(5, 1, 0),
        seasonal_order=(1, 0, 0, 7)
    )
    
    ets_model = ExponentialSmoothingModel(
        name="ets",
        trend='add',
        seasonal='add',
        seasonal_periods=7
    )
    
    # Deep learning model
    lstm_model = LSTMModel(
        name="lstm",
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        sequence_length=10,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=50  # Reduced for example
    )
    
    # Ensemble model
    ensemble_model = EnsembleModel(
        name="ensemble",
        models=[arima_model, ets_model, lstm_model],
        weights=[0.4, 0.3, 0.3],
        ensemble_method="weighted_average"
    )
    
    # Fit models
    print("\nFitting models...")
    
    # Fit statistical models
    arima_model.fit(train_data, target_column='value')
    ets_model.fit(train_data, target_column='value')
    
    # Fit deep learning model
    lstm_model.fit(train_data, target_column='value')
    
    # Fit ensemble model
    ensemble_model.fit(train_data, target_column='value')
    
    # Generate forecasts
    print("\nGenerating forecasts...")
    horizon = len(test_data)
    
    arima_forecast = arima_model.predict(train_data, horizon)
    ets_forecast = ets_model.predict(train_data, horizon)
    lstm_forecast = lstm_model.predict(train_data, horizon)
    ensemble_forecast = ensemble_model.predict(train_data, horizon)
    
    # Evaluate forecasts using metrics
    print("\nEvaluating forecasts using metrics...")
    
    # Calculate metrics for each model
    arima_metrics = Metrics.calculate_metrics(
        test_data['value'],
        arima_forecast['value'],
        train_data['value']
    )
    
    ets_metrics = Metrics.calculate_metrics(
        test_data['value'],
        ets_forecast['value'],
        train_data['value']
    )
    
    lstm_metrics = Metrics.calculate_metrics(
        test_data['value'],
        lstm_forecast['value'],
        train_data['value']
    )
    
    ensemble_metrics = Metrics.calculate_metrics(
        test_data['value'],
        ensemble_forecast['value'],
        train_data['value']
    )
    
    # Print metrics
    print("\nARIMA Model Metrics:")
    for metric, value in arima_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nETS Model Metrics:")
    for metric, value in ets_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nLSTM Model Metrics:")
    for metric, value in lstm_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nEnsemble Model Metrics:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual', color='black')
    plt.plot(arima_forecast.index, arima_forecast['value'], label='ARIMA', linestyle='--')
    plt.plot(ets_forecast.index, ets_forecast['value'], label='ETS', linestyle='--')
    plt.plot(lstm_forecast.index, lstm_forecast['value'], label='LSTM', linestyle='--')
    plt.plot(ensemble_forecast.index, ensemble_forecast['value'], label='Ensemble', linestyle='--')
    plt.axvline(x=train_data.index[-1], color='gray', linestyle='-.')
    plt.title('Forecasts vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Backtesting
    print("\nPerforming backtesting...")
    
    # Create backtesting instance
    backtesting = Backtesting(
        strategy=BacktestingStrategy.EXPANDING_WINDOW,
        initial_train_size=int(len(data) * 0.5),
        step_size=10
    )
    
    # Perform backtesting for each model
    arima_backtest = backtesting.backtest(
        model=arima_model,
        data=data,
        target_column='value',
        horizon=7  # 1-week forecast
    )
    
    ets_backtest = backtesting.backtest(
        model=ets_model,
        data=data,
        target_column='value',
        horizon=7  # 1-week forecast
    )
    
    lstm_backtest = backtesting.backtest(
        model=lstm_model,
        data=data,
        target_column='value',
        horizon=7  # 1-week forecast
    )
    
    ensemble_backtest = backtesting.backtest(
        model=ensemble_model,
        data=data,
        target_column='value',
        horizon=7  # 1-week forecast
    )
    
    # Print backtesting results
    print("\nBacktesting Results (RMSE):")
    print(f"  ARIMA: {arima_backtest['metrics']['rmse']:.4f}")
    print(f"  ETS: {ets_backtest['metrics']['rmse']:.4f}")
    print(f"  LSTM: {lstm_backtest['metrics']['rmse']:.4f}")
    print(f"  Ensemble: {ensemble_backtest['metrics']['rmse']:.4f}")
    
    # Plot backtesting results
    backtesting.plot_results(arima_backtest, title='ARIMA Backtesting Results')
    backtesting.plot_metrics_by_horizon(arima_backtest, metric='rmse', title='ARIMA RMSE by Horizon')
    
    # Compare models using backtesting
    print("\nComparing models using backtesting...")
    
    comparison = backtesting.compare_models(
        models=[arima_model, ets_model, lstm_model, ensemble_model],
        data=data,
        target_column='value',
        horizon=7  # 1-week forecast
    )
    
    # Print comparison results
    print("\nModel Comparison Results:")
    for model_name, metrics in comparison.items():
        print(f"  {model_name}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    # Plot model comparison
    backtesting.plot_model_comparison(comparison, title='Model Comparison')
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    
    # Create cross-validation splits
    cv_splits = time_series_split(data, n_splits=5, test_size=30)
    
    # Plot cross-validation splits
    plot_time_series_split(data, cv_splits, 'value', title='Time Series Cross-Validation')
    
    # Perform cross-validation for each model
    arima_cv = backtesting.cross_validation(
        model=arima_model,
        data=data,
        target_column='value',
        n_splits=5,
        horizon=7  # 1-week forecast
    )
    
    ets_cv = backtesting.cross_validation(
        model=ets_model,
        data=data,
        target_column='value',
        n_splits=5,
        horizon=7  # 1-week forecast
    )
    
    lstm_cv = backtesting.cross_validation(
        model=lstm_model,
        data=data,
        target_column='value',
        n_splits=5,
        horizon=7  # 1-week forecast
    )
    
    ensemble_cv = backtesting.cross_validation(
        model=ensemble_model,
        data=data,
        target_column='value',
        n_splits=5,
        horizon=7  # 1-week forecast
    )
    
    # Print cross-validation results
    print("\nCross-Validation Results (RMSE):")
    print(f"  ARIMA: {arima_cv['avg_metrics']['rmse']:.4f}")
    print(f"  ETS: {ets_cv['avg_metrics']['rmse']:.4f}")
    print(f"  LSTM: {lstm_cv['avg_metrics']['rmse']:.4f}")
    print(f"  Ensemble: {ensemble_cv['avg_metrics']['rmse']:.4f}")
    
    print("\nExample completed.")


if __name__ == "__main__":
    main()
