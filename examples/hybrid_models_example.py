"""
Example usage of hybrid models in TimeFusion.

This example demonstrates how to use the hybrid models in TimeFusion,
including EnsembleModel and ResidualModel.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from timefusion.models.statistical import ARIMAModel, ExponentialSmoothingModel
from timefusion.models.deep_learning import LSTMModel, SimpleRNNModel
from timefusion.models.hybrid import EnsembleModel, ResidualModel
from timefusion.preprocessing.base import Pipeline
from timefusion.preprocessing.normalizer import Normalizer
from timefusion.evaluation.metrics import Metrics
from timefusion.utils.visualization import plot_forecast, plot_comparison


def generate_sample_data(n_samples=200, freq='D', seed=42):
    """
    Generate sample time series data with trend, seasonality, and noise.
    
    Args:
        n_samples: Number of samples to generate
        freq: Frequency of the time series
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Sample time series data
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq=freq)
    
    # Create trend component
    trend = np.linspace(0, 10, n_samples)
    
    # Create seasonal component (weekly seasonality)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
    
    # Create noise component
    noise = np.random.normal(0, 1, n_samples)
    
    # Combine components
    values = trend + seasonal + noise
    
    # Create DataFrame
    data = pd.DataFrame({'value': values}, index=dates)
    
    # Add some additional features for multivariate forecasting
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['day_of_year'] = data.index.dayofyear
    
    return data


def main():
    """Run the example."""
    print("TimeFusion Hybrid Models Example")
    print("===========================")
    
    # Generate sample data
    print("\nGenerating sample data...")
    data = generate_sample_data()
    print(f"Generated {len(data)} data points")
    print(data.head())
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'])
    plt.title('Sample Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sample_data.png')
    plt.close()
    
    print("\nSample data plot saved as 'sample_data.png'")
    
    # Split data into train and test
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"\nSplit data into train ({len(train_data)} samples) and test ({len(test_data)} samples)")
    
    # Create preprocessing pipeline
    print("\nCreating preprocessing pipeline...")
    normalizer = Normalizer(method='min-max')
    pipeline = Pipeline([
        ('normalizer', normalizer)
    ])
    
    # Preprocess data
    train_data_processed = pipeline.fit_transform(train_data)
    test_data_processed = pipeline.transform(test_data)
    
    print("\nPreprocessed data:")
    print(train_data_processed.head())
    
    # Create individual models
    print("\nCreating individual models...")
    
    # Statistical models
    arima_model = ARIMAModel(
        name="arima",
        order=(5, 1, 0),
        seasonal_order=(1, 0, 0, 7)
    )
    
    ets_model = ExponentialSmoothingModel(
        name="ets",
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        damped_trend=True
    )
    
    # Deep learning models
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
    
    rnn_model = SimpleRNNModel(
        name="simple_rnn",
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        sequence_length=10,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=50  # Reduced for example
    )
    
    # Fit individual models
    print("\nFitting individual models...")
    
    # Fit statistical models
    arima_model.fit(train_data, target_column='value')
    ets_model.fit(train_data, target_column='value')
    
    # Fit deep learning models
    lstm_model.fit(
        train_data_processed, 
        target_column='value',
        feature_columns=['value', 'day_of_week', 'month', 'day_of_year']
    )
    
    rnn_model.fit(
        train_data_processed, 
        target_column='value',
        feature_columns=['value', 'day_of_week', 'month', 'day_of_year']
    )
    
    # Create and fit ensemble model
    print("\nCreating and fitting ensemble model...")
    ensemble_model = EnsembleModel(
        name="ensemble",
        models=[arima_model, ets_model, lstm_model, rnn_model],
        weights=[0.3, 0.2, 0.3, 0.2],
        ensemble_method="weighted_average"
    )
    
    ensemble_model.fit(train_data, target_column='value')
    
    # Create and fit residual model
    print("\nCreating and fitting residual model...")
    residual_model = ResidualModel(
        name="residual",
        statistical_model=arima_model,
        deep_learning_model=lstm_model
    )
    
    residual_model.fit(train_data, target_column='value')
    
    # Generate forecasts
    print("\nGenerating forecasts...")
    horizon = len(test_data)
    
    # Generate individual forecasts
    arima_forecast = arima_model.predict(train_data, horizon=horizon)
    ets_forecast = ets_model.predict(train_data, horizon=horizon)
    
    lstm_forecast = lstm_model.predict(train_data_processed, horizon=horizon)
    lstm_forecast_original = pipeline.inverse_transform(lstm_forecast)
    
    rnn_forecast = rnn_model.predict(train_data_processed, horizon=horizon)
    rnn_forecast_original = pipeline.inverse_transform(rnn_forecast)
    
    # Generate hybrid forecasts
    ensemble_forecast = ensemble_model.predict(train_data, horizon=horizon)
    residual_forecast = residual_model.predict(train_data, horizon=horizon)
    
    # Evaluate forecasts
    print("\nEvaluating forecasts...")
    arima_metrics = Metrics.calculate_metrics(test_data['value'], arima_forecast['value'])
    ets_metrics = Metrics.calculate_metrics(test_data['value'], ets_forecast['value'])
    lstm_metrics = Metrics.calculate_metrics(test_data['value'], lstm_forecast_original['value'])
    rnn_metrics = Metrics.calculate_metrics(test_data['value'], rnn_forecast_original['value'])
    ensemble_metrics = Metrics.calculate_metrics(test_data['value'], ensemble_forecast['value'])
    residual_metrics = Metrics.calculate_metrics(test_data['value'], residual_forecast['value'])
    
    print("\nARIMA model metrics:")
    for metric, value in arima_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nETS model metrics:")
    for metric, value in ets_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nLSTM model metrics:")
    for metric, value in lstm_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nSimpleRNN model metrics:")
    for metric, value in rnn_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nEnsemble model metrics:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nResidual model metrics:")
    for metric, value in residual_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot forecasts
    print("\nPlotting forecasts...")
    
    # Plot comparison of all models
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(arima_forecast.index, arima_forecast['value'], label='ARIMA')
    plt.plot(ets_forecast.index, ets_forecast['value'], label='ETS')
    plt.plot(lstm_forecast_original.index, lstm_forecast_original['value'], label='LSTM')
    plt.plot(rnn_forecast_original.index, rnn_forecast_original['value'], label='SimpleRNN')
    plt.plot(ensemble_forecast.index, ensemble_forecast['value'], label='Ensemble')
    plt.plot(residual_forecast.index, residual_forecast['value'], label='Residual')
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('forecast_comparison.png')
    plt.close()
    
    # Plot ensemble model forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(ensemble_forecast.index, ensemble_forecast['value'], label='Ensemble Forecast')
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('Ensemble Model Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ensemble_forecast.png')
    plt.close()
    
    # Plot residual model forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(residual_forecast.index, residual_forecast['value'], label='Residual Forecast')
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('Residual Model Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('residual_forecast.png')
    plt.close()
    
    # Get and plot component forecasts from residual model
    component_forecasts = residual_model.get_component_forecasts(train_data, horizon)
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(component_forecasts['statistical'].index, component_forecasts['statistical']['value'], label='Statistical Component')
    plt.plot(component_forecasts['residual'].index, component_forecasts['residual']['value'], label='Residual Component')
    plt.plot(component_forecasts['combined'].index, component_forecasts['combined']['value'], label='Combined Forecast')
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('Residual Model Components')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('residual_components.png')
    plt.close()
    
    print("\nForecast plots saved as 'forecast_comparison.png', 'ensemble_forecast.png', 'residual_forecast.png', and 'residual_components.png'")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
