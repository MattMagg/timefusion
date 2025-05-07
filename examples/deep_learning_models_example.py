"""
Example usage of deep learning models in TimeFusion.

This example demonstrates how to use the deep learning models in TimeFusion,
including LSTMModel and SimpleRNNModel.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from timefusion.models.deep_learning import LSTMModel, SimpleRNNModel
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
    print("TimeFusion Deep Learning Models Example")
    print("==================================")
    
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
    
    # Create LSTM model
    print("\nCreating and fitting LSTM model...")
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
    
    # Fit LSTM model
    lstm_model.fit(
        train_data_processed, 
        target_column='value',
        feature_columns=['value', 'day_of_week', 'month', 'day_of_year']
    )
    
    # Generate LSTM forecasts
    horizon = len(test_data)
    lstm_forecast = lstm_model.predict(train_data_processed, horizon=horizon)
    
    # Inverse transform forecasts
    lstm_forecast_original = pipeline.inverse_transform(lstm_forecast)
    
    # Create SimpleRNN model
    print("\nCreating and fitting SimpleRNN model...")
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
    
    # Fit SimpleRNN model
    rnn_model.fit(
        train_data_processed, 
        target_column='value',
        feature_columns=['value', 'day_of_week', 'month', 'day_of_year']
    )
    
    # Generate SimpleRNN forecasts
    rnn_forecast = rnn_model.predict(train_data_processed, horizon=horizon)
    
    # Inverse transform forecasts
    rnn_forecast_original = pipeline.inverse_transform(rnn_forecast)
    
    # Evaluate forecasts
    print("\nEvaluating forecasts...")
    lstm_metrics = Metrics.calculate_metrics(test_data['value'], lstm_forecast_original['value'])
    rnn_metrics = Metrics.calculate_metrics(test_data['value'], rnn_forecast_original['value'])
    
    print("\nLSTM model metrics:")
    for metric, value in lstm_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nSimpleRNN model metrics:")
    for metric, value in rnn_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot forecasts
    print("\nPlotting forecasts...")
    
    # Plot LSTM forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(lstm_forecast_original.index, lstm_forecast_original['value'], label='LSTM Forecast')
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('LSTM Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lstm_forecast.png')
    plt.close()
    
    # Plot SimpleRNN forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(rnn_forecast_original.index, rnn_forecast_original['value'], label='SimpleRNN Forecast')
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('SimpleRNN Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('rnn_forecast.png')
    plt.close()
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(lstm_forecast_original.index, lstm_forecast_original['value'], label='LSTM Forecast')
    plt.plot(rnn_forecast_original.index, rnn_forecast_original['value'], label='SimpleRNN Forecast')
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('forecast_comparison.png')
    plt.close()
    
    print("\nForecast plots saved as 'lstm_forecast.png', 'rnn_forecast.png', and 'forecast_comparison.png'")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
