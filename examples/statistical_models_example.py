"""
Example usage of statistical models in TimeFusion.

This example demonstrates how to use the statistical models in TimeFusion,
including ARIMAModel and ExponentialSmoothingModel.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from timefusion.models.statistical import ARIMAModel, ExponentialSmoothingModel
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
    
    return data


def main():
    """Run the example."""
    print("TimeFusion Statistical Models Example")
    print("================================")
    
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
    
    # Create ARIMA model
    print("\nCreating and fitting ARIMA model...")
    arima_model = ARIMAModel(
        name="arima",
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7)
    )
    
    # Fit ARIMA model
    arima_model.fit(train_data_processed, target_column='value')
    
    # Generate ARIMA forecasts
    horizon = len(test_data)
    arima_forecast = arima_model.predict(train_data_processed, horizon=horizon)
    
    # Inverse transform forecasts
    arima_forecast_original = pipeline.inverse_transform(arima_forecast)
    
    # Create Exponential Smoothing model
    print("\nCreating and fitting Exponential Smoothing model...")
    ets_model = ExponentialSmoothingModel(
        name="ets",
        trend='add',
        seasonal='add',
        seasonal_periods=7,
        damped_trend=True
    )
    
    # Fit Exponential Smoothing model
    ets_model.fit(train_data_processed, target_column='value')
    
    # Generate Exponential Smoothing forecasts
    ets_forecast = ets_model.predict(train_data_processed, horizon=horizon)
    
    # Inverse transform forecasts
    ets_forecast_original = pipeline.inverse_transform(ets_forecast)
    
    # Evaluate forecasts
    print("\nEvaluating forecasts...")
    arima_metrics = Metrics.calculate_metrics(test_data['value'], arima_forecast_original['value'])
    ets_metrics = Metrics.calculate_metrics(test_data['value'], ets_forecast_original['value'])
    
    print("\nARIMA model metrics:")
    for metric, value in arima_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nExponential Smoothing model metrics:")
    for metric, value in ets_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot forecasts
    print("\nPlotting forecasts...")
    
    # Plot ARIMA forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(arima_forecast_original.index, arima_forecast_original['value'], label='ARIMA Forecast')
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('ARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('arima_forecast.png')
    plt.close()
    
    # Plot Exponential Smoothing forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(ets_forecast_original.index, ets_forecast_original['value'], label='ETS Forecast')
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('Exponential Smoothing Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ets_forecast.png')
    plt.close()
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(arima_forecast_original.index, arima_forecast_original['value'], label='ARIMA Forecast')
    plt.plot(ets_forecast_original.index, ets_forecast_original['value'], label='ETS Forecast')
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('forecast_comparison.png')
    plt.close()
    
    print("\nForecast plots saved as 'arima_forecast.png', 'ets_forecast.png', and 'forecast_comparison.png'")
    
    # Generate forecasts with confidence intervals
    print("\nGenerating forecasts with confidence intervals...")
    arima_forecast_ci = arima_model.predict_with_confidence(train_data_processed, horizon=horizon)
    ets_forecast_ci = ets_model.predict_with_confidence(train_data_processed, horizon=horizon)
    
    # Inverse transform forecasts with confidence intervals
    arima_forecast_ci_original = pipeline.inverse_transform(arima_forecast_ci)
    ets_forecast_ci_original = pipeline.inverse_transform(ets_forecast_ci)
    
    # Plot forecasts with confidence intervals
    print("\nPlotting forecasts with confidence intervals...")
    
    # Plot ARIMA forecast with confidence intervals
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(arima_forecast_ci_original.index, arima_forecast_ci_original['value'], label='ARIMA Forecast')
    plt.fill_between(
        arima_forecast_ci_original.index,
        arima_forecast_ci_original['value_lower_95'],
        arima_forecast_ci_original['value_upper_95'],
        alpha=0.2,
        label='95% Confidence Interval'
    )
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('ARIMA Forecast with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('arima_forecast_ci.png')
    plt.close()
    
    # Plot Exponential Smoothing forecast with confidence intervals
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Actual')
    plt.plot(ets_forecast_ci_original.index, ets_forecast_ci_original['value'], label='ETS Forecast')
    plt.fill_between(
        ets_forecast_ci_original.index,
        ets_forecast_ci_original['value_lower_95'],
        ets_forecast_ci_original['value_upper_95'],
        alpha=0.2,
        label='95% Confidence Interval'
    )
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('Exponential Smoothing Forecast with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ets_forecast_ci.png')
    plt.close()
    
    print("\nForecast plots with confidence intervals saved as 'arima_forecast_ci.png' and 'ets_forecast_ci.png'")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
