"""
Cryptocurrency forecasting example using Nixtla's statsforecast.

This example demonstrates how to:
1. Load and preprocess cryptocurrency time series data
2. Filter for more dynamic periods in the data
3. Train multiple statistical models
4. Generate forecasts
5. Evaluate model performance
6. Visualize the results with detailed plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import sys
from pathlib import Path

# Import Nixtla's statsforecast
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, 
    AutoETS, 
    MSTL, 
    SeasonalNaive, 
    Theta,
    TBATS,
    DynamicOptimizedTheta
)


def load_and_prepare_data(file_path, filter_zero_volume=True, recent_days=None):
    """
    Load and prepare the cryptocurrency data for Nixtla's statsforecast.
    
    Args:
        file_path: Path to the CSV file
        filter_zero_volume: Whether to filter out periods with zero trading volume
        recent_days: If provided, only use data from the last N days
        
    Returns:
        pd.DataFrame: Prepared time series data in the format required by statsforecast
    """
    print(f"Loading data from {file_path}...")
    
    # Skip the first row which contains the website URL
    data = pd.read_csv(file_path, skiprows=1)
    
    # Convert unix timestamp to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort by date (ascending order)
    data = data.sort_values('date')
    
    # Filter for more dynamic data
    if filter_zero_volume:
        original_len = len(data)
        data = data[data['Volume BAT'] > 0]
        print(f"Filtered out {original_len - len(data)} periods with zero volume ({len(data)} periods remaining)")
    
    if recent_days:
        end_date = data['date'].max()
        start_date = end_date - timedelta(days=recent_days)
        data = data[data['date'] >= start_date]
        print(f"Using only the last {recent_days} days of data ({len(data)} periods)")
    
    # Select only the 'date' and 'close' price for forecasting
    data = data[['date', 'close']]
    
    # Rename columns to match statsforecast requirements
    # statsforecast requires 'ds' for dates, 'y' for values, and 'unique_id' for time series identifier
    data = data.rename(columns={'date': 'ds', 'close': 'y'})
    
    # Add a unique_id column (required by statsforecast)
    data['unique_id'] = 'BATBTC'
    
    print(f"Final dataset: {len(data)} data points")
    print(f"Date range: {data['ds'].min()} to {data['ds'].max()}")
    print(f"Price range: {data['y'].min()} to {data['y'].max()}")
    print(f"Number of unique prices: {data['y'].nunique()}")
    
    return data


def plot_data_characteristics(data, output_dir):
    """Plot characteristics of the data to understand its patterns."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot the time series
    plt.figure(figsize=(15, 7))
    plt.plot(data['ds'], data['y'])
    plt.title('BATBTC Exchange Rate')
    plt.xlabel('Date')
    plt.ylabel('Price (BTC)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batbtc_data.png')
    plt.close()
    
    # Plot the distribution of prices
    plt.figure(figsize=(12, 6))
    plt.hist(data['y'], bins=50)
    plt.title('Distribution of BATBTC Prices')
    plt.xlabel('Price (BTC)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batbtc_price_distribution.png')
    plt.close()
    
    # Plot the autocorrelation
    from pandas.plotting import autocorrelation_plot
    plt.figure(figsize=(12, 6))
    autocorrelation_plot(data['y'])
    plt.title('Autocorrelation of BATBTC Prices')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batbtc_autocorrelation.png')
    plt.close()
    
    # Plot price changes
    data_diff = data.copy()
    data_diff['price_change'] = data_diff['y'].diff()
    plt.figure(figsize=(15, 7))
    plt.plot(data_diff['ds'][1:], data_diff['price_change'][1:])
    plt.title('BATBTC Price Changes')
    plt.xlabel('Date')
    plt.ylabel('Price Change (BTC)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batbtc_price_changes.png')
    plt.close()


def run_forecasts(data, output_dir, season_length=24):
    """Run forecasts using multiple models and evaluate their performance."""
    # Split data into train and test
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"\nSplit data into train ({len(train_data)} samples) and test ({len(test_data)} samples)")
    
    # Define forecast horizon
    horizon = len(test_data)
    
    # Create models with different parameters
    models = [
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
        MSTL(season_length=season_length),
        Theta(),
        TBATS(season_length=season_length),
        DynamicOptimizedTheta(),
        SeasonalNaive(season_length=season_length)
    ]
    
    # Create StatsForecast instance
    sf = StatsForecast(
        models=models,
        freq='h',  # Hourly data
        n_jobs=-1  # Use all available cores
    )
    
    print("\nFitting models...")
    
    # Fit models and generate forecasts
    forecasts = sf.forecast(df=train_data, h=horizon)
    
    print("\nForecasts generated:")
    print(forecasts.head())
    
    # Evaluate forecasts
    print("\nEvaluating forecasts...")
    
    # Merge forecasts with actual values for evaluation
    evaluation = test_data.copy()
    evaluation = evaluation.reset_index(drop=True)
    
    # Calculate metrics
    metrics = {}
    for model in [m.__class__.__name__ for m in models]:
        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(evaluation['y'] - forecasts[model].values))
        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((evaluation['y'] - forecasts[model].values)**2))
        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((evaluation['y'] - forecasts[model].values) / evaluation['y'])) * 100
        
        metrics[model] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    # Print metrics
    print("\nForecast Metrics:")
    for model, model_metrics in metrics.items():
        print(f"\n{model} model:")
        for metric_name, metric_value in model_metrics.items():
            print(f"  {metric_name}: {metric_value:.8f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv(f'{output_dir}/forecast_metrics.csv')
    
    # Plot forecasts
    plot_forecasts(train_data, test_data, forecasts, models, output_dir)
    
    return metrics


def plot_forecasts(train_data, test_data, forecasts, models, output_dir):
    """Plot forecasts for each model and a comparison of all models."""
    print("\nPlotting forecasts...")
    
    # Combine train and test data for plotting
    full_data = pd.concat([train_data, test_data.reset_index(drop=True)])
    
    # Create forecast dates
    horizon = len(test_data)
    forecast_dates = pd.date_range(
        start=train_data['ds'].iloc[-1], 
        periods=horizon + 1, 
        freq='h'
    )[1:]
    
    # Plot each model's forecast
    for model in [m.__class__.__name__ for m in models]:
        plt.figure(figsize=(15, 7))
        
        # Plot actual data
        plt.plot(full_data['ds'], full_data['y'], label='Actual', linewidth=2)
        
        # Plot forecast
        forecast_values = forecasts[model].values
        plt.plot(forecast_dates, forecast_values, label=f'{model} Forecast', linewidth=2)
        
        # Add vertical line at train/test split
        plt.axvline(x=train_data['ds'].iloc[-1], color='r', linestyle='--', label='Train/Test Split')
        
        # Highlight the forecast area
        plt.fill_between(
            forecast_dates, 
            forecast_values * 0.95,  # Lower bound (5% below forecast)
            forecast_values * 1.05,  # Upper bound (5% above forecast)
            alpha=0.2, 
            color='blue'
        )
        
        # Format the plot
        plt.title(f'{model} Forecast for BATBTC', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (BTC)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Format date axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/batbtc_{model.lower()}_forecast.png')
        plt.close()
    
    # Plot comparison of all models
    plt.figure(figsize=(15, 10))
    
    # Plot actual data
    plt.plot(full_data['ds'], full_data['y'], label='Actual', linewidth=3, color='black')
    
    # Plot each model's forecast with different colors
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
    for i, model in enumerate([m.__class__.__name__ for m in models]):
        forecast_values = forecasts[model].values
        plt.plot(
            forecast_dates, 
            forecast_values, 
            label=f'{model} Forecast', 
            linewidth=2,
            color=colors[i % len(colors)]
        )
    
    # Add vertical line at train/test split
    plt.axvline(x=train_data['ds'].iloc[-1], color='red', linestyle='--', linewidth=2, label='Train/Test Split')
    
    # Format the plot
    plt.title('Forecast Comparison for BATBTC', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (BTC)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batbtc_forecast_comparison.png')
    plt.close()
    
    print(f"\nForecast plots saved in the '{output_dir}' directory")


def main():
    """Run the cryptocurrency forecasting example."""
    print("Cryptocurrency Forecasting Example with Nixtla")
    print("=============================================")
    
    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    output_dir = script_dir / "outputs" / "cryptocurrency"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    data_path = data_dir / "Gemini_BATBTC_1h.csv"
    data = load_and_prepare_data(
        data_path, 
        filter_zero_volume=True,
        recent_days=90
    )
    
    # Plot data characteristics
    plot_data_characteristics(data, output_dir)
    
    # Run forecasts
    run_forecasts(data, output_dir, season_length=24)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
