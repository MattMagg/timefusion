"""
Example usage of preprocessing components in TimeFusion.

This example demonstrates how to use the preprocessing components in TimeFusion,
including Cleaner, Imputer, Normalizer, FeatureEngineering, and Pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from timefusion.preprocessing import (
    Cleaner, Imputer, Normalizer, FeatureEngineering, Pipeline
)


# Create a sample time series dataset with outliers and missing values
def create_sample_data(n_samples=100, freq='D'):
    """Create a sample time series dataset with outliers and missing values."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_samples, freq=freq)
    trend = np.linspace(0, 2, n_samples)
    seasonality = np.sin(np.linspace(0, 8 * np.pi, n_samples))
    noise = np.random.normal(0, 0.2, n_samples)
    values = trend + seasonality + noise
    
    # Add some outliers
    outlier_indices = [10, 30, 70]
    for idx in outlier_indices:
        values[idx] += 5.0
    
    # Add some missing values
    missing_indices = [20, 40, 60, 80]
    for idx in missing_indices:
        values[idx] = np.nan
    
    return pd.DataFrame({
        'value': values
    }, index=dates)


def main():
    """Run the example."""
    # Create sample data
    data = create_sample_data()
    print("Sample data:")
    print(data.head())
    
    # Plot the original data
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Original')
    plt.title('Original Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Create preprocessing components
    cleaner = Cleaner(
        method='z-score',
        threshold=3.0,
        strategy='clip'
    )
    
    imputer = Imputer(
        method='linear'
    )
    
    normalizer = Normalizer(
        method='min-max'
    )
    
    feature_engineering = FeatureEngineering(
        lag_features=[1, 7],
        window_features={'mean': [7, 14], 'std': [7]},
        date_features=['day_of_week', 'month'],
        fourier_features={'7': 1, '30': 2},  # Weekly and monthly seasonality
        target_column='value'
    )
    
    # Create preprocessing pipeline
    pipeline = Pipeline([
        ('cleaner', cleaner),
        ('imputer', imputer),
        ('normalizer', normalizer),
        ('feature_engineering', feature_engineering)
    ])
    
    # Apply the pipeline
    processed_data = pipeline.fit_transform(data)
    print("\nProcessed data:")
    print(processed_data.head())
    
    # Plot the processed data
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Original', alpha=0.5)
    plt.plot(processed_data.index, processed_data['value'], label='Processed')
    plt.title('Original vs Processed Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot some of the engineered features
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(processed_data.index, processed_data['value'], label='Value')
    plt.plot(processed_data.index, processed_data['value_lag_7'], label='Lag 7')
    plt.title('Value vs Lag 7')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(processed_data.index, processed_data['value'], label='Value')
    plt.plot(processed_data.index, processed_data['value_mean_7'], label='Rolling Mean 7')
    plt.title('Value vs Rolling Mean 7')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.scatter(processed_data['day_of_week'], processed_data['value'], alpha=0.5)
    plt.title('Value by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Value')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(processed_data.index, processed_data['fourier_sin_7_1'], label='Sin')
    plt.plot(processed_data.index, processed_data['fourier_cos_7_1'], label='Cos')
    plt.title('Fourier Features (Weekly)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate inverse transformation for normalizer
    normalized_data = normalizer.transform(data)
    original_scale_data = normalizer.inverse_transform(normalized_data)
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], label='Original', alpha=0.5)
    plt.plot(normalized_data.index, normalized_data['value'], label='Normalized')
    plt.plot(original_scale_data.index, original_scale_data['value'], label='Inverse Transformed')
    plt.title('Normalization and Inverse Transformation')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("\nFeature columns created:")
    for column in processed_data.columns:
        if column != 'value':
            print(f"- {column}")


if __name__ == "__main__":
    main()
