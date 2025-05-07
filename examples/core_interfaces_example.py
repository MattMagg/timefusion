"""
Example usage of core interfaces in TimeFusion.

This example demonstrates how to use the core interfaces in TimeFusion,
including BaseModel, BasePreprocessor, Pipeline, and ModelRegistry.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from timefusion.models.base import BaseModel, ModelRegistry
from timefusion.preprocessing.base import BasePreprocessor, Pipeline
from timefusion.evaluation.metrics import Metrics
from timefusion.utils.visualization import plot_forecast


# Create a simple time series dataset
def create_sample_data(n_samples=100, freq='D'):
    """Create a sample time series dataset."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_samples, freq=freq)
    trend = np.linspace(0, 2, n_samples)
    seasonality = np.sin(np.linspace(0, 8 * np.pi, n_samples))
    noise = np.random.normal(0, 0.2, n_samples)
    values = trend + seasonality + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values
    }).set_index('date')


# Create a simple preprocessor
class SimpleNormalizer(BasePreprocessor):
    """Simple normalizer that scales data to [0, 1] range."""
    
    def __init__(self, name="simple_normalizer"):
        """Initialize the normalizer."""
        super().__init__(name=name)
        self.min_val = None
        self.max_val = None
    
    def fit(self, data, **kwargs):
        """Fit the normalizer to the data."""
        self.min_val = data.min()
        self.max_val = data.max()
        self.is_fitted = True
        return self
    
    def transform(self, data, **kwargs):
        """Transform the data."""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted")
        
        return (data - self.min_val) / (self.max_val - self.min_val)
    
    def inverse_transform(self, data, **kwargs):
        """Inverse transform the data."""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted")
        
        return data * (self.max_val - self.min_val) + self.min_val


# Create a simple model
class SimpleAverageModel(BaseModel):
    """Simple model that predicts the average of the last n values."""
    
    def __init__(self, name="simple_average", window_size=5):
        """Initialize the model."""
        super().__init__(name=name, window_size=window_size)
        self.window_size = window_size
        self.target_column = None
    
    def fit(self, data, target_column, **kwargs):
        """Fit the model to the data."""
        self.target_column = target_column
        self.is_fitted = True
        return self
    
    def predict(self, data, horizon, **kwargs):
        """Generate forecasts."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Get the last window_size values
        last_values = data[self.target_column].iloc[-self.window_size:]
        
        # Calculate the average
        avg = last_values.mean()
        
        # Create forecast DataFrame
        forecast = pd.DataFrame({
            self.target_column: [avg] * horizon
        })
        
        return forecast


def main():
    """Run the example."""
    # Create sample data
    data = create_sample_data()
    print("Sample data:")
    print(data.head())
    
    # Create preprocessor
    normalizer = SimpleNormalizer()
    
    # Create pipeline
    pipeline = Pipeline([
        ("normalizer", normalizer)
    ])
    
    # Preprocess data
    data_processed = pipeline.fit_transform(data)
    print("\nProcessed data:")
    print(data_processed.head())
    
    # Split data into train and test
    train_size = int(len(data) * 0.8)
    train_data = data_processed.iloc[:train_size]
    test_data = data_processed.iloc[train_size:]
    
    # Create model
    model = SimpleAverageModel(window_size=10)
    
    # Create model registry
    registry = ModelRegistry()
    registry.register(model)
    
    # Fit model
    model.fit(train_data, "value")
    
    # Generate forecasts
    horizon = len(test_data)
    forecast = model.predict(train_data, horizon)
    
    # Evaluate forecasts
    metrics = Metrics.calculate_metrics(test_data["value"], forecast["value"])
    print("\nEvaluation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(data_processed.index[:train_size], train_data["value"], label="Train")
    plt.plot(data_processed.index[train_size:], test_data["value"], label="Test")
    plt.plot(data_processed.index[train_size:], forecast["value"], label="Forecast")
    plt.legend()
    plt.title("Simple Average Model Forecast")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
