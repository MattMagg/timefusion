# TimeFusion: Advanced Hybrid Time Series Forecasting System

A Python library that combines statistical methods with deep learning approaches for time series forecasting. This project provides a unified framework that leverages both traditional statistical methods and modern deep learning approaches to create powerful hybrid forecasting models.

## Key Features

- **Hybrid Approach**: Combines statistical and deep learning methods in a unified framework
- **Ready-to-Use Models**: Implementations of ARIMA, Exponential Smoothing, LSTM, and ensemble models
- **Preprocessing Pipeline**: Tools for data cleaning, imputation, normalization, and feature engineering
- **Evaluation Framework**: Comprehensive metrics and backtesting utilities
- **Extensibility**: Clear extension points for custom implementations
- **Minimal Dependencies**: Core functionality requires only widely-used libraries

## Installation

```bash
pip install timefusion
```

For optional deep learning components:

```bash
pip install timefusion[deep]
```

## Quick Start

```python
import pandas as pd
from timefusion.preprocessing import Cleaner, Imputer, Normalizer, Pipeline
from timefusion.models import ARIMAModel, LSTMModel, EnsembleModel
from timefusion.evaluation import Metrics, Backtesting

# Load data
data = pd.read_csv('your_data.csv', parse_dates=['date_column'])
data.set_index('date_column', inplace=True)

# Create preprocessing pipeline
pipeline = Pipeline([
    ('cleaner', Cleaner()),
    ('imputer', Imputer()),
    ('normalizer', Normalizer())
])

# Preprocess data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Create and train models
arima = ARIMAModel(order=(1,1,1))
lstm = LSTMModel(hidden_size=64, num_layers=2)
ensemble = EnsembleModel([arima, lstm], weights=[0.4, 0.6])

arima.fit(X_train)
lstm.fit(X_train_processed)
ensemble.fit(X_train)

# Generate forecasts
forecast = ensemble.predict(horizon=30)

# Evaluate performance
metrics = Metrics.calculate(y_test, forecast)
```

## Documentation

For full documentation, visit [timefusion.readthedocs.io](https://timefusion.readthedocs.io/en/latest/index.html#).

## Features

### Preprocessing

- **Data Cleaning**: Outlier detection and handling, duplicate removal
- **Imputation**: Multiple strategies for handling missing values
- **Normalization**: Various scaling methods (min-max, z-score, robust)
- **Feature Engineering**: Lag features, window features, date-based features

### Models

- **Statistical Models**: ARIMA, Exponential Smoothing, Naive methods
- **Deep Learning Models**: LSTM, SimpleRNN
- **Hybrid Models**: Ensemble, Residual

### Evaluation

- **Metrics**: MSE, RMSE, MAE, MAPE, SMAPE, R²
- **Backtesting**: Walk-forward validation, expanding window, sliding window
- **Cross-Validation**: Time series cross-validation strategies

### Utilities

- **Configuration**: Loading and saving configuration from/to files
- **Logging**: Configurable logging levels and formats
- **Visualization**: Plotting functions for time series data and forecasts
- **HPO**: Hyperparameter optimization utilities

## Testing

The OSHFS system has been thoroughly tested with a comprehensive test suite covering all components:

- **Preprocessing Module**: Tests for Cleaner, Imputer, Normalizer, and Feature Engineering
- **Models Module**: Tests for Statistical Models (ARIMA, Exponential Smoothing), Deep Learning Models (LSTM, SimpleRNN), and Hybrid Models (Ensemble, Residual)
- **Evaluation Module**: Tests for Metrics, Backtesting, and Cross-validation
- **Utilities Module**: Tests for Configuration, Logging, Visualization, and HPO

To run the tests:

```bash
# Run all tests
python -m pytest

# Run tests for a specific module
python -m pytest tests/test_preprocessing/
python -m pytest tests/test_models/
python -m pytest tests/test_evaluation/
python -m pytest tests/test_utils/
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
