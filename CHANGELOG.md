# Changelog

All notable changes to the TimeFusion project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added comprehensive test suite for all components
- Added Testing section to README.md
- Added CHANGELOG.md file
- Added cryptocurrency forecasting example using Nixtla's statsforecast library
- Added integration with Nixtla libraries for enhanced forecasting capabilities
- Added examples section to README.md

### Fixed
- Fixed Imputer component to correctly handle window_size parameter
- Fixed Normalizer component to correctly handle robust scaling
- Fixed Feature Engineering component to correctly handle lag features with NaN values
- Fixed Config component to correctly handle loading from files and environment variables
- Fixed deep learning models to correctly handle input shapes and parameter retrieval

### Changed
- Updated pyproject.toml to include seaborn as a dependency
- Refactored examples directory structure for better organization

## [0.1.0] - 2023-01-01

### Added
- Initial release of TimeFusion
- Preprocessing module with Cleaner, Imputer, Normalizer, and Feature Engineering
- Models module with Statistical Models (ARIMA, Exponential Smoothing), Deep Learning Models (LSTM, SimpleRNN), and Hybrid Models (Ensemble, Residual)
- Evaluation module with Metrics, Backtesting, and Cross-validation
- Utilities module with Configuration, Logging, Visualization, and HPO
