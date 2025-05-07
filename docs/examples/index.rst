Examples
========

This section provides examples of how to use TimeFusion for various time series forecasting tasks.

Basic Forecasting
----------------

The basic forecasting example demonstrates how to:

1. Load and preprocess time series data
2. Train statistical and deep learning models
3. Create an ensemble model
4. Generate forecasts
5. Evaluate model performance

.. literalinclude:: ../../examples/basic_forecasting.py
   :language: python
   :linenos:

Statistical Models
----------------

The statistical models example demonstrates how to use the statistical models in TimeFusion,
including ARIMAModel and ExponentialSmoothingModel.

.. literalinclude:: ../../examples/statistical_models_example.py
   :language: python
   :linenos:

Deep Learning Models
------------------

The deep learning models example demonstrates how to use the deep learning models in TimeFusion,
including LSTMModel and SimpleRNNModel.

.. literalinclude:: ../../examples/deep_learning_models_example.py
   :language: python
   :linenos:

Hybrid Models
-----------

The hybrid models example demonstrates how to use the hybrid models in TimeFusion,
including EnsembleModel and ResidualModel.

.. literalinclude:: ../../examples/hybrid_models_example.py
   :language: python
   :linenos:

Evaluation
--------

The evaluation example demonstrates how to use the evaluation tools in TimeFusion,
including metrics, backtesting, and cross-validation.

.. literalinclude:: ../../examples/evaluation_example.py
   :language: python
   :linenos:

Preprocessing
-----------

The preprocessing example demonstrates how to use the preprocessing tools in TimeFusion,
including data cleaning, imputation, normalization, and feature engineering.

.. literalinclude:: ../../examples/preprocessing_example.py
   :language: python
   :linenos:

Utilities
-------

The utilities example demonstrates how to use the utility functions in TimeFusion,
including configuration, logging, visualization, and hyperparameter optimization.

.. literalinclude:: ../../examples/utilities_example.py
   :language: python
   :linenos:

Core Interfaces
------------

The core interfaces example demonstrates how to use the core interfaces in TimeFusion,
including BaseModel, BasePreprocessor, Pipeline, and ModelRegistry.

.. literalinclude:: ../../examples/core_interfaces_example.py
   :language: python
   :linenos:

Cryptocurrency Forecasting
-----------------------

The cryptocurrency forecasting example demonstrates how to use TimeFusion with Nixtla's statsforecast library
to forecast cryptocurrency prices. This example shows:

1. Loading and preprocessing cryptocurrency time series data
2. Filtering for more dynamic periods in the data
3. Training multiple statistical models (AutoARIMA, AutoETS, MSTL, Theta, TBATS, etc.)
4. Generating forecasts
5. Evaluating model performance with various metrics
6. Visualizing the results with detailed plots

.. literalinclude:: ../../examples/cryptocurrency_forecasting.py
   :language: python
   :linenos:
