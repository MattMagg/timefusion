"""
Base class for hybrid forecasting models.

This module provides the HybridModel base class, which extends the BaseModel
class and provides common functionality for hybrid forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from abc import abstractmethod

from ..base import BaseModel


class HybridModel(BaseModel):
    """
    Base class for all hybrid forecasting models.

    This class extends the BaseModel class and provides common functionality
    for hybrid forecasting models, including support for model combination
    and contribution analysis.

    Attributes:
        name (str): Name of the model
        is_fitted (bool): Whether the model has been fitted
        params (Dict[str, Any]): Model parameters
        models (List[BaseModel]): List of component models
        target_column (str): Name of the target column
    """

    def __init__(self, name: str, models: List[BaseModel] = None, **kwargs):
        """
        Initialize the hybrid model.

        Args:
            name: Name of the model
            models: List of component models
            **kwargs: Additional parameters for the model
        """
        super().__init__(name, **kwargs)
        self.models = models or []
        self.target_column = None

    def add_model(self, model: BaseModel) -> None:
        """
        Add a model to the hybrid model.

        Args:
            model: Model to add
        """
        self.models.append(model)

    @abstractmethod
    def fit(self, data: pd.DataFrame, target_column: str, **kwargs) -> 'HybridModel':
        """
        Fit the hybrid model to the data.

        Args:
            data: Input data
            target_column: Name of the target column
            **kwargs: Additional parameters for fitting

        Returns:
            self: The fitted model
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, horizon: int, **kwargs) -> pd.DataFrame:
        """
        Generate forecasts using the hybrid model.

        Args:
            data: Input data
            horizon: Forecast horizon
            **kwargs: Additional parameters for prediction

        Returns:
            pd.DataFrame: Forecasts
        """
        pass

    def get_model_contributions(self, data: pd.DataFrame, horizon: int) -> Dict[str, pd.DataFrame]:
        """
        Get the contribution of each component model to the final forecast.

        Args:
            data: Input data
            horizon: Forecast horizon

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping model names to their forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        contributions = {}
        for model in self.models:
            if not model.is_fitted:
                continue
            forecast = model.predict(data, horizon)
            contributions[model.name] = forecast

        return contributions

    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.

        Returns:
            Dict[str, Any]: Model parameters
        """
        params = {
            "name": self.name,
            "is_fitted": self.is_fitted,
            "models": [model.name for model in self.models]
        }
        params.update(self.params)
        return params
