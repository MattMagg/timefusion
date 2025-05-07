"""
Base class for deep learning forecasting models.

This module provides the DeepLearningModel base class, which extends the BaseModel
class and provides common functionality for deep learning forecasting models.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
from abc import abstractmethod

from ...models.base import BaseModel
from ...utils.time_series import create_forecast_index, validate_fitted, prepare_target_and_features


class DeepLearningModel(BaseModel):
    """
    Base class for all deep learning forecasting models.
    
    This class extends the BaseModel class and provides common functionality
    for deep learning forecasting models, including support for sequence preparation,
    batching, and PyTorch integration.
    
    Attributes:
        name (str): Name of the model
        is_fitted (bool): Whether the model has been fitted
        params (Dict[str, Any]): Model parameters
        model: The underlying PyTorch model
        device (str): Device to use for computation ('cuda' or 'cpu')
        sequence_length (int): Length of input sequences
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimization
        num_epochs (int): Number of training epochs
        target_column (str): Name of the target column
        feature_columns (List[str]): List of feature column names
    """
    
    def __init__(
        self,
        name: str,
        sequence_length: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize the deep learning model.
        
        Args:
            name: Name of the model
            sequence_length: Length of input sequences
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            device: Device to use for computation ('cuda' or 'cpu')
            **kwargs: Additional parameters for the model
        """
        super().__init__(name, **kwargs)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        self.model = None
        self.target_column = None
        self.feature_columns = None
    
    @abstractmethod
    def _create_model(self, input_size: int, output_size: int = 1) -> nn.Module:
        """
        Create the PyTorch model.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            
        Returns:
            nn.Module: PyTorch model
        """
        pass
    
    def fit(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> 'DeepLearningModel':
        """
        Fit the deep learning model to the data.
        
        Args:
            data: Input data
            target_column: Name of the target column
            feature_columns: List of feature column names (if None, use all columns except target)
            **kwargs: Additional parameters for fitting
            
        Returns:
            self: The fitted model
        """
        self.target_column = target_column
        
        # Determine feature columns and extract data
        self.feature_columns, X_data, y_data = prepare_target_and_features(
            data, target_column, feature_columns
        )
        
        # Prepare data
        X, y = self._prepare_data(data)
        
        # Create model
        input_size = len(self.feature_columns)
        self.model = self._create_model(input_size)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Train the model
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame, horizon: int, **kwargs) -> pd.DataFrame:
        """
        Generate forecasts using the deep learning model.
        
        Args:
            data: Input data
            horizon: Forecast horizon
            **kwargs: Additional parameters for prediction
            
        Returns:
            pd.DataFrame: Forecasts
        """
        validate_fitted(self.is_fitted, "predict")
        
        # Prepare input data
        X = self._prepare_prediction_data(data)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate forecasts
        with torch.no_grad():
            forecasts = []
            current_X = X
            
            for _ in range(horizon):
                # Generate forecast for the next step
                output = self.model(current_X)
                forecasts.append(output.cpu().numpy()[0, 0])
                
                # Update input for the next step
                # This is a simplified approach; in practice, you would need to update
                # all features, not just the target
                new_X = current_X.clone()
                new_X[0, -1, 0] = output[0, 0]  # Update the last time step with the prediction
                current_X = new_X
        
        # Create forecast DataFrame
        forecast_index = create_forecast_index(data, horizon)
        forecast_df = pd.DataFrame({self.target_column: forecasts}, index=forecast_index)
        
        return forecast_df
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for training.
        
        Args:
            data: Input data
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input sequences and target values
        """
        # Extract features and target
        X_data = data[self.feature_columns].values
        y_data = data[self.target_column].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(data) - self.sequence_length):
            X_sequences.append(X_data[i:i+self.sequence_length])
            y_sequences.append(y_data[i+self.sequence_length])
        
        # Convert to tensors
        X = torch.tensor(np.array(X_sequences), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(y_sequences), dtype=torch.float32).view(-1, 1).to(self.device)
        
        return X, y
    
    def _prepare_prediction_data(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Prepare data for prediction.
        
        Args:
            data: Input data
            
        Returns:
            torch.Tensor: Input sequence
        """
        # Extract features
        X_data = data[self.feature_columns].values
        
        # Use the last sequence_length time steps
        X_sequence = X_data[-self.sequence_length:]
        
        # Convert to tensor
        X = torch.tensor(np.array([X_sequence]), dtype=torch.float32).to(self.device)
        
        return X
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters
        """
        validate_fitted(self.is_fitted, "get_params")
        
        return {
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "device": self.device
        }
