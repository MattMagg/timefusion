"""
LSTM model for time series forecasting.

This module provides the LSTMModel class, which implements the LSTM
(Long Short-Term Memory) model for time series forecasting.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple

from .base import DeepLearningModel


class LSTMNet(nn.Module):
    """
    LSTM neural network for time series forecasting.
    
    This class implements a PyTorch LSTM model for time series forecasting.
    
    Attributes:
        hidden_size (int): Number of hidden units
        num_layers (int): Number of LSTM layers
        lstm (nn.LSTM): LSTM layer
        fc (nn.Linear): Fully connected output layer
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 1,
        dropout: float = 0.0
    ):
        """
        Initialize the LSTM network.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_size)
        
        # Get the output from the last time step
        out = out[:, -1, :]  # out: (batch_size, hidden_size)
        
        # Pass through the fully connected layer
        out = self.fc(out)  # out: (batch_size, output_size)
        
        return out


class LSTMModel(DeepLearningModel):
    """
    LSTM model for time series forecasting.
    
    This class provides an implementation of the LSTM (Long Short-Term Memory)
    model for time series forecasting.
    
    Attributes:
        name (str): Name of the model
        is_fitted (bool): Whether the model has been fitted
        params (Dict[str, Any]): Model parameters
        hidden_size (int): Number of hidden units
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        model (LSTMNet): The underlying LSTM model
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
        name: str = "lstm",
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize the LSTM model.
        
        Args:
            name: Name of the model
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            sequence_length: Length of input sequences
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            device: Device to use for computation ('cuda' or 'cpu')
            **kwargs: Additional parameters for the model
        """
        super().__init__(
            name=name,
            sequence_length=sequence_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=device,
            **kwargs
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
    
    def _create_model(self, input_size: int, output_size: int = 1) -> nn.Module:
        """
        Create the LSTM model.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            
        Returns:
            nn.Module: LSTM model
        """
        return LSTMNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=output_size,
            dropout=self.dropout
        ).to(self.device)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters
        """
        params = super().get_params()
        params.update({
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout
        })
        return params
