"""
Logging utilities for TimeFusion.

This module provides utilities for setting up and configuring
logging for the TimeFusion system.
"""

import logging
import sys
import os
import time
from typing import Optional, Dict, Any, Union, List, Tuple, Callable
from contextlib import contextmanager


class Logger:
    """
    Logger class for TimeFusion.

    This class provides methods for logging messages, errors, and performance
    metrics for the TimeFusion system.

    Attributes:
        logger (logging.Logger): Underlying logger
        name (str): Logger name
    """

    def __init__(
        self,
        name: Optional[str] = None,
        level: Union[int, str] = logging.INFO,
        log_file: Optional[str] = None,
        console: bool = True,
        format_str: Optional[str] = None
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name (default: root logger)
            level: Logging level (default: INFO)
            log_file: Path to log file (if None, no file logging)
            console: Whether to log to console
            format_str: Log message format
        """
        self.name = name
        self.logger = logging.getLogger(name)

        # Convert string level to int if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        self.logger.setLevel(level)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create formatters
        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_str)

        # Add console handler if requested
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Add file handler if requested
        if log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str, *args, **kwargs) -> None:
        """
        Log a debug message.

        Args:
            message: Message to log
            *args: Additional arguments for message formatting
            **kwargs: Additional keyword arguments for logging
        """
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """
        Log an info message.

        Args:
            message: Message to log
            *args: Additional arguments for message formatting
            **kwargs: Additional keyword arguments for logging
        """
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """
        Log a warning message.

        Args:
            message: Message to log
            *args: Additional arguments for message formatting
            **kwargs: Additional keyword arguments for logging
        """
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """
        Log an error message.

        Args:
            message: Message to log
            *args: Additional arguments for message formatting
            **kwargs: Additional keyword arguments for logging
        """
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """
        Log a critical message.

        Args:
            message: Message to log
            *args: Additional arguments for message formatting
            **kwargs: Additional keyword arguments for logging
        """
        self.logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args, exc_info=True, **kwargs) -> None:
        """
        Log an exception message.

        Args:
            message: Message to log
            *args: Additional arguments for message formatting
            exc_info: Whether to include exception info
            **kwargs: Additional keyword arguments for logging
        """
        self.logger.exception(message, *args, exc_info=exc_info, **kwargs)

    @contextmanager
    def timer(self, operation: str) -> None:
        """
        Context manager for timing operations.

        Args:
            operation: Name of the operation

        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.info(f"{operation} completed in {elapsed_time:.4f} seconds")

    def log_model_metrics(self, model_name: str, metrics: Dict[str, float], additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Log model metrics.

        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            additional_info: Additional information to log
        """
        message = f"Model: {model_name}, Metrics: {metrics}"
        if additional_info:
            message += f", Additional Info: {additional_info}"
        self.info(message)

    def log_prediction(self, model_name: str, prediction: Any, timestamp: Optional[str] = None, prediction_id: Optional[str] = None) -> None:
        """
        Log a prediction.

        Args:
            model_name: Name of the model
            prediction: Prediction value
            timestamp: Timestamp of the prediction
            prediction_id: Prediction ID
        """
        message = f"Model: {model_name}, Prediction: {prediction}"
        if timestamp:
            message += f", Timestamp: {timestamp}"
        if prediction_id:
            message += f", ID: {prediction_id}"
        self.info(message)

    def log_error(self, error_message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error.

        Args:
            error_message: Error message
            exception: Exception object
            context: Error context
        """
        message = error_message
        if context:
            message += f", Context: {context}"
        if exception:
            self.exception(message)
        else:
            self.error(message)

    def log_training_progress(
        self,
        model_name: str,
        epoch: int,
        total_epochs: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        validation_loss: Optional[float] = None,
        validation_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log training progress.

        Args:
            model_name: Name of the model
            epoch: Current epoch
            total_epochs: Total number of epochs
            loss: Training loss
            metrics: Training metrics
            validation_loss: Validation loss
            validation_metrics: Validation metrics
        """
        message = f"Model: {model_name}, Epoch: {epoch}/{total_epochs}, Loss: {loss:.4f}"
        if metrics:
            message += f", Metrics: {metrics}"
        if validation_loss is not None:
            message += f", Validation Loss: {validation_loss:.4f}"
        if validation_metrics:
            message += f", Validation Metrics: {validation_metrics}"
        self.info(message)


def setup_logger(
    name: Optional[str] = None,
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    format_str: Optional[str] = None
) -> Logger:
    """
    Set up a logger with the specified configuration.

    Args:
        name: Logger name (default: root logger)
        level: Logging level (default: INFO)
        log_file: Path to log file (if None, no file logging)
        console: Whether to log to console
        format_str: Log message format

    Returns:
        Logger: Configured logger
    """
    return Logger(name, level, log_file, console, format_str)


@contextmanager
def log_time(logger: Logger, operation: str) -> None:
    """
    Context manager for timing operations.

    Args:
        logger: Logger to use
        operation: Name of the operation

    Yields:
        None
    """
    with logger.timer(operation):
        yield
