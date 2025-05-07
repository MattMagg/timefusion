"""
Tests for the logging utilities.
"""

import pytest
import os
import logging
import tempfile
import time
from contextlib import contextmanager
from timefusion.utils.logging import Logger, setup_logger, log_time


@contextmanager
def captured_logs(logger):
    """Context manager to capture logs."""
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.logger.addHandler(handler)
    
    class LogCapture:
        def __init__(self):
            self.records = []
        
        def handle(self, record):
            self.records.append(record)
    
    capture = LogCapture()
    handler.emit = capture.handle
    
    try:
        yield capture
    finally:
        logger.logger.removeHandler(handler)


def test_logger_init():
    """Test Logger initialization."""
    # Test with default parameters
    logger = Logger()
    assert logger.name is None
    assert logger.logger.level == logging.INFO
    
    # Test with custom parameters
    logger = Logger(name="test", level=logging.DEBUG)
    assert logger.name == "test"
    assert logger.logger.level == logging.DEBUG


def test_logger_log_levels():
    """Test Logger log levels."""
    logger = Logger(name="test")
    
    with captured_logs(logger) as capture:
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")
    
    # Debug message should not be logged at INFO level
    assert len([r for r in capture.records if "debug message" in r.getMessage()]) == 0
    
    # Other messages should be logged
    assert len([r for r in capture.records if "info message" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "warning message" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "error message" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "critical message" in r.getMessage()]) == 1


def test_logger_file_output():
    """Test Logger file output."""
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        log_path = f.name
    
    try:
        # Create logger with file output
        logger = Logger(name="test", log_file=log_path)
        
        # Log a message
        logger.info("test message")
        
        # Check that the message was written to the file
        with open(log_path, "r") as f:
            content = f.read()
            assert "test message" in content
    finally:
        # Clean up
        if os.path.exists(log_path):
            os.remove(log_path)


def test_logger_timer():
    """Test Logger timer context manager."""
    logger = Logger(name="test")
    
    with captured_logs(logger) as capture:
        with logger.timer("test operation"):
            # Simulate some work
            time.sleep(0.1)
    
    # Check that the timer message was logged
    assert len([r for r in capture.records if "test operation completed in" in r.getMessage()]) == 1


def test_logger_log_model_metrics():
    """Test Logger log_model_metrics method."""
    logger = Logger(name="test")
    
    with captured_logs(logger) as capture:
        logger.log_model_metrics(
            model_name="test_model",
            metrics={"rmse": 0.1, "mae": 0.05},
            additional_info={"epochs": 10}
        )
    
    # Check that the metrics were logged
    assert len([r for r in capture.records if "test_model" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "rmse" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "mae" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "epochs" in r.getMessage()]) == 1


def test_logger_log_prediction():
    """Test Logger log_prediction method."""
    logger = Logger(name="test")
    
    with captured_logs(logger) as capture:
        logger.log_prediction(
            model_name="test_model",
            prediction=0.5,
            timestamp="2023-01-01",
            prediction_id="123"
        )
    
    # Check that the prediction was logged
    assert len([r for r in capture.records if "test_model" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "0.5" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "2023-01-01" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "123" in r.getMessage()]) == 1


def test_logger_log_error():
    """Test Logger log_error method."""
    logger = Logger(name="test")
    
    with captured_logs(logger) as capture:
        # Test with message only
        logger.log_error("test error")
        
        # Test with exception
        try:
            raise ValueError("test exception")
        except ValueError as e:
            logger.log_error("test error with exception", exception=e)
        
        # Test with context
        logger.log_error("test error with context", context={"key": "value"})
    
    # Check that the errors were logged
    assert len([r for r in capture.records if "test error" in r.getMessage()]) == 3
    assert len([r for r in capture.records if "test exception" in r.getMessage()]) == 0  # Exception details are not in the message
    assert len([r for r in capture.records if "key" in r.getMessage()]) == 1


def test_logger_log_training_progress():
    """Test Logger log_training_progress method."""
    logger = Logger(name="test")
    
    with captured_logs(logger) as capture:
        logger.log_training_progress(
            model_name="test_model",
            epoch=1,
            total_epochs=10,
            loss=0.5,
            metrics={"accuracy": 0.8},
            validation_loss=0.6,
            validation_metrics={"accuracy": 0.7}
        )
    
    # Check that the training progress was logged
    assert len([r for r in capture.records if "test_model" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "1/10" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "0.5" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "accuracy" in r.getMessage()]) == 1
    assert len([r for r in capture.records if "0.6" in r.getMessage()]) == 1


def test_setup_logger():
    """Test setup_logger function."""
    # Test with default parameters
    logger = setup_logger()
    assert isinstance(logger, Logger)
    assert logger.name is None
    assert logger.logger.level == logging.INFO
    
    # Test with custom parameters
    logger = setup_logger(name="test", level=logging.DEBUG, log_file=None, console=True)
    assert isinstance(logger, Logger)
    assert logger.name == "test"
    assert logger.logger.level == logging.DEBUG


def test_log_time():
    """Test log_time context manager."""
    logger = Logger(name="test")
    
    with captured_logs(logger) as capture:
        with log_time(logger, "test operation"):
            # Simulate some work
            time.sleep(0.1)
    
    # Check that the timer message was logged
    assert len([r for r in capture.records if "test operation completed in" in r.getMessage()]) == 1
