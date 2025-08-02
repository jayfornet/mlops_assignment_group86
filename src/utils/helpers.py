"""
Utility functions for the California Housing MLOps Pipeline.

This module provides:
- Configuration management
- Logging utilities
- Common helper functions
- Model utilities
"""

import os
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class Config:
    """Configuration management for the MLOps pipeline."""
    
    # Model configuration
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # API configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_VERSION = "1.0.0"
    
    # MLflow configuration
    MLFLOW_TRACKING_URI = "file:./mlruns"
    EXPERIMENT_NAME = "california_housing_prediction"
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Data configuration
    DATA_DIR = "data"
    MODEL_DIR = "models"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"
    
    # Model performance thresholds
    MAX_ACCEPTABLE_RMSE = 1.0
    MIN_ACCEPTABLE_R2 = 0.5
    
    @classmethod
    def get_data_dir(cls) -> Path:
        """Get data directory path."""
        return Path(cls.DATA_DIR)
    
    @classmethod
    def get_model_dir(cls) -> Path:
        """Get model directory path."""
        return Path(cls.MODEL_DIR)
    
    @classmethod
    def get_log_dir(cls) -> Path:
        """Get log directory path."""
        return Path(cls.LOG_DIR)
    
    @classmethod
    def get_results_dir(cls) -> Path:
        """Get results directory path."""
        return Path(cls.RESULTS_DIR)


def setup_logging(
    level: str = Config.LOG_LEVEL,
    log_file: Optional[str] = None,
    logger_name: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        logger_name: Optional logger name
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(Config.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_directories():
    """Ensure all necessary directories exist."""
    directories = [
        Config.get_data_dir(),
        Config.get_model_dir(),
        Config.get_log_dir(),
        Config.get_results_dir(),
        Path("mlruns"),
        Path("mlflow-artifacts")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[Any, Any], filepath: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[Any, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.datetime.now().isoformat()


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, float]:
    """
    Format metrics to specified precision.
    
    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places
        
    Returns:
        Formatted metrics dictionary
    """
    return {key: round(value, precision) for key, value in metrics.items()}


def validate_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean input data for prediction.
    
    Args:
        data: Input data dictionary
        
    Returns:
        Validated data dictionary
        
    Raises:
        ValueError: If validation fails
    """
    required_features = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    
    # Check required features
    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Validate data types and ranges
    validated_data = {}
    
    for feature in required_features:
        value = data[feature]
        
        # Convert to float
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value for {feature}: {value}")
        
        # Basic range validation
        if feature == 'MedInc' and (value < 0 or value > 20):
            raise ValueError(f"MedInc must be between 0 and 20, got {value}")
        elif feature == 'HouseAge' and (value < 0 or value > 60):
            raise ValueError(f"HouseAge must be between 0 and 60, got {value}")
        elif feature == 'AveRooms' and (value < 1 or value > 20):
            raise ValueError(f"AveRooms must be between 1 and 20, got {value}")
        elif feature == 'AveBedrms' and (value < 0 or value > 5):
            raise ValueError(f"AveBedrms must be between 0 and 5, got {value}")
        elif feature == 'Population' and (value < 1 or value > 50000):
            raise ValueError(f"Population must be between 1 and 50000, got {value}")
        elif feature == 'AveOccup' and (value < 1 or value > 20):
            raise ValueError(f"AveOccup must be between 1 and 20, got {value}")
        elif feature == 'Latitude' and (value < 30 or value > 45):
            raise ValueError(f"Latitude must be between 30 and 45, got {value}")
        elif feature == 'Longitude' and (value < -130 or value > -110):
            raise ValueError(f"Longitude must be between -130 and -110, got {value}")
        
        validated_data[feature] = value
    
    # Additional validation: bedrooms shouldn't exceed rooms
    if validated_data['AveBedrms'] > validated_data['AveRooms']:
        raise ValueError("Average bedrooms cannot exceed average rooms")
    
    return validated_data


def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive model evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary of calculated metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,  # Mean Absolute Percentage Error
        'max_error': np.max(np.abs(y_true - y_pred)),
        'mean_residual': np.mean(y_true - y_pred)
    }
    
    return format_metrics(metrics)


def check_model_performance(metrics: Dict[str, float]) -> Dict[str, bool]:
    """
    Check if model performance meets acceptable thresholds.
    
    Args:
        metrics: Model performance metrics
        
    Returns:
        Dictionary of performance checks
    """
    checks = {
        'rmse_acceptable': metrics.get('rmse', float('inf')) <= Config.MAX_ACCEPTABLE_RMSE,
        'r2_acceptable': metrics.get('r2_score', 0) >= Config.MIN_ACCEPTABLE_R2,
        'no_nan_predictions': not any(np.isnan([v for v in metrics.values() if isinstance(v, (int, float))]))
    }
    
    checks['overall_acceptable'] = all(checks.values())
    
    return checks


def generate_model_report(
    model_name: str,
    metrics: Dict[str, float],
    training_time: float,
    feature_importance: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive model report.
    
    Args:
        model_name: Name of the model
        metrics: Performance metrics
        training_time: Training time in seconds
        feature_importance: Optional feature importance scores
        
    Returns:
        Complete model report
    """
    performance_checks = check_model_performance(metrics)
    
    report = {
        'model_name': model_name,
        'timestamp': get_timestamp(),
        'metrics': metrics,
        'training_time_seconds': training_time,
        'performance_checks': performance_checks,
        'feature_importance': feature_importance or {},
        'model_acceptable': performance_checks['overall_acceptable']
    }
    
    return report


def load_model_safe(model_path: str):
    """
    Safely load a model with error handling.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model or None if failed
    """
    try:
        import joblib
        return joblib.load(model_path)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None


def create_sample_request() -> Dict[str, float]:
    """
    Create a sample request for testing the API.
    
    Returns:
        Sample housing data dictionary
    """
    return {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984,
        "AveBedrms": 1.024,
        "Population": 322.0,
        "AveOccup": 2.555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }


def format_prediction_response(
    prediction: float,
    model_info: Dict[str, Any],
    input_data: Dict[str, Any],
    prediction_id: str
) -> Dict[str, Any]:
    """
    Format prediction response for API.
    
    Args:
        prediction: Model prediction
        model_info: Model information
        input_data: Input features
        prediction_id: Unique prediction identifier
        
    Returns:
        Formatted response dictionary
    """
    return {
        'prediction': round(prediction, 4),
        'prediction_id': prediction_id,
        'model_version': f"{model_info.get('model_name', 'unknown')}_v{model_info.get('version', '1.0')}",
        'timestamp': get_timestamp(),
        'input_features': input_data,
        'units': 'hundreds_of_thousands_usd',
        'confidence': 'high' if prediction > 0 else 'low'
    }


# Initialize logging for this module
logger = setup_logging(logger_name=__name__)

# Ensure directories exist when module is imported
ensure_directories()
