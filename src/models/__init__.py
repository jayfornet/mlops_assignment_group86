"""
Model training and evaluation module.

This module provides functionality for:
- Training multiple ML models
- MLflow experiment tracking
- Model comparison and selection
- Model evaluation and metrics
"""

from .train_model import ModelTrainer

__all__ = ['ModelTrainer']
