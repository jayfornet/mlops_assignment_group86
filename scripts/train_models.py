#!/usr/bin/env python3
"""
Model Training Script for MLOps Pipeline

This script trains multiple machine learning models on the California Housing dataset,
tracks experiments with MLflow, and saves the best performing models.
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os
import json
import sys
from datetime import datetime
from pathlib import Path


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }


def setup_mlflow(experiment_name='california_housing_prediction'):
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri('file:./mlruns')
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    return experiment_name


def load_preprocessed_data(data_path='data/processed/preprocessed_data.joblib'):
    """Load preprocessed data."""
    try:
        data = joblib.load(data_path)
        return (
            data['X_train'], data['X_val'], data['X_test'],
            data['y_train'], data['y_val'], data['y_test']
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Preprocessed data not found at {data_path}")


def get_models_config():
    """Get configuration for all models to train."""
    return {
        'random_forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42
        ),
        'linear_regression': LinearRegression()
    }


def train_model(model, model_name, X_train, y_train, X_val, y_val, logger):
    """Train a single model and log results to MLflow."""
    with mlflow.start_run(run_name=f'{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'):
        logger.info(f'Training {model_name}...')
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_val_pred = model.predict(X_val)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        
        # Log to MLflow
        mlflow.log_param('model_type', model_name)
        mlflow.log_metric('val_rmse', val_metrics['rmse'])
        mlflow.log_metric('val_mae', val_metrics['mae'])
        mlflow.log_metric('val_r2', val_metrics['r2_score'])
        mlflow.log_metric('val_mse', val_metrics['mse'])
        mlflow.sklearn.log_model(model, f'{model_name}_model')
        
        logger.info(f'{model_name} - Val RMSE: {val_metrics["rmse"]:.4f}')
        
        return model, val_metrics


def save_models(results, models_dir='models'):
    """Save all trained models."""
    os.makedirs(models_dir, exist_ok=True)
    
    for model_name, result in results.items():
        model_path = f'{models_dir}/{model_name}_best_model.joblib'
        joblib.dump(result['model'], model_path)
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'metrics': result['val_metrics'],
            'saved_at': datetime.now().isoformat(),
            'model_file': f'{model_name}_best_model.joblib'
        }
        
        metadata_path = f'{models_dir}/{model_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def evaluate_best_model(best_model, best_model_name, X_test, y_test, logger):
    """Evaluate the best model on test set."""
    logger.info(f'Testing best model: {best_model_name}')
    y_test_pred = best_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    logger.info(
        f'Training completed! Best: {best_model_name}, '
        f'Test RMSE: {test_metrics["rmse"]:.4f}'
    )
    
    return test_metrics


def save_comparison_results(results, best_model_name, results_dir='results'):
    """Save model comparison results."""
    os.makedirs(results_dir, exist_ok=True)
    
    comparison_data = [
        {
            'Model': name,
            'Val_RMSE': result['val_metrics']['rmse'],
            'Val_MAE': result['val_metrics']['mae'],
            'Val_R2': result['val_metrics']['r2_score'],
            'Is_Best': name == best_model_name
        }
        for name, result in results.items()
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f'{results_dir}/model_comparison.csv', index=False)
    
    return comparison_df


def main():
    """Main training function."""
    logger = setup_logging()
    logger.info("Starting model training pipeline...")
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
        logger.info("Preprocessed data loaded successfully")
        
        # Get model configurations
        models_config = get_models_config()
        
        # Train all models
        results = {}
        best_model, best_model_name, best_score = None, None, float('inf')
        
        for model_name, model in models_config.items():
            trained_model, val_metrics = train_model(
                model, model_name, X_train, y_train, X_val, y_val, logger
            )
            
            # Track best model
            if val_metrics['rmse'] < best_score:
                best_score = val_metrics['rmse']
                best_model = trained_model
                best_model_name = model_name
            
            results[model_name] = {
                'model': trained_model,
                'val_metrics': val_metrics
            }
        
        # Save all models
        save_models(results)
        logger.info("All models saved successfully")
        
        # Evaluate best model on test set
        test_metrics = evaluate_best_model(
            best_model, best_model_name, X_test, y_test, logger
        )
        
        # Save comparison results
        comparison_df = save_comparison_results(results, best_model_name)
        logger.info("Model comparison results saved")
        
        # Print summary
        logger.info(f"Training Summary:")
        logger.info(f"  Best Model: {best_model_name}")
        logger.info(f"  Validation RMSE: {best_score:.4f}")
        logger.info(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
