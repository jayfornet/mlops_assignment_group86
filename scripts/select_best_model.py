#!/usr/bin/env python3
"""
Model Selection and Deployment Preparation Script

This script identifies the best model from MLflow experiments and prepares it for deployment.
"""

import os
import json
import joblib
import mlflow
import pandas as pd
import sys
import logging
from pathlib import Path
from datetime import datetime


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def setup_mlflow():
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri('file:./mlruns')
    return mlflow


def get_best_model_from_mlflow(experiment_name='california_housing_prediction'):
    """Get the best model from MLflow experiments."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['metrics.val_rmse ASC']
    )
    
    if runs.empty:
        raise ValueError("No runs found in the experiment")
    
    best_run = runs.iloc[0]
    return {
        'model_name': best_run['params.model_type'],
        'rmse': best_run['metrics.val_rmse'],
        'run_id': best_run['run_id'],
        'mae': best_run.get('metrics.val_mae', None),
        'r2_score': best_run.get('metrics.val_r2', None)
    }


def copy_best_model(best_model_info, source_dir='models', target_dir='deployment/models'):
    """Copy the best model to deployment directory."""
    os.makedirs(target_dir, exist_ok=True)
    
    source_model = f'{source_dir}/{best_model_info["model_name"]}_best_model.joblib'
    target_model = f'{target_dir}/best_model.joblib'
    
    if not os.path.exists(source_model):
        raise FileNotFoundError(f"Model file {source_model} not found")
    
    import shutil
    shutil.copy2(source_model, target_model)
    
    return target_model


def create_deployment_metadata(best_model_info, target_dir='deployment/models'):
    """Create deployment metadata file."""
    metadata = {
        'model_name': best_model_info['model_name'],
        'rmse': float(best_model_info['rmse']),
        'run_id': best_model_info['run_id'],
        'deployment_timestamp': datetime.now().isoformat(),
        'model_file': 'best_model.joblib'
    }
    
    # Add optional metrics if available
    if best_model_info.get('mae'):
        metadata['mae'] = float(best_model_info['mae'])
    if best_model_info.get('r2_score'):
        metadata['r2_score'] = float(best_model_info['r2_score'])
    
    metadata_path = f'{target_dir}/model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


def validate_model_file(model_path):
    """Validate that the model file can be loaded."""
    try:
        joblib.load(model_path)  # Just verify it loads successfully
        return True
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        return False


def main():
    """Main function for model selection and deployment preparation."""
    logger = setup_logging()
    logger.info("Starting model selection and deployment preparation...")
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Get best model from MLflow
        best_model_info = get_best_model_from_mlflow()
        logger.info(
            f"Best model identified: {best_model_info['model_name']} "
            f"with RMSE: {best_model_info['rmse']:.4f}"
        )
        
        # Copy best model to deployment directory
        target_model = copy_best_model(best_model_info)
        logger.info(f"Model copied to {target_model}")
        
        # Validate the copied model
        if not validate_model_file(target_model):
            raise ValueError("Failed to validate the copied model file")
        
        # Create deployment metadata
        metadata_path = create_deployment_metadata(best_model_info)
        logger.info(f"Deployment metadata created at {metadata_path}")
        
        # Create additional deployment artifacts
        deployment_info = {
            'status': 'ready',
            'model_name': best_model_info['model_name'],
            'rmse': best_model_info['rmse'],
            'prepared_at': datetime.now().isoformat(),
            'files': {
                'model': 'best_model.joblib',
                'metadata': 'model_metadata.json'
            }
        }
        
        with open('deployment/models/deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info("Model preparation for deployment completed successfully")
        
        # Print summary
        print("Best model prepared for deployment:")
        print(f"   Model: {best_model_info['model_name']}")
        print(f"   RMSE: {best_model_info['rmse']:.4f}")
        print(f"   Run ID: {best_model_info['run_id']}")
        print(f"   Location: {target_model}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Model selection failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
