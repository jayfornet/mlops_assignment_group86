#!/usr/bin/env python3
"""
MLflow Setup and Experiment Tracking Script

This script sets up MLflow tracking, manages experiments, and generates experiment summaries.
"""

import mlflow
import os
import json
import sys
import logging
from datetime import datetime
from pathlib import Path

# Constants for MLflow parameter and metric names
PARAM_MODEL_TYPE = 'params.model_type'
METRIC_VAL_RMSE = 'metrics.val_rmse'
METRIC_VAL_R2 = 'metrics.val_r2'


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def setup_mlflow_tracking(tracking_dir='./mlruns', artifacts_dir='./mlflow-artifacts'):
    """Setup MLflow tracking configuration."""
    # Ensure directories exist
    os.makedirs(tracking_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Set tracking URI - use absolute path directly on Windows
    tracking_path = os.path.abspath(tracking_dir)
    mlflow.set_tracking_uri(tracking_path)
    
    return tracking_path, os.path.abspath(artifacts_dir)


def ensure_experiment_exists(experiment_name, artifact_location):
    """Ensure MLflow experiment exists, create if not."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
            return experiment_id, True  # True indicates new experiment created
        else:
            return experiment.experiment_id, False  # False indicates existing experiment
    except Exception as e:
        raise RuntimeError(f"Failed to setup experiment '{experiment_name}': {e}")


def generate_experiment_summary(experiment_name):
    """Generate a summary of the experiment runs."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return {"error": f"Experiment '{experiment_name}' not found"}
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        summary = {
            "experiment_id": experiment.experiment_id,
            "experiment_name": experiment_name,
            "total_runs": len(runs),
            "created_at": datetime.now().isoformat(),
            "runs": []
        }
        
        if len(runs) > 0:
            # Add recent runs summary
            for _, run in runs.head(10).iterrows():  # Last 10 runs
                run_info = {
                    "run_id": run['run_id'],
                    "model_type": run.get(PARAM_MODEL_TYPE, 'unknown'),
                    "val_rmse": run.get(METRIC_VAL_RMSE, None),
                    "val_r2": run.get(METRIC_VAL_R2, None),
                    "status": run.get('status', 'unknown'),
                    "start_time": run.get('start_time', None)
                }
                summary["runs"].append(run_info)
            
            # Find best model
            if METRIC_VAL_RMSE in runs.columns:
                best_run = runs.loc[runs[METRIC_VAL_RMSE].idxmin()]
                summary["best_model"] = {
                    "run_id": best_run['run_id'],
                    "model_type": best_run.get(PARAM_MODEL_TYPE, 'unknown'),
                    "rmse": float(best_run.get(METRIC_VAL_RMSE, 0)),
                    "r2_score": float(best_run.get(METRIC_VAL_R2, 0)) if best_run.get(METRIC_VAL_R2) else None
                }
        
        return summary
        
    except Exception as e:
        return {"error": f"Failed to generate experiment summary: {e}"}


def save_best_model_info(experiment_name, pipeline_run_number=None):
    """Save best model information for deployment."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if len(runs) > 0 and METRIC_VAL_RMSE in runs.columns:
            best_run = runs.loc[runs[METRIC_VAL_RMSE].idxmin()]
            best_model_info = {
                'run_id': best_run['run_id'],
                'model_type': best_run.get(PARAM_MODEL_TYPE, 'unknown'),
                'rmse': float(best_run.get(METRIC_VAL_RMSE, 0)),
                'r2_score': float(best_run.get(METRIC_VAL_R2, 0)) if best_run.get(METRIC_VAL_R2) else None,
                'experiment_id': experiment.experiment_id,
                'last_updated': datetime.now().isoformat(),
                'pipeline_run': pipeline_run_number
            }
            
            os.makedirs('deployment/models', exist_ok=True)
            with open('deployment/models/best_model_info.json', 'w') as f:
                json.dump(best_model_info, f, indent=2)
            
            return best_model_info
        else:
            return None
            
    except Exception as e:
        logging.error(f"Failed to save best model info: {e}")
        return None


def print_experiment_summary(summary):
    """Print formatted experiment summary."""
    if "error" in summary:
        print(f"❌ Error: {summary['error']}")
        return
    
    print(f"\n📊 Experiment Summary: {summary['total_runs']} total runs")
    
    if summary['total_runs'] > 0:
        print("Recent runs:")
        for run in summary['runs'][:5]:  # Show first 5
            model_type = run['model_type']
            rmse = run['val_rmse']
            status = run['status']
            rmse_str = f"RMSE={rmse:.4f}" if rmse else "RMSE=N/A"
            print(f"  - {model_type}: {rmse_str} ({status})")
        
        if "best_model" in summary:
            best = summary["best_model"]
            print(f"\n🏆 Best Model: {best['model_type']} (RMSE: {best['rmse']:.4f})")


def main():
    """Main function for MLflow setup and tracking."""
    logger = setup_logging()
    
    # Get parameters from environment or use defaults
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'california_housing_prediction')
    pipeline_run = os.getenv('GITHUB_RUN_NUMBER', None)
    
    try:
        logger.info("Setting up MLflow tracking...")
        
        # Setup MLflow tracking
        tracking_uri, artifact_location = setup_mlflow_tracking()
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"Artifact location: {artifact_location}")
        
        # Ensure experiment exists
        experiment_id, is_new = ensure_experiment_exists(experiment_name, artifact_location)
        if is_new:
            logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
        # Generate experiment summary
        summary = generate_experiment_summary(experiment_name)
        print_experiment_summary(summary)
        
        # Save best model info if runs exist
        best_model_info = save_best_model_info(experiment_name, pipeline_run)
        if best_model_info:
            logger.info("Best model information saved")
        
        # Save experiment summary to file
        summary_file = 'mlflow_experiment_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Experiment summary saved to {summary_file}")
        
        logger.info("MLflow setup and tracking completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"MLflow setup failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
