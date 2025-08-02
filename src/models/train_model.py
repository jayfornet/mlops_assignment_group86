"""
Model training module with MLflow experiment tracking.

This module handles:
- Training multiple ML models (Linear Regression, Random Forest, Gradient Boosting)
- MLflow experiment tracking for parameters, metrics, and models
- Model evaluation and comparison
- Best model selection and registration
- Model artifact saving
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import logging
import os
import sys
import joblib
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_processor import DataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model trainer class for California Housing price prediction.
    
    Handles training multiple models, tracking experiments, and model selection.
    """
    
    def __init__(self, experiment_name="california_housing_prediction"):
        """
        Initialize ModelTrainer with MLflow experiment setup.
        
        Args:
            experiment_name (str): Name of MLflow experiment
        """
        self.experiment_name = experiment_name
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('inf')
        
        # Setup MLflow
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow experiment and tracking."""
        try:
            # Set MLflow tracking URI (local file store)
            mlflow.set_tracking_uri("file:./mlruns")
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new MLflow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            raise
    
    def define_models(self):
        """
        Define the models to train and their hyperparameters.
        
        Returns:
            dict: Dictionary of model configurations
        """
        models_config = {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            }
        }
        
        # Update model parameters
        for model_name, config in models_config.items():
            if config['params']:
                config['model'].set_params(**config['params'])
        
        return models_config
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate regression metrics.
        
        Args:
            y_true (array): True target values
            y_pred (array): Predicted values
            
        Returns:
            dict: Dictionary of calculated metrics
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
        
        return metrics
    
    def train_model(self, model_name, model_config, X_train, y_train, X_val, y_val):
        """
        Train a single model and track the experiment with MLflow.
        
        Args:
            model_name (str): Name of the model
            model_config (dict): Model configuration
            X_train (DataFrame): Training features
            y_train (Series): Training target
            X_val (DataFrame): Validation features
            y_val (Series): Validation target
            
        Returns:
            dict: Model results and metrics
        """
        logger.info(f"Training {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            try:
                # Get model and parameters
                model = model_config['model']
                params = model_config['params']
                
                # Log parameters
                mlflow.log_params(params)
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("training_samples", len(X_train))
                mlflow.log_param("validation_samples", len(X_val))
                
                # Train the model
                start_time = datetime.now()
                model.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                # Calculate metrics
                train_metrics = self.calculate_metrics(y_train, y_train_pred)
                val_metrics = self.calculate_metrics(y_val, y_val_pred)
                
                # Log metrics
                for metric_name, value in train_metrics.items():
                    mlflow.log_metric(f"train_{metric_name}", value)
                
                for metric_name, value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", value)
                
                mlflow.log_metric("training_time_seconds", training_time)
                
                # Log model
                signature = infer_signature(X_train, y_train_pred)
                mlflow.sklearn.log_model(
                    model, 
                    f"{model_name}_model",
                    signature=signature,
                    registered_model_name=f"{model_name}_housing_model"
                )
                
                # Store results
                results = {
                    'model': model,
                    'model_name': model_name,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'training_time': training_time,
                    'run_id': mlflow.active_run().info.run_id
                }
                
                logger.info(f"{model_name} training completed:")
                logger.info(f"  - Train RMSE: {train_metrics['rmse']:.4f}")
                logger.info(f"  - Val RMSE: {val_metrics['rmse']:.4f}")
                logger.info(f"  - Val R²: {val_metrics['r2_score']:.4f}")
                logger.info(f"  - Training time: {training_time:.2f}s")
                
                return results
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                mlflow.log_param("error", str(e))
                raise
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """
        Train all defined models and track experiments.
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training target
            X_val (DataFrame): Validation features
            y_val (Series): Validation target
            
        Returns:
            dict: Results for all trained models
        """
        logger.info("Starting training for all models...")
        
        # Get model configurations
        models_config = self.define_models()
        
        # Train each model
        for model_name, model_config in models_config.items():
            try:
                results = self.train_model(
                    model_name, model_config, X_train, y_train, X_val, y_val
                )
                self.results[model_name] = results
                
                # Track best model based on validation RMSE
                val_rmse = results['val_metrics']['rmse']
                if val_rmse < self.best_score:
                    self.best_score = val_rmse
                    self.best_model = results['model']
                    self.best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        logger.info(f"All models trained. Best model: {self.best_model_name} (RMSE: {self.best_score:.4f})")
        
        return self.results
    
    def evaluate_on_test(self, X_test, y_test):
        """
        Evaluate the best model on test set.
        
        Args:
            X_test (DataFrame): Test features
            y_test (Series): Test target
            
        Returns:
            dict: Test set evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("No trained model found. Train models first.")
        
        logger.info(f"Evaluating best model ({self.best_model_name}) on test set...")
        
        # Make predictions
        y_test_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        # Log test metrics to MLflow
        with mlflow.start_run(run_name=f"{self.best_model_name}_test_evaluation"):
            mlflow.log_param("model_name", self.best_model_name)
            mlflow.log_param("evaluation_type", "test_set")
            
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
        
        logger.info(f"Test set evaluation completed:")
        logger.info(f"  - Test RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"  - Test MAE: {test_metrics['mae']:.4f}")
        logger.info(f"  - Test R²: {test_metrics['r2_score']:.4f}")
        
        return test_metrics
    
    def save_best_model(self, model_dir="models"):
        """
        Save the best model to disk.
        
        Args:
            model_dir (str): Directory to save the model
        """
        if self.best_model is None:
            raise ValueError("No trained model found. Train models first.")
        
        logger.info(f"Saving best model ({self.best_model_name})...")
        
        try:
            # Create model directory
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, f"{self.best_model_name}_best_model.joblib")
            joblib.dump(self.best_model, model_path)
            
            # Save model metadata
            metadata = {
                'model_name': self.best_model_name,
                'best_score': self.best_score,
                'training_timestamp': datetime.now().isoformat(),
                'feature_names': list(self.results[self.best_model_name]['model'].feature_names_in_)
                if hasattr(self.results[self.best_model_name]['model'], 'feature_names_in_') else None
            }
            
            metadata_path = os.path.join(model_dir, f"{self.best_model_name}_metadata.json")
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Best model saved to: {model_path}")
            logger.info(f"Model metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def get_model_comparison(self):
        """
        Generate model comparison summary.
        
        Returns:
            DataFrame: Comparison of all trained models
        """
        if not self.results:
            raise ValueError("No trained models found. Train models first.")
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train_RMSE': results['train_metrics']['rmse'],
                'Val_RMSE': results['val_metrics']['rmse'],
                'Val_MAE': results['val_metrics']['mae'],
                'Val_R2': results['val_metrics']['r2_score'],
                'Training_Time': results['training_time'],
                'Is_Best': model_name == self.best_model_name
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Val_RMSE')
        
        return comparison_df


def main():
    """
    Main function to demonstrate the complete training pipeline.
    """
    try:
        # Initialize data processor and model trainer
        data_processor = DataProcessor(random_state=42)
        model_trainer = ModelTrainer()
        
        # Load and process data
        logger.info("Loading and processing data...")
        X, y = data_processor.load_data()
        X_processed, y_processed = data_processor.preprocess_data(X, y)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
            X_processed, y_processed
        )
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = data_processor.scale_features(
            X_train, X_val, X_test
        )
        
        # Train all models
        logger.info("Training models...")
        results = model_trainer.train_all_models(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        # Evaluate best model on test set
        test_metrics = model_trainer.evaluate_on_test(X_test_scaled, y_test)
        
        # Save best model
        model_trainer.save_best_model()
        
        # Generate model comparison
        comparison_df = model_trainer.get_model_comparison()
        logger.info("Model Comparison:")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        # Save comparison to CSV
        os.makedirs("results", exist_ok=True)
        comparison_df.to_csv("results/model_comparison.csv", index=False)
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Best model: {model_trainer.best_model_name}")
        logger.info(f"Best validation RMSE: {model_trainer.best_score:.4f}")
        logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
        
        return model_trainer, test_metrics
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
