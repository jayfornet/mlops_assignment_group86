import os
import sys
import logging
import joblib
import mlflow
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.train_model import ModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Train models using preprocessed data"""
    logger.info("Loading preprocessed data...")
    
    try:
        # Load preprocessed data
        preprocessed_data = joblib.load('data/processed/preprocessed_data.joblib')
        
        X_train = preprocessed_data['X_train']
        X_val = preprocessed_data['X_val']
        X_test = preprocessed_data['X_test']
        y_train = preprocessed_data['y_train']
        y_val = preprocessed_data['y_val']
        y_test = preprocessed_data['y_test']
        
        logger.info(f"Preprocessed data loaded. Training set shape: {X_train.shape}")
        
        # Initialize model trainer
        model_trainer = ModelTrainer()
        
        # Train all models
        logger.info("Training models...")
        results = model_trainer.train_all_models(
            X_train, y_train, X_val, y_val
        )
        
        # Evaluate best model on test set
        test_metrics = model_trainer.evaluate_on_test(X_test, y_test)
        
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
