#!/usr/bin/env python3
"""
Model Validation Script

This script validates trained models by checking if they can be loaded
and make predictions on sample data.
"""

import os
import sys
import joblib
import logging
import argparse
import numpy as np
import warnings
from pathlib import Path


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_model_file(model_path, logger):
    """Validate a single model file."""
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Check file size (should not be empty)
        file_size = os.path.getsize(model_path)
        if file_size == 0:
            logger.error(f"Model file is empty: {model_path}")
            return False
        
        logger.debug(f"Model file size: {file_size} bytes")
        
        # Suppress sklearn version warnings during model loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            warnings.filterwarnings("ignore", message=".*version.*")
            
            # Try to load the model
            model = joblib.load(model_path)
            logger.info(f"✓ Successfully loaded model from {model_path}")
            
            # Check if model has required methods
            if not hasattr(model, 'predict'):
                logger.error(f"Model does not have predict method: {model_path}")
                return False
            
            # Create sample data for prediction test
            # Using feature shape consistent with California housing dataset
            sample_data = np.array([[8.3252, 41.0, 6.984, 1.023, 322.0, 2.555, 37.88, -122.23]])
            
            # Try to make a prediction
            prediction = model.predict(sample_data)
            
            # Validate prediction output
            if prediction is None or len(prediction) == 0:
                logger.error(f"Model prediction returned invalid result: {model_path}")
                return False
                
            logger.info(f"✓ Model prediction test successful: {prediction[0]:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model validation failed for {model_path}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def create_dummy_model(output_path, logger):
    """Create a dummy model for testing purposes."""
    try:
        from sklearn.linear_model import LinearRegression
        
        # Suppress sklearn warnings during model creation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            
            # Create a simple dummy model
            model = LinearRegression()
            
            # Train on dummy data (8 features like California housing)
            rng = np.random.default_rng(42)  # Use new random generator
            x_dummy = rng.random((100, 8))
            y_dummy = rng.random(100)
            model.fit(x_dummy, y_dummy)
            
            # Save the dummy model
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            joblib.dump(model, output_path)
            
            logger.info(f"✓ Created dummy model at {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create dummy model: {str(e)}")
        return False


def validate_models_directory(models_dir, logger, create_dummy=False):
    """Validate all models in a directory."""
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        logger.warning(f"Models directory does not exist: {models_dir}")
        if create_dummy:
            models_dir.mkdir(parents=True, exist_ok=True)
            dummy_path = models_dir / "dummy_model.joblib"
            if create_dummy_model(str(dummy_path), logger):
                return validate_model_file(str(dummy_path), logger)
        return False
    
    # Find all .joblib files
    model_files = list(models_dir.glob("*.joblib"))
    
    if not model_files:
        logger.warning(f"No .joblib files found in {models_dir}")
        if create_dummy:
            dummy_path = models_dir / "dummy_model.joblib"
            if create_dummy_model(str(dummy_path), logger):
                return validate_model_file(str(dummy_path), logger)
        return False
    
    logger.info(f"Found {len(model_files)} model files to validate")
    
    all_valid = True
    failed_models = []
    
    for model_file in model_files:
        if not validate_model_file(str(model_file), logger):
            all_valid = False
            failed_models.append(str(model_file))
    
    if failed_models:
        logger.error(f"Failed to validate {len(failed_models)} models: {failed_models}")
    
    return all_valid


def main():
    """Main function for model validation."""
    parser = argparse.ArgumentParser(description='Validate trained models')
    parser.add_argument(
        '--models-dir', 
        default='models',
        help='Directory containing model files (default: models)'
    )
    parser.add_argument(
        '--model-file',
        help='Specific model file to validate'
    )
    parser.add_argument(
        '--create-dummy',
        action='store_true',
        help='Create dummy model if no models found'
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    try:
        if args.model_file:
            # Validate specific model file
            logger.info(f"Validating specific model: {args.model_file}")
            success = validate_model_file(args.model_file, logger)
        else:
            # Validate all models in directory
            logger.info(f"Validating models in directory: {args.models_dir}")
            success = validate_models_directory(args.models_dir, logger, args.create_dummy)
        
        if success:
            logger.info("✅ All model validations passed")
            return 0
        else:
            logger.error("❌ Model validation failed")
            return 1
            
    except Exception as e:
        logger.error(f"Model validation failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
