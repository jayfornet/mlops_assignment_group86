#!/usr/bin/env python
"""
Model Validation Script

This script validates that model files exist and can be loaded correctly.
It's used in CI/CD pipelines to ensure that models are properly packaged
with the application.
"""

import os
import sys
import logging
import argparse
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def find_model_files(models_dir):
    """Find model files in the specified directory.
    
    Args:
        models_dir (str): Path to the models directory
        
    Returns:
        list: List of model file paths
    """
    logger.info(f"Searching for model files in {models_dir}")
    
    # Search for common model file extensions
    model_files = []
    for ext in ['*.joblib', '*.pkl', '*.h5', '*.pth', '*.onnx', '*.pb']:
        model_files.extend(glob.glob(os.path.join(models_dir, ext)))
    
    if model_files:
        for model_file in model_files:
            logger.info(f"Found model file: {model_file}")
    else:
        logger.warning(f"No model files found in {models_dir}")
    
    return model_files

def validate_model_load(model_file):
    """Attempt to load the model file.
    
    Args:
        model_file (str): Path to the model file
        
    Returns:
        bool: True if model can be loaded, False otherwise
    """
    logger.info(f"Validating model file: {model_file}")
    
    # Determine the file type based on extension
    ext = os.path.splitext(model_file)[1].lower()
    
    try:
        if ext == '.joblib':
            import joblib
            model = joblib.load(model_file)
            logger.info(f"Successfully loaded joblib model: {type(model)}")
            return True
        elif ext == '.pkl':
            import pickle
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Successfully loaded pickle model: {type(model)}")
            return True
        elif ext == '.h5':
            # Try loading with keras
            try:
                from tensorflow import keras
                model = keras.models.load_model(model_file)
                logger.info(f"Successfully loaded Keras model: {type(model)}")
                return True
            except ImportError:
                logger.warning("Tensorflow/Keras not available, skipping .h5 model validation")
                return False
        elif ext == '.pth':
            # Try loading with PyTorch
            try:
                import torch
                model = torch.load(model_file)
                logger.info(f"Successfully loaded PyTorch model: {type(model)}")
                return True
            except ImportError:
                logger.warning("PyTorch not available, skipping .pth model validation")
                return False
        else:
            logger.warning(f"Unsupported model file type: {ext}")
            return False
    except Exception as e:
        logger.error(f"Error loading model {model_file}: {e}")
        return False

def create_dummy_model(models_dir):
    """Create a dummy model for testing purposes.
    
    Args:
        models_dir (str): Path to the models directory
        
    Returns:
        str: Path to the created model file
    """
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    
    logger.info(f"Creating dummy model in {models_dir}")
    
    # Create the directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a simple model
    model_path = os.path.join(models_dir, 'dummy_model.joblib')
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(np.array([[1, 2, 3, 4, 5, 6, 7, 8]]), np.array([4.5]))
    
    # Save the model
    joblib.dump(model, model_path)
    logger.info(f"Dummy model created at {model_path}")
    
    return model_path

def main(models_dir="models", create_dummy=False):
    """Main function to validate model files.
    
    Args:
        models_dir (str): Path to the models directory
        create_dummy (bool): Whether to create a dummy model if no models are found
        
    Returns:
        bool: True if validation was successful, False otherwise
    """
    # Check if models directory exists
    if not os.path.isdir(models_dir):
        logger.warning(f"Models directory {models_dir} does not exist")
        if create_dummy:
            os.makedirs(models_dir, exist_ok=True)
        else:
            return False
    
    # Find model files
    model_files = find_model_files(models_dir)
    
    # Create dummy model if requested and no models found
    if not model_files and create_dummy:
        logger.info("No model files found, creating dummy model")
        model_files = [create_dummy_model(models_dir)]
    
    if not model_files:
        logger.error("No model files found and dummy model creation not requested")
        return False
    
    # Validate each model file
    validation_results = []
    for model_file in model_files:
        validation_results.append(validate_model_load(model_file))
    
    # Check if all validations passed
    if all(validation_results):
        logger.info("All model validations passed!")
        return True
    else:
        logger.error("Some model validations failed")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate model files")
    parser.add_argument("--models-dir", default="models", help="Path to models directory")
    parser.add_argument("--create-dummy", action="store_true", help="Create dummy model if no models found")
    
    args = parser.parse_args()
    success = main(models_dir=args.models_dir, create_dummy=args.create_dummy)
    
    if not success:
        sys.exit(1)
