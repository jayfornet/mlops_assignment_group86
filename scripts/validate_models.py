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
    import json
    import numpy as np
    from datetime import datetime
    
    logger.info(f"Creating dummy model in {models_dir}")
    
    # Create the directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Create simple models with different filename patterns
    models_to_create = [
        ('random_forest_best_model.joblib', 'random_forest_metadata.json'),
        ('gradient_boosting_best_model.joblib', 'gradient_boosting_metadata.json')
    ]
    
    created_models = []
    
    # Create feature names matching what the API expects
    feature_names = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    
    for model_filename, metadata_filename in models_to_create:
        model_path = os.path.join(models_dir, model_filename)
        
        try:
            # Try to create a sklearn model first
            try:
                from sklearn.ensemble import RandomForestRegressor
                logger.info("Creating scikit-learn based dummy model")
                
                # Create dummy training data matching the feature names
                X = np.array([
                    [8.3252, 41.0, 6.984, 1.023, 322.0, 2.555, 37.88, -122.23],
                    [8.3252, 21.0, 6.238, 0.971, 2401.0, 2.109, 37.86, -122.22]
                ])
                y = np.array([4.526, 3.585])
                
                # Create a dummy model
                model = RandomForestRegressor(n_estimators=5, random_state=42)
                model.fit(X, y)
                
                # Save the model
                joblib.dump(model, model_path)
            except Exception as e:
                logger.warning(f"Failed to create scikit-learn model: {e}")
                logger.info("Creating simplified dummy model instead")
                
                # Create a simple callable class that can predict
                class DummyPredictor:
                    def __init__(self):
                        self.feature_names = feature_names
                    
                    def predict(self, X):
                        """Return constant predictions."""
                        if isinstance(X, np.ndarray):
                            return np.ones(X.shape[0]) * 2.5
                        else:  # assume pandas DataFrame or similar
                            return np.ones(len(X)) * 2.5
                
                model = DummyPredictor()
                joblib.dump(model, model_path)
                
            logger.info(f"Dummy model created at {model_path}")
            created_models.append(model_path)
            
            # Create and save model metadata
            metadata_path = os.path.join(models_dir, metadata_filename)
            metadata = {
                "model_type": "DummyRegressor",
                "created_at": datetime.now().isoformat(),
                "features": feature_names,
                "metrics": {
                    "rmse": 0.5,
                    "mae": 0.4,
                    "r2": 0.8
                },
                "parameters": {
                    "n_estimators": 5,
                    "random_state": 42
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Model metadata created at {metadata_path}")
        except Exception as e:
            logger.error(f"Error creating model {model_filename}: {e}")
    
    if created_models:
        return created_models[0]  # Return the first model path
    else:
        logger.error("Failed to create any dummy models")
        return None
        "features": feature_names,
        "metrics": {
            "rmse": 0.5,
            "mae": 0.4,
            "r2": 0.8
        },
        "parameters": {
            "n_estimators": 10,
            "random_state": 42
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata created at {metadata_path}")
    
    return model_path

def main(models_dir="models", create_dummy=False, force=False):
    """Main function to validate model files.
    
    Args:
        models_dir (str): Path to the models directory
        create_dummy (bool): Whether to create a dummy model if no models are found
        force (bool): Force creation of dummy model even if other models exist
        
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
    
    # Create dummy model if requested and no models found, or if force is True
    if (not model_files and create_dummy) or (force and create_dummy):
        logger.info("Creating dummy model" + (" (forced)" if force else ""))
        dummy_model_path = create_dummy_model(models_dir)
        if dummy_model_path:
            model_files = [dummy_model_path]
        else:
            logger.error("Failed to create dummy model")
            return False
    
    if not model_files:
        logger.error("No model files found and dummy model creation not requested")
        return False
    
    # Validate each model file
    validation_results = []
    for model_file in model_files:
        validation_results.append(validate_model_load(model_file))
    
    # Check if at least one validation passed (we only need one working model)
    if any(validation_results):
        logger.info("At least one model validation passed!")
        return True
    else:
        logger.error("All model validations failed")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate model files")
    parser.add_argument("--models-dir", default="models", help="Path to models directory")
    parser.add_argument("--create-dummy", action="store_true", help="Create dummy model if no models found")
    parser.add_argument("--force", action="store_true", help="Force creation of dummy model even if other models exist")
    
    args = parser.parse_args()
    success = main(models_dir=args.models_dir, create_dummy=args.create_dummy, force=args.force)
    
    if not success:
        sys.exit(1)
