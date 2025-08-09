#!/usr/bin/env python3
"""
End-to-End Pipeline Test

This script creates sample data and runs the complete pipeline end-to-end.
"""

import os
import sys
import subprocess
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def run_command(command, timeout=300):
    """Run a command safely."""
    logger = logging.getLogger(__name__)
    
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{os.getcwd()}/src:{env.get('PYTHONPATH', '')}"
        
        logger.info(f"Running: {command}")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Success: {command}")
            return True
        else:
            logger.error(f"❌ Failed: {command}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Timeout: {command}")
        return False
    except Exception as e:
        logger.error(f"💥 Exception: {command} - {e}")
        return False


def create_sample_data():
    """Create sample California housing data."""
    logger = logging.getLogger(__name__)
    logger.info("📊 Creating sample data...")
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate synthetic data similar to California housing
    rng = np.random.default_rng(42)
    n_samples = 2000
    
    # Feature names from California housing dataset
    feature_names = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    
    # Generate realistic-looking data
    data = {
        'MedInc': rng.normal(5.0, 2.0, n_samples),
        'HouseAge': rng.uniform(1, 52, n_samples),
        'AveRooms': rng.normal(6.0, 1.0, n_samples),
        'AveBedrms': rng.normal(1.0, 0.2, n_samples),
        'Population': rng.uniform(100, 5000, n_samples),
        'AveOccup': rng.normal(3.0, 0.5, n_samples),
        'Latitude': rng.uniform(32.5, 42.0, n_samples),
        'Longitude': rng.uniform(-124.3, -114.3, n_samples)
    }
    
    # Create target (house prices) with some realistic correlation
    target = (
        data['MedInc'] * 50000 +
        (52 - data['HouseAge']) * 1000 +
        data['AveRooms'] * 10000 +
        rng.normal(0, 20000, n_samples)
    )
    target = np.clip(target, 50000, 800000)  # Realistic price range
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['MedHouseVal'] = target
    
    # Save raw data
    df.to_csv('data/california_housing.csv', index=False)
    logger.info(f"✅ Created raw data: {df.shape}")
    
    # Create preprocessed data
    X = df[feature_names].values
    y = df['MedHouseVal'].values
    
    X_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    
    preprocessed_data = {
        'X_train': X_train,
        'X_val': x_val,
        'X_test': x_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names
    }
    
    joblib.dump(preprocessed_data, 'data/processed/preprocessed_data.joblib')
    logger.info("✅ Created preprocessed data")
    
    return True


def setup_environment():
    """Setup the environment for testing."""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Setting up environment...")
    
    # Create all necessary directories
    directories = [
        'data', 'data/processed', 'models', 'logs', 'results',
        'mlruns', 'mlflow-artifacts', 'deployment/models', 'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Set environment variables
    os.environ['MLFLOW_EXPERIMENT_NAME'] = 'california_housing_prediction'
    os.environ['GITHUB_RUN_NUMBER'] = '1'
    
    logger.info("✅ Environment setup complete")
    return True


def test_full_pipeline():
    """Test the complete pipeline end-to-end."""
    logger = setup_logging()
    logger.info("🚀 Starting End-to-End Pipeline Test")
    logger.info("=" * 50)
    
    # Pipeline steps
    steps = [
        ("Environment Setup", setup_environment),
        ("Sample Data Creation", create_sample_data),
        ("MLflow Initialization", lambda: run_command("python scripts/init_mlflow.py")),
        ("Model Training", lambda: run_command("python scripts/train_models.py")),
        ("Model Selection", lambda: run_command("python scripts/select_best_model.py")),
        ("Model Validation", lambda: run_command("python scripts/validate_models_enhanced.py --models-dir models")),
        ("MLflow Setup", lambda: run_command("python scripts/setup_mlflow.py")),
    ]
    
    results = {}
    
    for step_name, step_function in steps:
        logger.info(f"\n{'='*30}")
        logger.info(f"Step: {step_name}")
        logger.info(f"{'='*30}")
        
        try:
            success = step_function()
            results[step_name] = success
            
            if success:
                logger.info(f"✅ {step_name} completed successfully")
            else:
                logger.error(f"❌ {step_name} failed")
                
        except Exception as e:
            logger.error(f"💥 {step_name} failed with exception: {e}")
            results[step_name] = False
    
    # Check outputs
    logger.info(f"\n{'='*30}")
    logger.info("Checking Outputs")
    logger.info(f"{'='*30}")
    
    expected_outputs = [
        'data/california_housing.csv',
        'data/processed/preprocessed_data.joblib',
        'models/',
        'mlruns/',
        'deployment/models/best_model.joblib',
        'deployment/models/model_metadata.json'
    ]
    
    output_check = True
    for output in expected_outputs:
        if os.path.exists(output):
            if os.path.isdir(output):
                contents = len(os.listdir(output)) if os.path.isdir(output) else 0
                logger.info(f"✅ Found: {output} ({contents} items)")
            else:
                size = os.path.getsize(output)
                logger.info(f"✅ Found: {output} ({size} bytes)")
        else:
            logger.warning(f"⚠️ Missing: {output}")
            output_check = False
    
    # Summary
    successful_steps = sum(results.values())
    total_steps = len(results)
    
    logger.info(f"\n{'='*50}")
    logger.info("PIPELINE TEST SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Steps: {successful_steps}/{total_steps} successful")
    logger.info(f"Outputs: {'✅ All found' if output_check else '⚠️ Some missing'}")
    
    if successful_steps == total_steps and output_check:
        logger.info("\n🎉 End-to-End Pipeline Test PASSED!")
        logger.info("Pipeline is working correctly! 🚀")
        return 0
    else:
        logger.warning("\n⚠️ End-to-End Pipeline Test FAILED!")
        logger.warning("Check the errors above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(test_full_pipeline())
