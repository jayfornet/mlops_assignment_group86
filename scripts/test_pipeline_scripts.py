#!/usr/bin/env python3
"""
Test Script for Pipeline Script Integration

This script tests all the refactored pipeline scripts to ensure they work correctly.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path


def run_command(command, cwd=None, env=None):
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def setup_test_environment():
    """Setup a test environment with minimal data."""
    print("🔧 Setting up test environment...")
    
    # Create necessary directories
    dirs = ['data/processed', 'models', 'logs', 'results', 'mlruns', 'deployment/models']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create dummy preprocessed data
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic California housing-like data
    rng = np.random.default_rng(42)
    n_samples = 1000
    n_features = 8
    
    x = rng.random((n_samples, n_features))
    y = rng.random(n_samples) * 500000  # House prices
    
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    
    # Save preprocessed data
    data = {
        'X_train': x_train,
        'X_val': x_val,
        'X_test': x_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    
    joblib.dump(data, 'data/processed/preprocessed_data.joblib')
    print("✅ Test data created")


def test_script(script_name, description, env_vars=None):
    """Test a single script."""
    print(f"\n🧪 Testing {description}...")
    
    # Setup environment
    test_env = os.environ.copy()
    test_env['PYTHONPATH'] = f"{os.getcwd()}/src:{test_env.get('PYTHONPATH', '')}"
    
    if env_vars:
        test_env.update(env_vars)
    
    # Run the script
    success, stdout, stderr = run_command(f"python scripts/{script_name}", env=test_env)
    
    if success:
        print(f"✅ {description} - PASSED")
        return True
    else:
        print(f"❌ {description} - FAILED")
        print(f"Error: {stderr}")
        if stdout:
            print(f"Output: {stdout}")
        return False


def test_all_scripts():
    """Test all refactored scripts."""
    print("🚀 Testing all pipeline scripts...")
    
    # Test results
    results = {}
    
    # Test 1: MLflow Initialization
    results['init_mlflow'] = test_script(
        'init_mlflow.py',
        'MLflow Initialization'
    )
    
    # Test 2: MLflow Setup
    results['setup_mlflow'] = test_script(
        'setup_mlflow.py',
        'MLflow Setup and Tracking',
        {'MLFLOW_EXPERIMENT_NAME': 'test_experiment', 'GITHUB_RUN_NUMBER': '1'}
    )
    
    # Test 3: Model Training
    results['train_models'] = test_script(
        'train_models.py',
        'Model Training'
    )
    
    # Test 4: Model Selection
    results['select_best_model'] = test_script(
        'select_best_model.py',
        'Best Model Selection'
    )
    
    # Test 5: Model Validation
    results['validate_models'] = test_script(
        'validate_models_enhanced.py',
        'Model Validation'
    )
    
    return results


def cleanup_test_environment():
    """Clean up test environment."""
    print("\n🧹 Cleaning up test environment...")
    
    # Directories to clean up
    cleanup_dirs = ['data', 'models', 'logs', 'results', 'mlruns', 'deployment']
    
    for dir_path in cleanup_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    
    # Clean up generated files
    cleanup_files = [
        'mlflow_experiment_summary.json',
        'mlflow_persistence_info.json'
    ]
    
    for file_path in cleanup_files:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    print("✅ Cleanup completed")


def main():
    """Main test function."""
    print("🔍 Pipeline Scripts Integration Test")
    print("=" * 50)
    
    try:
        # Setup test environment
        setup_test_environment()
        
        # Test all scripts
        results = test_all_scripts()
        
        # Print summary
        print("\n📊 Test Results Summary:")
        print("=" * 30)
        
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        for script, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{script:20} - {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Pipeline scripts are working correctly.")
            return 0
        else:
            print("⚠️  Some tests failed. Check the errors above.")
            return 1
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return 1
        
    finally:
        # Cleanup
        cleanup_test_environment()


if __name__ == "__main__":
    sys.exit(main())
