#!/usr/bin/env python3
"""
Quick Model Validation Script

A lightweight version of model validation specifically for integration tests.
Focuses on validating the deployment model (best_model.joblib) which is 
sufficient for assignment purposes.
"""

import os
import sys
import joblib
import warnings
import numpy as np
from pathlib import Path


def quick_validate(models_dir="models"):
    """Quick validation of models directory - focus on deployment-ready models."""
    print(f"🔍 Quick validation of models in: {models_dir}")
    
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    # Priority order: look for best_model.joblib first (deployment model)
    priority_models = ["best_model.joblib", "deployment_model.joblib"]
    deployment_model = None
    
    for model_name in priority_models:
        model_path = models_path / model_name
        if model_path.exists():
            deployment_model = model_path
            break
    
    if deployment_model:
        print(f"📦 Found deployment model: {deployment_model.name}")
        
        # Validate only the deployment model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            try:
                # Load model
                model = joblib.load(deployment_model)
                
                # Quick prediction test
                sample_data = np.array([[8.3252, 41.0, 6.984, 1.023, 322.0, 2.555, 37.88, -122.23]])
                prediction = model.predict(sample_data)
                
                print(f"✅ {deployment_model.name}: OK (prediction: {prediction[0]:.2f})")
                print(f"🎉 Deployment model validated successfully!")
                return True
                
            except Exception as e:
                print(f"❌ {deployment_model.name}: FAILED ({str(e)[:50]}...)")
                return False
    else:
        # Fallback: look for any .joblib files if no deployment model found
        model_files = list(models_path.glob("*.joblib"))
        
        if not model_files:
            print(f"⚠️ No .joblib files found in {models_dir}")
            return False
        
        print(f"⚠️ No deployment model found, checking {len(model_files)} available models...")
        
        # Try to validate at least one working model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            for model_file in model_files:
                try:
                    model = joblib.load(model_file)
                    sample_data = np.array([[8.3252, 41.0, 6.984, 1.023, 322.0, 2.555, 37.88, -122.23]])
                    prediction = model.predict(sample_data)
                    
                    print(f"✅ {model_file.name}: OK (prediction: {prediction[0]:.2f})")
                    print(f"🎉 At least one model validated successfully!")
                    return True
                    
                except Exception as e:
                    print(f"❌ {model_file.name}: FAILED ({str(e)[:50]}...)")
                    continue
        
        print("❌ No models could be validated successfully")
        return False


if __name__ == "__main__":
    models_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
    success = quick_validate(models_dir)
    sys.exit(0 if success else 1)
