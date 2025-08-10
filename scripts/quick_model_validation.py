#!/usr/bin/env python3
"""
Quick Model Validation Script

A lightweight version of model validation specifically for integration tests.
"""

import os
import sys
import joblib
import warnings
import numpy as np
from pathlib import Path


def quick_validate(models_dir="models"):
    """Quick validation of models directory."""
    print(f"🔍 Quick validation of models in: {models_dir}")
    
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    # Find all .joblib files
    model_files = list(models_path.glob("*.joblib"))
    
    if not model_files:
        print(f"⚠️ No .joblib files found in {models_dir}")
        return False
    
    print(f"📦 Found {len(model_files)} model files")
    
    success_count = 0
    
    # Suppress all sklearn warnings for clean output
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        for model_file in model_files:
            try:
                # Load model
                model = joblib.load(model_file)
                
                # Quick prediction test
                sample_data = np.array([[8.3252, 41.0, 6.984, 1.023, 322.0, 2.555, 37.88, -122.23]])
                prediction = model.predict(sample_data)
                
                print(f"✅ {model_file.name}: OK (prediction: {prediction[0]:.2f})")
                success_count += 1
                
            except Exception as e:
                print(f"❌ {model_file.name}: FAILED ({str(e)[:50]}...)")
    
    if success_count == len(model_files):
        print(f"🎉 All {success_count} models validated successfully!")
        return True
    else:
        print(f"⚠️ Only {success_count}/{len(model_files)} models validated successfully")
        return False


if __name__ == "__main__":
    models_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
    success = quick_validate(models_dir)
    sys.exit(0 if success else 1)
