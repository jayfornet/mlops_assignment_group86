#!/usr/bin/env python
"""
Prepares the testing environment by creating all necessary directories.
Run this script before running tests to ensure all required directories exist.
"""

import os
from pathlib import Path

def prepare_test_environment():
    """Create all necessary directories for tests."""
    print("Creating necessary directories for testing...")
    
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Change to project root
    os.chdir(project_root)
    
    # Create necessary directories
    directories = [
        "data", "models", "logs", "results", "mlruns", "mlflow-artifacts"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")
    
    print("Environment preparation complete.")

if __name__ == "__main__":
    prepare_test_environment()
