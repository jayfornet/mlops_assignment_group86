#!/usr/bin/env python3
"""
MLflow Data Verification Script

This script verifies that MLflow data is properly mounted and accessible
in the Docker environment.
"""

import os
import json
import sys
from pathlib import Path
import subprocess


def check_directory_structure():
    """Check if MLflow directories exist and are accessible."""
    in_docker = Path('/.dockerenv').exists()
    print(f"📦 Running in Docker: {'Yes' if in_docker else 'No'}")
    
    mlflow_paths = {
        'mlruns': '/mlflow/mlruns' if in_docker else './mlruns',
        'artifacts': '/mlflow/artifacts' if in_docker else './mlflow-artifacts',
        'models': '/mlflow/models' if in_docker else './models',
        'deployment': '/mlflow/deployment' if in_docker else './deployment',
        'results': '/mlflow/results' if in_docker else './results'
    }
    
    print("\n📂 Checking MLflow data directories:")
    
    for name, path in mlflow_paths.items():
        path_obj = Path(path)
        exists = path_obj.exists()
        print(f"  {'✅' if exists else '❌'} {name}: {path}")
        
        if exists and path_obj.is_dir():
            try:
                file_count = len(list(path_obj.rglob('*')))
                print(f"     📁 Contains {file_count} files/directories")
            except PermissionError:
                print("     🔒 Permission denied")
    
    return mlflow_paths, in_docker


def check_experiments(mlflow_paths):
    """Check MLflow experiments and runs."""
    print("\n🔬 Checking MLflow experiments:")
    
    mlruns_path = Path(mlflow_paths['mlruns'])
    if mlruns_path.exists():
        experiments = [d for d in mlruns_path.iterdir() if d.is_dir()]
        print(f"  📊 Found {len(experiments)} experiments")
        
        for exp in experiments[:5]:  # Show first 5
            runs = [d for d in exp.iterdir() if d.is_dir() and d.name != 'models']
            print(f"    📈 Experiment {exp.name}: {len(runs)} runs")
    else:
        print(f"  ❌ MLruns directory not found at {mlruns_path}")


def check_model_files(mlflow_paths):
    """Check available model files."""
    print("\n🎯 Checking model files:")
    
    models_path = Path(mlflow_paths['models'])
    if models_path.exists():
        model_files = list(models_path.glob('*.joblib')) + list(models_path.glob('*.pkl'))
        print(f"  🏆 Found {len(model_files)} model files")
        for model in model_files[:5]:  # Show first 5
            print(f"    📄 {model.name}")
    else:
        print(f"  ❌ Models directory not found at {models_path}")


def check_persistence_metadata():
    """Check MLflow persistence metadata."""
    print("\n📝 Checking persistence metadata:")
    
    info_file = Path('./mlflow_persistence_info.json')
    if info_file.exists():
        try:
            with open(info_file, 'r') as f:
                info = json.load(f)
            print("  ✅ Persistence info found")
            print(f"    🕒 Last update: {info.get('timestamp', 'Unknown')}")
            print(f"    🏃 Total runs: {info.get('total_runs', 'Unknown')}")
            print(f"    🥇 Best model: {info.get('best_model_type', 'Unknown')}")
        except Exception as e:
            print(f"  ⚠️  Error reading persistence info: {e}")
    else:
        print("  ❌ Persistence info file not found")


def test_mlflow_connectivity(in_docker):
    """Test MLflow server connectivity if running in Docker."""
    if not in_docker:
        return
        
    print("\n🌐 Testing MLflow server connectivity:")
    try:
        result = subprocess.run(
            ['curl', '-f', 'http://localhost:5000/health'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("  ✅ MLflow server is accessible")
        else:
            print("  ❌ MLflow server not accessible")
    except subprocess.TimeoutExpired:
        print("  ⏰ MLflow server connection timeout")
    except FileNotFoundError:
        print("  ❌ curl not available for testing")


def check_mlflow_data():
    """Check if MLflow data is properly mounted and accessible."""
    print("🔍 Verifying MLflow Data Mounting in Docker Environment")
    print("=" * 60)
    
    mlflow_paths, in_docker = check_directory_structure()
    check_experiments(mlflow_paths)
    check_model_files(mlflow_paths)
    check_persistence_metadata()
    test_mlflow_connectivity(in_docker)
    
    print("\n" + "="*60)
    print("🎉 MLflow data verification complete!")
    
    return True


def main():
    """Main function."""
    try:
        check_mlflow_data()
    except KeyboardInterrupt:
        print("\n⏹️  Verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
