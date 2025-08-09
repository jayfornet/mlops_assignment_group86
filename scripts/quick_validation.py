#!/usr/bin/env python3
"""
Quick Pipeline Validation Script

This script quickly validates that all pipeline components are working correctly.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path


def setup_logging():
    """Setup simple logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    return logging.getLogger(__name__)


def run_command(command, timeout=60):
    """Run a command with timeout."""
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{os.getcwd()}/src:{env.get('PYTHONPATH', '')}"
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)


def quick_validation():
    """Run quick validation of pipeline components."""
    logger = setup_logging()
    logger.info("🔍 Quick Pipeline Validation")
    logger.info("=" * 40)
    
    # Create temp directories
    temp_dirs = ['data/processed', 'models', 'logs', 'results', 'mlruns', 'deployment/models']
    for dir_path in temp_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    validation_tests = [
        ("Python Environment", "python --version"),
        ("Required Packages", "python -c 'import pandas, numpy, sklearn, mlflow, joblib; print(\"All packages available\")'"),
        ("MLflow Init Script", "python scripts/init_mlflow.py"),
        ("Model Training Script", "python scripts/train_models.py --help || python -c 'from scripts.train_models import main; print(\"Script can be imported\")'"),
        ("Model Selection Script", "python scripts/select_best_model.py --help || python -c 'from scripts.select_best_model import main; print(\"Script can be imported\")'"),
        ("MLflow Setup Script", "python scripts/setup_mlflow.py --help || python -c 'from scripts.setup_mlflow import main; print(\"Script can be imported\")'"),
        ("Model Validation Script", "python scripts/validate_models_enhanced.py --help"),
        ("API Module", "python -c 'from src.api.app import app; print(\"API module available\")'"),
    ]
    
    results = {}
    
    for test_name, command in validation_tests:
        logger.info(f"\n🧪 Testing: {test_name}")
        success, stdout, stderr = run_command(command)
        
        if success:
            logger.info(f"✅ {test_name} - PASSED")
            if stdout.strip():
                logger.info(f"   Output: {stdout.strip()}")
        else:
            logger.error(f"❌ {test_name} - FAILED")
            if stderr.strip():
                logger.error(f"   Error: {stderr.strip()}")
        
        results[test_name] = success
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    logger.info(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All validations passed! Pipeline is ready.")
        return 0
    else:
        logger.warning("⚠️ Some validations failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(quick_validation())
