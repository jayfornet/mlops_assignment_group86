#!/usr/bin/env python3
"""
Quick Pipeline Test

Simple test that avoids unicode and Windows path issues.
"""

import os
import sys
import subprocess
import logging


def setup_logging():
    """Setup simple logging."""
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
            logger.info(f"SUCCESS: {command}")
            return True
        else:
            logger.error(f"FAILED: {command}")
            if result.stderr:
                logger.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"EXCEPTION: {command} - {e}")
        return False


def main():
    """Test the pipeline components."""
    logger = setup_logging()
    logger.info("Starting Pipeline Component Test")
    
    # Test individual components
    tests = [
        ("Quick Validation", "python scripts/quick_validation.py"),
        ("Train Models", "python scripts/train_models.py"),
        ("Select Best Model", "python scripts/select_best_model.py"),
        ("Validate Models", "python scripts/validate_models_enhanced.py --models-dir models")
    ]
    
    results = []
    
    for test_name, command in tests:
        logger.info(f"\n--- Testing: {test_name} ---")
        success = run_command(command)
        results.append((test_name, success))
        
        if success:
            logger.info(f"PASS: {test_name}")
        else:
            logger.error(f"FAIL: {test_name}")
    
    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    logger.info("\n--- SUMMARY ---")
    logger.info(f"Tests passed: {passed}/{total}")
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"  {status}: {test_name}")
    
    if passed == total:
        logger.info("\nAll tests passed! Pipeline is working correctly.")
        return 0
    else:
        logger.warning(f"\n{total - passed} tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
