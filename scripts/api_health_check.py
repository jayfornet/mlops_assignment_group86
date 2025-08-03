#!/usr/bin/env python
"""
API Health Check Script

This script checks the health of the API by making requests to various endpoints
and verifying the responses. It can be used in CI/CD pipelines to ensure the API
is functioning correctly after deployment.
"""

import sys
import time
import json
import argparse
import logging
from urllib.parse import urljoin
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_health(base_url, max_retries=5, retry_delay=5):
    """Check the health endpoint of the API.
    
    Args:
        base_url (str): Base URL of the API
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        bool: True if health check passed, False otherwise
    """
    health_url = urljoin(base_url, "/health")
    logger.info(f"Checking API health at {health_url}")
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Health check attempt {attempt}/{max_retries}")
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Health check passed: {response.text}")
                return True
            else:
                logger.warning(f"Health check failed with status code {response.status_code}: {response.text}")
        except requests.RequestException as e:
            logger.warning(f"Health check request failed: {e}")
        
        if attempt < max_retries:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    logger.error("All health check attempts failed")
    return False

def check_docs(base_url):
    """Check if the API documentation is accessible.
    
    Args:
        base_url (str): Base URL of the API
        
    Returns:
        bool: True if docs check passed, False otherwise
    """
    docs_url = urljoin(base_url, "/docs")
    logger.info(f"Checking API docs at {docs_url}")
    
    try:
        response = requests.get(docs_url, timeout=10)
        if response.status_code == 200:
            logger.info("Docs endpoint is accessible")
            return True
        else:
            logger.warning(f"Docs endpoint check failed with status code {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.warning(f"Docs endpoint request failed: {e}")
        return False

def test_prediction(base_url):
    """Test the prediction endpoint with sample data.
    
    Args:
        base_url (str): Base URL of the API
        
    Returns:
        bool: True if prediction test passed, False otherwise
    """
    predict_url = urljoin(base_url, "/predict")
    logger.info(f"Testing prediction endpoint at {predict_url}")
    
    sample_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984,
        "AveBedrms": 1.024,
        "Population": 322.0,
        "AveOccup": 2.555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    
    try:
        response = requests.post(predict_url, 
                                json=sample_data, 
                                headers={"Content-Type": "application/json"},
                                timeout=10)
        
        if response.status_code == 200:
            logger.info(f"Prediction test passed: {response.text}")
            return True
        else:
            logger.warning(f"Prediction test failed with status code {response.status_code}: {response.text}")
            return False
    except requests.RequestException as e:
        logger.warning(f"Prediction request failed: {e}")
        return False

def check_metrics(base_url):
    """Check the metrics endpoint.
    
    Args:
        base_url (str): Base URL of the API
        
    Returns:
        bool: True if metrics check passed, False otherwise
    """
    metrics_url = urljoin(base_url, "/metrics")
    logger.info(f"Checking metrics endpoint at {metrics_url}")
    
    try:
        response = requests.get(metrics_url, timeout=10)
        if response.status_code == 200:
            logger.info("Metrics endpoint is accessible")
            return True
        else:
            logger.warning(f"Metrics endpoint check failed with status code {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.warning(f"Metrics endpoint request failed: {e}")
        return False

def run_all_checks(host="localhost", port=8000, exit_on_failure=True):
    """Run all API health checks.
    
    Args:
        host (str): API host
        port (int): API port
        exit_on_failure (bool): Exit with error code if any check fails
        
    Returns:
        bool: True if all checks passed, False otherwise
    """
    base_url = f"http://{host}:{port}"
    logger.info(f"Running health checks against {base_url}")
    
    checks = [
        ("Health check", lambda: check_health(base_url)),
        ("Docs check", lambda: check_docs(base_url)),
        ("Prediction test", lambda: test_prediction(base_url)),
        ("Metrics check", lambda: check_metrics(base_url))
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
            logger.info(f"{name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error during {name}: {e}")
            results.append(False)
    
    all_passed = all(results)
    logger.info(f"Overall health check: {'PASSED' if all_passed else 'FAILED'}")
    
    if exit_on_failure and not all_passed:
        sys.exit(1)
    
    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run API health checks")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--no-exit", action="store_true", help="Don't exit with error code on failure")
    
    args = parser.parse_args()
    run_all_checks(host=args.host, port=args.port, exit_on_failure=not args.no_exit)
