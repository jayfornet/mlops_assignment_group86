import pytest
from fastapi.testclient import TestClient
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import app
from src.api.health_check import check_system_health

client = TestClient(app)

@patch('src.api.health_check.check_system_health')
def test_health_endpoint_success(mock_check_system):
    """Test that the health endpoint returns 200 when everything is OK."""
    # Mock a healthy system
    mock_check_system.return_value = {
        "status": "ok",
        "cpu": {"percent": 50.0, "status": "ok", "message": "CPU usage: 50.0%"},
        "memory": {"percent": 60.0, "status": "ok", "message": "Memory usage: 60.0%"},
        "disk": {"percent": 70.0, "status": "ok", "message": "Disk usage: 70.0%"}
    }
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "checks" in data
    assert isinstance(data["checks"], dict)

@patch('src.api.health_check.check_system_health')
def test_health_endpoint_with_resource_warning(mock_check_system):
    """Test that the health endpoint handles resource warnings correctly."""
    # Mock a system with warnings
    mock_check_system.return_value = {
        "status": "warning",
        "cpu": {
            "percent": 85.0, 
            "status": "warning", 
            "message": "High CPU usage detected: 85.0%"
        },
        "memory": {"percent": 60.0, "status": "ok", "message": "Memory usage: 60.0%"},
        "disk": {"percent": 70.0, "status": "ok", "message": "Disk usage: 70.0%"}
    }
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "warning"
    assert "High CPU usage" in str(data["checks"]["system"])

@patch('src.api.health_check.check_system_health')
def test_health_endpoint_with_error(mock_check_system):
    """Test that the health endpoint handles errors gracefully."""
    # Mock an exception in the health check
    mock_check_system.side_effect = Exception("Test error")
    
    response = client.get("/health")
    assert response.status_code == 500
    data = response.json()
    assert data["status"] == "error"
    assert "Failed to perform health check" in data["message"]
    assert "Failed to perform health check" in data["message"]

def test_system_health_check():
    """Test the system health check function directly."""
    health_info = check_system_health()
    assert isinstance(health_info, dict)
    assert "cpu" in health_info
    assert "memory" in health_info
    assert "disk" in health_info
