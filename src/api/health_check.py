"""
Health Check API for California Housing Price Prediction Service

This module provides a more robust health check endpoint for the API.
It handles errors gracefully and provides detailed system information.
"""

import os
import sys
import platform
import psutil
import logging
from datetime import datetime
import json
from typing import Dict, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """Get system information.
    
    Returns:
        dict: System information
    """
    try:
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "cpu_count": psutil.cpu_count(),
            "memory_available": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
            "memory_total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "disk_free": f"{psutil.disk_usage('/').free / (1024**3):.2f} GB",
            "disk_total": f"{psutil.disk_usage('/').total / (1024**3):.2f} GB"
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {"error": f"Error getting system info: {str(e)}"}

def check_system_health() -> Dict[str, Any]:
    """Check system health (CPU, memory, disk usage).
    
    Returns:
        dict: System health metrics and statuses
    """
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.5)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Determine status levels
        cpu_status = "ok" if cpu_percent < 80 else "warning" if cpu_percent < 95 else "critical"
        memory_status = "ok" if memory_percent < 80 else "warning" if memory_percent < 95 else "critical"
        disk_status = "ok" if disk_percent < 80 else "warning" if disk_percent < 95 else "critical"
        
        # Overall status is the worst of all statuses
        if "critical" in [cpu_status, memory_status, disk_status]:
            overall_status = "critical"
        elif "warning" in [cpu_status, memory_status, disk_status]:
            overall_status = "warning"
        else:
            overall_status = "ok"
        
        return {
            "status": overall_status,
            "cpu": {
                "percent": cpu_percent,
                "status": cpu_status,
                "cores": psutil.cpu_count(),
                "message": f"CPU usage: {cpu_percent}%"
            },
            "memory": {
                "percent": memory_percent,
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3),
                "status": memory_status,
                "message": f"Memory usage: {memory_percent}%"
            },
            "disk": {
                "percent": disk_percent,
                "free_gb": disk.free / (1024**3),
                "total_gb": disk.total / (1024**3),
                "status": disk_status,
                "message": f"Disk usage: {disk_percent}%"
            }
        }
    except Exception as e:
        logger.error(f"Error checking system health: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to check system health"
        }

def get_model_info() -> Dict[str, Any]:
    """Get model information.
    
    Returns:
        dict: Model information
    """
    try:
        model_dir = "models"
        if not os.path.exists(model_dir):
            return {"error": f"Model directory not found: {model_dir}"}
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
        if not model_files:
            return {"error": "No model files found"}
        
        model_info = []
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            model_size = os.path.getsize(model_path) / (1024**2)  # Convert to MB
            model_modified = datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
            
            model_info.append({
                "name": model_file,
                "size_mb": f"{model_size:.2f}",
                "last_modified": model_modified
            })
        
        return {
            "model_count": len(model_files),
            "models": model_info
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {"error": f"Error getting model info: {str(e)}"}

def check_health() -> Dict[str, Any]:
    """Perform a health check.
    
    Returns:
        dict: Health check results
    """
    try:
        start_time = datetime.now()
        
        # Basic API info
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "api_version": "1.0.0",
            "uptime": "N/A"  # Would be calculated in a real service
        }
        
        # System information
        health_data["system"] = get_system_info()
        
        # Model information
        health_data["model"] = get_model_info()
        
        # Add response time
        health_data["response_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
        
        return health_data
    except Exception as e:
        logger.error(f"Error during health check: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    # Can be run as a standalone script for testing
    print(json.dumps(check_health(), indent=2))
