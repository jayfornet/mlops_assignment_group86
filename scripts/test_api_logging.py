"""
Test script for the enhanced request/response logging functionality.

This script demonstrates and tests the new logging features.
"""

import requests
import json
import time
import random
from datetime import datetime


def test_api_logging():
    """Test the API logging functionality."""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing API Request/Response Logging")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response time: {response.elapsed.total_seconds() * 1000:.2f}ms")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(base_url)
        print(f"   Status: {response.status_code}")
        print(f"   Response time: {response.elapsed.total_seconds() * 1000:.2f}ms")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Multiple prediction requests
    print("\n3. Testing prediction endpoint with multiple requests...")
    
    test_data = [
        {
            "MedInc": 8.3252,
            "HouseAge": 41.0,
            "AveRooms": 6.984,
            "AveBedrms": 1.023,
            "Population": 322.0,
            "AveOccup": 2.555,
            "Latitude": 37.88,
            "Longitude": -122.23
        },
        {
            "MedInc": 7.2574,
            "HouseAge": 21.0,
            "AveRooms": 6.238,
            "AveBedrms": 0.971,
            "Population": 2401.0,
            "AveOccup": 2.109,
            "Latitude": 37.86,
            "Longitude": -122.22
        },
        {
            "MedInc": 5.6431,
            "HouseAge": 52.0,
            "AveRooms": 5.817,
            "AveBedrms": 1.073,
            "Population": 496.0,
            "AveOccup": 2.802,
            "Latitude": 37.85,
            "Longitude": -122.24
        }
    ]
    
    for i, data in enumerate(test_data, 1):
        print(f"   Prediction {i}...")
        try:
            response = requests.post(
                f"{base_url}/predict",
                headers={"Content-Type": "application/json"},
                json=data
            )
            print(f"     Status: {response.status_code}")
            print(f"     Response time: {response.elapsed.total_seconds() * 1000:.2f}ms")
            
            if response.status_code == 200:
                result = response.json()
                print(f"     Prediction: ${result['prediction']:.2f}k")
        except Exception as e:
            print(f"     Error: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Test 4: Test error handling
    print("\n4. Testing error handling (invalid data)...")
    try:
        invalid_data = {
            "MedInc": -1,  # Invalid negative value
            "HouseAge": 200,  # Invalid high value
            "AveRooms": 0,  # Invalid zero value
        }
        response = requests.post(
            f"{base_url}/predict",
            headers={"Content-Type": "application/json"},
            json=invalid_data
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response time: {response.elapsed.total_seconds() * 1000:.2f}ms")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 5: Check logging endpoints
    print("\n5. Testing logging endpoints...")
    
    # Wait a moment for logs to be written
    time.sleep(2)
    
    try:
        # Get recent requests
        print("   Fetching recent requests...")
        response = requests.get(f"{base_url}/logs/requests?limit=10")
        if response.status_code == 200:
            data = response.json()
            print(f"     Found {data['total']} recent requests")
        
        # Get statistics
        print("   Fetching request statistics...")
        response = requests.get(f"{base_url}/logs/stats?hours=1")
        if response.status_code == 200:
            stats = response.json()
            print(f"     Total requests (last hour): {stats.get('total_requests', 0)}")
            print(f"     Success rate: {stats.get('success_rate', 0)}%")
            print(f"     Average response time: {stats.get('average_response_time_ms', 0):.2f}ms")
    
    except Exception as e:
        print(f"   Error testing logging endpoints: {e}")
    
    print("\n✅ Testing completed!")
    print("\nTo view the logs:")
    print(f"- Request logs database: logs/api_requests.db")
    print(f"- Request logs file: logs/api_requests.log")
    print(f"- API logs: logs/api.log")
    print(f"- View recent requests: {base_url}/logs/requests")
    print(f"- View statistics: {base_url}/logs/stats")


def generate_load_test():
    """Generate some load for testing logging."""
    
    base_url = "http://localhost:8000"
    
    print("\n🚀 Generating load test for logging...")
    print("=" * 50)
    
    # Sample data variations
    base_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984,
        "AveBedrms": 1.023,
        "Population": 322.0,
        "AveOccup": 2.555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    
    print("Sending 20 prediction requests...")
    
    for i in range(20):
        # Vary the data slightly
        data = base_data.copy()
        data["MedInc"] += random.uniform(-2, 2)
        data["HouseAge"] += random.uniform(-10, 10)
        data["AveRooms"] += random.uniform(-1, 1)
        data["Population"] += random.uniform(-100, 100)
        
        try:
            response = requests.post(
                f"{base_url}/predict",
                headers={"Content-Type": "application/json"},
                json=data
            )
            
            status_icon = "✅" if response.status_code == 200 else "❌"
            print(f"  {status_icon} Request {i+1:2d}: {response.status_code} ({response.elapsed.total_seconds() * 1000:.1f}ms)")
            
        except Exception as e:
            print(f"  ❌ Request {i+1:2d}: Error - {e}")
        
        # Random delay between requests
        time.sleep(random.uniform(0.1, 0.5))
    
    print("\n📊 Load test completed! Check the logs:")
    print(f"- View statistics: {base_url}/logs/stats")
    print(f"- Download logs: {base_url}/logs/download?format=json&hours=1")


if __name__ == "__main__":
    print("🏠 California Housing API - Logging Test")
    print("=" * 50)
    print()
    print("Make sure the API is running on http://localhost:8000")
    print()
    
    choice = input("Choose test type:\n1. Basic logging test\n2. Load test\n3. Both\nEnter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        test_api_logging()
    
    if choice in ["2", "3"]:
        if choice == "3":
            input("\nPress Enter to continue with load test...")
        generate_load_test()
    
    print("\n🎉 All tests completed!")
