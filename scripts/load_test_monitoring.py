#!/usr/bin/env python3
"""
Load Testing Script for MLOps API Monitoring

This script generates prediction requests to test the monitoring system
and create meaningful metrics for Prometheus and Grafana visualization.
"""

import requests
import time
import random
import json
import concurrent.futures
import argparse
from datetime import datetime
import sys


class MLOpsLoadTester:
    def __init__(self, api_url="http://localhost:8000", concurrency=5):
        self.api_url = api_url.rstrip('/')
        self.concurrency = concurrency
        self.success_count = 0
        self.error_count = 0
        self.response_times = []
        
    def generate_housing_data(self):
        """Generate realistic housing data for predictions."""
        # Generate realistic California housing data
        return {
            "MedInc": round(random.uniform(1.0, 15.0), 4),  # Median income
            "HouseAge": round(random.uniform(1.0, 52.0), 1),  # House age
            "AveRooms": round(random.uniform(3.0, 8.0), 3),  # Average rooms
            "AveBedrms": round(random.uniform(0.8, 2.0), 3),  # Average bedrooms
            "Population": round(random.uniform(100.0, 5000.0), 1),  # Population
            "AveOccup": round(random.uniform(1.5, 5.0), 3),  # Average occupancy
            "Latitude": round(random.uniform(32.0, 42.0), 2),  # California latitude
            "Longitude": round(random.uniform(-124.0, -114.0), 2)  # California longitude
        }
    
    def make_prediction_request(self, request_id):
        """Make a single prediction request."""
        try:
            data = self.generate_housing_data()
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_url}/predict",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            if response.status_code == 200:
                self.success_count += 1
                result = response.json()
                print(f"✅ Request {request_id}: Prediction={result['prediction']:.2f}, Time={response_time:.3f}s")
                return True, response_time, result['prediction']
            else:
                self.error_count += 1
                print(f"❌ Request {request_id}: HTTP {response.status_code}, Time={response_time:.3f}s")
                return False, response_time, None
                
        except Exception as e:
            self.error_count += 1
            print(f"❌ Request {request_id}: Error - {str(e)}")
            return False, 0, None
    
    def check_api_health(self):
        """Check if the API is healthy before starting load test."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API Health: {health_data.get('status', 'unknown')}")
                print(f"📊 Model Loaded: {health_data.get('model_loaded', False)}")
                return True
            else:
                print(f"❌ API Health Check Failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API Health Check Error: {str(e)}")
            return False
    
    def run_load_test(self, total_requests=100, duration_seconds=None):
        """Run the load test with specified parameters."""
        print(f"🚀 Starting MLOps API Load Test")
        print(f"📡 API URL: {self.api_url}")
        print(f"🔄 Concurrency: {self.concurrency}")
        print(f"📊 Target Requests: {total_requests}")
        print(f"⏱️ Duration: {duration_seconds or 'Until completion'} seconds")
        print("=" * 60)
        
        # Check API health first
        if not self.check_api_health():
            print("❌ API health check failed. Exiting.")
            return False
        
        start_time = time.time()
        requests_made = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = []
            
            while requests_made < total_requests:
                # Check duration limit if specified
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                # Submit batch of requests
                batch_size = min(self.concurrency, total_requests - requests_made)
                for i in range(batch_size):
                    future = executor.submit(self.make_prediction_request, requests_made + i + 1)
                    futures.append(future)
                    requests_made += 1
                
                # Wait for current batch to complete
                concurrent.futures.wait(futures[-batch_size:], timeout=30)
                
                # Add small delay between batches to create realistic load
                time.sleep(random.uniform(0.1, 0.5))
            
            # Wait for all remaining requests to complete
            concurrent.futures.wait(futures, timeout=60)
        
        total_time = time.time() - start_time
        self.print_summary(total_time, requests_made)
        return True
    
    def run_continuous_load(self, duration_seconds=300):
        """Run continuous load for monitoring demonstration."""
        print(f"🔄 Starting Continuous Load Test for {duration_seconds} seconds")
        print("📊 This will generate metrics for monitoring visualization")
        print("=" * 60)
        
        if not self.check_api_health():
            print("❌ API health check failed. Exiting.")
            return False
        
        start_time = time.time()
        request_count = 0
        
        while (time.time() - start_time) < duration_seconds:
            # Vary the load to create interesting patterns
            current_concurrency = random.randint(1, self.concurrency)
            batch_size = random.randint(1, 5)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=current_concurrency) as executor:
                futures = []
                for i in range(batch_size):
                    future = executor.submit(self.make_prediction_request, request_count + i + 1)
                    futures.append(future)
                
                concurrent.futures.wait(futures, timeout=10)
                request_count += batch_size
            
            # Variable delay to simulate realistic usage patterns
            delay = random.uniform(0.5, 3.0)
            time.sleep(delay)
            
            # Print progress every 30 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0:
                print(f"⏱️ Progress: {elapsed:.0f}s / {duration_seconds}s, Requests: {request_count}")
        
        total_time = time.time() - start_time
        self.print_summary(total_time, request_count)
        return True
    
    def print_summary(self, total_time, total_requests):
        """Print load test summary."""
        print("\n" + "=" * 60)
        print("📊 Load Test Summary")
        print("=" * 60)
        print(f"⏱️ Total Time: {total_time:.2f} seconds")
        print(f"📊 Total Requests: {total_requests}")
        print(f"✅ Successful: {self.success_count}")
        print(f"❌ Failed: {self.error_count}")
        print(f"📈 Success Rate: {(self.success_count/max(total_requests,1))*100:.1f}%")
        print(f"🚀 Requests/Second: {total_requests/total_time:.2f}")
        
        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            min_time = min(self.response_times)
            max_time = max(self.response_times)
            print(f"⚡ Avg Response Time: {avg_time:.3f}s")
            print(f"⚡ Min Response Time: {min_time:.3f}s")
            print(f"⚡ Max Response Time: {max_time:.3f}s")
        
        print("=" * 60)
        print("🎯 Check your monitoring dashboards for real-time metrics!")
        print("📊 Prometheus: http://localhost:9090")
        print("📈 Grafana: http://localhost:3000")


def main():
    parser = argparse.ArgumentParser(description='MLOps API Load Tester for Monitoring')
    parser.add_argument('--api-url', default='http://localhost:8000', 
                       help='API base URL (default: http://localhost:8000)')
    parser.add_argument('--requests', type=int, default=50, 
                       help='Total number of requests (default: 50)')
    parser.add_argument('--concurrency', type=int, default=3, 
                       help='Number of concurrent requests (default: 3)')
    parser.add_argument('--duration', type=int, 
                       help='Test duration in seconds (overrides --requests)')
    parser.add_argument('--continuous', action='store_true', 
                       help='Run continuous load test (5 minutes)')
    
    args = parser.parse_args()
    
    # Create load tester
    tester = MLOpsLoadTester(api_url=args.api_url, concurrency=args.concurrency)
    
    try:
        if args.continuous:
            success = tester.run_continuous_load(duration_seconds=300)
        elif args.duration:
            success = tester.run_load_test(total_requests=args.requests, duration_seconds=args.duration)
        else:
            success = tester.run_load_test(total_requests=args.requests)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Load test interrupted by user")
        tester.print_summary(time.time(), tester.success_count + tester.error_count)
        sys.exit(0)


if __name__ == "__main__":
    main()
