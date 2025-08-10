#!/usr/bin/env python3
"""
MLOps Monitoring Setup Script

This script helps set up and manage the monitoring stack for the MLOps pipeline.
"""

import subprocess
import time
import sys
import os
import requests
import json


class MonitoringSetup:
    def __init__(self):
        self.services = {
            'prometheus': {'port': 9090, 'path': '/-/healthy'},
            'grafana': {'port': 3000, 'path': '/api/health'},
            'node-exporter': {'port': 9100, 'path': '/metrics'},
            'cadvisor': {'port': 8080, 'path': '/metrics'}
        }
        
    def check_docker(self):
        """Check if Docker is running."""
        try:
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def check_docker_compose(self):
        """Check if Docker Compose is available."""
        try:
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def start_monitoring_stack(self):
        """Start the monitoring stack using Docker Compose."""
        print("🚀 Starting monitoring stack...")
        
        if not self.check_docker():
            print("❌ Docker is not running. Please start Docker first.")
            return False
        
        if not self.check_docker_compose():
            print("❌ Docker Compose is not available.")
            return False
        
        if not os.path.exists('docker-compose.monitoring.yml'):
            print("❌ docker-compose.monitoring.yml not found.")
            return False
        
        try:
            # Start the stack
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.monitoring.yml', 'up', '-d'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to start monitoring stack: {result.stderr}")
                return False
            
            print("✅ Monitoring stack started successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error starting monitoring stack: {e}")
            return False
    
    def stop_monitoring_stack(self):
        """Stop the monitoring stack."""
        print("🛑 Stopping monitoring stack...")
        
        try:
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.monitoring.yml', 'down'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Monitoring stack stopped successfully!")
                return True
            else:
                print(f"❌ Error stopping monitoring stack: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error stopping monitoring stack: {e}")
            return False
    
    def check_service_health(self, service_name, max_retries=30):
        """Check if a service is healthy."""
        service = self.services.get(service_name)
        if not service:
            return False
        
        url = f"http://localhost:{service['port']}{service['path']}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            time.sleep(2)
        
        return False
    
    def wait_for_services(self):
        """Wait for all services to be healthy."""
        print("⏳ Waiting for services to start...")
        
        for service_name in self.services.keys():
            print(f"   Checking {service_name}...")
            if self.check_service_health(service_name):
                print(f"   ✅ {service_name} is healthy")
            else:
                print(f"   ❌ {service_name} is not responding")
                return False
        
        return True
    
    def show_service_status(self):
        """Show the status of all monitoring services."""
        print("📊 Monitoring Services Status:")
        print("=" * 50)
        
        for service_name, service in self.services.items():
            if self.check_service_health(service_name, max_retries=1):
                status = "🟢 Running"
                url = f"http://localhost:{service['port']}"
            else:
                status = "🔴 Not Running"
                url = "N/A"
            
            print(f"{service_name:15} {status:12} {url}")
        
        print("=" * 50)
    
    def show_access_info(self):
        """Show access information for monitoring services."""
        print("\n🌐 Access Information:")
        print("=" * 50)
        print("📊 Prometheus:  http://localhost:9090")
        print("📈 Grafana:     http://localhost:3000")
        print("   └─ Login:    admin / admin123")
        print("🖥️  Node Exporter: http://localhost:9100")
        print("📦 cAdvisor:    http://localhost:8080")
        print("")
        print("🎯 MLOps API should be running on:")
        print("🏠 API:         http://localhost:8000")
        print("❤️  API Health:  http://localhost:8000/health")
        print("📊 API Metrics: http://localhost:8000/metrics")
        print("=" * 50)
    
    def check_api_connectivity(self):
        """Check if the MLOps API is accessible."""
        print("\n🔍 Checking MLOps API connectivity...")
        
        try:
            # Check health endpoint
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                print("✅ MLOps API is healthy")
                
                # Check metrics endpoint
                response = requests.get('http://localhost:8000/metrics', timeout=5)
                if response.status_code == 200:
                    print("✅ MLOps API metrics endpoint is accessible")
                    return True
                else:
                    print("⚠️  MLOps API metrics endpoint not accessible")
                    return False
            else:
                print("❌ MLOps API health check failed")
                return False
                
        except Exception as e:
            print(f"❌ MLOps API is not accessible: {e}")
            print("💡 Make sure to start the MLOps API first:")
            print("   docker run -d --name housing-api -p 8000:8000 <your-username>/housing-prediction-api:latest")
            return False
    
    def setup_complete_monitoring(self):
        """Complete monitoring setup process."""
        print("🔧 MLOps Monitoring Setup")
        print("=" * 50)
        
        # Step 1: Start monitoring stack
        if not self.start_monitoring_stack():
            return False
        
        # Step 2: Wait for services
        if not self.wait_for_services():
            print("❌ Some services failed to start properly")
            return False
        
        # Step 3: Check API connectivity
        api_accessible = self.check_api_connectivity()
        
        # Step 4: Show status and access info
        self.show_service_status()
        self.show_access_info()
        
        print("\n🎉 Monitoring setup complete!")
        
        if api_accessible:
            print("✅ Ready to monitor your MLOps API!")
            print("\n💡 Next steps:")
            print("1. Open Grafana: http://localhost:3000")
            print("2. Login with admin/admin123")
            print("3. View the MLOps dashboard")
            print("4. Make some API predictions to see metrics")
        else:
            print("⚠️  MLOps API not detected. Start it first, then monitoring will work!")
        
        return True
    
    def run_load_test(self):
        """Run a load test to generate monitoring data."""
        print("🧪 Running load test to generate monitoring data...")
        
        try:
            result = subprocess.run([
                sys.executable, 'scripts/load_test_monitoring.py', 
                '--requests', '20', '--concurrency', '2'
            ], text=True)
            
            if result.returncode == 0:
                print("✅ Load test completed successfully!")
            else:
                print("⚠️  Load test had some issues, but may have generated useful data")
                
        except Exception as e:
            print(f"❌ Error running load test: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MLOps Monitoring Setup')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'setup', 'test'],
                       help='Action to perform')
    
    args = parser.parse_args()
    
    setup = MonitoringSetup()
    
    if args.action == 'start':
        setup.start_monitoring_stack()
        setup.wait_for_services()
        setup.show_access_info()
        
    elif args.action == 'stop':
        setup.stop_monitoring_stack()
        
    elif args.action == 'status':
        setup.show_service_status()
        setup.show_access_info()
        
    elif args.action == 'setup':
        setup.setup_complete_monitoring()
        
    elif args.action == 'test':
        if setup.check_service_health('prometheus', max_retries=1):
            setup.run_load_test()
        else:
            print("❌ Monitoring stack not running. Use 'setup' first.")


if __name__ == "__main__":
    main()
