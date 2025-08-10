#!/usr/bin/env python3
"""
Test script for the data monitoring pipeline.

This script simulates various scenarios to test the data monitoring system:
- Data changes detection
- Quality validation
- Webhook triggers
- Pipeline integration
"""

import os
import sys
import json
import time
import requests
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class DataMonitoringTester:
    """Test suite for data monitoring functionality."""
    
    def __init__(self):
        self.base_url = "http://localhost:5555"
        self.data_dir = Path("data")
        self.test_data_file = self.data_dir / "california_housing.csv"
        
    def setup_test_environment(self):
        """Set up test environment."""
        print("🔧 Setting up test environment...")
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Create initial test dataset
        self.create_test_dataset()
        
        print("✅ Test environment ready")
    
    def create_test_dataset(self, rows=1000, seed=42):
        """Create a test dataset."""
        print(f"📊 Creating test dataset with {rows} rows...")
        
        np.random.seed(seed)
        
        data = {
            'MedInc': np.random.uniform(1, 15, rows),
            'HouseAge': np.random.uniform(1, 52, rows),
            'AveRooms': np.random.uniform(3, 10, rows),
            'AveBedrms': np.random.uniform(0.5, 2, rows),
            'Population': np.random.uniform(100, 5000, rows),
            'AveOccup': np.random.uniform(1, 6, rows),
            'Latitude': np.random.uniform(32, 42, rows),
            'Longitude': np.random.uniform(-125, -114, rows),
            'target': np.random.uniform(0.5, 5, rows)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(self.test_data_file, index=False)
        
        print(f"✅ Test dataset created: {self.test_data_file}")
    
    def modify_test_dataset(self, change_type="add_rows"):
        """Modify the test dataset to trigger change detection."""
        print(f"🔄 Modifying dataset: {change_type}")
        
        if not self.test_data_file.exists():
            self.create_test_dataset()
            return
        
        df = pd.read_csv(self.test_data_file)
        
        if change_type == "add_rows":
            # Add new rows
            new_rows = 100
            new_data = {
                'MedInc': np.random.uniform(1, 15, new_rows),
                'HouseAge': np.random.uniform(1, 52, new_rows),
                'AveRooms': np.random.uniform(3, 10, new_rows),
                'AveBedrms': np.random.uniform(0.5, 2, new_rows),
                'Population': np.random.uniform(100, 5000, new_rows),
                'AveOccup': np.random.uniform(1, 6, new_rows),
                'Latitude': np.random.uniform(32, 42, new_rows),
                'Longitude': np.random.uniform(-125, -114, new_rows),
                'target': np.random.uniform(0.5, 5, new_rows)
            }
            new_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_df], ignore_index=True)
            
        elif change_type == "modify_values":
            # Modify some existing values
            df.loc[0:50, 'MedInc'] *= 1.1
            df.loc[0:50, 'target'] *= 1.05
            
        elif change_type == "add_noise":
            # Add noise to data
            noise_factor = 0.01
            for col in ['MedInc', 'HouseAge', 'AveRooms']:
                noise = np.random.normal(0, df[col].std() * noise_factor, len(df))
                df[col] += noise
        
        # Save modified dataset
        df.to_csv(self.test_data_file, index=False)
        print(f"✅ Dataset modified: {len(df)} rows")
    
    def test_data_monitoring_script(self):
        """Test the data monitoring script directly."""
        print("\n🧪 Testing data monitoring script...")
        
        try:
            result = subprocess.run([
                sys.executable, "scripts/data_monitoring.py",
                "--config", "config/data_monitoring.json",
                "--data-source", "california_housing"
            ], capture_output=True, text=True, timeout=60)
            
            print(f"Return code: {result.returncode}")
            print(f"stdout: {result.stdout}")
            if result.stderr:
                print(f"stderr: {result.stderr}")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("❌ Script timed out")
            return False
        except Exception as e:
            print(f"❌ Error running script: {e}")
            return False
    
    def test_webhook_server_health(self):
        """Test webhook server health endpoint."""
        print("\n🧪 Testing webhook server health...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Webhook server healthy: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Cannot connect to webhook server: {e}")
            print("💡 Make sure to start the webhook server first:")
            print("   python scripts/webhook_server.py")
            return False
    
    def test_manual_webhook_trigger(self):
        """Test manual webhook trigger."""
        print("\n🧪 Testing manual webhook trigger...")
        
        payload = {
            "data_source": "california_housing",
            "force_trigger": True,
            "trigger_reason": "Test trigger from test script"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/webhook/manual-trigger",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Manual trigger successful: {data}")
                return True
            else:
                print(f"❌ Manual trigger failed: {response.status_code} - {response.text}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Error triggering webhook: {e}")
            return False
    
    def test_data_update_webhook(self):
        """Test data update webhook."""
        print("\n🧪 Testing data update webhook...")
        
        payload = {
            "data_source": "california_housing",
            "trigger_reason": "Test data update from test script",
            "force_trigger": False,
            "run_local": True
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/webhook/data-updated",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Data update webhook successful: {data}")
                return True
            else:
                print(f"❌ Data update webhook failed: {response.status_code} - {response.text}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Error with data update webhook: {e}")
            return False
    
    def test_complete_workflow(self):
        """Test complete data monitoring workflow."""
        print("\n🧪 Testing complete workflow...")
        
        # Step 1: Create initial dataset
        self.create_test_dataset(rows=500, seed=42)
        
        # Step 2: Run initial monitoring (should detect as first version)
        print("📋 Step 1: Initial monitoring run...")
        initial_success = self.test_data_monitoring_script()
        
        # Step 3: Modify dataset
        print("📋 Step 2: Modifying dataset...")
        self.modify_test_dataset("add_rows")
        
        # Step 4: Run monitoring again (should detect changes)
        print("📋 Step 3: Second monitoring run (should detect changes)...")
        change_success = self.test_data_monitoring_script()
        
        # Step 5: Test webhook trigger
        print("📋 Step 4: Testing webhook trigger...")
        webhook_success = self.test_manual_webhook_trigger()
        
        workflow_success = initial_success and change_success and webhook_success
        
        if workflow_success:
            print("✅ Complete workflow test passed!")
        else:
            print("❌ Complete workflow test failed!")
        
        return workflow_success
    
    def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting Data Monitoring Tests")
        print("=" * 50)
        
        # Setup
        self.setup_test_environment()
        
        # Test results
        results = {}
        
        # Test 1: Data monitoring script
        results["monitoring_script"] = self.test_data_monitoring_script()
        
        # Test 2: Webhook server health
        results["webhook_health"] = self.test_webhook_server_health()
        
        # Test 3: Manual webhook trigger (only if server is running)
        if results["webhook_health"]:
            results["manual_trigger"] = self.test_manual_webhook_trigger()
            results["data_update_webhook"] = self.test_data_update_webhook()
        else:
            results["manual_trigger"] = False
            results["data_update_webhook"] = False
        
        # Test 4: Complete workflow
        results["complete_workflow"] = self.test_complete_workflow()
        
        # Results summary
        print("\n📊 Test Results Summary")
        print("=" * 50)
        
        passed = 0
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name:<20}: {status}")
            if success:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
            return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Data Monitoring Pipeline")
    parser.add_argument("--webhook-url", default="http://localhost:5555",
                       help="Webhook server URL")
    parser.add_argument("--test", choices=["all", "script", "webhook", "workflow"],
                       default="all", help="Which tests to run")
    
    args = parser.parse_args()
    
    tester = DataMonitoringTester()
    tester.base_url = args.webhook_url
    
    if args.test == "all":
        success = tester.run_all_tests()
    elif args.test == "script":
        tester.setup_test_environment()
        success = tester.test_data_monitoring_script()
    elif args.test == "webhook":
        success = (tester.test_webhook_server_health() and 
                  tester.test_manual_webhook_trigger())
    elif args.test == "workflow":
        tester.setup_test_environment()
        success = tester.test_complete_workflow()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
