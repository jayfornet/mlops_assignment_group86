#!/usr/bin/env python3
"""
Simple Data Update Simulator for MLOps Assignment

This script simulates data updates and can trigger the MLOps pipeline.
Perfect for assignment demonstrations without complex infrastructure.

Usage:
    python update_data_and_trigger.py                    # Check for changes
    python update_data_and_trigger.py --simulate-update  # Simulate new data
    python update_data_and_trigger.py --force-trigger    # Force trigger pipeline
"""

import os
import sys
import json
import hashlib
import argparse
import pandas as pd
import numpy as np
import subprocess
from datetime import datetime
from pathlib import Path


def create_sample_data(rows=1000, seed=None):
    """Create sample California housing data."""
    if seed:
        np.random.seed(seed)
    
    print(f"🏠 Creating sample housing data with {rows} rows...")
    
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
    
    return pd.DataFrame(data)


def simulate_data_update(data_file="data/california_housing.csv", update_type="add_rows"):
    """Simulate different types of data updates."""
    print(f"📊 Simulating data update: {update_type}")
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Load existing data or create new
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        print(f"📋 Loaded existing data: {len(df)} rows")
    else:
        df = create_sample_data(500, seed=42)
        print(f"📋 Created new dataset: {len(df)} rows")
    
    if update_type == "add_rows":
        # Add new rows (simulates new data arrival)
        new_rows = 50
        new_data = create_sample_data(new_rows, seed=int(datetime.now().timestamp()))
        df = pd.concat([df, new_data], ignore_index=True)
        print(f"➕ Added {new_rows} new rows")
        
    elif update_type == "modify_values":
        # Modify some existing values (simulates data corrections)
        n_modify = min(100, len(df) // 10)
        indices = np.random.choice(len(df), n_modify, replace=False)
        df.loc[indices, 'MedInc'] *= np.random.uniform(0.95, 1.05, n_modify)
        print(f"🔧 Modified {n_modify} rows")
        
    elif update_type == "add_noise":
        # Add small noise to simulate sensor updates
        for col in ['MedInc', 'HouseAge', 'target']:
            noise = np.random.normal(0, df[col].std() * 0.01, len(df))
            df[col] += noise
        print(f"🌊 Added noise to simulate sensor updates")
    
    # Save updated data
    df.to_csv(data_file, index=False)
    print(f"💾 Saved updated dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return len(df)


def check_data_changes(data_file="data/california_housing.csv"):
    """Check if data has changed since last check."""
    print("🔍 Checking for data changes...")
    
    if not os.path.exists(data_file):
        print("❌ Data file not found!")
        return False, {}
    
    # Calculate current hash
    df = pd.read_csv(data_file)
    current_hash = hashlib.md5(df.to_string().encode()).hexdigest()
    
    # Check previous hash
    hash_file = "data/last_hash.txt"
    previous_hash = ""
    
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            previous_hash = f.read().strip()
    
    data_changed = current_hash != previous_hash
    
    # Save current hash
    with open(hash_file, 'w') as f:
        f.write(current_hash)
    
    info = {
        "current_hash": current_hash,
        "previous_hash": previous_hash,
        "data_changed": data_changed,
        "rows": len(df),
        "columns": len(df.columns),
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"📊 Data Analysis:")
    print(f"  - Changed: {data_changed}")
    print(f"  - Rows: {len(df)}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Current Hash: {current_hash[:8]}...")
    
    return data_changed, info


def trigger_github_workflow():
    """Trigger the MLOps pipeline via GitHub CLI or API."""
    print("🚀 Triggering MLOps Pipeline...")
    
    # Method 1: Try GitHub CLI (if available)
    try:
        result = subprocess.run([
            'gh', 'workflow', 'run', 'mlops-pipeline.yml',
            '--field', 'triggered_by=data-update-script',
            '--field', 'trigger_reason=Data updated by simulation script'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Pipeline triggered via GitHub CLI")
            return True
        else:
            print(f"⚠️ GitHub CLI failed: {result.stderr}")
    except FileNotFoundError:
        print("⚠️ GitHub CLI not available")
    
    # Method 2: Manual instructions
    print("\n🔧 Manual Trigger Instructions:")
    print("1. Go to: https://github.com/jayfornet/mlops_assignment_group86/actions")
    print("2. Click on 'MLOps Pipeline - California Housing Prediction'")
    print("3. Click 'Run workflow' button")
    print("4. Fill in:")
    print("   - triggered_by: data-update-script")
    print("   - trigger_reason: Data updated by simulation script")
    print("5. Click 'Run workflow'")
    
    return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple Data Update and Trigger Tool")
    parser.add_argument("--simulate-update", choices=["add_rows", "modify_values", "add_noise"],
                       help="Simulate a data update")
    parser.add_argument("--force-trigger", action="store_true",
                       help="Force trigger pipeline even if no changes")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check for changes, don't trigger")
    
    args = parser.parse_args()
    
    print("🚀 Simple Data Update and Trigger Tool")
    print("=" * 50)
    
    # Simulate data update if requested
    if args.simulate_update:
        rows = simulate_data_update(update_type=args.simulate_update)
        print(f"✅ Data simulation complete: {rows} total rows")
    
    # Check for changes
    changed, info = check_data_changes()
    
    # Decide whether to trigger
    should_trigger = args.force_trigger or (changed and not args.check_only)
    
    if should_trigger:
        print(f"\n🎯 Triggering MLOps Pipeline...")
        print(f"Reason: {'Force trigger' if args.force_trigger else 'Data changed'}")
        
        # Save trigger info
        trigger_info = {
            **info,
            "trigger_reason": "Force trigger" if args.force_trigger else "Data changed",
            "triggered_by": "data-update-script"
        }
        
        with open("data/trigger_info.json", "w") as f:
            json.dump(trigger_info, f, indent=2)
        
        success = trigger_github_workflow()
        
        if success:
            print("✅ Pipeline triggered successfully!")
        else:
            print("⚠️ Use manual trigger instructions above")
    else:
        if args.check_only:
            print("📋 Check complete - no trigger requested")
        else:
            print("📋 No changes detected - no trigger needed")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"Data Changed: {changed}")
    print(f"Should Trigger: {should_trigger}")
    print(f"Timestamp: {datetime.now()}")


if __name__ == "__main__":
    main()
