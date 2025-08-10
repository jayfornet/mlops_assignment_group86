#!/usr/bin/env python3
"""
Simple File-Based Data Monitoring (No Webhooks Required)

This script only monitors local files and doesn't require any webhook server.
Perfect for simpler setups where external integrations aren't needed.
"""

import os
import sys
import json
import hashlib
import pandas as pd
import subprocess
from datetime import datetime
from pathlib import Path


class SimpleDataMonitor:
    """Simple file-based data monitoring without webhooks."""
    
    def __init__(self, data_file="data/california_housing.csv"):
        self.data_file = Path(data_file)
        self.metadata_file = Path("data/simple_metadata.json")
        
    def calculate_file_hash(self):
        """Calculate hash of the data file."""
        if not self.data_file.exists():
            return None
            
        try:
            df = pd.read_csv(self.data_file)
            content = df.to_string()
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            print(f"Error calculating hash: {e}")
            return None
    
    def load_metadata(self):
        """Load previous metadata."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_metadata(self, metadata):
        """Save metadata."""
        self.metadata_file.parent.mkdir(exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def check_for_changes(self):
        """Check if data has changed."""
        current_hash = self.calculate_file_hash()
        if not current_hash:
            print("❌ Could not calculate file hash")
            return False, {}
        
        metadata = self.load_metadata()
        previous_hash = metadata.get('last_hash')
        
        data_changed = current_hash != previous_hash
        
        # Update metadata
        new_metadata = {
            'last_hash': current_hash,
            'last_check': datetime.now().isoformat(),
            'data_changed': data_changed,
            'file_size': os.path.getsize(self.data_file)
        }
        
        if self.data_file.exists():
            df = pd.read_csv(self.data_file)
            new_metadata.update({
                'rows': len(df),
                'columns': len(df.columns)
            })
        
        self.save_metadata(new_metadata)
        
        return data_changed, new_metadata
    
    def trigger_pipeline_if_changed(self):
        """Check for changes and trigger pipeline if needed."""
        print("🔍 Checking for data changes...")
        
        changed, metadata = self.check_for_changes()
        
        if changed:
            print("📊 Data change detected!")
            print(f"  - File: {self.data_file}")
            print(f"  - Rows: {metadata.get('rows', 'N/A')}")
            print(f"  - Size: {metadata.get('file_size', 'N/A')} bytes")
            
            # Trigger GitHub Actions (if running in CI)
            if os.getenv('GITHUB_ACTIONS'):
                print("🚀 Triggering MLOps pipeline via GitHub Actions...")
                # Set output for GitHub Actions
                with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a') as f:
                    f.write("data_changed=true\n")
            else:
                print("🏠 Running locally - would trigger pipeline here")
            
            return True
        else:
            print("📋 No changes detected")
            return False


def main():
    """Main entry point."""
    monitor = SimpleDataMonitor()
    
    # Download latest data first
    print("📥 Downloading latest dataset...")
    try:
        subprocess.run([sys.executable, "src/data/download_dataset.py"], check=True)
    except subprocess.CalledProcessError:
        print("⚠️ Download failed, checking existing data...")
    
    # Check for changes and trigger if needed
    changed = monitor.trigger_pipeline_if_changed()
    
    # Exit code for GitHub Actions
    sys.exit(0 if changed else 1)


if __name__ == "__main__":
    main()
