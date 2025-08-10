#!/usr/bin/env python3
"""
Data Monitoring Script for MLOps Pipeline

This script monitors data sources for changes and can trigger the MLOps pipeline
when new data versions are detected.

Features:
- Automatic data change detection
- Data quality validation
- Version management
- External trigger support via webhooks
- Integration with GitHub Actions
"""

import os
import sys
import json
import hashlib
import logging
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import subprocess


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataMonitor:
    """Monitors data sources and triggers MLOps pipeline on changes."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.data_dir = Path("data")
        self.metadata_file = self.data_dir / "data_metadata.json"
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration."""
        default_config = {
            "data_sources": {
                "california_housing": {
                    "file_path": "data/california_housing.csv",
                    "check_interval": 3600,  # 1 hour
                    "quality_checks": True,
                    "auto_trigger": True
                }
            },
            "triggers": {
                "github_actions": {
                    "enabled": True,
                    "workflow": "mlops-pipeline.yml",
                    "repository": "jayfornet/mlops_assignment_group86"
                },
                "webhook": {
                    "enabled": False,
                    "url": None,
                    "secret": None
                }
            },
            "validation": {
                "min_rows": 100,
                "required_columns": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
                                   "Population", "AveOccup", "Latitude", "Longitude"],
                "latitude_range": [30, 45],
                "longitude_range": [-130, -110]
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def calculate_data_hash(self, file_path: str) -> Optional[str]:
        """Calculate hash of the dataset for change detection."""
        if not os.path.exists(file_path):
            return None
        
        try:
            df = pd.read_csv(file_path)
            # Create a hash based on data content and structure
            data_str = df.to_string(index=False) + str(df.dtypes.to_dict())
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load existing data metadata."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}
    
    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save data metadata."""
        try:
            self.data_dir.mkdir(exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def detect_data_changes(self, data_source: str) -> Tuple[bool, Dict[str, Any]]:
        """Detect changes in the specified data source."""
        config = self.config["data_sources"][data_source]
        file_path = config["file_path"]
        
        logger.info(f"Checking for changes in {data_source}")
        
        # Calculate current hash
        current_hash = self.calculate_data_hash(file_path)
        if current_hash is None:
            logger.warning(f"Could not calculate hash for {file_path}")
            return False, {}
        
        # Load previous metadata
        metadata = self.load_metadata()
        previous_hash = metadata.get(data_source, {}).get("hash")
        previous_version = metadata.get(data_source, {}).get("version", "v1.0.0")
        
        # Check if data changed
        data_changed = current_hash != previous_hash
        
        # Generate new version if data changed
        if data_changed:
            version_parts = previous_version.replace('v', '').split('.')
            new_version = f"v{version_parts[0]}.{int(version_parts[1]) + 1}.0"
        else:
            new_version = previous_version
        
        # Analyze dataset
        df = pd.read_csv(file_path)
        
        change_info = {
            "data_changed": data_changed,
            "hash": current_hash,
            "previous_hash": previous_hash,
            "version": new_version,
            "previous_version": previous_version,
            "rows": len(df),
            "columns": len(df.columns),
            "file_size": os.path.getsize(file_path),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            "check_timestamp": datetime.now().isoformat()
        }
        
        # Update metadata
        if data_source not in metadata:
            metadata[data_source] = {}
        metadata[data_source].update(change_info)
        self.save_metadata(metadata)
        
        logger.info(f"Data change detection results for {data_source}:")
        logger.info(f"  Changed: {data_changed}")
        logger.info(f"  Version: {previous_version} -> {new_version}")
        logger.info(f"  Rows: {change_info['rows']}")
        logger.info(f"  Hash: {current_hash}")
        
        return data_changed, change_info
    
    def validate_data_quality(self, data_source: str) -> Dict[str, Any]:
        """Perform data quality validation."""
        config = self.config["data_sources"][data_source]
        file_path = config["file_path"]
        validation_config = self.config["validation"]
        
        logger.info(f"Validating data quality for {data_source}")
        
        try:
            df = pd.read_csv(file_path)
            
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "stats": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "missing_values": int(df.isnull().sum().sum()),
                    "duplicate_rows": int(df.duplicated().sum()),
                    "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
                }
            }
            
            # Check minimum rows
            if len(df) < validation_config["min_rows"]:
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"Dataset too small: {len(df)} < {validation_config['min_rows']}"
                )
            
            # Check required columns
            missing_cols = set(validation_config["required_columns"]) - set(df.columns)
            if missing_cols:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Missing columns: {missing_cols}")
            
            # Check geographic bounds
            if 'Latitude' in df.columns:
                lat_range = (float(df['Latitude'].min()), float(df['Latitude'].max()))
                expected_lat = validation_config["latitude_range"]
                if lat_range[0] < expected_lat[0] or lat_range[1] > expected_lat[1]:
                    validation_results["warnings"].append(
                        f"Latitude range {lat_range} outside expected {expected_lat}"
                    )
            
            if 'Longitude' in df.columns:
                lon_range = (float(df['Longitude'].min()), float(df['Longitude'].max()))
                expected_lon = validation_config["longitude_range"]
                if lon_range[0] < expected_lon[0] or lon_range[1] > expected_lon[1]:
                    validation_results["warnings"].append(
                        f"Longitude range {lon_range} outside expected {expected_lon}"
                    )
            
            # Check for excessive missing data
            missing_percentage = (validation_results["stats"]["missing_values"] / 
                                (len(df) * len(df.columns))) * 100
            if missing_percentage > 10:
                validation_results["warnings"].append(
                    f"High missing data percentage: {missing_percentage:.1f}%"
                )
            
            logger.info("Data quality validation results:")
            logger.info(f"  Valid: {validation_results['valid']}")
            logger.info(f"  Errors: {len(validation_results['errors'])}")
            logger.info(f"  Warnings: {len(validation_results['warnings'])}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "stats": {}
            }
    
    def trigger_mlops_pipeline(self, data_source: str, change_info: Dict[str, Any]) -> bool:
        """Trigger the MLOps pipeline via GitHub Actions."""
        trigger_config = self.config["triggers"]["github_actions"]
        
        if not trigger_config["enabled"]:
            logger.info("GitHub Actions trigger disabled")
            return False
        
        # Check if we have GitHub token
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            logger.warning("GITHUB_TOKEN not available, cannot trigger pipeline")
            return False
        
        repository = trigger_config["repository"]
        workflow = trigger_config["workflow"]
        
        # Prepare dispatch payload
        payload = {
            "event_type": "new-data-version",
            "client_payload": {
                "data_source": data_source,
                "data_version": change_info["version"],
                "trigger_reason": "Data change detected by monitoring script",
                "change_info": change_info,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        try:
            # Trigger via repository dispatch
            url = f"https://api.github.com/repos/{repository}/dispatches"
            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 204:
                logger.info("✅ MLOps pipeline triggered successfully via GitHub Actions")
                return True
            else:
                logger.error(f"Failed to trigger pipeline: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering MLOps pipeline: {e}")
            return False
    
    def send_webhook_notification(self, data_source: str, change_info: Dict[str, Any]) -> bool:
        """Send webhook notification about data changes."""
        webhook_config = self.config["triggers"]["webhook"]
        
        if not webhook_config["enabled"] or not webhook_config["url"]:
            return False
        
        payload = {
            "event": "data_change_detected",
            "data_source": data_source,
            "change_info": change_info,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(
                webhook_config["url"],
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("✅ Webhook notification sent successfully")
                return True
            else:
                logger.error(f"Webhook failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return False
    
    def monitor_data_sources(self) -> Dict[str, Any]:
        """Monitor all configured data sources."""
        results = {}
        
        for data_source in self.config["data_sources"]:
            logger.info(f"Monitoring data source: {data_source}")
            
            # Check for data changes
            data_changed, change_info = self.detect_data_changes(data_source)
            
            source_results = {
                "data_changed": data_changed,
                "change_info": change_info,
                "validation": None,
                "pipeline_triggered": False,
                "webhook_sent": False
            }
            
            if data_changed:
                logger.info(f"📊 Data change detected in {data_source}")
                
                # Validate data quality
                if self.config["data_sources"][data_source].get("quality_checks", True):
                    validation_results = self.validate_data_quality(data_source)
                    source_results["validation"] = validation_results
                    
                    if not validation_results["valid"]:
                        logger.error(f"❌ Data validation failed for {data_source}")
                        logger.error(f"Errors: {validation_results['errors']}")
                        continue
                
                # Trigger pipeline if auto-trigger is enabled
                if self.config["data_sources"][data_source].get("auto_trigger", True):
                    pipeline_triggered = self.trigger_mlops_pipeline(data_source, change_info)
                    source_results["pipeline_triggered"] = pipeline_triggered
                
                # Send webhook notification
                webhook_sent = self.send_webhook_notification(data_source, change_info)
                source_results["webhook_sent"] = webhook_sent
                
            else:
                logger.info(f"📋 No changes detected in {data_source}")
            
            results[data_source] = source_results
        
        return results
    
    def download_latest_data(self, data_source: str = "california_housing") -> bool:
        """Download the latest version of the dataset."""
        logger.info(f"Downloading latest data for {data_source}")
        
        try:
            # Run the download script
            script_path = "src/data/download_dataset.py"
            if os.path.exists(script_path):
                result = subprocess.run([sys.executable, script_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("✅ Data download completed successfully")
                    return True
                else:
                    logger.error(f"Data download failed: {result.stderr}")
                    return False
            else:
                logger.warning(f"Download script not found: {script_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            return False


def main():
    """Main entry point for the data monitoring script."""
    parser = argparse.ArgumentParser(description="Data Monitoring for MLOps Pipeline")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--download", "-d", action="store_true", 
                       help="Download latest data before monitoring")
    parser.add_argument("--force-trigger", "-f", action="store_true",
                       help="Force trigger pipeline even if no changes detected")
    parser.add_argument("--data-source", "-s", default="california_housing",
                       help="Data source to monitor")
    parser.add_argument("--validate-only", "-v", action="store_true",
                       help="Only perform data validation, don't trigger pipeline")
    
    args = parser.parse_args()
    
    try:
        # Initialize monitor
        monitor = DataMonitor(args.config)
        
        # Download latest data if requested
        if args.download:
            if not monitor.download_latest_data(args.data_source):
                logger.error("Failed to download latest data")
                sys.exit(1)
        
        # Monitor data sources
        if args.validate_only:
            # Only perform validation
            validation_results = monitor.validate_data_quality(args.data_source)
            print(json.dumps(validation_results, indent=2))
            if not validation_results["valid"]:
                sys.exit(1)
        else:
            # Full monitoring
            results = monitor.monitor_data_sources()
            
            # Force trigger if requested
            if args.force_trigger:
                logger.info("🔄 Force trigger requested")
                change_info = {"version": "manual", "data_changed": True, 
                             "trigger_reason": "Manual force trigger"}
                monitor.trigger_mlops_pipeline(args.data_source, change_info)
            
            # Print results
            print("\n📊 Data Monitoring Results:")
            print("=" * 50)
            for source, result in results.items():
                print(f"\n{source}:")
                print(f"  Changed: {result['data_changed']}")
                if result['data_changed']:
                    print(f"  Version: {result['change_info']['version']}")
                    print(f"  Pipeline Triggered: {result['pipeline_triggered']}")
                    if result['validation']:
                        print(f"  Validation: {'✅ Passed' if result['validation']['valid'] else '❌ Failed'}")
            
            # Exit with error if any critical issues
            for result in results.values():
                if result['validation'] and not result['validation']['valid']:
                    sys.exit(1)
        
        logger.info("✅ Data monitoring completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Data monitoring interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Data monitoring failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
