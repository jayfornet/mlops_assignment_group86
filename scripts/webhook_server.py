#!/usr/bin/env python3
"""
Data Webhook Server for MLOps Pipeline

This Flask server receives webhook notifications about data changes
and triggers the data monitoring pipeline accordingly.

Features:
- Secure webhook endpoints with authentication
- Support for multiple data sources
- Validation of incoming payloads
- Integration with GitHub Actions
- Monitoring and logging
"""

import os
import json
import hmac
import hashlib
import logging
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, abort
import requests


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class WebhookServer:
    """Handles webhook requests for data monitoring."""
    
    def __init__(self):
        self.webhook_secret = os.getenv('WEBHOOK_SECRET', 'default-secret-change-me')
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repository = os.getenv('GITHUB_REPOSITORY', 'jayfornet/mlops_assignment_group86')
        
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature for security."""
        if not signature:
            return False
            
        try:
            expected_signature = hmac.new(
                self.webhook_secret.encode(),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures safely
            return hmac.compare_digest(f"sha256={expected_signature}", signature)
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False
    
    def trigger_github_workflow(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """Trigger GitHub Actions workflow via repository dispatch."""
        if not self.github_token:
            logger.error("GitHub token not available")
            return False
        
        url = f"https://api.github.com/repos/{self.repository}/dispatches"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        dispatch_payload = {
            "event_type": event_type,
            "client_payload": payload
        }
        
        try:
            response = requests.post(url, json=dispatch_payload, headers=headers, timeout=30)
            
            if response.status_code == 204:
                logger.info(f"✅ GitHub workflow triggered: {event_type}")
                return True
            else:
                logger.error(f"GitHub API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering GitHub workflow: {e}")
            return False
    
    def run_local_monitoring(self, data_source: str, force: bool = False) -> Dict[str, Any]:
        """Run local data monitoring script."""
        try:
            script_path = "scripts/data_monitoring.py"
            cmd = [sys.executable, script_path, "--data-source", data_source]
            
            if force:
                cmd.append("--force-trigger")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Data monitoring script timed out")
            return {"success": False, "error": "Script timeout"}
        except Exception as e:
            logger.error(f"Error running monitoring script: {e}")
            return {"success": False, "error": str(e)}


webhook_server = WebhookServer()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


@app.route('/webhook/data-updated', methods=['POST'])
def handle_data_updated():
    """Handle webhook notifications for data updates."""
    try:
        # Verify signature if provided
        signature = request.headers.get('X-Hub-Signature-256')
        if signature:
            if not webhook_server.verify_signature(request.data, signature):
                logger.warning("Invalid webhook signature")
                abort(401)
        
        # Parse payload
        payload = request.get_json()
        if not payload:
            abort(400, "Invalid JSON payload")
        
        # Extract data source information
        data_source = payload.get('data_source', 'california_housing')
        force_trigger = payload.get('force_trigger', False)
        trigger_reason = payload.get('trigger_reason', 'External webhook trigger')
        
        logger.info(f"Received data update webhook for: {data_source}")
        logger.info(f"Trigger reason: {trigger_reason}")
        
        # Prepare response data
        response_data = {
            "status": "received",
            "data_source": data_source,
            "timestamp": datetime.now().isoformat(),
            "trigger_reason": trigger_reason
        }
        
        # Trigger GitHub Actions workflow
        github_payload = {
            "data_source": data_source,
            "trigger_reason": trigger_reason,
            "force_trigger": force_trigger,
            "webhook_payload": payload,
            "timestamp": datetime.now().isoformat()
        }
        
        workflow_triggered = webhook_server.trigger_github_workflow(
            "external-data-trigger", 
            github_payload
        )
        
        response_data["github_workflow_triggered"] = workflow_triggered
        
        # Optionally run local monitoring
        if payload.get('run_local', False):
            monitoring_result = webhook_server.run_local_monitoring(data_source, force_trigger)
            response_data["local_monitoring"] = monitoring_result
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error handling webhook: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/webhook/kaggle-update', methods=['POST'])
def handle_kaggle_update():
    """Handle webhook notifications from Kaggle dataset updates."""
    try:
        payload = request.get_json()
        if not payload:
            abort(400, "Invalid JSON payload")
        
        # Extract Kaggle-specific information
        dataset_id = payload.get('dataset_id')
        version = payload.get('version')
        
        logger.info(f"Received Kaggle update webhook for dataset: {dataset_id}, version: {version}")
        
        # Trigger data monitoring with download
        github_payload = {
            "data_source": "california_housing",
            "trigger_reason": f"Kaggle dataset update: {dataset_id} v{version}",
            "kaggle_dataset_id": dataset_id,
            "kaggle_version": version,
            "download_latest": True,
            "timestamp": datetime.now().isoformat()
        }
        
        workflow_triggered = webhook_server.trigger_github_workflow(
            "new-data-version",
            github_payload
        )
        
        return jsonify({
            "status": "processed",
            "dataset_id": dataset_id,
            "version": version,
            "workflow_triggered": workflow_triggered,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error handling Kaggle webhook: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/webhook/manual-trigger', methods=['POST'])
def handle_manual_trigger():
    """Handle manual webhook triggers (for testing and debugging)."""
    try:
        payload = request.get_json() or {}
        
        data_source = payload.get('data_source', 'california_housing')
        force_trigger = payload.get('force_trigger', True)
        
        logger.info(f"Manual trigger requested for: {data_source}")
        
        # Run both local monitoring and GitHub trigger
        monitoring_result = webhook_server.run_local_monitoring(data_source, force_trigger)
        
        github_payload = {
            "data_source": data_source,
            "trigger_reason": "Manual webhook trigger",
            "force_trigger": force_trigger,
            "timestamp": datetime.now().isoformat()
        }
        
        workflow_triggered = webhook_server.trigger_github_workflow(
            "external-data-trigger",
            github_payload
        )
        
        return jsonify({
            "status": "completed",
            "data_source": data_source,
            "local_monitoring": monitoring_result,
            "github_workflow_triggered": workflow_triggered,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error handling manual trigger: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/webhook/status', methods=['GET'])
def webhook_status():
    """Get webhook server status and configuration."""
    return jsonify({
        "status": "active",
        "endpoints": [
            "/webhook/data-updated",
            "/webhook/kaggle-update", 
            "/webhook/manual-trigger"
        ],
        "github_repository": webhook_server.repository,
        "github_token_configured": bool(webhook_server.github_token),
        "webhook_secret_configured": bool(webhook_server.webhook_secret != 'default-secret-change-me'),
        "timestamp": datetime.now().isoformat()
    })


@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "message": str(error)}), 400


@app.errorhandler(401)
def unauthorized(error):
    return jsonify({"error": "Unauthorized", "message": "Invalid signature"}), 401


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "message": str(error)}), 500


def main():
    """Main entry point for the webhook server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Webhook Server for MLOps")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5555, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    logger.info(f"Starting webhook server on {args.host}:{args.port}")
    logger.info(f"GitHub repository: {webhook_server.repository}")
    logger.info(f"GitHub token configured: {bool(webhook_server.github_token)}")
    
    # Run the Flask app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == "__main__":
    main()
