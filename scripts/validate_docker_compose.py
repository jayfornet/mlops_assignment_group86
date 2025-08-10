#!/usr/bin/env python3
"""
Docker Compose Validation Script

This script validates the Docker Compose configuration for the monitoring stack.
"""

import subprocess
import sys
import os


def check_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker available: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker not available")
            return False
    except FileNotFoundError:
        print("❌ Docker not found")
        return False


def check_docker_compose():
    """Check if Docker Compose is available."""
    # Try modern docker compose first
    try:
        result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker Compose (modern) available: {result.stdout.strip()}")
            return 'modern'
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    
    # Try legacy docker-compose
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker Compose (legacy) available: {result.stdout.strip()}")
            return 'legacy'
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    
    print("❌ Docker Compose not available")
    return None


def validate_compose_file():
    """Validate the Docker Compose configuration."""
    if not os.path.exists('docker-compose.monitoring.yml'):
        print("❌ docker-compose.monitoring.yml not found")
        return False
    
    print("📁 Found docker-compose.monitoring.yml")
    
    # Check which compose command to use
    compose_type = check_docker_compose()
    if not compose_type:
        return False
    
    try:
        if compose_type == 'modern':
            cmd = ['docker', 'compose', '-f', 'docker-compose.monitoring.yml', 'config']
        else:
            cmd = ['docker-compose', '-f', 'docker-compose.monitoring.yml', 'config']
        
        print(f"🔍 Validating configuration with: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Docker Compose configuration is valid")
            return True
        else:
            print(f"❌ Docker Compose validation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error validating Docker Compose: {e}")
        return False


def show_services():
    """Show the services defined in the compose file."""
    compose_type = check_docker_compose()
    if not compose_type:
        return
    
    try:
        if compose_type == 'modern':
            cmd = ['docker', 'compose', '-f', 'docker-compose.monitoring.yml', 'config', '--services']
        else:
            cmd = ['docker-compose', '-f', 'docker-compose.monitoring.yml', 'config', '--services']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            services = result.stdout.strip().split('\n')
            print(f"📊 Services defined in monitoring stack:")
            for service in services:
                if service.strip():
                    print(f"  - {service}")
        else:
            print(f"⚠️ Could not list services: {result.stderr}")
            
    except Exception as e:
        print(f"⚠️ Error listing services: {e}")


def main():
    """Main validation function."""
    print("🔧 Docker Compose Validation for MLOps Monitoring")
    print("=" * 60)
    
    # Check Docker
    if not check_docker():
        sys.exit(1)
    
    # Check Docker Compose
    if not check_docker_compose():
        sys.exit(1)
    
    # Validate compose file
    if not validate_compose_file():
        sys.exit(1)
    
    # Show services
    show_services()
    
    print("=" * 60)
    print("✅ All Docker Compose validations passed!")
    print("")
    print("🚀 Ready to start monitoring stack:")
    
    compose_type = check_docker_compose()
    if compose_type == 'modern':
        print("   docker compose -f docker-compose.monitoring.yml up -d")
    else:
        print("   docker-compose -f docker-compose.monitoring.yml up -d")


if __name__ == "__main__":
    main()
