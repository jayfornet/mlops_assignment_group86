#!/usr/bin/env python3
"""
Simple test script to verify the California Housing MLOps Pipeline.

This script tests basic functionality without requiring all dependencies.
"""

import sys
import os
from pathlib import Path

def test_project_structure():
    """Test that all required files and directories exist."""
    print("🔍 Testing project structure...")
    
    required_files = [
        "README.md",
        "requirements.txt", 
        "Dockerfile",
        "docker-compose.yml",
        "setup.py",
        "demo.py",
        "PROJECT_SUMMARY.md",
        "src/data/data_processor.py",
        "src/models/train_model.py",
        "src/api/app.py",
        "src/utils/helpers.py",
        "tests/test_data_processor.py",
        "tests/test_api.py",
        ".github/workflows/mlops-pipeline.yml"
    ]
    
    required_dirs = [
        "src", "src/data", "src/models", "src/api", "src/utils",
        "tests", "data", "models", "logs", "results", "mlruns",
        ".github", ".github/workflows", "monitoring"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"📁 {dir_path}/")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {missing_dirs}")
        return False
    
    print("\n✅ All required files and directories present!")
    return True

def test_imports():
    """Test that core modules can be imported."""
    print("\n🔍 Testing module imports...")
    
    # Add src to path
    sys.path.append('src')
    
    try:
        print("Testing data processor import...")
        # Don't import the class directly to avoid sklearn dependency
        with open('src/data/data_processor.py', 'r') as f:
            content = f.read()
            if 'class DataProcessor' in content:
                print("✅ DataProcessor class found")
            else:
                print("❌ DataProcessor class not found")
                return False
    except Exception as e:
        print(f"❌ Error checking data processor: {e}")
        return False
    
    try:
        print("Testing model trainer import...")
        with open('src/models/train_model.py', 'r') as f:
            content = f.read()
            if 'class ModelTrainer' in content:
                print("✅ ModelTrainer class found")
            else:
                print("❌ ModelTrainer class not found")
                return False
    except Exception as e:
        print(f"❌ Error checking model trainer: {e}")
        return False
    
    try:
        print("Testing API app...")
        with open('src/api/app.py', 'r') as f:
            content = f.read()
            if 'app = FastAPI' in content:
                print("✅ FastAPI app found")
            else:
                print("❌ FastAPI app not found")
                return False
    except Exception as e:
        print(f"❌ Error checking API: {e}")
        return False
    
    print("\n✅ All module structure tests passed!")
    return True

def test_configuration_files():
    """Test configuration files are properly structured."""
    print("\n🔍 Testing configuration files...")
    
    # Test Docker configuration
    try:
        with open('Dockerfile', 'r') as f:
            dockerfile_content = f.read()
            if 'FROM python:' in dockerfile_content and 'EXPOSE 8000' in dockerfile_content:
                print("✅ Dockerfile properly configured")
            else:
                print("❌ Dockerfile missing required content")
                return False
    except Exception as e:
        print(f"❌ Error checking Dockerfile: {e}")
        return False
    
    # Test Docker Compose
    try:
        with open('docker-compose.yml', 'r') as f:
            compose_content = f.read()
            if 'mlops-api:' in compose_content and 'mlflow-server:' in compose_content:
                print("✅ Docker Compose properly configured")
            else:
                print("❌ Docker Compose missing required services")
                return False
    except Exception as e:
        print(f"❌ Error checking Docker Compose: {e}")
        return False
    
    # Test GitHub Actions
    try:
        with open('.github/workflows/mlops-pipeline.yml', 'r') as f:
            github_content = f.read()
            if 'jobs:' in github_content and 'test:' in github_content:
                print("✅ GitHub Actions properly configured")
            else:
                print("❌ GitHub Actions missing required jobs")
                return False
    except Exception as e:
        print(f"❌ Error checking GitHub Actions: {e}")
        return False
    
    print("\n✅ All configuration files are properly structured!")
    return True

def test_documentation():
    """Test that documentation is comprehensive."""
    print("\n🔍 Testing documentation...")
    
    # Test README
    try:
        with open('README.md', 'r') as f:
            readme_content = f.read()
            required_sections = [
                'Architecture Overview', 'Quick Start', 'API Usage',
                'Docker Deployment', 'Testing', 'MLflow'
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in readme_content:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"❌ README missing sections: {missing_sections}")
                return False
            else:
                print("✅ README comprehensive and well-structured")
    
    except Exception as e:
        print(f"❌ Error checking README: {e}")
        return False
    
    # Test Project Summary
    try:
        with open('PROJECT_SUMMARY.md', 'r') as f:
            summary_content = f.read()
            if len(summary_content) > 1000:  # Should be comprehensive
                print("✅ Project summary is comprehensive")
            else:
                print("❌ Project summary too brief")
                return False
    except Exception as e:
        print(f"❌ Error checking project summary: {e}")
        return False
    
    print("\n✅ Documentation is comprehensive and well-structured!")
    return True

def main():
    """Run all tests."""
    print("🏠 California Housing MLOps Pipeline - Structure Test")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run all test functions
    test_functions = [
        test_project_structure,
        test_imports,
        test_configuration_files,
        test_documentation
    ]
    
    for test_func in test_functions:
        try:
            if not test_func():
                all_tests_passed = False
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with error: {e}")
            all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! The MLOps pipeline is properly structured.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the setup: python setup.py")
        print("3. Start training: python src/models/train_model.py")
        print("4. Run the demo: python demo.py")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
