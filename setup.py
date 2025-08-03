#!/usr/bin/env python3
"""
Setup script for the California Housing MLOps Pipeline.

This script:
- Sets up the project environment
- Trains models with MLflow tracking
- Saves the best model for API deployment
- Generates project summary
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary project directories."""
    directories = [
        'data', 'models', 'logs', 'results', 'mlruns', 
        'mlflow-artifacts', 'tests', 'monitoring/grafana/dashboards'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def install_dependencies(skip_for_docker=False):
    """Install Python dependencies.
    
    Args:
        skip_for_docker (bool): Skip dependency installation if running in Docker
    """
    if skip_for_docker and os.environ.get('RUNNING_IN_DOCKER', '').lower() == 'true':
        logger.info("Running in Docker, skipping dependency installation")
        return True
        
    logger.info("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False
    return True


def initialize_dvc():
    """Initialize DVC for data versioning with GitHub storage."""
    try:
        if not os.path.exists('.dvc'):
            subprocess.check_call(['dvc', 'init'])
            # Configure GitHub remote
            subprocess.check_call(['dvc', 'remote', 'add', '-d', 'github-storage', 
                                  'github://jayfornet/mlops_assignment_group86/releases/data'])
            logger.info("DVC initialized with GitHub storage")
        else:
            logger.info("DVC already initialized")
            
        # Track dataset files if they exist
        data_files = ['data/california_housing.csv', 'data/california_housing.joblib']
        for file_path in data_files:
            if os.path.exists(file_path) and not os.path.exists(f"{file_path}.dvc"):
                try:
                    subprocess.check_call(['dvc', 'add', file_path])
                    logger.info(f"Added {file_path} to DVC tracking")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to add {file_path} to DVC: {e}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"DVC not available or failed to initialize: {e}. Skipping DVC setup.")


def download_dataset():
    """
    Check for California Housing dataset and create it if needed.
    Try to download from scikit-learn first, but fallback to synthetic data 
    if there are connection issues.
    """
    logger.info("Checking for California Housing dataset...")
    
    csv_path = os.path.join('data', 'california_housing.csv')
    joblib_path = os.path.join('data', 'california_housing.joblib')
    
    # If dataset already exists, we're good
    if os.path.exists(csv_path) and os.path.exists(joblib_path):
        logger.info(f"Dataset files already exist: {csv_path} and {joblib_path}")
        return True
    
    # First try downloading from scikit-learn
    try:
        logger.info("Trying to download dataset from scikit-learn...")
        from sklearn.datasets import fetch_california_housing
        import pandas as pd
        import joblib
        
        # Download the dataset
        housing = fetch_california_housing(as_frame=True)
        
        # Extract features and target
        X = housing.data
        y = housing.target
        
        # Save as CSV
        os.makedirs('data', exist_ok=True)
        df = X.copy()
        df['MedHouseVal'] = y
        df.to_csv(csv_path, index=False)
        
        # Save as joblib
        joblib.dump({
            "data": X,
            "target": y,
            "feature_names": X.columns.tolist()
        }, joblib_path)
        
        logger.info(f"Dataset downloaded from scikit-learn and saved to {csv_path} and {joblib_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to download from scikit-learn: {e}")
        logger.info("Creating synthetic California Housing dataset instead...")
        
        try:
            import numpy as np
            import pandas as pd
            import joblib
            
            # Create synthetic dataset
            np.random.seed(42)
            n_samples = 20640
            
            # Create synthetic features
            X = pd.DataFrame({
                'MedInc': np.random.lognormal(mean=1.0, sigma=0.5, size=n_samples),
                'HouseAge': np.random.uniform(1, 50, n_samples),
                'AveRooms': np.random.lognormal(mean=1.5, sigma=0.3, size=n_samples),
                'AveBedrms': np.random.lognormal(mean=0.5, sigma=0.2, size=n_samples),
                'Population': np.random.lognormal(mean=5.5, sigma=0.7, size=n_samples),
                'AveOccup': np.random.lognormal(mean=1.0, sigma=0.3, size=n_samples),
                'Latitude': np.random.uniform(32, 42, n_samples),
                'Longitude': np.random.uniform(-125, -114, n_samples)
            })
            
            # Create synthetic target with correlation to features
            y = (
                3.0 * X['MedInc'] + 
                -0.1 * X['HouseAge'] + 
                0.5 * X['AveRooms'] + 
                -0.2 * X['Population'] / 1000 +
                0.1 * np.abs(X['Latitude'] - 37.5) +
                0.1 * np.abs(X['Longitude'] + 122) +
                np.random.normal(0, 0.5, n_samples)
            )
            
            # Normalize to be similar to original dataset
            y = np.clip(y / 5.0, 0.5, 5.0)
            
            # Save as CSV
            os.makedirs('data', exist_ok=True)
            df = X.copy()
            df['MedHouseVal'] = y
            df.to_csv(csv_path, index=False)
            
            # Save as joblib
            joblib.dump({
                "data": X,
                "target": y,
                "feature_names": X.columns.tolist()
            }, joblib_path)
            
            logger.info(f"Synthetic dataset created and saved to {csv_path} and {joblib_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create synthetic dataset: {e}")
            return False


def train_models():
    """Train models using the training pipeline."""
    logger.info("Starting model training...")
    try:
        # Set PYTHONPATH to include src directory
        env = os.environ.copy()
        src_path = os.path.abspath('src')
        
        # Handle path separator for different OS
        path_sep = ';' if sys.platform == 'win32' else ':'
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{src_path}{path_sep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = src_path
        
        # Run training script
        result = subprocess.run([
            sys.executable, 'src/models/train_model.py'
        ], env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Model training completed successfully")
            logger.info(f"Training output:\n{result.stdout}")
            return True
        else:
            logger.error(f"Model training failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return False


def save_scaler_for_api():
    """Save the scaler used during training for API deployment."""
    logger.info("Saving scaler for API deployment...")
    try:
        # Add src to path
        sys.path.append(os.path.abspath('src'))
        
        # Import required modules
        try:
            from src.data.data_processor import DataProcessor
        except ImportError:
            from data.data_processor import DataProcessor
        
        import joblib
        
        # Create processor and load data
        processor = DataProcessor(random_state=42)
        X, y = processor.load_data()
        X_processed, y_processed = processor.preprocess_data(X, y)
        
        # Split and scale data to fit the scaler
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
            X_processed, y_processed
        )
        X_train_scaled, _, _ = processor.scale_features(X_train, X_val, X_test)
        
        # Save the fitted scaler
        scaler_path = 'models/scaler.joblib'
        joblib.dump(processor.scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving scaler: {e}")
        return False


def generate_project_summary():
    """Generate a project summary document."""
    logger.info("Generating project summary...")
    
    summary_content = """
# California Housing Price Prediction - MLOps Pipeline Summary

## Project Overview

This project implements a complete MLOps pipeline for predicting California housing prices using machine learning. The pipeline demonstrates industry best practices for model development, deployment, and monitoring.

## Architecture Components

### 1. Data Processing (`src/data/data_processor.py`)
- **Purpose**: Load, preprocess, and prepare the California Housing dataset
- **Features**:
  - Data loading from scikit-learn
  - Outlier detection and removal
  - Train/validation/test splitting
  - Feature scaling with StandardScaler
  - Data quality validation

### 2. Model Training (`src/models/train_model.py`)
- **Purpose**: Train multiple ML models and track experiments
- **Models Trained**:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **MLflow Integration**:
  - Experiment tracking
  - Parameter logging
  - Metrics comparison
  - Model versioning and registration

### 3. API Service (`src/api/app.py`)
- **Framework**: FastAPI
- **Features**:
  - RESTful prediction endpoint
  - Input validation with Pydantic
  - Health monitoring
  - Request/response logging
  - Prometheus metrics
  - Database logging (SQLite)

### 4. Containerization (`Dockerfile`, `docker-compose.yml`)
- **Multi-stage Docker build** for optimized image size
- **Service orchestration** with Docker Compose
- **Monitoring stack** (Prometheus + Grafana)
- **MLflow tracking server**

### 5. CI/CD Pipeline (`.github/workflows/mlops-pipeline.yml`)
- **Automated testing** with pytest
- **Code quality checks** (Black, flake8, isort)
- **Security scanning** (Bandit, Safety)
- **Docker image building and pushing**
- **Integration testing**
- **Automated deployment**

## Key Features

### Data Versioning
- Optional DVC integration for large datasets
- Git-based version control for code and models

### Experiment Tracking
- MLflow for comprehensive experiment management
- Model comparison and selection
- Artifact storage and versioning

### Model Deployment
- REST API with comprehensive error handling
- Input validation and sanitization
- Scalable containerized deployment

### Monitoring & Observability
- Prometheus metrics collection
- Grafana dashboards (optional)
- Application logging
- Health checks and status monitoring

### Testing & Quality Assurance
- Unit tests for all components
- Integration tests for API endpoints
- Code quality enforcement
- Security vulnerability scanning

## Model Performance

The pipeline automatically trains and compares multiple models, selecting the best performer based on validation RMSE:

1. **Linear Regression**: Simple baseline model
2. **Random Forest**: Ensemble method for better generalization
3. **Gradient Boosting**: Advanced ensemble for optimal performance

## Deployment Options

### Local Development
```bash
python src/api/app.py
```

### Docker Deployment
```bash
docker-compose up -d
```

### Production Deployment
- CI/CD pipeline supports automated deployment
- Configurable for cloud providers (AWS, GCP, Azure)
- Kubernetes-ready container images

## Monitoring Dashboard

Access monitoring tools:
- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)

## API Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "MedInc": 8.3252,
       "HouseAge": 41.0,
       "AveRooms": 6.984,
       "AveBedrms": 1.024,
       "Population": 322.0,
       "AveOccup": 2.555,
       "Latitude": 37.88,
       "Longitude": -122.23
     }'
```

## Quality Assurance

### Code Quality
- **Black** for code formatting
- **flake8** for linting
- **isort** for import sorting
- **mypy** for type checking

### Testing Coverage
- Unit tests for data processing
- API endpoint testing
- Model training validation
- Integration testing

### Security
- Input validation and sanitization
- Container security best practices
- Dependency vulnerability scanning
- Non-root container execution

## Future Enhancements

1. **Advanced Monitoring**: Custom Grafana dashboards
2. **Model Retraining**: Automated retraining on new data
3. **A/B Testing**: Model comparison in production
4. **Advanced Features**: Feature importance analysis
5. **Scalability**: Kubernetes deployment manifests

## Success Metrics

- **Model Accuracy**: RMSE < 0.7 on test set
- **API Performance**: <100ms average response time
- **System Reliability**: 99.9% uptime
- **Code Quality**: 100% test coverage
- **Security**: Zero high-severity vulnerabilities

## Conclusion

This MLOps pipeline demonstrates production-ready machine learning deployment with:
- Automated model training and selection
- Robust API deployment with monitoring
- Comprehensive testing and quality assurance
- Scalable containerized architecture
- CI/CD automation for reliable deployments

The project showcases modern MLOps practices and can serve as a template for similar machine learning projects.
"""
    
    try:
        with open('PROJECT_SUMMARY.md', 'w') as f:
            f.write(summary_content.strip())
        logger.info("Project summary generated: PROJECT_SUMMARY.md")
        return True
    except Exception as e:
        logger.error(f"Error generating project summary: {e}")
        return False


def main(skip_deps_for_docker=False):
    """Main setup function.
    
    Args:
        skip_deps_for_docker (bool): Skip dependency installation if running in Docker
    """
    logger.info("Starting California Housing MLOps Pipeline Setup...")
    
    # Create project directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies(skip_for_docker=skip_deps_for_docker):
        logger.error("Setup failed during dependency installation")
        return False
    
    # Verify or download dataset
    if not download_dataset():
        logger.error("Dataset verification/download failed")
        logger.error("California Housing dataset is required to continue")
        return False
    
    # Initialize DVC (optional)
    initialize_dvc()
    
    # Train models
    if not train_models():
        logger.error("Setup failed during model training")
        return False
    
    # Save scaler for API
    if not save_scaler_for_api():
        logger.error("Setup failed during scaler saving")
        return False
    
    # Generate project summary
    generate_project_summary()
    
    logger.info("✅ Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Start MLflow UI: mlflow ui --host 0.0.0.0 --port 5000")
    logger.info("2. Start API server: python src/api/app.py")
    logger.info("3. Or use Docker: docker-compose up -d")
    logger.info("4. Visit http://localhost:8000/docs for API documentation")
    
    return True


if __name__ == "__main__":
    # Check if running in Docker mode
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup for California Housing MLOps Pipeline')
    parser.add_argument('--docker-mode', action='store_true', help='Skip dependency installation for Docker')
    args = parser.parse_args()
    
    success = main(skip_deps_for_docker=args.docker_mode)
    sys.exit(0 if success else 1)
