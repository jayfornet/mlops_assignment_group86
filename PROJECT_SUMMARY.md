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
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
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