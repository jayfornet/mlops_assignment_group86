# MLOps Pipeline - California Housing Price Prediction

A complete MLOps pipeline demonstrating model training, versioning, deployment, and monitoring using modern MLOps tools.

## 🏗️ Architecture Overview

This project implements a complete MLOps pipeline with the following components:

- **Data Versioning**: DVC for dataset tracking
- **Experiment Tracking**: MLflow for model versioning and metrics with Git persistence
- **API Service**: FastAPI for model serving
- **Containerization**: Docker for deployment
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Monitoring**: Comprehensive stack with Prometheus, Grafana, and MLflow tracking
- **Model Registry**: Centralized model management with MLflow
- **Metrics & Observability**: Real-time monitoring dashboards

## 📁 Project Structure

```
mlops-pipeline/
├── data/                   # Dataset storage
├── src/                    # Source code
│   ├── data/              # Data processing scripts
│   ├── models/            # Model training scripts
│   ├── api/               # FastAPI application
│   └── utils/             # Utility functions
├── tests/                 # Unit tests
├── logs/                  # Application logs
├── mlruns/                # MLflow tracking (Git-persisted)
├── mlflow-artifacts/      # MLflow artifacts storage
├── models/                # Trained model files
├── monitoring/            # Monitoring configurations
│   ├── grafana/          # Grafana dashboards & datasources
│   └── prometheus.yml    # Prometheus configuration
├── .github/workflows/     # CI/CD pipelines
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
└── docker-compose.yml    # Multi-service orchestration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Git

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-organization/california-housing-mlops.git
cd california-housing-mlops

# Or if you're setting up a new repository:
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-organization/california-housing-mlops.git
git push -u origin main
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Initialize DVC (Optional)

```bash
dvc init
dvc remote add -d myremote <your-remote-storage>
```

### 4. Train Models

```bash
python src/models/train_model.py
```

### 5. Start MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### 6. Run API Service

```bash
python src/api/app.py
```

### 7. Docker Deployment

```bash
docker build -t mlops-api .
docker run -p 8000:8000 mlops-api
```

## 🔧 API Usage

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

### View Metrics
```bash
curl http://localhost:8000/metrics
```

## 📊 Model Performance

The pipeline trains and compares multiple models:

- **Linear Regression**: Baseline model
- **Random Forest**: Tree-based ensemble
- **Gradient Boosting**: Advanced ensemble method

Best model is automatically selected based on RMSE and registered in MLflow.

## 🔍 Monitoring

- **Request Logging**: All API requests logged to `logs/api.log`
- **Model Metrics**: Performance metrics tracked in MLflow
- **Health Monitoring**: `/health` endpoint for service status
- **Prediction Tracking**: Input/output logging for model monitoring

## 🧪 Running Tests

### Prepare Test Environment
Before running tests, ensure all necessary directories exist:

```bash
# On Linux/macOS
python scripts/prepare_test_env.py

# On Windows PowerShell
.\scripts\prepare_test_env.ps1
```

### Run Tests
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## 📊 MLflow Experiment Tracking & Persistence

### How MLflow Persistence Works

This project implements a sophisticated MLflow persistence strategy that ensures experiment data survives across pipeline runs:

#### **Git-Based Persistence**
- MLflow runs and artifacts are automatically committed to the Git repository
- Each pipeline execution adds to the experiment history (no data loss)
- Best models are promoted to dedicated deployment folders
- Experiment metadata is tracked with Git commits

#### **Pipeline Integration**
1. **Training Phase**: Models are trained with MLflow tracking
2. **Persistence Phase**: MLflow data is committed to Git repository
3. **Docker Phase**: Container builds include all historical experiment data
4. **Deployment Phase**: MLflow UI shows complete experiment history

#### **Key Benefits**
- **Experiment Continuity**: All runs accumulate over time
- **Model Comparison**: Compare models across different pipeline executions
- **Deployment Tracking**: Track which models are deployed when
- **Collaboration**: Team members see complete experiment history

### Accessing MLflow Data

```bash
# View experiment tracking locally
mlflow ui --backend-store-uri file:./mlruns

# Check experiment data in repository
git log --oneline | grep "MLflow"

# View best model information
cat mlflow_persistence_info.json
```

### MLflow in Docker Environment

```bash
# Start monitoring stack (includes MLflow server)
docker-compose up -d

# Access MLflow UI
open http://localhost:5001
```

## 🛠️ Development

### Code Style
```bash
black src/
flake8 src/
```

### Add New Features
1. Create feature branch
2. Implement changes with tests
3. Submit pull request
4. CI/CD pipeline runs automatically

## 📝 Logging Configuration

Logs are structured and include:
- Timestamp
- Log level
- Component
- Message
- Request ID (for API calls)

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**: Change port in configuration
2. **MLflow Not Starting**: Check if port 5000 is available
3. **Docker Build Fails**: Ensure Docker daemon is running
4. **DVC Remote Issues**: Verify remote storage configuration

### Debug Commands

```bash
# Check API logs
tail -f logs/api.log

# MLflow server logs
mlflow server --help

# Docker container logs
docker logs <container-id>
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Scikit-learn for ML algorithms
- MLflow for experiment tracking
- FastAPI for API framework
- Docker for containerization
