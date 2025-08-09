# 🚀 MLOps Stack with Monitoring and Experiment Tracking

This comprehensive MLOps stack includes:
- **Housing Prediction API** (FastAPI)
- **MLflow** for experiment tracking and model management
- **Prometheus** for metrics collection
- **Grafana** for visualization and monitoring
- **Node Exporter** for system metrics
- **cAdvisor** for container metrics

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Grafana       │    │   Prometheus    │    │   MLflow        │
│   (Port 3000)   │◄───┤   (Port 9090)   │    │   (Port 5000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                        ▲                        ▲
         │                        │                        │
         │                        │                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Housing API   │    │  Node Exporter  │    │   cAdvisor      │
│   (Port 8000)   │    │   (Port 9100)   │    │   (Port 8080)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Option 1: Windows PowerShell
```powershell
.\start-mlops.ps1
```

### Option 2: Linux/Mac Bash
```bash
chmod +x start-mlops.sh
./start-mlops.sh
```

### Option 3: Manual Docker Compose
```bash
docker-compose up -d
```

## 📊 Access Your Services

Once all services are running, you can access:

| Service | URL | Description |
|---------|-----|-------------|
| 🏠 **Housing API** | http://localhost:8000 | Main prediction API |
| 📖 **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| 🔬 **MLflow** | http://localhost:5000 | Complete experiment tracking with Git-persisted data |
| 📊 **Prometheus** | http://localhost:9090 | Metrics collection and querying |
| 📈 **Grafana** | http://localhost:3000 | Dashboards and visualization |
| 📋 **Node Exporter** | http://localhost:9100/metrics | System metrics |
| 🐳 **cAdvisor** | http://localhost:8080 | Container metrics |

### 🔐 Default Credentials
- **Grafana**: `admin` / `admin123`

## 🎯 Key Features

### 1. 🔬 MLflow Integration with Git Persistence
- **Complete Experiment History**: Browse all experiments from previous pipeline runs
- **Model Registry**: Manage model versions with full historical context
- **Artifact Storage**: Access all model files and training artifacts
- **Metrics Comparison**: Compare model performance across multiple pipeline executions
- **Best Model Tracking**: View automatically selected best models from each run
- **Git-Synchronized Data**: All experiment data persisted and versioned in Git

### 2. 📊 Comprehensive Monitoring
- **API Performance**: Request rates, latency, error rates
- **System Resources**: CPU, memory, disk usage
- **Container Metrics**: Resource usage per service
- **Custom Dashboards**: Pre-built Grafana dashboards

### 3. 🚀 Production-Ready API
- **Health Checks**: Automated service health monitoring
- **Prometheus Metrics**: Built-in metrics export
- **Request Logging**: Detailed API request logging
- **Error Handling**: Robust error handling and reporting

## 📈 Pre-built Dashboards

### 1. MLOps Dashboard (`mlops-dashboard`)
- API prediction rate and latency
- Model performance metrics
- Error rate monitoring
- System resource usage

### 2. System Performance Dashboard (`system-performance`)
- CPU, memory, and disk usage
- Container resource consumption
- Service health status
- Network and I/O metrics

## 🔧 Configuration

### Environment Variables
The following environment variables are automatically configured:

```bash
# API Configuration
PYTHONPATH=/app
LOG_LEVEL=INFO
MLFLOW_TRACKING_URI=http://mlflow-server:5000
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc_dir

# MLflow Configuration
MLFLOW_BACKEND_STORE_URI=file:///mlflow/mlruns
MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
MLFLOW_SERVE_ARTIFACTS=true

# Grafana Configuration
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin123
GF_USERS_ALLOW_SIGN_UP=false
```

### Volume Mounts
- `./logs:/app/logs` - API logs
- `./models:/app/models:ro` - Model files (read-only)
- `./mlruns:/mlflow/mlruns` - MLflow experiments (complete history)
- `./mlflow-artifacts:/mlflow/artifacts` - MLflow artifacts storage
- `./deployment:/mlflow/deployment:ro` - Production-ready models
- `./results:/mlflow/results:ro` - Model comparison results

## 🔍 Browsing MLflow Data in Docker

### Accessing Complete Experiment History

When you start the Docker stack, MLflow will have access to:

1. **All Historical Experiments**: Every pipeline run's experiments are preserved
2. **Cross-Run Comparisons**: Compare models from different pipeline executions
3. **Best Model Evolution**: Track how best models change over time
4. **Complete Artifact Library**: Access all model files and training outputs

### MLflow UI Features

Visit http://localhost:5000 to explore:

#### **Experiments View**
- Browse all experiments from multiple pipeline runs
- Filter and sort runs by metrics, parameters, or dates
- View experiment evolution over time

#### **Model Registry**
- See all registered models with version history
- Track model stage transitions (staging → production)
- Download any model version for analysis

#### **Run Comparison**
- Select multiple runs from different pipeline executions
- Compare metrics side-by-side
- Visualize parameter impact on performance

#### **Artifacts Browser**
- Download model files (.joblib, .pkl)
- View training plots and visualizations
- Access model metadata and configuration

### Example Browsing Workflow

```bash
# 1. Start the monitoring stack
docker-compose up -d

# 2. Wait for services to be ready (30-60 seconds)
docker-compose logs -f mlflow-server

# 3. Open MLflow UI
open http://localhost:5000

# 4. Browse experiments:
#    - Click on experiment names to see runs
#    - Compare runs using checkboxes
#    - Download artifacts from the Artifacts tab
#    - View metrics evolution in the Charts tab
```

### MLflow Data Structure in Docker

The MLflow container will have access to the following data structure:

```
/mlflow/
├── mlruns/                     # Complete experiment history
│   ├── 0/                     # Default experiment
│   ├── <experiment_id>/       # Pipeline experiments
│   │   ├── <run_id>/         # Individual training runs
│   │   │   ├── meta.yaml     # Run metadata
│   │   │   ├── metrics/      # Logged metrics (RMSE, MAE, etc.)
│   │   │   ├── params/       # Hyperparameters
│   │   │   ├── tags/         # Run tags and info
│   │   │   └── artifacts/    # Model artifacts
│   │   └── models/           # Registered models
│   └── models/               # Model registry
├── artifacts/                 # MLflow artifacts storage
│   └── <run_id>/             # Run-specific artifacts
│       └── model/            # Serialized models
├── models/                    # Best models from pipeline
│   ├── linear_regression_best_model.joblib
│   ├── random_forest_best_model.joblib
│   ├── gradient_boosting_best_model.joblib
│   └── *_metadata.json       # Model metadata
├── deployment/               # Production-ready models
│   └── models/
│       └── best_model/       # Current production model
└── results/                  # Model comparison results
    └── model_comparison.csv  # Performance comparison
```

### Pipeline-Generated Data Available

Each time the CI/CD pipeline runs, it adds:
- New experiment runs with metrics and parameters
- Best model selections and promotions
- Updated model comparison results
- Training artifacts and model files

This means your Docker MLflow instance will show the complete evolution of your models across all pipeline executions!

## 🔧 Verifying MLflow Data in Docker

### Quick Verification Script

Use the provided script to verify MLflow data mounting:

```bash
# Run verification script
python scripts/verify_mlflow_data.py

# Or run inside Docker container
docker-compose exec mlflow-server python /tmp/verify_mlflow_data.py
```

The script will check:
- ✅ Directory structure and file counts
- ✅ MLflow experiments and runs
- ✅ Available model files
- ✅ Persistence metadata
- ✅ MLflow server connectivity

### Manual Verification

You can also manually check the mounted data:

```bash
# Check Docker volumes
docker-compose exec mlflow-server ls -la /mlflow/

# View experiment structure
docker-compose exec mlflow-server find /mlflow/mlruns -name "*.yaml" | head -10

# Check model files
docker-compose exec mlflow-server ls -la /mlflow/models/

# View logs
docker-compose logs mlflow-server | tail -20
```

## 🧪 Testing the Setup

### 1. Test API Health
```bash
curl http://localhost:8000/health
```

### 2. Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "MedInc": 8.3252,
       "HouseAge": 41.0,
       "AveRooms": 6.984,
       "AveBedrms": 1.023,
       "Population": 322.0,
       "AveOccup": 2.555,
       "Latitude": 37.88,
       "Longitude": -122.23
     }'
```

### 3. Check Metrics
```bash
curl http://localhost:8000/metrics
```

## 📝 Monitoring Metrics

### API Metrics
- `predictions_total` - Total number of predictions made
- `prediction_duration_seconds` - Histogram of prediction durations
- `prediction_errors_total` - Total number of prediction errors

### System Metrics
- `node_cpu_seconds_total` - CPU usage by core
- `node_memory_MemTotal_bytes` - Total system memory
- `node_filesystem_size_bytes` - Filesystem size

### Container Metrics
- `container_memory_usage_bytes` - Memory usage per container
- `container_cpu_usage_seconds_total` - CPU usage per container

## 🛠️ Management Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mlops-api
docker-compose logs -f mlflow-server
docker-compose logs -f grafana
docker-compose logs -f prometheus
```

### Stop Services
```bash
docker-compose down
```

### Restart Services
```bash
docker-compose restart [service-name]
```

### Scale Services
```bash
docker-compose up -d --scale mlops-api=2
```

## 🚨 Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   docker-compose down --remove-orphans
   docker-compose up -d
   ```

2. **Prometheus not scraping metrics**
   - Check that all target services are running
   - Verify network connectivity between containers

3. **Grafana dashboards not loading**
   - Check that datasource configuration is correct
   - Verify Prometheus is accessible from Grafana

4. **MLflow experiments not showing**
   - Ensure `mlruns` directory has proper permissions
   - Check MLflow server logs for errors

### Log Locations
- API logs: `./logs/api.log`
- Docker logs: `docker-compose logs [service]`
- Prometheus config: `./monitoring/prometheus.yml`
- Grafana dashboards: `./monitoring/grafana/dashboard-configs/`

## 🔄 Updates and Maintenance

### Updating Dashboards
1. Modify JSON files in `./monitoring/grafana/dashboard-configs/`
2. Restart Grafana: `docker-compose restart grafana`

### Adding New Metrics
1. Update Prometheus configuration in `./monitoring/prometheus.yml`
2. Add metric collection in your application code
3. Restart Prometheus: `docker-compose restart prometheus`

### Backing Up Data
```bash
# Backup MLflow experiments
tar -czf mlflow-backup.tar.gz mlruns/ mlflow-artifacts/

# Backup Grafana dashboards
tar -czf grafana-backup.tar.gz monitoring/grafana/
```

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🎉 Happy Monitoring!

Your comprehensive MLOps stack is now ready for production-grade machine learning operations with full observability and experiment tracking!
