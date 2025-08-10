# 🎯 MLOps Monitoring Assignment Instructions

## 📋 Assignment Overview

This assignment demonstrates a complete MLOps pipeline with comprehensive monitoring using Prometheus and Grafana. Both the API and monitoring stack run locally on different ports for evaluation.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MLOps Assignment Setup                    │
├─────────────────────────────────────────────────────────────┤
│  🏠 API Container (Port 8000)                               │
│  ├─ Housing Prediction API                                   │
│  ├─ Health Check: /health                                    │
│  ├─ Predictions: /predict                                    │
│  └─ Metrics: /metrics (Prometheus format)                    │
│                                                             │
│  📊 Monitoring Stack (Ports 9090, 3000, 9100, 8080)        │
│  ├─ Prometheus (9090) - Metrics Collection                  │
│  ├─ Grafana (3000) - Visualization Dashboard               │
│  ├─ Node Exporter (9100) - System Metrics                  │
│  └─ cAdvisor (8080) - Container Metrics                     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start Guide

### Step 1: Start the API Container

```powershell
# Build and start the API container
docker run -d --name housing-api -p 8000:8000 <your-dockerhub-username>/housing-prediction-api:latest

# Verify API is running
curl http://localhost:8000/health
```

### Step 2: Start Monitoring Stack

```powershell
# Use the monitoring setup script
python scripts/setup_monitoring.py setup

# Or manually with Docker Compose
docker compose -f docker-compose.monitoring.yml up -d
```

### Step 3: Access Monitoring Services

| Service | URL | Login | Purpose |
|---------|-----|-------|---------|
| **API** | http://localhost:8000 | - | Housing predictions |
| **Prometheus** | http://localhost:9090 | - | Metrics collection |
| **Grafana** | http://localhost:3000 | admin/admin123 | Dashboards |
| **Node Exporter** | http://localhost:9100 | - | System metrics |
| **cAdvisor** | http://localhost:8080 | - | Container metrics |

## 🧪 Testing the Setup

### Automated Testing

```powershell
# Generate monitoring data with load test
python scripts/load_test_monitoring.py --requests 50 --concurrency 3

# Check all services status
python scripts/setup_monitoring.py status
```

### Manual Testing

1. **Health Check**: `GET http://localhost:8000/health`
2. **Make Prediction**: 
   ```json
   POST http://localhost:8000/predict
   {
     "MedInc": 8.3252,
     "HouseAge": 41.0,
     "AveRooms": 6.984,
     "AveBedrms": 1.024,
     "Population": 322.0,
     "AveOccup": 2.556,
     "Latitude": 37.88,
     "Longitude": -122.23
   }
   ```

## 📊 Monitoring Dashboards

### Grafana Dashboard Features

1. **API Performance Metrics**
   - Request rate (requests/second)
   - Response time percentiles (p50, p95, p99)
   - Error rate and status codes
   - Request duration histograms

2. **System Health Metrics**
   - CPU and memory usage
   - Disk I/O and network traffic
   - Container resource utilization
   - Service uptime and availability

3. **Business Metrics**
   - Prediction volume trends
   - Model performance indicators
   - User activity patterns
   - Response accuracy tracking

## 🔧 GitHub Actions Workflows

### API Pipeline (`.github/workflows/mlops-pipeline.yml`)
- Builds and tests the prediction API
- Pushes Docker image to registry
- Runs integration tests

### Monitoring Pipeline (`.github/workflows/monitoring-pipeline.yml`)
- Validates Prometheus configuration
- Builds monitoring Docker image
- Tests monitoring stack integration
- Generates monitoring documentation

## 📈 Key Metrics Being Monitored

### Application Metrics
```
http_requests_total - Total HTTP requests
http_request_duration_seconds - Request duration
prediction_requests_total - ML prediction requests
model_inference_duration_seconds - Model inference time
```

### Infrastructure Metrics
```
cpu_usage_percent - CPU utilization
memory_usage_bytes - Memory consumption
disk_io_bytes - Disk I/O activity
network_bytes - Network traffic
```

## 🛠️ Management Commands

```powershell
# Setup complete monitoring
python scripts/setup_monitoring.py setup

# Start monitoring only
python scripts/setup_monitoring.py start

# Check status
python scripts/setup_monitoring.py status

# Stop monitoring
python scripts/setup_monitoring.py stop

# Run load test
python scripts/setup_monitoring.py test
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```powershell
   # Check what's using ports
   netstat -ano | findstr :8000
   netstat -ano | findstr :9090
   netstat -ano | findstr :3000
   ```

2. **Docker Issues**
   ```powershell
   # Check Docker status
   docker ps -a
   docker logs housing-api
   docker logs prometheus
   docker logs grafana
   ```

3. **Service Health**
   ```powershell
   # Check individual services
   curl http://localhost:8000/health
   curl http://localhost:9090/-/healthy
   curl http://localhost:3000/api/health
   ```

### Reset Everything

```powershell
# Stop all containers
docker compose -f docker-compose.monitoring.yml down
docker stop housing-api
docker rm housing-api

# Clean up
docker system prune -f

# Restart fresh
python scripts/setup_monitoring.py setup
```

## 📝 Assignment Deliverables

### What to Demonstrate

1. **Working API** - Housing prediction service running on port 8000
2. **Monitoring Stack** - Prometheus and Grafana running on ports 9090 and 3000
3. **Metrics Collection** - Real-time monitoring of API requests
4. **Dashboard Visualization** - Grafana dashboards showing live metrics
5. **Load Testing** - Automated testing generating monitoring data
6. **CI/CD Integration** - GitHub Actions workflows for deployment

### Evaluation Criteria

- ✅ Both containers running on different ports
- ✅ API responds to health checks and predictions
- ✅ Prometheus collecting metrics from API
- ✅ Grafana displaying real-time dashboards
- ✅ Load testing generates observable metrics
- ✅ Monitoring captures all requests and responses
- ✅ System health monitoring functional

## 🎓 Learning Outcomes

1. **MLOps Pipeline Design** - Complete CI/CD with monitoring
2. **Containerization** - Docker multi-service deployment
3. **Observability** - Metrics, monitoring, and alerting
4. **Infrastructure as Code** - Docker Compose orchestration
5. **Performance Testing** - Load testing and validation
6. **Dashboard Design** - Effective monitoring visualization

## 📞 Support

If you encounter issues:

1. Check the logs: `docker logs <container-name>`
2. Verify ports: `netstat -ano | findstr :<port>`
3. Test connectivity: `curl http://localhost:<port>/health`
4. Review documentation in `/docs` folder
5. Use the management scripts in `/scripts` folder

---

**🎯 Assignment Success**: When you can simultaneously access the prediction API on port 8000 and see its metrics visualized in Grafana on port 3000, with Prometheus collecting data on port 9090, you've successfully completed the monitoring setup!
