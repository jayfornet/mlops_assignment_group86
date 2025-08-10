# GitHub Actions Workflow Separation

## 🔄 Workflow Overview

This project uses **two separate GitHub Actions workflows** to handle different aspects of the MLOps pipeline:

### 1. **MLOps Pipeline** (`.github/workflows/mlops-pipeline.yml`)
**Triggered by changes to:**
- Source code (`src/`)
- Tests (`tests/`)
- Data processing scripts
- Model training scripts
- API implementation
- Docker configuration for API
- Requirements and dependencies

**Does NOT trigger for:**
- Monitoring configuration changes
- Grafana/Prometheus setup
- Monitoring scripts
- Documentation updates

**Purpose:** Handles data processing, model training, API building, and main application deployment.

### 2. **Monitoring Pipeline** (`.github/workflows/monitoring-pipeline.yml`)
**Triggered by changes to:**
- `monitoring/` directory (Prometheus, Grafana configs)
- `docker-compose.monitoring.yml`
- `Dockerfile.monitoring*`
- Monitoring-related scripts (`setup_monitoring.py`, `load_test_monitoring.py`)
- The monitoring workflow file itself

**Purpose:** Builds and deploys the monitoring stack (Prometheus + Grafana) separately from the main application.

## 🎯 Why Separate Workflows?

### ✅ **Benefits:**

1. **Faster Builds**: 
   - Monitoring changes don't rebuild the entire ML pipeline
   - ML changes don't rebuild monitoring infrastructure

2. **Independent Deployment**: 
   - Update monitoring dashboards without affecting API
   - Deploy new models without touching monitoring

3. **Resource Efficiency**: 
   - Only run necessary tests and builds
   - Parallel development of monitoring and ML features

4. **Assignment Clarity**: 
   - Clear separation between API and monitoring components
   - Each has its own Docker image and deployment process

### 📋 **Workflow Behavior:**

| Change Type | MLOps Pipeline | Monitoring Pipeline |
|-------------|:--------------:|:------------------:|
| Update model training | ✅ Triggers | ❌ Skipped |
| Change API code | ✅ Triggers | ❌ Skipped |
| Update Prometheus config | ❌ Skipped | ✅ Triggers |
| Modify Grafana dashboard | ❌ Skipped | ✅ Triggers |
| Update requirements.txt | ✅ Triggers | ❌ Skipped |
| Change monitoring scripts | ❌ Skipped | ✅ Triggers |

## 🚀 **Local Development**

### For ML/API Development:
```bash
# Your changes to src/, tests/, models/, etc. will trigger MLOps pipeline
git add src/models/new_model.py
git commit -m "Add new model architecture"
git push  # Triggers MLOps pipeline only
```

### For Monitoring Development:
```bash
# Your changes to monitoring/ will trigger monitoring pipeline
git add monitoring/grafana/dashboards/new-dashboard.json
git commit -m "Add new Grafana dashboard"
git push  # Triggers monitoring pipeline only
```

### For Both:
```bash
# If you change both ML and monitoring code
git add src/api/new_endpoint.py monitoring/prometheus.yml
git commit -m "Add new endpoint and monitoring"
git push  # Triggers BOTH pipelines
```

## 🔧 **Manual Triggers**

Both workflows can be triggered manually:

1. Go to **Actions** tab in GitHub
2. Select the workflow you want to run
3. Click **"Run workflow"**
4. Choose the branch and click **"Run workflow"**

## 📊 **Assignment Usage**

This separation is particularly useful for the assignment because:

1. **API Container** (Port 8000): Built by MLOps pipeline
2. **Monitoring Stack** (Ports 9090, 3000, etc.): Built by monitoring pipeline
3. **Independent Updates**: Can update dashboards without rebuilding ML models
4. **Faster Iterations**: Monitoring tweaks don't require full ML pipeline execution

## 🛠️ **Debugging Workflows**

### Check Which Workflow Should Trigger:
```bash
# Check what files you've changed
git diff --name-only HEAD~1

# Files in these paths trigger MLOps pipeline:
# - src/, tests/, scripts/train_models.py, requirements.txt, etc.

# Files in these paths trigger Monitoring pipeline:
# - monitoring/, docker-compose.monitoring.yml, scripts/*monitoring*
```

### Force a Specific Workflow:
- **MLOps Pipeline**: Make a change to `src/` or `requirements.txt`
- **Monitoring Pipeline**: Make a change to `monitoring/prometheus.yml`
- **Manual Trigger**: Use GitHub Actions UI

---

This separation ensures efficient, targeted builds while maintaining clear boundaries between ML operations and monitoring concerns! 🎯
