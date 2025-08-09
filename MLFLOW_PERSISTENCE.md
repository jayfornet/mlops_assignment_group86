# MLflow Experiment Persistence Strategy

## Overview

This document explains how MLflow experiment data persists across multiple pipeline executions in our MLOps pipeline.

## Problem Statement

By default, MLflow experiments in CI/CD environments are ephemeral - they exist only during the pipeline execution and are lost afterwards. This makes it impossible to:

- Compare models across different pipeline runs
- Track experiment evolution over time
- Maintain a historical record of model performance
- Enable collaborative model development

## Solution: Git-Based MLflow Persistence

We've implemented a comprehensive persistence strategy that commits MLflow data to the Git repository, ensuring experiment continuity across pipeline executions.

## Architecture

### Pipeline Jobs

```
data-loading → data-preprocessing → model-training → mlflow-persistence
                                                    ↓
                  docker ← integration-tests ← [mlflow data committed]
                     ↓
                  deploy
```

### MLflow Persistence Job

The dedicated `mlflow-persistence` job handles:

1. **Experiment Collection**: Downloads MLflow data from training job
2. **Git Integration**: Commits experiment data to repository
3. **Model Promotion**: Moves best models to deployment folders
4. **Metadata Creation**: Generates tracking information
5. **Artifact Management**: Organizes MLflow artifacts

## Implementation Details

### Job Configuration

```yaml
mlflow-persistence:
  runs-on: ubuntu-latest
  name: Persist MLflow Data
  needs: [model-training]
  steps:
    - name: Setup Git and commit MLflow data
    - name: Identify and promote best models
    - name: Create metadata for tracking
```

### File Structure After Persistence

```
mlruns/                          # Complete experiment history
├── <experiment_id>/
│   ├── <run_id>/               # Individual runs
│   │   ├── meta.yaml           # Run metadata
│   │   ├── metrics/            # Logged metrics
│   │   ├── params/             # Hyperparameters
│   │   └── tags/               # Run tags
│   └── models/                 # Registered models

mlflow-artifacts/               # Model artifacts
├── <run_id>/
│   └── model/                  # Serialized models

models/                         # Best models for deployment
├── <algorithm>_best_model.joblib
├── <algorithm>_best_model.pkl
└── <algorithm>_metadata.json

deployment/models/              # Production-ready models
└── best_model/
    ├── model.joblib
    ├── model.pkl
    └── metadata.json

mlflow_persistence_info.json   # Tracking metadata
```

## Benefits

### 1. Experiment Continuity
- Each pipeline run adds to the experiment history
- No data loss between executions
- Complete model evolution tracking

### 2. Model Comparison
- Compare models across different commits
- Track performance trends over time
- Identify best performing approaches

### 3. Collaborative Development
- Team members see complete experiment history
- Shared understanding of model performance
- Easy collaboration on model improvements

### 4. Production Tracking
- Clear trail of which models were deployed when
- Easy rollback to previous model versions
- Audit trail for model changes

## Usage Examples

### Viewing Experiment History

```bash
# View all experiments
mlflow ui --backend-store-uri file:./mlruns

# Check Git history of MLflow commits
git log --oneline --grep="MLflow"

# View best model information
cat mlflow_persistence_info.json
```

### In Docker Environment

```bash
# Start MLflow server with historical data
docker-compose up mlflow

# Access MLflow UI with complete history
open http://localhost:5001
```

### Pipeline Monitoring

Each pipeline execution will:
1. Train new models with MLflow tracking
2. Add experiment data to Git repository
3. Update best model selections
4. Maintain complete experiment history

## Technical Implementation

### Git Operations

```bash
# Configure Git for automation
git config user.name "MLOps Pipeline"
git config user.email "mlops@github.actions"

# Add MLflow data to repository
git add mlruns/ mlflow-artifacts/ models/ deployment/

# Commit with meaningful message
git commit -m "MLflow: Add experiment data from run <hash>"

# Push to repository
git push origin main
```

### Model Selection Logic

```python
# Find best model based on validation metrics
best_model = min(runs, key=lambda x: x.metrics['val_rmse'])

# Copy to deployment folder
shutil.copy(best_model.artifacts['model'], 'deployment/models/')

# Create metadata
metadata = {
    'run_id': best_model.run_id,
    'model_type': best_model.params['model_type'],
    'metrics': best_model.metrics,
    'timestamp': datetime.now().isoformat()
}
```

## Monitoring and Validation

### Pipeline Checks

The pipeline includes validation steps to ensure:
- MLflow data is properly committed
- Best models are correctly identified
- Deployment artifacts are available
- Git operations complete successfully

### Error Handling

Robust error handling ensures:
- Pipeline continues even if some MLflow operations fail
- Clear error messages for debugging
- Graceful degradation when Git operations fail

## Future Enhancements

1. **Model Versioning**: Implement semantic versioning for models
2. **Automated Rollback**: Add capability to rollback to previous model versions
3. **Performance Alerts**: Alert when model performance degrades
4. **A/B Testing**: Enable deployment of multiple model versions

## Conclusion

This MLflow persistence strategy provides a robust foundation for experiment tracking in CI/CD environments, ensuring that valuable model development insights are preserved and accessible across the entire team and project lifecycle.
