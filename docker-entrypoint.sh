#!/bin/bash
set -e

# Create necessary directories
mkdir -p data models logs results mlruns mlflow-artifacts

# Check if dataset exists before trying to download
if [ ! -f "data/california_housing.csv" ] || [ ! -f "data/california_housing.joblib" ]; then
    echo "Creating synthetic California Housing dataset..."
    python -c "
import numpy as np
import pandas as pd
import joblib

# Create synthetic dataset with same structure as California Housing
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
df = X.copy()
df['MedHouseVal'] = y
df.to_csv('data/california_housing.csv', index=False)

# Save as joblib
joblib.dump({
    'data': X,
    'target': y,
    'feature_names': X.columns.tolist()
}, 'data/california_housing.joblib')

print('Synthetic California Housing dataset created and saved.')
"
else
    echo "California Housing dataset already exists."
fi

# Run setup script with Docker mode flag
echo "Running setup script in Docker mode..."
python setup.py --docker-mode

# Start the API server
echo "Starting API server..."
exec python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
