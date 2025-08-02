"""
Create a simpler synthetic California Housing dataset.

This script creates a synthetic dataset that matches the structure of the
California Housing dataset but doesn't require downloading from Kaggle.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path('data')
data_dir.mkdir(parents=True, exist_ok=True)

# Define dataset path
dataset_path = data_dir / 'california_housing.csv'

# Check if dataset already exists
if dataset_path.exists():
    print(f"Dataset already exists at {dataset_path}")
    exit(0)

# Generate synthetic data
print("Creating synthetic California Housing dataset...")
np.random.seed(42)

# Generate random data
n_samples = 20000
data = {
    'MedInc': np.random.uniform(1, 15, n_samples),
    'HouseAge': np.random.uniform(1, 52, n_samples),
    'AveRooms': np.random.uniform(3, 10, n_samples),
    'AveBedrms': np.random.uniform(0.5, 2, n_samples),
    'Population': np.random.uniform(100, 5000, n_samples),
    'AveOccup': np.random.uniform(1, 6, n_samples),
    'Latitude': np.random.uniform(32, 42, n_samples),
    'Longitude': np.random.uniform(-125, -114, n_samples)
}

# Create target variable with realistic relationship to features
target = (
    0.5 * data['MedInc'] +
    0.2 * data['HouseAge'] +
    0.1 * data['AveRooms'] -
    0.05 * data['AveBedrms'] -
    0.01 * data['Population'] -
    0.1 * data['AveOccup'] +
    0.05 * np.abs(data['Latitude'] - 37) +  # Distance from central California
    0.05 * np.abs(data['Longitude'] + 120) +  # Distance from central California
    np.random.normal(0, 0.5, n_samples)  # Add some noise
)

# Scale target to realistic housing prices (in 100k)
target = (target - target.min()) / (target.max() - target.min()) * 4.5 + 0.5

# Create DataFrame
df = pd.DataFrame(data)
df['target'] = target

# Save to CSV
df.to_csv(dataset_path, index=False)
print(f"Synthetic dataset created and saved to {dataset_path}")
print(f"Dataset shape: {df.shape}")
print(f"Feature names: {list(df.columns[:-1])}")
print(f"Target name: target")
print(f"Target statistics: min={target.min():.2f}, max={target.max():.2f}, mean={target.mean():.2f}")
