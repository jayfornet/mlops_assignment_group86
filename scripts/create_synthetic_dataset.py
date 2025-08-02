#!/usr/bin/env python3
"""
Create a synthetic California Housing dataset for demonstration purposes.
This script creates a synthetic dataset with the same structure as the original.
"""

import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

print("Creating synthetic California Housing dataset...")

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 20000

# Create synthetic features with appropriate distributions
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

# Create synthetic target (median house value) with some correlation to features
y = (
    3.0 * X['MedInc'] +  # Higher income → higher house value
    -0.1 * X['HouseAge'] +  # Newer houses are more valuable
    0.5 * X['AveRooms'] +  # More rooms → higher value
    -0.2 * X['Population'] / 1000 +  # Less crowded areas → higher value
    0.1 * np.abs(X['Latitude'] - 37.5) +  # Proximity to certain latitude
    0.1 * np.abs(X['Longitude'] + 122) +  # Proximity to certain longitude
    np.random.normal(0, 0.5, n_samples)  # Random noise
)

# Normalize target to be similar to original dataset (values around 0.5 to 5.0)
y = np.clip(y / 5.0, 0.5, 5.0)

# Save as CSV
csv_path = data_dir / "california_housing.csv"
print(f"Saving dataset to {csv_path}...")

# Combine features and target into a single DataFrame
df = X.copy()
df['MedHouseVal'] = y

# Save to CSV
df.to_csv(csv_path, index=False)
print(f"Dataset saved as CSV with {len(df)} rows and {len(df.columns)} columns")

# Additionally save as joblib for faster loading
joblib_path = data_dir / "california_housing.joblib"
print(f"Saving dataset to {joblib_path} for faster loading...")
joblib.dump({
    "data": X, 
    "target": y, 
    "feature_names": X.columns.tolist()
}, joblib_path)
print("Dataset saved as joblib")

print("Done! The synthetic California Housing dataset is now ready for use.")
print("This synthetic dataset preserves the structure of the original California Housing dataset,")
print("but uses generated data for demonstration purposes.")
