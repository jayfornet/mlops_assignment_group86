#!/usr/bin/env python3
"""
Download the California Housing dataset once and save it to disk.
This script should be run once to prepare the data for the pipeline.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import joblib
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

print("Downloading California Housing dataset...")
# Load the dataset from scikit-learn
housing = fetch_california_housing(as_frame=True)

# Extract features and target
X = housing.data
y = housing.target

# Save as CSV
dataset_path = data_dir / "california_housing.csv"
print(f"Saving dataset to {dataset_path}...")

# Combine features and target into a single DataFrame
df = X.copy()
df['MedHouseVal'] = y

# Save to CSV
df.to_csv(dataset_path, index=False)
print(f"Dataset saved as CSV with {len(df)} rows and {len(df.columns)} columns")

# Additionally save as joblib for faster loading
joblib_path = data_dir / "california_housing.joblib"
print(f"Saving dataset to {joblib_path} for faster loading...")
joblib.dump({"data": X, "target": y, "feature_names": X.columns.tolist()}, joblib_path)
print("Dataset saved as joblib")

print("Done! The California Housing dataset is now ready for use.")
