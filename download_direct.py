"""
Simple script to fetch California Housing dataset from scikit-learn.
"""
from sklearn.datasets import fetch_california_housing
import pandas as pd
import joblib
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

try:
    # Download dataset directly from scikit-learn
    print("Downloading California Housing dataset from scikit-learn...")
    housing = fetch_california_housing(as_frame=True)
    
    # Extract features and target
    X = housing.data
    y = housing.target
    
    # Save as CSV
    print("Saving dataset as CSV...")
    df = X.copy()
    df['MedHouseVal'] = y
    df.to_csv('data/california_housing.csv', index=False)
    
    # Save as joblib
    print("Saving dataset as joblib...")
    joblib.dump({
        'data': X,
        'target': y,
        'feature_names': X.columns.tolist()
    }, 'data/california_housing.joblib')
    
    print('California Housing dataset downloaded directly from scikit-learn and saved.')
except Exception as e:
    print(f"Error downloading dataset: {e}")
