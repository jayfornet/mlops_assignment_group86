"""
Data loading and preprocessing module for California Housing dataset.

This module handles:
- Loading the California Housing dataset from local file or Kaggle
- Data preprocessing and feature engineering
- Data validation and quality checks
- Train/test splitting with proper random state
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os
import joblib
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processor class for California Housing dataset.
    
    Handles data loading, preprocessing, and splitting operations.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize DataProcessor with random state for reproducibility.
        
        Args:
            random_state (int): Random state for reproducible results
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = 'MedHouseVal'
        
    def load_data(self):
        """
        Load California Housing dataset from local file or create synthetic data.
        
        Returns:
            tuple: (X, y) features and target arrays
        """
        logger.info("Loading California Housing dataset...")
        
        try:
            # First try to load from local files
            csv_path = os.path.join('data', 'california_housing.csv')
            joblib_path = os.path.join('data', 'california_housing.joblib')
            
            if os.path.exists(csv_path):
                logger.info(f"Loading dataset from local CSV file: {csv_path}")
                df = pd.read_csv(csv_path)
                
                # Assume the last column is the target (MedHouseVal)
                if 'MedHouseVal' in df.columns:
                    X = df.drop('MedHouseVal', axis=1)
                    y = df['MedHouseVal']
                else:
                    # If MedHouseVal column not found, assume last column is target
                    X = df.iloc[:, :-1]
                    y = df.iloc[:, -1]
                
                self.feature_names = X.columns.tolist()
                logger.info(f"Dataset loaded from local CSV with shape: {X.shape}")
                return X, y
                
            elif os.path.exists(joblib_path):
                logger.info(f"Loading dataset from local joblib file")
                housing_dict = joblib.load(joblib_path)
                
                X = housing_dict['data']
                y = housing_dict['target']
                self.feature_names = housing_dict['feature_names']
                
                logger.info(f"Dataset loaded from local joblib with shape: {X.shape}")
                return X, y
                
            # If local files don't exist, try to load from scikit-learn
            logger.info("Local dataset files not found. Trying to load from scikit-learn...")
            try:
                housing = fetch_california_housing(as_frame=True)
                
                # Extract features and target
                X = housing.data
                y = housing.target
                
                # Store feature names for later use
                self.feature_names = X.columns.tolist()
                
                logger.info(f"Dataset loaded from scikit-learn with shape: {X.shape}")
                
                # Save dataset to local files for faster access later
                os.makedirs('data', exist_ok=True)
                
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
                
                logger.info("Dataset saved to local files for future use")
                return X, y
                
            except Exception as e:
                logger.warning(f"Failed to load from scikit-learn: {e}")
                logger.info("Creating synthetic California Housing dataset...")
                
                # Create synthetic dataset
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
                os.makedirs('data', exist_ok=True)
                df = X.copy()
                df['MedHouseVal'] = y
                df.to_csv('data/california_housing.csv', index=False)
                
                # Save as joblib
                joblib.dump({
                    'data': X,
                    'target': y,
                    'feature_names': X.columns.tolist()
                }, 'data/california_housing.joblib')
                
                self.feature_names = X.columns.tolist()
                logger.info(f"Synthetic dataset created with shape: {X.shape}")
                return X, y
                
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
            
            logger.info(f"Features: {self.feature_names}")
            logger.info(f"Target: {self.target_name}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def preprocess_data(self, X, y):
        """
        Preprocess the data by handling missing values and outliers.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): Target vector
            
        Returns:
            tuple: (X_processed, y_processed) preprocessed data
        """
        logger.info("Starting data preprocessing...")
        
        # Create copies to avoid modifying original data
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Check for missing values
        missing_values = X_processed.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Found missing values: {missing_values[missing_values > 0]}")
        else:
            logger.info("No missing values found")
        
        # Basic data quality checks
        logger.info("Performing data quality checks...")
        
        # Check for negative values in features that should be positive
        positive_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
        for feature in positive_features:
            if feature in X_processed.columns:
                negative_count = (X_processed[feature] < 0).sum()
                if negative_count > 0:
                    logger.warning(f"Found {negative_count} negative values in {feature}")
        
        # Remove extreme outliers (beyond 3 standard deviations)
        logger.info("Removing extreme outliers...")
        initial_size = len(X_processed)
        
        # Calculate Z-scores for numerical features
        z_scores = np.abs((X_processed - X_processed.mean()) / X_processed.std())
        
        # Keep rows where all features have Z-score < 3
        outlier_mask = (z_scores < 3).all(axis=1)
        X_processed = X_processed[outlier_mask]
        y_processed = y_processed[outlier_mask]
        
        removed_count = initial_size - len(X_processed)
        logger.info(f"Removed {removed_count} outliers ({removed_count/initial_size*100:.2f}%)")
        
        # Log final data statistics
        logger.info(f"Final dataset shape: {X_processed.shape}")
        logger.info("Data preprocessing completed successfully")
        
        return X_processed, y_processed
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): Target vector
            test_size (float): Proportion of data for test set
            val_size (float): Proportion of training data for validation set
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Splitting data (test: {test_size}, val: {val_size})...")
        
        try:
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=None
            )
            
            # Second split: separate train and validation from remaining data
            val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
            )
            
            logger.info(f"Train set shape: {X_train.shape}")
            logger.info(f"Validation set shape: {X_val.shape}")
            logger.info(f"Test set shape: {X_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def scale_features(self, X_train, X_val, X_test):
        """
        Scale features using StandardScaler fitted on training data.
        
        Args:
            X_train (DataFrame): Training features
            X_val (DataFrame): Validation features
            X_test (DataFrame): Test features
            
        Returns:
            tuple: (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        logger.info("Scaling features using StandardScaler...")
        
        try:
            # Fit scaler on training data only
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            
            # Transform validation and test data using fitted scaler
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            logger.info("Feature scaling completed successfully")
            
            return X_train_scaled, X_val_scaled, X_test_scaled
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise
    
    def save_data(self, data_dict, data_dir="data"):
        """
        Save processed data to CSV files.
        
        Args:
            data_dict (dict): Dictionary containing data splits
            data_dir (str): Directory to save data files
        """
        logger.info(f"Saving processed data to {data_dir}/...")
        
        try:
            # Create data directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Save each data split
            for name, data in data_dict.items():
                filepath = os.path.join(data_dir, f"{name}.csv")
                data.to_csv(filepath, index=False)
                logger.info(f"Saved {name} to {filepath}")
            
            logger.info("Data saving completed successfully")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def get_data_summary(self, X, y):
        """
        Generate summary statistics for the dataset.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): Target vector
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist(),
            'target_stats': {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max())
            },
            'feature_stats': X.describe().to_dict()
        }
        
        return summary


def main():
    """
    Main function to demonstrate data processing pipeline.
    """
    # Initialize data processor
    processor = DataProcessor(random_state=42)
    
    # Load raw data
    X, y = processor.load_data()
    
    # Preprocess data
    X_processed, y_processed = processor.preprocess_data(X, y)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
        X_processed, y_processed
    )
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_features(
        X_train, X_val, X_test
    )
    
    # Prepare data dictionary for saving
    data_dict = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    
    # Save processed data
    processor.save_data(data_dict)
    
    # Generate and log summary
    summary = processor.get_data_summary(X_processed, y_processed)
    logger.info(f"Data processing completed. Summary: {summary}")
    
    return processor, data_dict


if __name__ == "__main__":
    main()
