"""
Download California Housing dataset or create synthetic data if download fails.

This script:
- Downloads the California Housing dataset from Kaggle
- Saves it to the data directory
- Requires a Kaggle API token in ~/.kaggle/kaggle.json
- Creates synthetic data as fallback if download fails
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_dataset():
    """Download California Housing dataset from Kaggle."""
    logger.info("Downloading California Housing dataset from Kaggle...")
    
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = data_dir / 'california_housing.csv'
    
    # Check if dataset already exists
    if dataset_path.exists():
        logger.info(f"Dataset already exists at {dataset_path}")
        return True
    
    try:
        # Check if kaggle CLI is installed
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])
            logger.info("Kaggle package installed")
        except subprocess.CalledProcessError:
            logger.error("Failed to install Kaggle package")
            return False
        
        # Download the dataset
        # URL: https://www.kaggle.com/code/subhradeep88/house-price-predict-decision-tree-random-forest
        # We need to extract the actual dataset name from this code notebook
        
        # Try to download a common California Housing dataset
        try:
            subprocess.check_call(['kaggle', 'datasets', 'download', 'camnugent/california-housing-prices', '--path', 'data'])
            logger.info("Dataset downloaded from Kaggle")
            
            # Extract the zip file
            subprocess.check_call(['powershell', 'Expand-Archive', '-Path', 'data/california-housing-prices.zip', '-DestinationPath', 'data', '-Force'])
            logger.info("Dataset extracted")
            
            # Rename the file if needed
            housing_csv = list(data_dir.glob('*.csv'))[0]
            housing_csv.rename(dataset_path)
            logger.info(f"Dataset saved to {dataset_path}")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download dataset from Kaggle: {e}")
            
            # If Kaggle download fails, create a sample dataset
            logger.info("Creating sample California Housing dataset...")
            create_sample_dataset(dataset_path)
            return True
            
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        
        # Create a sample dataset as fallback
        logger.info("Creating sample California Housing dataset as fallback...")
        create_sample_dataset(dataset_path)
        return True


def create_sample_dataset(output_path):
    """Create a sample California Housing dataset for testing."""
    try:
        from sklearn.datasets import fetch_california_housing
        import numpy as np
        
        # Load the sklearn dataset
        housing = fetch_california_housing(as_frame=True)
        
        # Convert to DataFrame
        df = housing.data
        df['target'] = housing.target
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Sample dataset created and saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating sample dataset: {e}")
        
        # Create completely synthetic dataset if sklearn fails
        columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                  'Population', 'AveOccup', 'Latitude', 'Longitude', 'target']
        
        import numpy as np
        np.random.seed(42)
        
        # Generate random data
        n_samples = 1000
        data = {
            'MedInc': np.random.uniform(1, 15, n_samples),
            'HouseAge': np.random.uniform(1, 52, n_samples),
            'AveRooms': np.random.uniform(3, 10, n_samples),
            'AveBedrms': np.random.uniform(0.5, 2, n_samples),
            'Population': np.random.uniform(100, 5000, n_samples),
            'AveOccup': np.random.uniform(1, 6, n_samples),
            'Latitude': np.random.uniform(32, 42, n_samples),
            'Longitude': np.random.uniform(-125, -114, n_samples),
            'target': np.random.uniform(0.5, 5, n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Synthetic dataset created and saved to {output_path}")
        return True


if __name__ == "__main__":
    download_dataset()
