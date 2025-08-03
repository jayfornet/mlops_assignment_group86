import sys
import os
import logging
from data_processor import DataProcessor
import pandas as pd
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Standalone preprocessing script"""
    logger.info("Starting data preprocessing...")
    
    # Initialize data processor
    data_processor = DataProcessor(random_state=42)
    
    # Load data
    X, y = data_processor.load_data()
    
    # Preprocess data
    X_processed, y_processed = data_processor.preprocess_data(X, y)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        X_processed, y_processed
    )
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = data_processor.scale_features(
        X_train, X_val, X_test
    )
    
    # Save preprocessed data
    preprocessed_data = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled, 
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': data_processor.scaler,
        'feature_names': data_processor.feature_names
    }
    
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump(preprocessed_data, 'data/processed/preprocessed_data.joblib')
    logger.info("Preprocessed data saved to data/processed/preprocessed_data.joblib")
    
    # Save metadata about preprocessing
    metadata = {
        'num_samples': {
            'total': len(X_processed),
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        },
        'features': data_processor.feature_names,
        'target': data_processor.target_name
    }
    
    import json
    with open('data/processed/preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Preprocessing complete!")

if __name__ == "__main__":
    main()
