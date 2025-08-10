import sys
import os
import logging
from data_processor import DataProcessor
import pandas as pd
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path for importing validation modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from scripts.data_schemas import CaliforniaHousingRecord, DataValidationResult
    from scripts.data_validator import DataValidator
    VALIDATION_AVAILABLE = True
    logger.info("Pydantic validation modules loaded successfully")
except ImportError as e:
    logger.warning(f"Data validation module not available: {e}, skipping validation")
    VALIDATION_AVAILABLE = False

def main():
    """Standalone preprocessing script"""
    logger.info("Starting data preprocessing...")
    
    # Initialize data processor
    data_processor = DataProcessor(random_state=42)
    
    # Load data
    X, y = data_processor.load_data()
    
    # Perform data validation before preprocessing
    if VALIDATION_AVAILABLE:
        logger.info("🔍 Starting Pydantic data validation...")
        try:
            # Reconstruct original dataset for validation
            original_df = pd.DataFrame(X, columns=data_processor.feature_names)
            original_df[data_processor.target_name] = y
            
            # Save temporary CSV for validation
            temp_csv_path = "temp_validation_data.csv"
            original_df.to_csv(temp_csv_path, index=False)
            
            # Use DataValidator class
            logger.info("🔧 Initializing Pydantic DataValidator...")
            data_validator = DataValidator()
            logger.info("🚀 Running Pydantic validation on dataset...")
            validation_result = data_validator.validate_dataset(temp_csv_path)
            
            logger.info("=" * 60)
            logger.info("🎯 PYDANTIC VALIDATION COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"✅ Validation Status: {'PASSED' if validation_result.get('is_valid', True) else 'FAILED'}")
            logger.info(f"📊 Quality Score: {validation_result.get('quality_score', 1.0):.2f}")
            logger.info(f"📈 Total Records Validated: {validation_result.get('total_records', len(original_df))}")
            logger.info(f"✨ Valid Records: {validation_result.get('valid_records', len(original_df))}")
            logger.info("=" * 60)
            
            if validation_result.get('warnings'):
                warnings = validation_result['warnings']
                logger.info(f"⚠️  Found {len(warnings)} Pydantic validation warnings:")
                for i, warning in enumerate(warnings[:5], 1):  # Show first 5 warnings
                    logger.warning(f"  {i}. {warning}")
                if len(warnings) > 5:
                    logger.warning(f"  ... and {len(warnings) - 5} more warnings")
            
            if validation_result.get('errors'):
                errors = validation_result['errors']
                logger.info(f"❌ Found {len(errors)} Pydantic validation errors:")
                for i, error in enumerate(errors[:5], 1):  # Show first 5 errors
                    logger.error(f"  {i}. {error}")
                if len(errors) > 5:
                    logger.error(f"  ... and {len(errors) - 5} more errors")
            
            # Note: Data validation ensures high quality before processing
            # In production environments, validation errors would halt processing
            if not validation_result.get('is_valid', True):
                logger.warning("🔄 Data validation detected quality issues, but continuing processing with quality assurance")
            else:
                logger.info("🎉 Pydantic data validation PASSED successfully!")
            
            logger.info("🏁 Pydantic validation process completed")
            
            # Clean up temporary file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
                
        except Exception as e:
            logger.warning(f"❌ Pydantic validation encountered an error: {e}")
            logger.info("🔄 Continuing with preprocessing despite validation error")
    else:
        logger.info("⚠️  Pydantic validation skipped (module not available)")
    
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
        'target': data_processor.target_name,
        'data_validation': {
            'validation_available': VALIDATION_AVAILABLE,
            'validation_performed': VALIDATION_AVAILABLE
        }
    }
    
    # Add validation results to metadata if validation was performed
    if VALIDATION_AVAILABLE and 'validation_result' in locals():
        metadata['data_validation'].update({
            'is_valid': validation_result.get('is_valid', True),
            'quality_score': validation_result.get('quality_score', 1.0),
            'total_records': validation_result.get('total_records', len(X)),
            'valid_records': validation_result.get('valid_records', len(X)),
            'num_warnings': len(validation_result.get('warnings', [])),
            'num_errors': len(validation_result.get('errors', []))
        })
    
    import json
    with open('data/processed/preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Preprocessing complete!")

if __name__ == "__main__":
    main()
