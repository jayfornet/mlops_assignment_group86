"""
FastAPI application for California Housing price prediction.

This module provides:
- RESTful API endpoints for housing price prediction
- Input validation using Pydantic models
- Health checks and monitoring endpoints
- Request/response logging
- Error handling and validation
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, field_validator
import pandas as pd
import numpy as np
import joblib
import logging
import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Setup logging
os.makedirs('logs', exist_ok=True)  # Create logs directory if it doesn't exist
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total number of predictions made')
PREDICTION_HISTOGRAM = Histogram('prediction_duration_seconds', 'Time spent on predictions')
ERROR_COUNTER = Counter('prediction_errors_total', 'Total number of prediction errors')

# Define lifespan context manager for FastAPI app
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for application startup and shutdown."""
    # Startup events
    logger.info("Starting California Housing Prediction API...")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Initialize model manager (defined later in the file)
    global model_manager
    logger.info(f"Model loaded: {model_manager.is_loaded() if 'model_manager' in globals() else False}")
    
    yield
    
    # Shutdown events
    logger.info("Shutting down California Housing Prediction API...")

# Initialize FastAPI app
app = FastAPI(
    title="California Housing Price Prediction API",
    description="MLOps pipeline for predicting housing prices using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HousingInput(BaseModel):
    """
    Input model for housing price prediction.
    
    All features from the California Housing dataset:
    - MedInc: Median income in block group
    - HouseAge: Median house age in block group
    - AveRooms: Average number of rooms per household
    - AveBedrms: Average number of bedrooms per household
    - Population: Block group population
    - AveOccup: Average number of household members
    - Latitude: Latitude of the block group
    - Longitude: Longitude of the block group
    """
    
    MedInc: float = Field(..., ge=0, le=20, description="Median income (in tens of thousands)")
    HouseAge: float = Field(..., ge=0, le=60, description="Median house age")
    AveRooms: float = Field(..., ge=1, le=20, description="Average rooms per household")
    AveBedrms: float = Field(..., ge=0, le=5, description="Average bedrooms per household")
    Population: float = Field(..., ge=1, le=50000, description="Block group population")
    AveOccup: float = Field(..., ge=1, le=20, description="Average occupancy")
    Latitude: float = Field(..., ge=30, le=45, description="Latitude")
    Longitude: float = Field(..., ge=-130, le=-110, description="Longitude")
    
    @field_validator('AveBedrms')
    def validate_bedrooms_ratio(cls, v, info):
        """Validate that bedrooms don't exceed rooms."""
        values = info.data
        if 'AveRooms' in values and v > values['AveRooms']:
            raise ValueError('Average bedrooms cannot exceed average rooms')
        return v
    
    @field_validator('AveOccup')
    def validate_occupancy(cls, v, info):
        """Validate reasonable occupancy levels."""
        if v > 10:  # Very high occupancy warning
            logger.warning(f"High occupancy detected: {v}")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.023,
                "Population": 322.0,
                "AveOccup": 2.555,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    prediction: float = Field(..., description="Predicted housing price (in hundreds of thousands)")
    prediction_id: str = Field(..., description="Unique identifier for this prediction")
    model_version: str = Field(..., description="Version of the model used")
    timestamp: str = Field(..., description="Timestamp of the prediction")
    input_features: Dict = Field(..., description="Input features used for prediction")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 4.526,
                "prediction_id": "pred_123456789",
                "model_version": "gradient_boosting_v1.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "input_features": {
                    "MedInc": 8.3252,
                    "HouseAge": 41.0,
                    "AveRooms": 6.984,
                    "AveBedrms": 1.024,
                    "Population": 322.0,
                    "AveOccup": 2.555,
                    "Latitude": 37.88,
                    "Longitude": -122.23
                }
            }
        }
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    checks: Optional[Dict[str, Any]] = Field(None, description="Detailed health check information")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "model_loaded": True,
                "version": "1.0.0",
                "checks": {
                    "system": {
                        "cpu": 0.2,
                        "memory": 0.4,
                        "disk": 0.1
                    },
                    "model": {
                        "model_name": "gradient_boosting",
                        "version": "1.0"
                    }
                }
            }
        }
    )


class ModelManager:
    """
    Model manager for loading and managing the trained model.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_metadata = None
        self.feature_names = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler."""
        try:
            # Find the best model file
            model_dir = "models"
            if not os.path.exists(model_dir):
                logger.error(f"Model directory not found: {model_dir}")
                return
            
            # Look for model files in order of preference
            model_files = []
            # First check for specific model types in order of preference
            for model_type in ['random_forest', 'gradient_boosting', 'linear_regression']:
                model_files = [f for f in os.listdir(model_dir) if f.startswith(model_type) and f.endswith('_best_model.joblib')]
                if model_files:
                    break
            
            # If no specific best model files, look for any best model files
            if not model_files:
                model_files = [f for f in os.listdir(model_dir) if f.endswith('_best_model.joblib')]
            
            # If still no files, look for any joblib files
            if not model_files:
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
            
            if not model_files:
                logger.error("No trained model found. Please train a model first.")
                return
            
            # Try loading models one by one until one succeeds
            for model_file in model_files:
                model_path = os.path.join(model_dir, model_file)
                try:
                    logger.info(f"Attempting to load model from: {model_path}")
                    try:
                        # First try with joblib (standard approach)
                        self.model = joblib.load(model_path)
                        logger.info(f"Successfully loaded model with joblib: {model_path}")
                    except Exception as joblib_error:
                        # If joblib fails with numpy incompatibility, try pickle as fallback
                        if "numpy.dtype size changed" in str(joblib_error):
                            logger.warning(f"Binary incompatibility detected with {model_path}: {joblib_error}")
                            logger.info("Trying alternative loading method with pickle...")
                            
                            import pickle
                            with open(model_path, 'rb') as f:
                                self.model = pickle.load(f)
                            logger.info(f"Successfully loaded model with pickle fallback: {model_path}")
                        else:
                            # Re-raise if it's not a compatibility issue
                            raise
                    
                    # Once we have a working model, try to load its metadata
                    self._load_model_metadata(model_dir, model_file)
                    
                    # Success - we have a working model
                    break
                except Exception as e:
                    logger.warning(f"Error loading model {model_file}: {str(e)}")
                    continue
            
            # If no model could be loaded after trying all files
            if self.model is None:
                logger.error("All model loading attempts failed.")
                return
                
            # Load scaler (would be saved during training)
            scaler_path = os.path.join(model_dir, "scaler.joblib")
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    logger.info("Scaler loaded successfully")
                except Exception as e:
                    logger.warning(f"Error loading scaler: {str(e)}")
                    self.scaler = None
            else:
                logger.warning("Scaler not found. Predictions may not work correctly.")
                
        except Exception as e:
            logger.error(f"Error in model loading process: {str(e)}")
            self.model = None
    
    def _load_model_metadata(self, model_dir, model_file):
        """Helper method to load model metadata."""
        try:
            # Try different possible metadata filenames
            base_name = os.path.splitext(model_file)[0]
            possible_metadata_files = [
                base_name.replace('_best_model', '_metadata') + '.json',
                base_name + '_metadata.json',
                base_name.replace('_model', '_metadata') + '.json',
                'model_metadata.json'
            ]
            
            for metadata_file in possible_metadata_files:
                metadata_path = os.path.join(model_dir, metadata_file)
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.model_metadata = json.load(f)
                    logger.info(f"Model metadata loaded successfully from {metadata_path}")
                    
                    # Extract feature names from metadata if available
                    if 'features' in self.model_metadata:
                        self.feature_names = self.model_metadata['features']
                        logger.info(f"Using feature names from metadata: {self.feature_names}")
                    
                    return True
            
            logger.warning("No model metadata found. Using default feature names.")
            return False
            
        except Exception as e:
            logger.warning(f"Error loading model metadata: {str(e)}")
            return False
    
    def predict(self, input_data: HousingInput) -> float:
        """
        Make prediction using the loaded model.
        
        Args:
            input_data (HousingInput): Input features
            
        Returns:
            float: Predicted housing price
        """
        if self.model is None:
            logger.warning("Model not loaded, using fallback prediction")
            # Use a simple fallback prediction based on median income
            # This ensures the API doesn't fail completely if model loading failed
            return float(input_data.MedInc * 0.5)
        
        try:
            # Convert input to DataFrame
            input_dict = input_data.model_dump()
            input_df = pd.DataFrame([input_dict])
            
            # Ensure correct feature order
            input_df = input_df[self.feature_names]
            
            # Scale features if scaler is available
            if self.scaler is not None:
                try:
                    input_scaled = self.scaler.transform(input_df)
                    input_df = pd.DataFrame(input_scaled, columns=self.feature_names)
                except Exception as e:
                    logger.warning(f"Error scaling input: {str(e)}")
                    # Continue with unscaled data
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            # Fallback to a simple rule-based prediction
            return float(input_data.MedInc * 0.5)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if self.model_metadata:
            return {
                "model_name": self.model_metadata.get("model_name", "unknown"),
                "training_timestamp": self.model_metadata.get("training_timestamp", "unknown"),
                "version": "1.0"
            }
        return {"model_name": "unknown", "version": "1.0"}


class DatabaseManager:
    """
    Database manager for logging predictions.
    """
    
    def __init__(self, db_path="logs/predictions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for logging."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    input_features TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    model_version TEXT,
                    response_time_ms REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def log_prediction(self, prediction_id: str, input_data: Dict, 
                      prediction: float, model_version: str, response_time: float):
        """Log prediction to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (id, timestamp, input_features, prediction, model_version, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                prediction_id,
                datetime.now().isoformat(),
                json.dumps(input_data),
                prediction,
                model_version,
                response_time
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")


# Initialize global instances
model_manager = ModelManager()
db_manager = DatabaseManager()


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "California Housing Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint that provides system and model status.
    
    Returns:
        JSON response with health status information
    """
    try:
        # Import health check module
        from src.api.health_check import check_health, check_system_health, get_model_info
        
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        # Get system health status
        system_health = check_system_health()
        
        # Check if model is loaded
        model_loaded = model_manager.is_loaded()
        
        # Get detailed model info
        model_info = {
            "loaded": model_loaded,
            "model_type": model_manager.model_metadata.get("model_type", "unknown") if model_manager.model_metadata else "unknown",
            "feature_count": len(model_manager.feature_names) if model_manager.feature_names else 0,
            "scaler_available": model_manager.scaler is not None
        }
        
        # Add model path information for debugging
        try:
            model_files = []
            if os.path.exists("models"):
                model_files = [f for f in os.listdir("models") if f.endswith('.joblib')]
            model_info["available_model_files"] = model_files
        except Exception as e:
            model_info["model_listing_error"] = str(e)
        
        # Determine overall status
        if not model_loaded:
            status = "warning"  # Model not loaded but API can still function with fallback
        elif system_health.get("status") == "critical" or "error" in system_health:
            status = "error"
        elif system_health.get("status") == "warning":
            status = "warning"
        else:
            status = "healthy"
        
        # Build response
        health_data = {
            "status": status,
            "timestamp": timestamp,
            "model_loaded": model_loaded,
            "version": "1.0.0",
            "checks": {
                "system": system_health,
                "model": model_info
            }
        }
        
        return health_data
    except Exception as e:
        logger.error(f"Error in health check endpoint: {e}")
        # Return a simplified response with 500 status code if the health check fails
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": False,
                "version": "1.0.0",
                "message": "Failed to perform health check",
                "error": str(e)
            }
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_housing_price(input_data: HousingInput):
    """
    Predict housing price based on input features.
    
    Args:
        input_data (HousingInput): Housing features
        
    Returns:
        PredictionResponse: Prediction result with metadata
    """
    start_time = datetime.now()
    prediction_id = f"pred_{uuid.uuid4().hex[:8]}"
    
    try:
        # Make prediction
        with PREDICTION_HISTOGRAM.time():
            prediction = model_manager.predict(input_data)
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get model info
        model_info = model_manager.get_model_info()
        
        # Create response
        response = PredictionResponse(
            prediction=prediction,
            prediction_id=prediction_id,
            model_version=f"{model_info['model_name']}_v{model_info['version']}",
            timestamp=start_time.isoformat(),
            input_features=input_data.model_dump()
        )
        
        # Log prediction
        db_manager.log_prediction(
            prediction_id=prediction_id,
            input_data=input_data.model_dump(),
            prediction=prediction,
            model_version=response.model_version,
            response_time=response_time
        )
        
        # Update metrics
        PREDICTION_COUNTER.inc()
        
        logger.info(f"Prediction made: {prediction_id}, price: {prediction:.4f}, time: {response_time:.2f}ms")
        
        return response
        
    except HTTPException:
        ERROR_COUNTER.inc()
        raise
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/predictions/recent")
async def get_recent_predictions(limit: int = 10):
    """Get recent predictions from database."""
    try:
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            predictions.append({
                "id": row[0],
                "timestamp": row[1],
                "input_features": json.loads(row[2]),
                "prediction": row[3],
                "model_version": row[4],
                "response_time_ms": row[5]
            })
        
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Error fetching recent predictions: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching predictions")


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True
    )
