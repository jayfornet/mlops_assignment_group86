"""
Data Schemas for California Housing Dataset using Pydantic

This module defines Pydantic models for validating the California Housing dataset
and data monitoring results.
"""

from pydantic import BaseModel, ValidationError, Field, validator
from typing import Optional, List, Literal
from datetime import datetime
import pandas as pd


class CaliforniaHousingRecord(BaseModel):
    """Pydantic model for California Housing dataset validation."""
    
    MedInc: float = Field(..., gt=0, le=20, description="Median income in block group")
    HouseAge: float = Field(..., ge=0, le=100, description="Median house age in block group")
    AveRooms: float = Field(..., gt=0, le=50, description="Average number of rooms per household")
    AveBedrms: float = Field(..., gt=0, le=10, description="Average number of bedrooms per household")
    Population: float = Field(..., gt=0, le=50000, description="Block group population")
    AveOccup: float = Field(..., gt=0, le=20, description="Average number of household members")
    Latitude: float = Field(..., ge=32, le=42, description="Block group latitude")
    Longitude: float = Field(..., ge=-125, le=-114, description="Block group longitude")
    target: Optional[float] = Field(None, gt=0, description="Median house value (target variable)")
    
    @validator('AveRooms')
    def validate_rooms_reasonable(cls, v, values):
        """Validate that average rooms per household is reasonable."""
        if v > 20:
            raise ValueError('Average rooms per household seems unrealistic (>20)')
        return v
    
    @validator('AveBedrms')
    def validate_bedrooms_vs_rooms(cls, v, values):
        """Validate that bedrooms don't exceed total rooms."""
        if 'AveRooms' in values and v > values['AveRooms']:
            raise ValueError('Average bedrooms cannot exceed average rooms')
        return v
    
    @validator('AveOccup')
    def validate_occupancy(cls, v):
        """Validate reasonable occupancy levels."""
        if v > 15:
            raise ValueError('Average occupancy seems too high (>15 people per household)')
        return v
    
    class Config:
        str_strip_whitespace = True
        validate_assignment = True
        schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.023,
                "Population": 322.0,
                "AveOccup": 2.555,
                "Latitude": 37.88,
                "Longitude": -122.23,
                "target": 4.526
            }
        }


class DataValidationResult(BaseModel):
    """Model for data validation results."""
    
    is_valid: bool = Field(..., description="Whether the dataset passed validation")
    total_records: int = Field(..., ge=0, description="Total number of records")
    valid_records: int = Field(..., ge=0, description="Number of valid records")
    invalid_records: int = Field(..., ge=0, description="Number of invalid records")
    validation_errors: List[str] = Field(default=[], description="List of validation errors")
    data_quality_score: float = Field(..., ge=0, le=100, description="Data quality score (0-100)")
    validation_timestamp: datetime = Field(default_factory=datetime.now, description="When validation was performed")
    schema_version: str = Field(default="1.0.0", description="Schema version used for validation")
    
    @property
    def quality_grade(self) -> str:
        """Return quality grade based on score."""
        if self.data_quality_score >= 95:
            return "A+"
        elif self.data_quality_score >= 90:
            return "A"
        elif self.data_quality_score >= 80:
            return "B"
        elif self.data_quality_score >= 70:
            return "C"
        else:
            return "F"
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.total_records == 0:
            return 0.0
        return (self.invalid_records / self.total_records) * 100
    
    class Config:
        schema_extra = {
            "example": {
                "is_valid": True,
                "total_records": 20640,
                "valid_records": 20635,
                "invalid_records": 5,
                "validation_errors": ["Row 1000: Latitude - ensure this value is greater than or equal to 32"],
                "data_quality_score": 99.98,
                "validation_timestamp": "2025-08-10T10:30:00",
                "schema_version": "1.0.0"
            }
        }


class DataMonitoringConfig(BaseModel):
    """Configuration for data monitoring and validation."""
    
    quality_threshold: float = Field(90.0, ge=0, le=100, description="Minimum quality threshold for valid data")
    max_errors_to_report: int = Field(10, ge=1, le=100, description="Maximum validation errors to report")
    enable_detailed_logging: bool = Field(True, description="Enable detailed validation logging")
    auto_trigger_pipeline: bool = Field(True, description="Auto-trigger MLOps pipeline on valid data")
    validation_sample_size: Optional[int] = Field(None, ge=100, description="Sample size for validation (None = full dataset)")
    strict_mode: bool = Field(False, description="Enable strict validation mode")
    
    class Config:
        schema_extra = {
            "example": {
                "quality_threshold": 95.0,
                "max_errors_to_report": 5,
                "enable_detailed_logging": True,
                "auto_trigger_pipeline": True,
                "validation_sample_size": 1000,
                "strict_mode": False
            }
        }


class DataStatistics(BaseModel):
    """Model for dataset statistics."""
    
    total_records: int
    total_columns: int
    missing_values: int
    duplicate_records: int
    memory_usage_mb: float
    min_values: dict
    max_values: dict
    mean_values: dict
    std_values: dict
    
    class Config:
        schema_extra = {
            "example": {
                "total_records": 20640,
                "total_columns": 9,
                "missing_values": 207,
                "duplicate_records": 0,
                "memory_usage_mb": 1.2,
                "min_values": {"MedInc": 0.4999, "HouseAge": 1.0},
                "max_values": {"MedInc": 15.0001, "HouseAge": 52.0},
                "mean_values": {"MedInc": 3.87, "HouseAge": 28.64},
                "std_values": {"MedInc": 1.9, "HouseAge": 12.59}
            }
        }
