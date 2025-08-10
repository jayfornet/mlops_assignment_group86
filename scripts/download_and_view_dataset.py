#!/usr/bin/env python3
"""
Download and View California Housing Dataset

This script downloads the California Housing dataset and saves it as CSV
in the data/csv folder for easy viewing and analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.datasets import fetch_california_housing
import joblib
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    directories = ['data', 'data/csv', 'data/processed']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created/verified directory: {directory}")

def download_and_save_dataset():
    """Download California Housing dataset and save in multiple formats."""
    print("🚀 Downloading California Housing dataset...")
    
    try:
        # Fetch the dataset
        housing = fetch_california_housing()
        
        # Create DataFrame
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['target'] = housing.target
        
        print(f"✅ Dataset downloaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Save as CSV in data/csv folder
        csv_path = 'data/csv/california_housing.csv'
        df.to_csv(csv_path, index=False)
        print(f"💾 Dataset saved as CSV: {csv_path}")
        
        # Also save as joblib for compatibility
        joblib_path = 'data/california_housing.joblib'
        joblib.dump(housing, joblib_path)
        print(f"💾 Dataset saved as joblib: {joblib_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def display_dataset_overview(df):
    """Display comprehensive dataset overview."""
    if df is None:
        print("❌ No dataset to display")
        return
    
    print("\n" + "="*60)
    print("🔍 CALIFORNIA HOUSING DATASET OVERVIEW")
    print("="*60)
    
    print(f"\n📈 Dataset Information:")
    print(f"   • Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📋 Column Information:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        print(f"   {i:2d}. {col:<12} | {dtype:<8} | {null_count:>4} nulls")
    
    print(f"\n📊 First 10 rows:")
    print(df.head(10).to_string(index=False))
    
    print(f"\n📈 Statistical Summary:")
    print(df.describe().round(3))
    
    print(f"\n🔍 Data Quality Check:")
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isnull().sum().sum()
    print(f"   • Total cells: {total_cells:,}")
    print(f"   • Null cells: {null_cells:,}")
    print(f"   • Data completeness: {((total_cells - null_cells) / total_cells * 100):.2f}%")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"   • Duplicate rows: {duplicates:,}")
    
    print(f"\n🌍 Geographic Distribution (Latitude/Longitude):")
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        print(f"   • Latitude range: {df['Latitude'].min():.3f} to {df['Latitude'].max():.3f}")
        print(f"   • Longitude range: {df['Longitude'].min():.3f} to {df['Longitude'].max():.3f}")
        print(f"   • Geographic spread: California housing blocks")
    
    print(f"\n💰 Target Variable (House Values):")
    if 'target' in df.columns:
        target_stats = df['target'].describe()
        print(f"   • Range: ${target_stats['min']:.1f}k to ${target_stats['max']:.1f}k")
        print(f"   • Median: ${target_stats['50%']:.1f}k")
        print(f"   • Mean: ${target_stats['mean']:.1f}k")
    
    print("\n" + "="*60)

def save_sample_files(df):
    """Save sample files for easy viewing."""
    if df is None:
        return
    
    try:
        # Save first 100 rows as sample
        sample_path = 'data/csv/california_housing_sample_100.csv'
        df.head(100).to_csv(sample_path, index=False)
        print(f"📝 Sample (100 rows) saved: {sample_path}")
        
        # Save summary statistics
        summary_path = 'data/csv/california_housing_summary.csv'
        df.describe().to_csv(summary_path)
        print(f"📊 Summary statistics saved: {summary_path}")
        
        # Create a markdown report
        report_path = 'data/csv/dataset_report.md'
        with open(report_path, 'w') as f:
            f.write("# California Housing Dataset Report\n\n")
            f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Dataset Overview\n")
            f.write(f"- **Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns\n")
            f.write(f"- **File Location:** `data/csv/california_housing.csv`\n")
            f.write(f"- **Sample File:** `data/csv/california_housing_sample_100.csv`\n\n")
            
            f.write("## Column Descriptions\n")
            descriptions = {
                'MedInc': 'Median income in block group',
                'HouseAge': 'Median house age in block group', 
                'AveRooms': 'Average number of rooms per household',
                'AveBedrms': 'Average number of bedrooms per household',
                'Population': 'Block group population',
                'AveOccup': 'Average number of household members',
                'Latitude': 'Block group latitude',
                'Longitude': 'Block group longitude',
                'target': 'Median house value (in hundreds of thousands of dollars)'
            }
            
            for col in df.columns:
                desc = descriptions.get(col, 'No description available')
                f.write(f"- **{col}:** {desc}\n")
            
            f.write("\n## How to View the Data\n")
            f.write("1. **Full dataset:** Open `data/csv/california_housing.csv` in Excel, VS Code, or any CSV viewer\n")
            f.write("2. **Sample data:** Open `data/csv/california_housing_sample_100.csv` for quick viewing\n")
            f.write("3. **Summary stats:** Open `data/csv/california_housing_summary.csv`\n")
            
        print(f"📄 Dataset report saved: {report_path}")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not save sample files: {e}")

def main():
    """Main function to download and display dataset."""
    print("🏠 California Housing Dataset Downloader & Viewer")
    print("="*50)
    
    # Create directories
    create_directories()
    
    # Download and save dataset
    df = download_and_save_dataset()
    
    # Display overview
    display_dataset_overview(df)
    
    # Save sample files
    save_sample_files(df)
    
    if df is not None:
        print(f"\n🎉 SUCCESS! Dataset is ready for viewing:")
        print(f"   📂 Full dataset: data/csv/california_housing.csv")
        print(f"   📋 Sample (100 rows): data/csv/california_housing_sample_100.csv") 
        print(f"   📊 Summary stats: data/csv/california_housing_summary.csv")
        print(f"   📄 Report: data/csv/dataset_report.md")
        print(f"\n💡 You can now open these files in Excel, VS Code, or any CSV viewer!")
    else:
        print(f"\n❌ Failed to download dataset. Please check your internet connection.")

if __name__ == "__main__":
    main()
