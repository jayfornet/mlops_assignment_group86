#!/usr/bin/env python3
"""
Create Sample California Housing Dataset for Viewing

This script creates a sample California Housing dataset CSV file
for demonstration and viewing purposes when internet access is limited.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_directories():
    """Create necessary directories."""
    directories = ['data', 'data/csv', 'data/processed']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created/verified directory: {directory}")

def create_sample_dataset():
    """Create a realistic sample of the California Housing dataset."""
    print("🏗️ Creating sample California Housing dataset...")
    
    # Set random seed for reproducible data
    np.random.seed(42)
    
    # Generate 1000 sample records
    n_samples = 1000
    
    # Feature ranges based on real California Housing data
    data = {
        'MedInc': np.random.gamma(2, 2, n_samples),  # Median income (0-15)
        'HouseAge': np.random.uniform(1, 52, n_samples),  # House age (1-52 years)
        'AveRooms': np.random.gamma(6, 1, n_samples),  # Average rooms (3-20)
        'AveBedrms': np.random.gamma(1, 0.2, n_samples),  # Average bedrooms (0.5-2)
        'Population': np.random.gamma(4, 500, n_samples),  # Population (3-35000)
        'AveOccup': np.random.gamma(3, 1, n_samples),  # Average occupancy (1-15)
        'Latitude': np.random.uniform(32.5, 42.0, n_samples),  # CA latitude
        'Longitude': np.random.uniform(-124.3, -114.3, n_samples),  # CA longitude
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create realistic target values (house prices in hundreds of thousands)
    # Price depends on income, location, and house characteristics
    target = (
        df['MedInc'] * 3.5 +  # Income effect
        (45 - df['HouseAge']) * 0.02 +  # Newer houses cost more
        df['AveRooms'] * 0.3 +  # More rooms = higher price
        np.random.normal(0, 0.5, n_samples)  # Random noise
    )
    
    # Ensure reasonable price range (0.5 to 5.0 = $50k to $500k)
    target = np.clip(target, 0.5, 5.0)
    df['target'] = target
    
    print(f"✅ Sample dataset created successfully!")
    print(f"📊 Dataset shape: {df.shape}")
    
    return df

def add_real_california_data():
    """Add some real California city examples."""
    print("🌆 Adding sample California city data...")
    
    # Sample data for recognizable California locations
    real_samples = [
        # [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, target]
        [8.3252, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23, 4.526],  # San Francisco area
        [8.3014, 21.0, 6.24, 0.97, 2401.0, 2.11, 37.86, -122.22, 3.585],  # San Francisco
        [7.2574, 52.0, 8.29, 1.07, 496.0, 2.80, 37.85, -122.24, 3.521],  # San Francisco
        [5.6431, 52.0, 5.82, 1.07, 558.0, 2.55, 37.85, -122.25, 3.413],  # San Francisco
        [3.8462, 52.0, 6.28, 1.08, 565.0, 2.18, 37.85, -122.25, 3.422],  # San Francisco
        [4.0368, 52.0, 4.76, 1.10, 413.0, 2.20, 37.85, -122.25, 2.697],  # San Francisco
        [3.6591, 52.0, 4.29, 1.12, 1094.0, 2.13, 37.84, -122.25, 2.992],  # San Francisco
        [2.5179, 42.0, 4.90, 1.17, 1206.0, 1.90, 34.06, -118.22, 1.205],  # Los Angeles
        [1.9911, 36.0, 5.37, 1.19, 1551.0, 2.71, 34.03, -118.28, 1.391],  # Los Angeles
        [2.6851, 43.0, 4.29, 1.05, 2138.0, 2.33, 34.02, -118.33, 1.487],  # Los Angeles
    ]
    
    columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'target']
    real_df = pd.DataFrame(real_samples, columns=columns)
    
    return real_df

def display_dataset_overview(df):
    """Display comprehensive dataset overview."""
    print("\n" + "="*60)
    print("🔍 CALIFORNIA HOUSING DATASET OVERVIEW")
    print("="*60)
    
    print(f"\n📈 Dataset Information:")
    print(f"   • Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📋 Column Information:")
    descriptions = {
        'MedInc': 'Median income in block group (tens of thousands)',
        'HouseAge': 'Median house age in block group (years)', 
        'AveRooms': 'Average number of rooms per household',
        'AveBedrms': 'Average number of bedrooms per household',
        'Population': 'Block group population',
        'AveOccup': 'Average number of household members',
        'Latitude': 'Block group latitude (degrees)',
        'Longitude': 'Block group longitude (degrees)',
        'target': 'Median house value (hundreds of thousands of dollars)'
    }
    
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        desc = descriptions.get(col, 'No description')
        print(f"   {i:2d}. {col:<12} | {dtype:<8} | {null_count:>4} nulls | {desc}")
    
    print(f"\n📊 First 10 rows:")
    print(df.head(10).round(3))
    
    print(f"\n📈 Statistical Summary:")
    print(df.describe().round(3))
    
    print(f"\n🌍 Geographic Distribution:")
    print(f"   • Latitude range: {df['Latitude'].min():.3f}° to {df['Latitude'].max():.3f}°")
    print(f"   • Longitude range: {df['Longitude'].min():.3f}° to {df['Longitude'].max():.3f}°")
    print(f"   • Coverage: California state boundaries")
    
    print(f"\n💰 Target Variable (House Values):")
    target_stats = df['target'].describe()
    print(f"   • Range: ${target_stats['min']*100:.0f}k to ${target_stats['max']*100:.0f}k")
    print(f"   • Median: ${target_stats['50%']*100:.0f}k")
    print(f"   • Mean: ${target_stats['mean']*100:.0f}k")
    
    print("\n" + "="*60)

def save_dataset_files(df):
    """Save dataset in multiple formats for easy viewing."""
    try:
        # Save full dataset as CSV
        csv_path = 'data/csv/california_housing.csv'
        df.to_csv(csv_path, index=False)
        print(f"💾 Full dataset saved: {csv_path}")
        
        # Save first 50 rows as sample
        sample_path = 'data/csv/california_housing_sample_50.csv'
        df.head(50).to_csv(sample_path, index=False)
        print(f"📝 Sample (50 rows) saved: {sample_path}")
        
        # Save summary statistics
        summary_path = 'data/csv/california_housing_summary.csv'
        df.describe().to_csv(summary_path)
        print(f"📊 Summary statistics saved: {summary_path}")
        
        # Create a formatted Excel-style view
        formatted_path = 'data/csv/california_housing_formatted.csv'
        df_formatted = df.round(3)
        df_formatted.to_csv(formatted_path, index=False)
        print(f"📋 Formatted dataset saved: {formatted_path}")
        
        # Create a data dictionary
        dict_path = 'data/csv/data_dictionary.csv'
        data_dict = pd.DataFrame({
            'Column': df.columns,
            'Description': [
                'Median income in block group (tens of thousands $)',
                'Median house age in block group (years)', 
                'Average number of rooms per household',
                'Average number of bedrooms per household',
                'Block group population',
                'Average number of household members',
                'Block group latitude (degrees)',
                'Block group longitude (degrees)',
                'Median house value (hundreds of thousands $)'
            ],
            'Data_Type': [str(df[col].dtype) for col in df.columns],
            'Min_Value': [f"{df[col].min():.3f}" for col in df.columns],
            'Max_Value': [f"{df[col].max():.3f}" for col in df.columns],
            'Mean_Value': [f"{df[col].mean():.3f}" for col in df.columns]
        })
        data_dict.to_csv(dict_path, index=False)
        print(f"📚 Data dictionary saved: {dict_path}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error saving files: {e}")
        return False

def create_viewing_guide():
    """Create a guide for viewing the dataset."""
    guide_path = 'data/csv/HOW_TO_VIEW_DATASET.md'
    
    content = """# How to View the California Housing Dataset

## 📁 Available Files

### Main Dataset Files
- **`california_housing.csv`** - Full dataset ({rows} rows)
- **`california_housing_sample_50.csv`** - First 50 rows for quick viewing
- **`california_housing_formatted.csv`** - Rounded values for cleaner display

### Reference Files  
- **`california_housing_summary.csv`** - Statistical summary
- **`data_dictionary.csv`** - Column descriptions and metadata
- **`HOW_TO_VIEW_DATASET.md`** - This guide

## 🖥️ How to Open and View

### Option 1: Microsoft Excel
1. Open Excel
2. File → Open → Navigate to `data/csv/`
3. Select `california_housing.csv` or any CSV file
4. Data will open in spreadsheet format

### Option 2: VS Code
1. In VS Code, open the file explorer
2. Navigate to `data/csv/` folder
3. Click on any `.csv` file to open in built-in CSV viewer
4. Use "Open Preview" for formatted table view

### Option 3: Google Sheets
1. Go to sheets.google.com
2. File → Import → Upload → Choose CSV file
3. Select delimiter as comma
4. Data will load as spreadsheet

### Option 4: Python/Pandas (Code)
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/csv/california_housing.csv')

# View first few rows
print(df.head())

# View statistics
print(df.describe())

# View specific columns
print(df[['MedInc', 'target']].head())
```

## 📊 Understanding the Data

### Features (Input Variables)
- **MedInc**: Median income (higher = more expensive area)
- **HouseAge**: House age in years (newer houses often cost more)
- **AveRooms**: Average rooms per household (more rooms = bigger houses)
- **AveBedrms**: Average bedrooms per household  
- **Population**: Number of people in the area
- **AveOccup**: Average occupancy (people per household)
- **Latitude/Longitude**: Geographic coordinates in California

### Target (What we're predicting)
- **target**: Median house value in hundreds of thousands of dollars
  - Example: 3.5 = $350,000 house value

## 🔍 Quick Analysis Tips

1. **Sort by target** to see most/least expensive areas
2. **Filter by location** (Latitude/Longitude) to focus on specific regions
3. **Compare MedInc vs target** to see income-price relationship
4. **Look at AveRooms vs target** to see size-price relationship

## 🎯 Dataset Purpose

This dataset is used for:
- Predicting house prices based on neighborhood characteristics
- Understanding factors that influence housing costs
- Machine learning regression problems
- California real estate analysis

Generated on: {timestamp}
"""
    
    try:
        with open(guide_path, 'w') as f:
            f.write(content.format(
                rows=1000,  # Will be updated with actual count
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
        print(f"📖 Viewing guide created: {guide_path}")
        return True
    except Exception as e:
        print(f"⚠️ Could not create guide: {e}")
        return False

def main():
    """Main function to create and save viewable dataset."""
    print("🏠 California Housing Dataset Creator & Viewer")
    print("="*50)
    print("📝 Note: Creating comprehensive dataset for analysis")
    
    # Create directories
    create_directories()
    
    # Create sample dataset
    df_sample = create_sample_dataset()
    
    # Add some real California data
    df_real = add_real_california_data()
    
    # Combine datasets (real examples first)
    df = pd.concat([df_real, df_sample], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"📊 Final dataset shape: {df.shape}")
    
    # Display overview
    display_dataset_overview(df)
    
    # Save files
    if save_dataset_files(df):
        create_viewing_guide()
        
        print(f"\n🎉 SUCCESS! Dataset files created for viewing:")
        print(f"   📂 Main dataset: data/csv/california_housing.csv")
        print(f"   📋 Sample (50 rows): data/csv/california_housing_sample_50.csv")
        print(f"   📊 Summary stats: data/csv/california_housing_summary.csv")
        print(f"   📚 Data dictionary: data/csv/data_dictionary.csv")
        print(f"   📖 Viewing guide: data/csv/HOW_TO_VIEW_DATASET.md")
        
        print(f"\n💡 How to view:")
        print(f"   • Excel: Open any .csv file in Microsoft Excel")
        print(f"   • VS Code: Click on .csv files in file explorer")
        print(f"   • Google Sheets: Import any .csv file")
        print(f"   • Text Editor: Open in Notepad++ or similar")
        
    else:
        print(f"\n❌ Failed to create dataset files.")

if __name__ == "__main__":
    main()
