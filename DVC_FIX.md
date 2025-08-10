# DVC Integration Fix

## Problem
The pipeline was failing with the error:
```
ERROR: output 'data/california_housing.csv' is already tracked by SCM (e.g. Git).
```

This happened because DVC cannot track files that are already being tracked by Git.

## Solution
1. **Updated Pipeline**: Modified the DVC tracking step in `.github/workflows/mlops-pipeline.yml` to:
   - Check if files are already tracked by DVC (`.dvc` files exist)
   - Check if files are already tracked by Git using `git ls-files --error-unmatch`
   - Remove files from Git tracking first using `git rm --cached`
   - Then add them to DVC tracking

2. **Updated .gitignore**: Added data files to `.gitignore` so they won't be tracked by Git:
   ```
   # DVC - Data files managed by DVC (only .dvc files are tracked by Git)
   /data/california_housing.csv
   /data/processed/preprocessed_data.joblib
   /data/processed/dummy_data.json
   ```

3. **Fixed Locally**: Removed `data/california_housing.csv` from Git tracking:
   ```bash
   git rm --cached data/california_housing.csv
   ```

## How DVC Works Now
- **Git tracks**: Only the `.dvc` files (metadata)
- **DVC tracks**: The actual data files
- **Pipeline**: Automatically handles the transition from Git to DVC tracking

## Benefits
- ✅ No more "already tracked by SCM" errors
- ✅ Data versioning with DVC while keeping Git repo lightweight
- ✅ Proper separation of code (Git) and data (DVC)
- ✅ Pipeline automatically handles existing files

# MLflow JSON Serialization Fix

## Problem
The MLflow setup was failing with:
```
ERROR: MLflow setup failed with error: Object of type Timestamp is not JSON serializable
```

This happened because pandas Timestamp objects in MLflow runs cannot be directly serialized to JSON.

## Solution
1. **Custom JSON Encoder**: Added `MLflowJSONEncoder` class to handle:
   - Pandas Timestamp objects → ISO format strings
   - NaN values → None
   - Other non-serializable objects → String representations

2. **Updated Functions**: Fixed `save_best_model_info()` and `generate_experiment_summary()` to:
   - Safely extract values from pandas DataFrames
   - Convert to JSON-serializable types
   - Use custom encoder for JSON operations

3. **Better Error Handling**: Added proper exception handling and type conversion

## How It Works Now
- ✅ All datetime objects are converted to ISO format strings
- ✅ NaN values are converted to None
- ✅ All JSON operations use the custom encoder
- ✅ Robust type checking and conversion

## Next Steps
Both DVC and MLflow issues are now resolved. The pipeline should run successfully without errors.
