# DVC Integration Fix

## Problem
The pipeline was failing with the error:
```
ERROR: output 'data/california_housing.csv' is already tracked by SCM (e.g. Git).
```

This happened because DVC cannot track files that are already being tracked by Git.

## Solution
1. **Updated Pipeline**: Modified the DVC tracking step in `.github/workflows/mlops-pipeline.yml` to:
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

## Next Steps
The pipeline should now run successfully through the DVC step without errors.
