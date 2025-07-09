# Kaggle API Setup Instructions

## Issue Resolution: "Missing username in configuration"

If you're getting the "Missing username in configuration" error, here are the steps to fix it:

### Method 1: Check Kaggle Credentials File

1. **Verify kaggle.json location**:
   ```bash
   # On macOS/Linux:
   ls -la ~/.kaggle/kaggle.json
   
   # On Windows:
   dir %USERPROFILE%\.kaggle\kaggle.json
   ```

2. **Check file permissions** (macOS/Linux only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Verify file contents**:
   ```bash
   cat ~/.kaggle/kaggle.json
   ```
   
   Should look like:
   ```json
   {"username":"your_username","key":"your_api_key"}
   ```

### Method 2: Set Environment Variables

Add these to your shell profile or set them directly:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

### Method 3: Use the No-Kaggle Demo

**Recommended for testing**: Use the new demo notebook that doesn't require Kaggle:
```bash
jupyter notebook notebooks/retail_semantic_analysis_demo_no_kaggle.ipynb
```

This notebook:
- ✅ Works without Kaggle credentials
- ✅ Creates realistic sample data
- ✅ Demonstrates all features
- ✅ Generates publication-ready results

### Method 4: Manual Kaggle Setup

1. **Get your API credentials**:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Download kaggle.json

2. **Place the file correctly**:
   ```bash
   # Create directory if it doesn't exist
   mkdir -p ~/.kaggle
   
   # Move downloaded file
   mv ~/Downloads/kaggle.json ~/.kaggle/
   
   # Set correct permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Test the setup**:
   ```python
   import kaggle
   kaggle.api.authenticate()
   print("Kaggle API configured successfully!")
   ```

### Method 5: Alternative Data Sources

If Kaggle setup continues to fail, you can use these alternatives:

1. **Direct Dataset Download**:
   - Download datasets manually from Kaggle
   - Place CSV files in `data/raw/` directory
   - Modify the data loader to point to your files

2. **Use Sample Data Generator**:
   ```python
   from src.data_loader import RetailDataLoader
   loader = RetailDataLoader()
   df = loader.create_sample_dataset(size=5000)  # Larger sample
   ```

3. **Use Other Public Datasets**:
   - Amazon product reviews from academic sources
   - Yelp dataset
   - Other e-commerce review datasets

## Quick Test

To verify everything works, run this quick test:

```python
# Test without Kaggle
from src.data_loader import RetailDataLoader
loader = RetailDataLoader()
df = loader.create_sample_dataset(size=100)
print(f"Created {len(df)} sample reviews successfully!")
```

## Recommended Approach

For the best experience with this project:

1. **Start with the no-Kaggle demo**: `retail_semantic_analysis_demo_no_kaggle.ipynb`
2. **Explore all features** with sample data
3. **Set up Kaggle later** if you need real datasets
4. **Use the framework** for your own data sources

The sample data generator creates realistic reviews that demonstrate all the analysis capabilities without requiring external data sources.