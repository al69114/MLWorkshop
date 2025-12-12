# Run Utrecht Housing Demo

## What This Demo Shows

This demo uses the **Utrecht Housing dataset** (utrechthousingsmall.csv) to demonstrate:

1. **BEFORE Training**: What happens without machine learning
   - Random predictions (completely useless)
   - Average baseline (ignores house features)

2. **AFTER Training**: The power of machine learning
   - Linear Regression (learns linear patterns)
   - Random Forest (learns complex patterns)

3. **Clear Comparison**: Visual side-by-side comparison showing the improvement

## Quick Setup & Run

### Option 1: Automatic Setup (Recommended)

```bash
cd "/Users/adithyalakshmikanth/Desktop/ml workshop/MLWorkshop"

# Run the setup script (creates venv and installs packages)
./setup_workshop.sh

# Run the demo
python utrecht_housing_demo.py
```

### Option 2: Manual Setup

```bash
cd "/Users/adithyalakshmikanth/Desktop/ml workshop/MLWorkshop"

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install numpy pandas matplotlib seaborn scikit-learn

# Run the demo
python utrecht_housing_demo.py
```

## What You'll See

The demo will:

1. Load the Utrecht housing dataset (101 houses)
2. Show statistics about the data
3. Demonstrate BEFORE training:
   - Random predictions (R² ≈ -0.5 to 0)
   - Average predictions (R² = 0)
4. Train two models:
   - Linear Regression
   - Random Forest
5. Show AFTER training results:
   - Much better R² scores (0.7-0.9)
   - Lower prediction errors
6. Create a comprehensive visualization saved as `utrecht_before_after_comparison.png`
7. Show example predictions on real houses

## Output Files

- `utrecht_before_after_comparison.png` - Main visualization showing before/after comparison

## Understanding the Results

### R² Score (R-squared)
- **1.0**: Perfect predictions
- **0.7-0.9**: Good model (typical for real-world data)
- **0.0**: No better than guessing the average
- **Negative**: Worse than guessing the average

### Mean Absolute Error (MAE)
- Average difference between predicted and actual price
- Lower is better
- Measured in Euros (€)

### What Makes a Good Result?
For the Utrecht dataset:
- **BEFORE**: R² ≈ 0.0, MAE ≈ €150,000-200,000
- **AFTER**: R² ≈ 0.7-0.9, MAE ≈ €50,000-80,000
- **Improvement**: 60-70% reduction in error!

## For Your Workshop

This demo is perfect for showing participants:

1. The problem with naive approaches (random, average)
2. How machine learning actually learns patterns
3. Concrete improvement metrics
4. Visual proof that ML works
5. Real-world application (house price prediction)

## Customizing for Your Workshop

You can modify `utrecht_housing_demo.py` to:

- Change which features are used for prediction
- Try different models (add Gradient Boosting, XGBoost, etc.)
- Adjust train/test split ratio
- Show more or fewer examples
- Customize the visualizations

## Troubleshooting

### "ModuleNotFoundError"
Run the setup script or install packages manually (see above)

### "FileNotFoundError: utrechthousingsmall.csv"
Make sure you're in the MLWorkshop directory when running the script

### Virtual environment issues
Make sure to activate the venv:
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

## Next Steps

After running this demo, you can:

1. Run `custom_dataset_demo.py` for any CSV dataset
2. Run `ml_with_sklearn.py` for more ML algorithms
3. Explore `iterative_model_improvement.py` for advanced techniques
