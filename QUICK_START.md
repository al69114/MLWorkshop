# Quick Start Guide - Using Your Own Dataset

## For Utrecht Dataset (or Any CSV Dataset)

### Step 1: Setup Virtual Environment (Recommended)

```bash
cd /Users/adithyalakshmikanth/Desktop/ml\ workshop/MLWorkshop

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1 (Alternative): Install Dependencies Without venv

```bash
cd /Users/adithyalakshmikanth/Desktop/ml\ workshop/MLWorkshop
pip3 install --user -r requirements.txt
```

### Step 2: Run Custom Dataset Demo

```bash
python custom_dataset_demo.py
```

### Step 3: Follow the Prompts

The script will ask you:

1. **File path**: Enter the path to your CSV file
   ```
   Example: /Users/yourusername/Downloads/utrecht_dataset.csv
   Or drag and drop the file into the terminal
   ```

2. **Target column**: Which column you want to predict
   ```
   Example: price, temperature, sales, etc.
   ```

The script will then:
- ✓ Load your data
- ✓ Clean it automatically
- ✓ Train 4 different models
- ✓ Show which one works best
- ✓ Create visualizations

---

## Want to See How to Improve Results?

Run the iterative improvement demo:

```bash
python iterative_model_improvement.py
```

This shows 5 different techniques to make your model better:

1. **Baseline** - Simple start
2. **Scaling** - Normalize features (usually +5-15% improvement)
3. **Regularization** - Prevent overfitting (+2-10% improvement)
4. **Polynomial** - Capture non-linear patterns (+10-20% improvement)
5. **Ensemble** - Combine multiple models (+15-30% improvement)

---

## Example: Training on Utrecht Dataset

```bash
# 1. Navigate to the workshop directory
cd /Users/adithyalakshmikanth/Desktop/ml\ workshop/MLWorkshop

# 2. Run the custom dataset script
python custom_dataset_demo.py

# 3. When prompted, enter your file path:
# /path/to/utrecht_dataset.csv

# 4. Enter target column name when asked
# Example: "temperature" or "price" or whatever you're predicting

# 5. Watch as it:
#    - Loads your data
#    - Cleans it
#    - Trains 4 models
#    - Shows the best one!
```

---

## Understanding the Output

### What the Models Do:

**Model 1: Linear Regression**
- Baseline model
- Simple and fast
- Good for linear relationships

**Model 2: Linear + Scaling**
- Same as Model 1 but with normalized features
- Usually improves performance by 5-15%

**Model 3: Ridge Regression**
- Adds regularization to prevent overfitting
- Better when you have many features

**Model 4: Random Forest**
- Ensemble of decision trees
- Usually the best performer
- Can capture complex patterns

### Metrics Explained:

**R² Score** (Higher is better, max is 1.0)
- 0.9 - 1.0: Excellent
- 0.7 - 0.9: Good
- 0.5 - 0.7: Moderate
- < 0.5: Poor

**MSE** (Mean Squared Error - Lower is better)
- Average squared difference between predictions and actual
- Heavily penalizes large errors

**MAE** (Mean Absolute Error - Lower is better)
- Average absolute difference
- More interpretable than MSE

---

## Tips for Best Results

### 1. Data Quality Matters
- Remove obviously wrong values
- Handle missing data
- More data = better results (aim for 1000+ samples)

### 2. Feature Selection
- Use features that logically relate to your target
- Remove duplicate or redundant features
- Create new features from existing ones

### 3. Try Different Models
- Start with simple (Linear Regression)
- Progress to complex (Random Forest, Gradient Boosting)
- The custom_dataset_demo.py does this automatically!

### 4. Check for Overfitting
- If Train R² >> Test R²: Model is overfitting
- Use regularization (Ridge, Lasso)
- Reduce model complexity

---

## Common Issues & Solutions

### Issue: "File not found"
**Solution**: Check the file path
```bash
# Make sure the path is correct
ls /path/to/your/file.csv

# Or use absolute path
python custom_dataset_demo.py
# Then paste: /Users/yourusername/Documents/data.csv
```

### Issue: "Target column not found"
**Solution**: Check column names
```python
import pandas as pd
df = pd.read_csv('your_file.csv')
print(df.columns)  # Shows all column names
```

### Issue: Poor model performance (R² < 0.5)
**Solutions**:
1. Check if you have enough data (need 100+ samples)
2. Check if features are related to target
3. Try removing outliers
4. Try feature engineering (create new features)
5. Use the iterative_model_improvement.py script

### Issue: "Import error" for packages
**Solution**: Install missing packages
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
# Or just:
pip install -r requirements.txt
```

---

## Next Steps After Your First Model

1. **Experiment with parameters**
   - Edit the scripts to try different settings
   - Change `n_estimators`, `max_depth` in Random Forest

2. **Feature engineering**
   - Create new features from existing ones
   - Example: If you have date, extract day/month/year

3. **Try more models**
   - Gradient Boosting
   - Support Vector Machines
   - Neural Networks

4. **Cross-validation**
   - Test model on multiple data splits
   - More reliable performance estimate

5. **Deploy your model**
   - Save the trained model
   - Use it to make predictions on new data

---

## Need Help?

1. Check the error message carefully
2. Read the DATASET_SUGGESTIONS.md for dataset ideas
3. Look at the code comments in the .py files
4. Ask workshop instructors

---

## Example Session

```
$ python custom_dataset_demo.py

======================================================================
UGA HACKS11 - ML WORKSHOP
CUSTOM DATASET TRAINING DEMO
======================================================================

📂 Enter the path to your CSV file:
   Example: /path/to/your/dataset.csv

File path: /Users/me/Downloads/utrecht_data.csv

✓ Successfully loaded: /Users/me/Downloads/utrecht_data.csv

Dataset shape: (1000, 10)
Rows: 1000, Columns: 10

📋 Column names:
  1. date (object)
  2. temperature (float64)
  3. humidity (float64)
  4. windspeed (float64)
  ...

🎯 Which column do you want to predict (target variable)?
Available columns: ['date', 'temperature', 'humidity', ...]

Target column name: temperature

... (training happens) ...

======================================================================
FINAL RESULTS COMPARISON
======================================================================

                        Model       R²      MSE      MAE
        Linear Regression  0.7234  12.45    2.89
     Linear + Scaling  0.7456  11.23    2.65
        Ridge Regression  0.7512  10.98    2.61
           Random Forest  0.8432   8.45    2.12

🏆 BEST MODEL: Random Forest
   R² Score: 0.8432
   Total Improvement: +16.57%

✓ Saved results as 'custom_dataset_results.png'
```

Good luck! 🚀
