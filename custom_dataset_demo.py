"""
UGA Hacks11 - ML Workshop: Custom Dataset Demo
Load and train on any CSV dataset

This script shows how to:
1. Load your own CSV dataset
2. Clean and prepare the data
3. Train multiple models
4. Improve results iteratively
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')


def load_custom_dataset(file_path):
    """
    Load a custom CSV dataset
    """
    print("="*70)
    print("LOADING CUSTOM DATASET")
    print("="*70)

    try:
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded: {file_path}")
        print(f"\nDataset shape: {df.shape}")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

        print("\n📋 Column names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col} ({df[col].dtype})")

        return df

    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        print("\nPlease check the file path and try again.")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return None


def explore_dataset(df):
    """
    Explore the dataset
    """
    print("\n" + "="*70)
    print("DATA EXPLORATION")
    print("="*70)

    print("\n📊 First 5 rows:")
    print(df.head())

    print("\n📈 Statistical Summary:")
    print(df.describe())

    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print("  ✓ No missing values!")
    else:
        print(f"  ⚠️  Found {total_missing} missing values:")
        print(missing[missing > 0])

    print("\n📉 Data Types:")
    print(df.dtypes.value_counts())

    return df


def clean_data(df, target_column):
    """
    Clean the dataset
    """
    print("\n" + "="*70)
    print("DATA CLEANING")
    print("="*70)

    # Make a copy
    df_clean = df.copy()

    # Handle missing values
    print("\n1. Handling missing values...")
    missing_before = df_clean.isnull().sum().sum()
    df_clean = df_clean.dropna()
    missing_after = df_clean.isnull().sum().sum()
    print(f"   Dropped rows with missing values: {len(df) - len(df_clean)}")

    # Separate features and target
    if target_column not in df_clean.columns:
        print(f"\n❌ Error: Target column '{target_column}' not found!")
        print(f"Available columns: {list(df_clean.columns)}")
        return None, None

    y = df_clean[target_column]
    X = df_clean.drop(target_column, axis=1)

    # Handle categorical variables
    print("\n2. Handling categorical variables...")
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"   Found {len(categorical_cols)} categorical columns:")
        for col in categorical_cols:
            print(f"     - {col}: {X[col].nunique()} unique values")

        # Encode categorical variables
        X_encoded = X.copy()
        for col in categorical_cols:
            if X[col].nunique() <= 10:  # One-hot encode if few categories
                print(f"   One-hot encoding: {col}")
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X_encoded = pd.concat([X_encoded.drop(col, axis=1), dummies], axis=1)
            else:  # Label encode if many categories
                print(f"   Label encoding: {col}")
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
        X = X_encoded

    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])

    print(f"\n✓ Clean dataset ready!")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Feature names: {list(X.columns)}")

    return X, y


def train_and_improve(X, y):
    """
    Train models and show progressive improvement
    """
    print("\n" + "="*70)
    print("MODEL TRAINING & IMPROVEMENT")
    print("="*70)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")

    results = []

    # Model 1: Baseline Linear Regression
    print("\n" + "-"*70)
    print("MODEL 1: Baseline Linear Regression")
    print("-"*70)

    model1 = LinearRegression()
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)

    mse1 = mean_squared_error(y_test, y_pred1)
    r2_1 = r2_score(y_test, y_pred1)
    mae1 = mean_absolute_error(y_test, y_pred1)

    print(f"Results:")
    print(f"  R² Score: {r2_1:.4f}")
    print(f"  MSE: {mse1:.4f}")
    print(f"  MAE: {mae1:.4f}")

    results.append({
        'Model': 'Linear Regression',
        'R²': r2_1,
        'MSE': mse1,
        'MAE': mae1
    })

    # Model 2: With Feature Scaling
    print("\n" + "-"*70)
    print("MODEL 2: Linear Regression + Feature Scaling")
    print("-"*70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model2 = LinearRegression()
    model2.fit(X_train_scaled, y_train)
    y_pred2 = model2.predict(X_test_scaled)

    mse2 = mean_squared_error(y_test, y_pred2)
    r2_2 = r2_score(y_test, y_pred2)
    mae2 = mean_absolute_error(y_test, y_pred2)

    print(f"Results:")
    print(f"  R² Score: {r2_2:.4f}")
    print(f"  MSE: {mse2:.4f}")
    print(f"  MAE: {mae2:.4f}")
    print(f"  Improvement: {((r2_2 - r2_1) / abs(r2_1) * 100):+.2f}%")

    results.append({
        'Model': 'Linear + Scaling',
        'R²': r2_2,
        'MSE': mse2,
        'MAE': mae2
    })

    # Model 3: Ridge Regression (Regularization)
    print("\n" + "-"*70)
    print("MODEL 3: Ridge Regression (Regularization)")
    print("-"*70)

    model3 = Ridge(alpha=1.0)
    model3.fit(X_train_scaled, y_train)
    y_pred3 = model3.predict(X_test_scaled)

    mse3 = mean_squared_error(y_test, y_pred3)
    r2_3 = r2_score(y_test, y_pred3)
    mae3 = mean_absolute_error(y_test, y_pred3)

    print(f"Results:")
    print(f"  R² Score: {r2_3:.4f}")
    print(f"  MSE: {mse3:.4f}")
    print(f"  MAE: {mae3:.4f}")
    print(f"  Improvement: {((r2_3 - r2_1) / abs(r2_1) * 100):+.2f}%")

    results.append({
        'Model': 'Ridge Regression',
        'R²': r2_3,
        'MSE': mse3,
        'MAE': mae3
    })

    # Model 4: Random Forest
    print("\n" + "-"*70)
    print("MODEL 4: Random Forest (Ensemble Method)")
    print("-"*70)

    model4 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model4.fit(X_train_scaled, y_train)
    y_pred4 = model4.predict(X_test_scaled)

    mse4 = mean_squared_error(y_test, y_pred4)
    r2_4 = r2_score(y_test, y_pred4)
    mae4 = mean_absolute_error(y_test, y_pred4)

    print(f"Results:")
    print(f"  R² Score: {r2_4:.4f}")
    print(f"  MSE: {mse4:.4f}")
    print(f"  MAE: {mae4:.4f}")
    print(f"  Improvement: {((r2_4 - r2_1) / abs(r2_1) * 100):+.2f}%")

    results.append({
        'Model': 'Random Forest',
        'R²': r2_4,
        'MSE': mse4,
        'MAE': mae4
    })

    # Show feature importance
    if hasattr(model4, 'feature_importances_'):
        print("\n  Top 5 Most Important Features:")
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model4.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(importance.head(5).to_string(index=False))

    # Final comparison
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    best_model = results_df.loc[results_df['R²'].idxmax(), 'Model']
    best_r2 = results_df['R²'].max()

    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   R² Score: {best_r2:.4f}")
    print(f"   Total Improvement: {((best_r2 - r2_1) / abs(r2_1) * 100):+.2f}%")

    # Visualize
    visualize_results(results_df, y_test, y_pred4, best_model)

    return results_df


def visualize_results(results_df, y_test, y_pred_best, best_model_name):
    """
    Create visualizations
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Model comparison
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    ax.barh(results_df['Model'], results_df['R²'], color=colors)
    ax.set_xlabel('R² Score (Higher is Better)')
    ax.set_title('Model Performance Comparison')
    ax.grid(True, alpha=0.3)

    # Plot 2: Predictions vs Actual (best model)
    ax = axes[1]
    ax.scatter(y_test, y_pred_best, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{best_model_name}: Predictions vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('custom_dataset_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved results as 'custom_dataset_results.png'")
    plt.show()


def main():
    """
    Main function to run the custom dataset demo
    """
    print("="*70)
    print("UGA HACKS11 - ML WORKSHOP")
    print("CUSTOM DATASET TRAINING DEMO")
    print("="*70)

    # Get file path from user
    print("\n📂 Enter the path to your CSV file:")
    print("   Example: /path/to/your/dataset.csv")
    print("   Or drag and drop the file here\n")

    file_path = input("File path: ").strip().strip("'\"")

    # Load dataset
    df = load_custom_dataset(file_path)
    if df is None:
        return

    # Explore dataset
    df = explore_dataset(df)

    # Get target column
    print("\n🎯 Which column do you want to predict (target variable)?")
    print(f"Available columns: {list(df.columns)}\n")
    target_column = input("Target column name: ").strip()

    # Clean data
    X, y = clean_data(df, target_column)
    if X is None or y is None:
        return

    print(f"\n✓ Ready to train models!")
    input("\nPress Enter to start training...")

    # Train and improve
    results = train_and_improve(X, y)

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\n🎓 What we did:")
    print("  1. Loaded your custom dataset")
    print("  2. Explored and cleaned the data")
    print("  3. Trained 4 different models")
    print("  4. Showed progressive improvement")
    print("  5. Identified the best model")
    print("\n🚀 You can now use these techniques on any dataset!")
    print("\nGood luck with your ML projects!\n")


if __name__ == "__main__":
    main()
