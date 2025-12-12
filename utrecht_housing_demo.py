"""
Intro to ML Workshop - Utrecht Housing Price Prediction Demo
Shows BEFORE vs AFTER training comparison

This script demonstrates:
1. What happens BEFORE training (random/baseline predictions)
2. Training the model on Utrecht housing data
3. What happens AFTER training (learned predictions)
4. Clear comparison showing the improvement
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)


def load_utrecht_data():
    """Load the Utrecht housing dataset"""
    print("="*80)
    print("INTRO TO ML WORKSHOP - UTRECHT HOUSING PRICE PREDICTION")
    print("="*80)
    print("\nStep 1: Loading Utrecht Housing Dataset...")

    # Load the dataset
    df = pd.read_csv('utrechthousingsmall.csv')

    print(f"✓ Dataset loaded successfully!")
    print(f"  Total houses: {len(df)}")
    print(f"  Features: {len(df.columns)}")

    # Show first few houses
    print("\nSample of the data:")
    print(df[['house-area', 'lot-area', 'buildyear', 'bathrooms', 'retailvalue']].head(10))

    return df


def prepare_data(df):
    """Prepare features and target"""
    print("\n" + "="*80)
    print("Step 2: Preparing Data for Training")
    print("="*80)

    # Select features to predict house price
    feature_columns = ['lot-area', 'house-area', 'garden-size', 'balcony',
                       'buildyear', 'bathrooms', 'energy-eff', 'monument']

    target_column = 'retailvalue'  # Retail value is what we want to predict

    print(f"\nFeatures we'll use to predict house price:")
    for i, feature in enumerate(feature_columns, 1):
        print(f"  {i}. {feature}")

    print(f"\nTarget: {target_column} (house retail value)")

    # Extract features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Handle any missing values
    X = X.fillna(X.mean())

    print(f"\n✓ Data prepared!")
    print(f"  Number of houses: {len(X)}")
    print(f"  Number of features: {X.shape[1]}")
    print(f"  Price range: €{y.min():,.0f} - €{y.max():,.0f}")
    print(f"  Average price: €{y.mean():,.0f}")

    return X, y


def show_before_training(X_train, X_test, y_train, y_test):
    """
    Show what happens BEFORE training
    We'll use random predictions and a simple average
    """
    print("\n" + "="*80)
    print("BEFORE TRAINING - What happens without Machine Learning?")
    print("="*80)

    results_before = {}

    # Method 1: Random predictions (worst case)
    print("\n1. Random Predictions (guessing randomly):")
    random_predictions = np.random.uniform(y_test.min(), y_test.max(), size=len(y_test))

    mse_random = mean_squared_error(y_test, random_predictions)
    r2_random = r2_score(y_test, random_predictions)
    mae_random = mean_absolute_error(y_test, random_predictions)

    print(f"   R² Score: {r2_random:.4f} (terrible!)")
    print(f"   Mean Absolute Error: €{mae_random:,.0f}")
    print(f"   → Predictions are completely random and useless!")

    results_before['Random'] = {
        'predictions': random_predictions,
        'r2': r2_random,
        'mae': mae_random,
        'mse': mse_random
    }

    # Method 2: Always predict the average (naive baseline)
    print("\n2. Average Prediction (always guessing the average price):")
    average_prediction = np.full(len(y_test), y_train.mean())

    mse_avg = mean_squared_error(y_test, average_prediction)
    r2_avg = r2_score(y_test, average_prediction)
    mae_avg = mean_absolute_error(y_test, average_prediction)

    print(f"   R² Score: {r2_avg:.4f}")
    print(f"   Mean Absolute Error: €{mae_avg:,.0f}")
    print(f"   → Better than random, but still ignores all house features!")

    results_before['Average'] = {
        'predictions': average_prediction,
        'r2': r2_avg,
        'mae': mae_avg,
        'mse': mse_avg
    }

    print("\n" + "-"*80)
    print("PROBLEM: Without training, we can't use house features to predict price!")
    print("-"*80)

    return results_before


def train_models(X_train, X_test, y_train, y_test):
    """
    TRAINING THE MODEL
    """
    print("\n" + "="*80)
    print("TRAINING - Teaching the Model to Learn from Data")
    print("="*80)

    # Scale features for better training
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results_after = {}

    # Model 1: Linear Regression
    print("\n1. Training Linear Regression...")
    print("   → Learning linear relationships between features and price")

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    print(f"   ✓ Training complete!")
    print(f"   R² Score: {r2_lr:.4f}")
    print(f"   Mean Absolute Error: €{mae_lr:,.0f}")

    results_after['Linear Regression'] = {
        'predictions': y_pred_lr,
        'r2': r2_lr,
        'mae': mae_lr,
        'mse': mse_lr,
        'model': lr_model
    }

    # Model 2: Random Forest (more powerful)
    print("\n2. Training Random Forest...")
    print("   → Learning complex non-linear patterns")

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)

    print(f"   ✓ Training complete!")
    print(f"   R² Score: {r2_rf:.4f}")
    print(f"   Mean Absolute Error: €{mae_rf:,.0f}")

    # Show feature importance
    print("\n   Most important features for pricing:")
    feature_names = X_train.columns
    importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    for idx, row in feature_importance.head(5).iterrows():
        print(f"      {row['Feature']}: {row['Importance']:.3f}")

    results_after['Random Forest'] = {
        'predictions': y_pred_rf,
        'r2': r2_rf,
        'mae': mae_rf,
        'mse': mse_rf,
        'model': rf_model,
        'feature_importance': feature_importance
    }

    return results_after, scaler


def show_improvement(results_before, results_after, y_test):
    """
    Show the dramatic improvement from BEFORE to AFTER training
    """
    print("\n" + "="*80)
    print("RESULTS: BEFORE vs AFTER Training")
    print("="*80)

    print("\n" + "-"*80)
    print("BEFORE TRAINING (No Machine Learning)")
    print("-"*80)

    for method, metrics in results_before.items():
        print(f"\n{method} Prediction:")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  Average Error: €{metrics['mae']:,.0f}")

    print("\n" + "-"*80)
    print("AFTER TRAINING (With Machine Learning)")
    print("-"*80)

    for method, metrics in results_after.items():
        print(f"\n{method}:")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  Average Error: €{metrics['mae']:,.0f}")

    # Calculate improvement
    baseline_mae = results_before['Average']['mae']
    best_mae = results_after['Random Forest']['mae']
    improvement = ((baseline_mae - best_mae) / baseline_mae) * 100

    baseline_r2 = results_before['Average']['r2']
    best_r2 = results_after['Random Forest']['r2']

    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    print(f"\nBefore Training (Average Baseline):")
    print(f"  R² Score: {baseline_r2:.4f}")
    print(f"  Average Error: €{baseline_mae:,.0f}")

    print(f"\nAfter Training (Random Forest):")
    print(f"  R² Score: {best_r2:.4f}")
    print(f"  Average Error: €{best_mae:,.0f}")

    print(f"\n🎯 IMPROVEMENT:")
    print(f"  Error Reduction: {improvement:.1f}%")
    print(f"  Money Saved: €{baseline_mae - best_mae:,.0f} per prediction on average!")

    return improvement


def visualize_comparison(results_before, results_after, y_test, X_test):
    """
    Create comprehensive visualization comparing BEFORE and AFTER
    """
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)

    fig = plt.figure(figsize=(18, 12))

    # Get predictions
    random_pred = results_before['Random']['predictions']
    avg_pred = results_before['Average']['predictions']
    lr_pred = results_after['Linear Regression']['predictions']
    rf_pred = results_after['Random Forest']['predictions']

    # Row 1: Predictions vs Actual
    # Plot 1: Random predictions (BEFORE)
    ax1 = plt.subplot(3, 3, 1)
    plt.scatter(y_test, random_pred, alpha=0.6, color='red', s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Price (€)', fontsize=11)
    plt.ylabel('Predicted Price (€)', fontsize=11)
    plt.title('BEFORE: Random Predictions\n(No Learning)', fontsize=12, fontweight='bold')
    r2 = results_before['Random']['r2']
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Average predictions (BEFORE)
    ax2 = plt.subplot(3, 3, 2)
    plt.scatter(y_test, avg_pred, alpha=0.6, color='orange', s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Price (€)', fontsize=11)
    plt.ylabel('Predicted Price (€)', fontsize=11)
    plt.title('BEFORE: Average Baseline\n(No Learning)', fontsize=12, fontweight='bold')
    r2 = results_before['Average']['r2']
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Random Forest (AFTER)
    ax3 = plt.subplot(3, 3, 3)
    plt.scatter(y_test, rf_pred, alpha=0.6, color='green', s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Price (€)', fontsize=11)
    plt.ylabel('Predicted Price (€)', fontsize=11)
    plt.title('AFTER: Random Forest\n(Trained Model)', fontsize=12, fontweight='bold')
    r2 = results_after['Random Forest']['r2']
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax3.transAxes,
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Row 2: Error distributions
    # Plot 4: Errors BEFORE (Random)
    ax4 = plt.subplot(3, 3, 4)
    errors_random = y_test - random_pred
    plt.hist(errors_random, bins=20, color='red', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Prediction Error (€)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Error Distribution: Random', fontsize=11)
    plt.grid(True, alpha=0.3)

    # Plot 5: Errors BEFORE (Average)
    ax5 = plt.subplot(3, 3, 5)
    errors_avg = y_test - avg_pred
    plt.hist(errors_avg, bins=20, color='orange', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Prediction Error (€)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Error Distribution: Average', fontsize=11)
    plt.grid(True, alpha=0.3)

    # Plot 6: Errors AFTER (Random Forest)
    ax6 = plt.subplot(3, 3, 6)
    errors_rf = y_test - rf_pred
    plt.hist(errors_rf, bins=20, color='green', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Prediction Error (€)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Error Distribution: Random Forest', fontsize=11)
    plt.grid(True, alpha=0.3)

    # Row 3: Comparison plots
    # Plot 7: R² Score comparison
    ax7 = plt.subplot(3, 3, 7)
    methods = ['Random\n(Before)', 'Average\n(Before)', 'Linear Reg\n(After)', 'Random Forest\n(After)']
    r2_scores = [
        results_before['Random']['r2'],
        results_before['Average']['r2'],
        results_after['Linear Regression']['r2'],
        results_after['Random Forest']['r2']
    ]
    colors_bar = ['red', 'orange', 'lightblue', 'green']
    bars = plt.bar(methods, r2_scores, color=colors_bar, edgecolor='black', linewidth=1.5)
    plt.ylabel('R² Score (Higher is Better)', fontsize=11)
    plt.title('Model Performance Comparison', fontsize=12, fontweight='bold')
    plt.ylim([-0.5, 1.0])
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 8: MAE comparison
    ax8 = plt.subplot(3, 3, 8)
    mae_values = [
        results_before['Random']['mae'],
        results_before['Average']['mae'],
        results_after['Linear Regression']['mae'],
        results_after['Random Forest']['mae']
    ]
    bars = plt.bar(methods, mae_values, color=colors_bar, edgecolor='black', linewidth=1.5)
    plt.ylabel('Mean Absolute Error €\n(Lower is Better)', fontsize=11)
    plt.title('Average Prediction Error', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mae in zip(bars, mae_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'€{mae:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 9: Feature Importance
    ax9 = plt.subplot(3, 3, 9)
    feature_importance = results_after['Random Forest']['feature_importance']
    top_features = feature_importance.head(8)
    plt.barh(top_features['Feature'], top_features['Importance'], color='green', edgecolor='black')
    plt.xlabel('Importance', fontsize=11)
    plt.title('Most Important Features\n(Random Forest)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('utrecht_before_after_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: 'utrecht_before_after_comparison.png'")
    plt.show()


def show_example_predictions(results_after, y_test, X_test):
    """
    Show specific examples of predictions
    """
    print("\n" + "="*80)
    print("Example Predictions on Real Houses")
    print("="*80)

    rf_pred = results_after['Random Forest']['predictions']

    # Show 10 random examples
    n_examples = min(10, len(y_test))
    indices = np.random.choice(len(y_test), n_examples, replace=False)

    print("\nHouse Details → Actual Price vs Predicted Price:\n")
    print(f"{'House Area':<12} {'Lot Area':<12} {'Build Year':<12} {'Actual Price':<15} {'Predicted':<15} {'Error':<15}")
    print("-" * 95)

    for idx in indices:
        actual = y_test.iloc[idx]
        predicted = rf_pred[idx]
        error = abs(actual - predicted)
        house_area = X_test.iloc[idx]['house-area']
        lot_area = X_test.iloc[idx]['lot-area']
        buildyear = X_test.iloc[idx]['buildyear']

        print(f"{house_area:<12.1f} {lot_area:<12.1f} {buildyear:<12.0f} "
              f"€{actual:<14,.0f} €{predicted:<14,.0f} €{error:,.0f}")


def main():
    """
    Main function to run the complete demo
    """
    # Load data
    df = load_utrecht_data()

    # Prepare data
    X, y = prepare_data(df)

    # Split into training and testing sets
    print("\n" + "="*80)
    print("Splitting Data into Training and Testing Sets")
    print("="*80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set: {len(X_train)} houses (80%)")
    print(f"Testing set: {len(X_test)} houses (20%)")
    print("\nWe'll train on the training set and test on houses the model has never seen!")

    # Show BEFORE training
    results_before = show_before_training(X_train, X_test, y_train, y_test)

    # Train models
    results_after, scaler = train_models(X_train, X_test, y_train, y_test)

    # Show improvement
    improvement = show_improvement(results_before, results_after, y_test)

    # Visualize
    visualize_comparison(results_before, results_after, y_test, X_test)

    # Show examples
    show_example_predictions(results_after, y_test, X_test)

    # Final summary
    print("\n" + "="*80)
    print("WORKSHOP COMPLETE!")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. BEFORE training: Random or average predictions perform poorly")
    print("2. Machine Learning LEARNS patterns from data during training")
    print("3. AFTER training: Model makes much better predictions")
    print(f"4. We reduced prediction error by {improvement:.1f}%!")
    print("\nThis is the power of Machine Learning!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
