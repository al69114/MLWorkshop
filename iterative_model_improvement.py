"""
UGA Hacks11 - ML Workshop: Iterative Model Improvement
Shows how to train a model and progressively improve results

This demo demonstrates:
1. Loading and exploring data
2. Building a baseline model
3. Iteratively improving performance through:
   - Feature engineering
   - Data preprocessing
   - Hyperparameter tuning
   - Algorithm selection
   - Cross-validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class MLModelImprovement:
    """
    Demonstrates iterative model improvement process
    """

    def __init__(self):
        self.results = []
        self.models = {}

    def load_data(self, dataset_choice='california'):
        """
        Load dataset for training

        Options:
        - 'california': California Housing dataset (built-in)
        - 'diabetes': Diabetes dataset (built-in)
        - 'custom': Load your own CSV file
        """
        print("="*70)
        print("STEP 1: LOADING DATA")
        print("="*70)

        if dataset_choice == 'california':
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            self.df = pd.DataFrame(data.data, columns=data.feature_names)
            self.df['target'] = data.target
            print(f"✓ Loaded California Housing Dataset")
            print(f"  - Predicting: Median house value")

        elif dataset_choice == 'diabetes':
            from sklearn.datasets import load_diabetes
            data = load_diabetes()
            self.df = pd.DataFrame(data.data, columns=data.feature_names)
            self.df['target'] = data.target
            print(f"✓ Loaded Diabetes Dataset")
            print(f"  - Predicting: Disease progression")

        elif dataset_choice == 'custom':
            # For custom CSV files
            file_path = input("Enter the path to your CSV file: ")
            self.df = pd.read_csv(file_path)
            print(f"✓ Loaded custom dataset from {file_path}")
            target_col = input("Enter the name of the target column: ")
            # Separate features and target
            print(f"  - Target variable: {target_col}")

        print(f"\nDataset shape: {self.df.shape}")
        print(f"Features: {len(self.df.columns) - 1}")
        print(f"Samples: {len(self.df)}")

        return self.df

    def explore_data(self):
        """
        Explore the dataset to understand it better
        """
        print("\n" + "="*70)
        print("STEP 2: DATA EXPLORATION")
        print("="*70)

        print("\n📊 First few rows:")
        print(self.df.head())

        print("\n📈 Statistical Summary:")
        print(self.df.describe())

        print("\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("  ✓ No missing values found!")
        else:
            print(missing[missing > 0])

        print("\n📉 Data Types:")
        print(self.df.dtypes)

        # Correlation with target
        print("\n🎯 Feature Correlation with Target:")
        correlations = self.df.corr()['target'].sort_values(ascending=False)
        print(correlations)

        # Visualize correlations
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved correlation heatmap as 'correlation_heatmap.png'")
        plt.close()

    def prepare_data(self):
        """
        Prepare data for training
        """
        print("\n" + "="*70)
        print("STEP 3: DATA PREPARATION")
        print("="*70)

        # Separate features and target
        X = self.df.drop('target', axis=1)
        y = self.df['target']

        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\n✓ Data split into train and test sets")
        print(f"  - Training samples: {len(self.X_train)}")
        print(f"  - Testing samples: {len(self.X_test)}")
        print(f"  - Train/Test ratio: 80/20")

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Evaluate model performance and store results
        """
        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # Store results
        self.results.append({
            'Model': model_name,
            'Train MSE': train_mse,
            'Test MSE': test_mse,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Train MAE': train_mae,
            'Test MAE': test_mae
        })

        self.models[model_name] = model

        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae
        }

    def iteration_1_baseline(self):
        """
        ITERATION 1: Build a simple baseline model
        """
        print("\n" + "="*70)
        print("ITERATION 1: BASELINE MODEL")
        print("="*70)
        print("\n🎯 Strategy: Simple Linear Regression with raw features")
        print("   - No preprocessing")
        print("   - No feature engineering")
        print("   - Default parameters\n")

        model = LinearRegression()
        metrics = self.evaluate_model(
            model, self.X_train, self.X_test,
            self.y_train, self.y_test,
            'Baseline Linear Regression'
        )

        print(f"📊 Results:")
        print(f"   Train R² Score: {metrics['train_r2']:.4f}")
        print(f"   Test R² Score:  {metrics['test_r2']:.4f}")
        print(f"   Test MSE:       {metrics['test_mse']:.4f}")
        print(f"   Test MAE:       {metrics['test_mae']:.4f}")

        if metrics['train_r2'] - metrics['test_r2'] > 0.1:
            print(f"\n⚠️  Large gap between train and test R² suggests overfitting")

    def iteration_2_scaling(self):
        """
        ITERATION 2: Add feature scaling
        """
        print("\n" + "="*70)
        print("ITERATION 2: FEATURE SCALING")
        print("="*70)
        print("\n🎯 Strategy: Standardize features to same scale")
        print("   - Use StandardScaler (mean=0, std=1)")
        print("   - Helps when features have different ranges\n")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        model = LinearRegression()
        metrics = self.evaluate_model(
            model, X_train_scaled, X_test_scaled,
            self.y_train, self.y_test,
            'Linear Regression + Scaling'
        )

        print(f"📊 Results:")
        print(f"   Train R² Score: {metrics['train_r2']:.4f}")
        print(f"   Test R² Score:  {metrics['test_r2']:.4f}")
        print(f"   Test MSE:       {metrics['test_mse']:.4f}")
        print(f"   Test MAE:       {metrics['test_mae']:.4f}")

        # Compare with baseline
        baseline_r2 = self.results[0]['Test R²']
        improvement = (metrics['test_r2'] - baseline_r2) / baseline_r2 * 100
        print(f"\n📈 Improvement over baseline: {improvement:+.2f}%")

        # Save scaler for later use
        self.scaler = scaler

    def iteration_3_regularization(self):
        """
        ITERATION 3: Add regularization to prevent overfitting
        """
        print("\n" + "="*70)
        print("ITERATION 3: REGULARIZATION")
        print("="*70)
        print("\n🎯 Strategy: Use Ridge regression to reduce overfitting")
        print("   - Adds penalty for large coefficients")
        print("   - Helps prevent overfitting\n")

        X_train_scaled = self.scaler.transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        # Try Ridge regression
        model = Ridge(alpha=1.0)
        metrics = self.evaluate_model(
            model, X_train_scaled, X_test_scaled,
            self.y_train, self.y_test,
            'Ridge Regression'
        )

        print(f"📊 Ridge Results:")
        print(f"   Train R² Score: {metrics['train_r2']:.4f}")
        print(f"   Test R² Score:  {metrics['test_r2']:.4f}")
        print(f"   Test MSE:       {metrics['test_mse']:.4f}")
        print(f"   Test MAE:       {metrics['test_mae']:.4f}")

        # Try Lasso regression
        model = Lasso(alpha=0.1)
        metrics = self.evaluate_model(
            model, X_train_scaled, X_test_scaled,
            self.y_train, self.y_test,
            'Lasso Regression'
        )

        print(f"\n📊 Lasso Results:")
        print(f"   Train R² Score: {metrics['train_r2']:.4f}")
        print(f"   Test R² Score:  {metrics['test_r2']:.4f}")
        print(f"   Test MSE:       {metrics['test_mse']:.4f}")
        print(f"   Test MAE:       {metrics['test_mae']:.4f}")

    def iteration_4_polynomial(self):
        """
        ITERATION 4: Add polynomial features to capture non-linear relationships
        """
        print("\n" + "="*70)
        print("ITERATION 4: POLYNOMIAL FEATURES")
        print("="*70)
        print("\n🎯 Strategy: Create polynomial features")
        print("   - Capture non-linear relationships")
        print("   - Add interaction terms between features\n")

        # Create polynomial features (degree 2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(self.X_train)
        X_test_poly = poly.transform(self.X_test)

        # Scale polynomial features
        scaler = StandardScaler()
        X_train_poly_scaled = scaler.fit_transform(X_train_poly)
        X_test_poly_scaled = scaler.transform(X_test_poly)

        print(f"   Original features: {self.X_train.shape[1]}")
        print(f"   Polynomial features: {X_train_poly.shape[1]}")

        # Use Ridge to prevent overfitting with many features
        model = Ridge(alpha=10.0)
        metrics = self.evaluate_model(
            model, X_train_poly_scaled, X_test_poly_scaled,
            self.y_train, self.y_test,
            'Polynomial + Ridge'
        )

        print(f"\n📊 Results:")
        print(f"   Train R² Score: {metrics['train_r2']:.4f}")
        print(f"   Test R² Score:  {metrics['test_r2']:.4f}")
        print(f"   Test MSE:       {metrics['test_mse']:.4f}")
        print(f"   Test MAE:       {metrics['test_mae']:.4f}")

    def iteration_5_ensemble(self):
        """
        ITERATION 5: Use ensemble methods (Random Forest, Gradient Boosting)
        """
        print("\n" + "="*70)
        print("ITERATION 5: ENSEMBLE METHODS")
        print("="*70)
        print("\n🎯 Strategy: Use ensemble of decision trees")
        print("   - Random Forest: Combines multiple trees")
        print("   - Gradient Boosting: Sequential tree building\n")

        X_train_scaled = self.scaler.transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        # Random Forest
        print("Training Random Forest...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        metrics = self.evaluate_model(
            model, X_train_scaled, X_test_scaled,
            self.y_train, self.y_test,
            'Random Forest'
        )

        print(f"📊 Random Forest Results:")
        print(f"   Train R² Score: {metrics['train_r2']:.4f}")
        print(f"   Test R² Score:  {metrics['test_r2']:.4f}")
        print(f"   Test MSE:       {metrics['test_mse']:.4f}")
        print(f"   Test MAE:       {metrics['test_mae']:.4f}")

        # Gradient Boosting
        print("\nTraining Gradient Boosting...")
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        metrics = self.evaluate_model(
            model, X_train_scaled, X_test_scaled,
            self.y_train, self.y_test,
            'Gradient Boosting'
        )

        print(f"\n📊 Gradient Boosting Results:")
        print(f"   Train R² Score: {metrics['train_r2']:.4f}")
        print(f"   Test R² Score:  {metrics['test_r2']:.4f}")
        print(f"   Test MSE:       {metrics['test_mse']:.4f}")
        print(f"   Test MAE:       {metrics['test_mae']:.4f}")

    def compare_all_models(self):
        """
        Compare all models and show improvement progression
        """
        print("\n" + "="*70)
        print("FINAL COMPARISON: ALL MODELS")
        print("="*70)

        # Create comparison dataframe
        results_df = pd.DataFrame(self.results)

        print("\n📊 Complete Results Table:")
        print(results_df.to_string(index=False))

        # Find best model
        best_idx = results_df['Test R²'].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_r2 = results_df.loc[best_idx, 'Test R²']

        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   Test R² Score: {best_r2:.4f}")
        print(f"   Test MSE: {results_df.loc[best_idx, 'Test MSE']:.4f}")
        print(f"   Test MAE: {results_df.loc[best_idx, 'Test MAE']:.4f}")

        # Calculate improvement from baseline
        baseline_r2 = results_df.loc[0, 'Test R²']
        total_improvement = (best_r2 - baseline_r2) / abs(baseline_r2) * 100
        print(f"\n📈 Total Improvement from Baseline: {total_improvement:+.2f}%")

        # Visualize comparison
        self.visualize_comparison(results_df)

    def visualize_comparison(self, results_df):
        """
        Create visualizations comparing all models
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Test R² comparison
        ax = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
        ax.barh(results_df['Model'], results_df['Test R²'], color=colors)
        ax.set_xlabel('Test R² Score (Higher is Better)')
        ax.set_title('Model Comparison: R² Score')
        ax.grid(True, alpha=0.3)

        # Plot 2: Test MSE comparison
        ax = axes[0, 1]
        ax.barh(results_df['Model'], results_df['Test MSE'], color=colors)
        ax.set_xlabel('Test MSE (Lower is Better)')
        ax.set_title('Model Comparison: Mean Squared Error')
        ax.grid(True, alpha=0.3)

        # Plot 3: Improvement progression
        ax = axes[1, 0]
        baseline_r2 = results_df['Test R²'].iloc[0]
        improvements = [(r2 - baseline_r2) / abs(baseline_r2) * 100
                       for r2 in results_df['Test R²']]
        ax.plot(range(len(improvements)), improvements, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Improvement over Baseline (%)')
        ax.set_title('Progressive Improvement')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # Plot 4: Train vs Test R² (overfitting check)
        ax = axes[1, 1]
        x = np.arange(len(results_df))
        width = 0.35
        ax.bar(x - width/2, results_df['Train R²'], width, label='Train R²', alpha=0.8)
        ax.bar(x + width/2, results_df['Test R²'], width, label='Test R²', alpha=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel('R² Score')
        ax.set_title('Train vs Test R² (Overfitting Check)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"M{i+1}" for i in range(len(results_df))], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved comparison charts as 'model_comparison.png'")
        plt.show()


def main():
    """
    Run the complete iterative improvement demo
    """
    print("="*70)
    print("UGA HACKS11 - ML WORKSHOP")
    print("ITERATIVE MODEL IMPROVEMENT DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows how to progressively improve ML model performance")
    print("through various techniques:\n")
    print("  1. Baseline Model")
    print("  2. Feature Scaling")
    print("  3. Regularization")
    print("  4. Polynomial Features")
    print("  5. Ensemble Methods\n")

    # Create improvement demo instance
    demo = MLModelImprovement()

    # Step 1: Load data
    print("\nWhich dataset would you like to use?")
    print("  1. California Housing (house prices)")
    print("  2. Diabetes (disease progression)")
    print("  3. Custom CSV file")

    choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"

    dataset_map = {'1': 'california', '2': 'diabetes', '3': 'custom'}
    demo.load_data(dataset_map.get(choice, 'california'))

    input("\n👉 Press Enter to explore the data...")

    # Step 2: Explore data
    demo.explore_data()

    input("\n👉 Press Enter to prepare data for training...")

    # Step 3: Prepare data
    demo.prepare_data()

    input("\n👉 Press Enter to start Iteration 1 (Baseline Model)...")

    # Run iterations
    demo.iteration_1_baseline()

    input("\n👉 Press Enter to continue to Iteration 2 (Feature Scaling)...")
    demo.iteration_2_scaling()

    input("\n👉 Press Enter to continue to Iteration 3 (Regularization)...")
    demo.iteration_3_regularization()

    input("\n👉 Press Enter to continue to Iteration 4 (Polynomial Features)...")
    demo.iteration_4_polynomial()

    input("\n👉 Press Enter to continue to Iteration 5 (Ensemble Methods)...")
    demo.iteration_5_ensemble()

    input("\n👉 Press Enter to see final comparison...")

    # Compare all models
    demo.compare_all_models()

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\n🎓 Key Takeaways:")
    print("  ✓ Always start with a simple baseline")
    print("  ✓ Feature scaling often helps linear models")
    print("  ✓ Regularization prevents overfitting")
    print("  ✓ Polynomial features capture non-linearity")
    print("  ✓ Ensemble methods usually perform best")
    print("  ✓ Monitor train vs test performance for overfitting")
    print("\n📊 All visualizations saved as PNG files!")
    print("\nGood luck with your ML projects! 🚀\n")


if __name__ == "__main__":
    main()
