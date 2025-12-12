"""
Machine Learning with Scikit-Learn - UGA Hacks11 ML Workshop
Demonstrates various ML algorithms using the popular sklearn library
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression

sns.set_style('whitegrid')


class MLWorkshopDemos:
    """Collection of ML algorithm demonstrations"""

    def __init__(self):
        self.results = {}

    def demo_linear_regression(self):
        """
        Demo 1: Linear Regression (Supervised - Regression)
        Predicts continuous values
        Use cases: House prices, stock prices, sales forecasting
        """
        print("\n" + "=" * 70)
        print("DEMO 1: LINEAR REGRESSION")
        print("=" * 70)
        print("Type: Supervised Learning - Regression")
        print("Purpose: Predict continuous numerical values\n")

        # Generate regression data
        X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        print("Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Slope (coefficient): {model.coef_[0]:.4f}")
        print(f"Intercept: {model.intercept_:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f} (closer to 1 is better)")

        # Visualize
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
        plt.plot(X_test, y_pred, 'r-', linewidth=2, label='Predicted')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title('Linear Regression: Predictions')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', linewidth=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted')

        plt.tight_layout()
        plt.savefig('linear_regression_sklearn.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'linear_regression_sklearn.png'\n")
        plt.show()

        self.results['Linear Regression'] = {'MSE': mse, 'R2': r2}

    def demo_logistic_regression(self):
        """
        Demo 2: Logistic Regression (Supervised - Classification)
        Predicts categorical outcomes (binary or multiclass)
        Use cases: Email spam detection, credit approval, disease diagnosis
        """
        print("\n" + "=" * 70)
        print("DEMO 2: LOGISTIC REGRESSION")
        print("=" * 70)
        print("Type: Supervised Learning - Classification")
        print("Purpose: Predict discrete classes (0 or 1, Yes or No)\n")

        # Generate classification data
        X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1,
                                   random_state=42)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        print("Training Logistic Regression model...")
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Correctly classified: {int(accuracy * len(y_test))}/{len(y_test)}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"  True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

        # Visualize
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(X_test[y_test==0][:, 0], X_test[y_test==0][:, 1],
                   c='blue', alpha=0.5, label='Class 0')
        plt.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1],
                   c='red', alpha=0.5, label='Class 1')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Logistic Regression: Test Data')
        plt.legend()

        plt.subplot(1, 2, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        plt.tight_layout()
        plt.savefig('logistic_regression_sklearn.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved as 'logistic_regression_sklearn.png'\n")
        plt.show()

        self.results['Logistic Regression'] = {'Accuracy': accuracy}

    def demo_decision_tree(self):
        """
        Demo 3: Decision Tree (Supervised - Classification)
        Makes decisions based on asking a series of questions
        Use cases: Customer churn prediction, loan approval, disease diagnosis
        """
        print("\n" + "=" * 70)
        print("DEMO 3: DECISION TREE")
        print("=" * 70)
        print("Type: Supervised Learning - Classification")
        print("Purpose: Make decisions using a tree-like model\n")

        # Generate data
        X, y = make_classification(n_samples=200, n_features=4, n_redundant=0,
                                   n_informative=3, random_state=42)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model with limited depth for visualization
        print("Training Decision Tree model...")
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Tree depth: {model.get_depth()}")
        print(f"Number of leaves: {model.get_n_leaves()}")

        # Visualize tree
        plt.figure(figsize=(15, 8))
        plot_tree(model, filled=True, feature_names=[f'Feature {i}' for i in range(4)],
                 class_names=['Class 0', 'Class 1'], rounded=True)
        plt.title('Decision Tree Visualization')
        plt.tight_layout()
        plt.savefig('decision_tree_sklearn.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'decision_tree_sklearn.png'\n")
        plt.show()

        self.results['Decision Tree'] = {'Accuracy': accuracy}

    def demo_random_forest(self):
        """
        Demo 4: Random Forest (Supervised - Regression)
        Ensemble of multiple decision trees
        Use cases: House price prediction, stock price forecasting
        """
        print("\n" + "=" * 70)
        print("DEMO 4: RANDOM FOREST")
        print("=" * 70)
        print("Type: Supervised Learning - Regression (Ensemble)")
        print("Purpose: Combine multiple decision trees for better predictions\n")

        # Generate data
        X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        print("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Number of trees: {model.n_estimators}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")

        # Feature importance
        feature_importance = model.feature_importances_
        print("\nFeature Importance:")
        for i, importance in enumerate(feature_importance):
            print(f"  Feature {i}: {importance:.4f}")

        # Visualize
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title('Feature Importance')

        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', linewidth=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Random Forest: Predictions')

        plt.tight_layout()
        plt.savefig('random_forest_sklearn.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved as 'random_forest_sklearn.png'\n")
        plt.show()

        self.results['Random Forest'] = {'MSE': mse, 'R2': r2}

    def demo_kmeans(self):
        """
        Demo 5: K-Means Clustering (Unsupervised)
        Groups data into clusters without labels
        Use cases: Customer segmentation, document clustering, image compression
        """
        print("\n" + "=" * 70)
        print("DEMO 5: K-MEANS CLUSTERING")
        print("=" * 70)
        print("Type: Unsupervised Learning")
        print("Purpose: Group similar data points together\n")

        # Generate clustered data
        from sklearn.datasets import make_blobs
        X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

        # Train model
        print("Training K-Means model with 4 clusters...")
        model = KMeans(n_clusters=4, random_state=42)
        y_pred = model.fit_predict(X)

        print(f"Number of clusters: {model.n_clusters}")
        print(f"Cluster centers found at:")
        for i, center in enumerate(model.cluster_centers_):
            print(f"  Cluster {i}: {center}")

        # Visualize
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.5)
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                   label='Centroids')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('K-Means Clustering Results')
        plt.legend()

        plt.subplot(1, 2, 2)
        # Elbow method
        inertias = []
        K_range = range(1, 10)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal K')

        plt.tight_layout()
        plt.savefig('kmeans_sklearn.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved as 'kmeans_sklearn.png'\n")
        plt.show()

        self.results['K-Means'] = {'n_clusters': model.n_clusters}

    def print_summary(self):
        """Print summary of all demos"""
        print("\n" + "=" * 70)
        print("WORKSHOP SUMMARY")
        print("=" * 70)
        print("\nResults from all demonstrations:")
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")


def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("UGA HACKS11 - INTRO TO MACHINE LEARNING WORKSHOP")
    print("Machine Learning with Scikit-Learn")
    print("=" * 70)

    demos = MLWorkshopDemos()

    print("\nThis demo will showcase 5 different ML algorithms:")
    print("1. Linear Regression (Supervised - Regression)")
    print("2. Logistic Regression (Supervised - Classification)")
    print("3. Decision Tree (Supervised - Classification)")
    print("4. Random Forest (Supervised - Regression)")
    print("5. K-Means Clustering (Unsupervised)")

    input("\nPress Enter to start the demonstrations...")

    # Run all demos
    demos.demo_linear_regression()
    input("\nPress Enter to continue to next demo...")

    demos.demo_logistic_regression()
    input("\nPress Enter to continue to next demo...")

    demos.demo_decision_tree()
    input("\nPress Enter to continue to next demo...")

    demos.demo_random_forest()
    input("\nPress Enter to continue to next demo...")

    demos.demo_kmeans()

    # Print summary
    demos.print_summary()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- Supervised Learning: Uses labeled data (Linear, Logistic, Trees)")
    print("- Unsupervised Learning: Finds patterns in unlabeled data (K-Means)")
    print("- Regression: Predicts continuous values (Linear, Random Forest)")
    print("- Classification: Predicts categories (Logistic, Decision Tree)")
    print("\nAll plots have been saved to your current directory!")


if __name__ == "__main__":
    main()
