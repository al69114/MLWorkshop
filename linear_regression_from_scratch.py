"""
Linear Regression from Scratch - UGA Hacks11 ML Workshop
Implementation using Gradient Descent as described in the workshop slides
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionScratch:
    """
    Linear Regression implementation from scratch using Gradient Descent

    Equation: y = mx + b (or y = β₁x + β₀)
    - m (slope): determines the steepness of the line
    - b (intercept): where the line crosses the y-axis

    Goal: Find the best fit line that minimizes the Mean Squared Error (MSE)
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize the Linear Regression model

        Args:
            learning_rate: How much to adjust weights during gradient descent
            iterations: Number of times to update weights
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.slope = 0  # m in y = mx + b
        self.intercept = 0  # b in y = mx + b
        self.mse_history = []  # Track MSE over iterations

    def calculate_mse(self, y_true, y_pred):
        """
        Calculate Mean Squared Error

        MSE = (1/n) * Σ(y_true - y_pred)²

        This measures how well our line fits the data.
        Lower MSE = better fit
        """
        n = len(y_true)
        mse = (1/n) * np.sum((y_true - y_pred) ** 2)
        return mse

    def fit(self, X, y):
        """
        Train the model using Gradient Descent

        Gradient Descent iteratively adjusts slope and intercept to minimize error

        Steps:
        1. Make predictions with current slope and intercept
        2. Calculate error
        3. Calculate gradients (how much to change slope and intercept)
        4. Update slope and intercept
        5. Repeat
        """
        n = len(X)

        print("Training Linear Regression Model...")
        print(f"Initial slope: {self.slope}, Initial intercept: {self.intercept}")

        for i in range(self.iterations):
            # Make predictions with current parameters
            y_pred = self.slope * X + self.intercept

            # Calculate MSE for this iteration
            mse = self.calculate_mse(y, y_pred)
            self.mse_history.append(mse)

            # Calculate gradients (partial derivatives)
            # Gradient for slope (m): -(2/n) * Σ(x * (y_true - y_pred))
            # Gradient for intercept (b): -(2/n) * Σ(y_true - y_pred)
            slope_gradient = -(2/n) * np.sum(X * (y - y_pred))
            intercept_gradient = -(2/n) * np.sum(y - y_pred)

            # Update parameters using gradient descent
            # New value = Old value - (learning_rate * gradient)
            self.slope = self.slope - (self.learning_rate * slope_gradient)
            self.intercept = self.intercept - (self.learning_rate * intercept_gradient)

            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.iterations} - MSE: {mse:.4f}, "
                      f"Slope: {self.slope:.4f}, Intercept: {self.intercept:.4f}")

        print("\nTraining Complete!")
        print(f"Final slope (m): {self.slope:.4f}")
        print(f"Final intercept (b): {self.intercept:.4f}")
        print(f"Final MSE: {self.mse_history[-1]:.4f}")

    def predict(self, X):
        """
        Make predictions using the trained model

        Formula: y = mx + b
        """
        return self.slope * X + self.intercept

    def plot_results(self, X_train, y_train, X_test=None, y_test=None):
        """
        Visualize the linear regression results
        """
        plt.figure(figsize=(15, 5))

        # Plot 1: Training data with best fit line
        plt.subplot(1, 3, 1)
        plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
        plt.plot(X_train, self.predict(X_train), color='red', linewidth=2,
                label=f'Best Fit Line: y = {self.slope:.2f}x + {self.intercept:.2f}')
        plt.xlabel('X (Independent Variable)')
        plt.ylabel('Y (Dependent Variable)')
        plt.title('Linear Regression: Best Fit Line')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: MSE over iterations (shows convergence)
        plt.subplot(1, 3, 2)
        plt.plot(self.mse_history, color='green', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Model Training: MSE Convergence')
        plt.grid(True, alpha=0.3)

        # Plot 3: Predictions vs Actual (if test data provided)
        if X_test is not None and y_test is not None:
            plt.subplot(1, 3, 3)
            y_pred = self.predict(X_test)
            plt.scatter(y_test, y_pred, color='purple', alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', linewidth=2, label='Perfect Prediction')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predictions vs Actual')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved as 'linear_regression_results.png'")
        plt.show()


def generate_sample_data(n_samples=100, noise=10):
    """
    Generate sample data for demonstration
    Creates data that follows: y = 2.5x + 5 + noise
    """
    np.random.seed(42)
    X = np.random.rand(n_samples) * 100  # Random X values between 0 and 100
    y = 2.5 * X + 5 + np.random.randn(n_samples) * noise  # Linear relationship with noise
    return X, y


if __name__ == "__main__":
    print("=" * 70)
    print("LINEAR REGRESSION FROM SCRATCH - UGA HACKS11 ML WORKSHOP")
    print("=" * 70)
    print("\nThis demo implements Linear Regression using Gradient Descent")
    print("Goal: Find the best fit line y = mx + b that minimizes error\n")

    # Generate sample data
    print("Step 1: Generating sample data...")
    X, y = generate_sample_data(n_samples=100, noise=10)
    print(f"Generated {len(X)} data points\n")

    # Split data into train and test (80/20 split)
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")

    # Create and train model
    print("Step 2: Training the model...\n")
    model = LinearRegressionScratch(learning_rate=0.0001, iterations=500)
    model.fit(X_train, y_train)

    # Make predictions
    print("\nStep 3: Making predictions on test data...")
    y_pred = model.predict(X_test)
    test_mse = model.calculate_mse(y_test, y_pred)
    print(f"Test MSE: {test_mse:.4f}")

    # Example predictions
    print("\nExample Predictions:")
    for i in range(min(5, len(X_test))):
        print(f"  X = {X_test[i]:.2f} -> Predicted: {y_pred[i]:.2f}, Actual: {y_test[i]:.2f}")

    # Visualize results
    print("\nStep 4: Visualizing results...")
    model.plot_results(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
