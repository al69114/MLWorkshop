# UGA Hacks11 - Intro to Machine Learning Workshop

Welcome to the ML Workshop! This repository contains everything you need to get started with Machine Learning.

## 📚 Workshop Overview

This workshop covers:
- **What is Machine Learning?** Understanding ML vs AI
- **Types of ML Models**: Supervised, Unsupervised, Reinforcement Learning
- **ML Workflow**: From data collection to model evaluation
- **Linear Regression from Scratch**: Implementation using Gradient Descent
- **Hands-on Demo**: Using scikit-learn for various ML algorithms

## 📁 Repository Contents

```
MLWorkshop/
├── linear_regression_from_scratch.py    # Linear Regression from scratch
├── ml_with_sklearn.py                   # Demos using scikit-learn library
├── iterative_model_improvement.py       # Show how to improve model results
├── custom_dataset_demo.py               # Train on ANY CSV dataset
├── DATASET_SUGGESTIONS.md               # Curated list of datasets for ML
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🚀 Getting Started

### Local Setup

#### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

#### Installation

1. **Clone or download this repository**
   ```bash
   cd /path/to/MLWorkshop
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Running the Demos

### Demo 1: Linear Regression from Scratch

Implements Linear Regression using Gradient Descent to understand the underlying math and logic.

```bash
python linear_regression_from_scratch.py
```

**What it does:**
- Generates synthetic data
- Trains a linear regression model from scratch
- Visualizes the best fit line and training progress
- Saves visualizations as PNG files

### Demo 2: ML with Scikit-Learn

Demonstrates multiple ML algorithms using the popular scikit-learn library.

```bash
python ml_with_sklearn.py
```

**What it covers:**
1. Linear Regression (Supervised - Regression)
2. Logistic Regression (Supervised - Classification)
3. Decision Tree (Supervised - Classification)
4. Random Forest (Supervised - Regression)
5. K-Means Clustering (Unsupervised)

### Demo 3: Iterative Model Improvement (NEW!)

Shows how to progressively improve your model through 5 iterations:

```bash
python iterative_model_improvement.py
```

**What it demonstrates:**
1. Baseline model (simple approach)
2. Feature scaling (standardization)
3. Regularization (Ridge/Lasso)
4. Polynomial features (non-linear relationships)
5. Ensemble methods (Random Forest, Gradient Boosting)

Each iteration shows improvement over the previous one!

### Demo 4: Custom Dataset Training (NEW!)

Train on YOUR OWN CSV dataset:

```bash
python custom_dataset_demo.py
```

**Perfect for:**
- Utrecht dataset (or any dataset you have)
- Your own collected data
- Kaggle competition datasets
- Research data

**Features:**
- Automatic data cleaning
- Handles categorical variables
- Trains 4 different models
- Shows which model works best
- Creates visualizations

## 📊 Working with Datasets

Check out `DATASET_SUGGESTIONS.md` for:
- Beginner-friendly datasets
- Dataset sources (Kaggle, UCI, etc.)
- Code examples for loading data
- Recommendations by task type

### Quick Dataset Loading Examples

```python
# Built-in sklearn datasets
from sklearn.datasets import load_iris, fetch_california_housing

iris = load_iris()
housing = fetch_california_housing()

# From CSV
import pandas as pd
df = pd.read_csv('data.csv')

# From Kaggle (requires API setup)
!kaggle datasets download -d <dataset-name>
```

## 🎯 Key Concepts

### Linear Regression
- **Equation**: `y = mx + b`
- **Goal**: Find the best fit line that minimizes error
- **Method**: Gradient Descent
- **Evaluation**: Mean Squared Error (MSE)

### ML Workflow
1. **Identify the problem**: What are we trying to predict?
2. **Data collection**: Gather relevant data
3. **Data cleaning**: Handle missing values, outliers
4. **Feature selection**: Choose important features
5. **Model building**: Select and train algorithm
6. **Model testing**: Evaluate on test data
7. **Evaluation**: Measure performance and iterate

### Types of ML
- **Supervised Learning**: Uses labeled data
  - Classification: Predict discrete classes (spam/not spam)
  - Regression: Predict continuous values (house prices)
- **Unsupervised Learning**: Find patterns in unlabeled data
  - Clustering: Group similar items (customer segmentation)
- **Reinforcement Learning**: Learn through trial and error

## 📈 Example Output

After running the demos, you'll see:
- **Visualizations**: Scatter plots, line plots, confusion matrices
- **Performance metrics**: MSE, R² score, accuracy
- **Model parameters**: Learned slope, intercept, feature importance

## 🔧 Troubleshooting

### Common Issues

**Import errors:**
```bash
# Make sure all packages are installed
pip install -r requirements.txt
```

**Display issues with plots:**
```python
# If plots don't show, add this at the top of your script
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

**Jupyter notebook not opening:**
```bash
# Install jupyter if not already installed
pip install jupyter notebook

# Then try again
jupyter notebook
```

## 📖 Additional Resources

### Learning Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Fast.ai](https://www.fast.ai/)
- [Machine Learning Crash Course (Google)](https://developers.google.com/machine-learning/crash-course)

### Dataset Sources
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)
- [Google Dataset Search](https://datasetsearch.research.google.com/)
- [Hugging Face Datasets](https://huggingface.co/datasets)

### Tools
- **Google Colab**: Free cloud-based Jupyter notebooks with GPU
- **Kaggle Kernels**: Run code in the browser
- **Jupyter Notebook**: Interactive Python environment

## 🎓 Workshop Challenges

Try these after completing the demos:

1. **Beginner**:
   - Load the Iris dataset and train a classifier
   - Visualize the results

2. **Intermediate**:
   - Load the Titanic dataset from Kaggle
   - Perform data cleaning and feature engineering
   - Train multiple models and compare accuracy

3. **Advanced**:
   - Implement polynomial regression from scratch
   - Build a neural network for image classification
   - Create an ensemble model

## 🤝 Contributing

Have suggestions or found issues? Feel free to:
- Open an issue
- Submit a pull request
- Share your ML projects!

## 📝 License

This project is for educational purposes as part of UGA Hacks11.

## 🙏 Acknowledgments

- UGA Hacks11 organizers
- scikit-learn contributors
- All open dataset providers

---

## 🎉 Happy Learning!

Remember: Machine Learning is a journey, not a destination. Start simple, practice often, and build amazing projects!

**For questions during the workshop, ask the instructors or check the documentation.**

Good luck with your ML projects! 🚀

---

**Workshop**: Intro to Machine Learning
**Event**: UGA Hacks11
