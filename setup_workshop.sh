#!/bin/bash
# Setup script for ML Workshop

echo "=========================================="
echo "ML Workshop - Environment Setup"
echo "=========================================="

# Navigate to workshop directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt -q

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the Utrecht housing demo:"
echo "  python utrecht_housing_demo.py"
echo ""
echo "To run other demos:"
echo "  python ml_with_sklearn.py"
echo "  python custom_dataset_demo.py"
echo "=========================================="
