#!/bin/bash
# Setup script for Gilbreth cluster
# Run this once to set up the environment

echo "Setting up environment on Gilbreth..."

# Load required modules
module load anaconda
module load cuda

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "drowsiness"; then
    echo "Creating conda environment..."
    conda create -n drowsiness python=3.10 -y
fi

# Activate environment
source activate drowsiness

# Install required packages
echo "Installing Python packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python-headless mediapipe scikit-learn pandas numpy tqdm kagglehub

echo "Setup complete!"
echo "To activate environment: source activate drowsiness"
