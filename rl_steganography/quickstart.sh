#!/bin/bash

# Quick start script for RL steganography training

echo "=================================================="
echo "RL Steganography - Quick Start"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Run setup test
echo ""
echo "Running setup tests..."
python -m rl_steganography.test_setup

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Setup complete! Ready to train."
    echo "=================================================="
    echo ""
    echo "To start training with default settings:"
    echo "  python -m rl_steganography.train"
    echo ""
    echo "To start training with custom settings:"
    echo "  python -m rl_steganography.train --num-episodes 5000 --password-bits 16"
    echo ""
    echo "For more options:"
    echo "  python -m rl_steganography.train --help"
    echo ""
else
    echo ""
    echo "Setup tests failed. Please check error messages above."
    exit 1
fi
