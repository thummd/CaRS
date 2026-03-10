#!/bin/bash
# =============================================================================
# CaRS Environment Setup Script
# Creates conda environment and installs all dependencies for AIDF server
# =============================================================================

set -e

echo "=================================================="
echo "CaRS Environment Setup"
echo "=================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

ENV_NAME="cars"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Warning: Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

# Create conda environment
echo ""
echo "Creating conda environment '${ENV_NAME}' with Python 3.10..."
conda create -n ${ENV_NAME} python=3.10 -y

# Activate environment
echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Detect CUDA version and install PyTorch accordingly
echo ""
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    echo "Detected CUDA version: $CUDA_VERSION"

    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
    fi
else
    echo "Warning: nvidia-smi not found. Installing CPU-only PyTorch."
    pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Install requirements
echo ""
echo "Installing Python packages..."
cd "$SCRIPT_DIR"
pip install -r requirements.txt
pip install -r requirements_fantom.txt

# Initialize git submodules (upstream repos)
echo ""
echo "Initializing git submodules..."
if [ -f ".gitmodules" ]; then
    git submodule update --init --recursive
    echo "Submodules initialized."
else
    echo "No .gitmodules found (submodules may need to be added)."
fi

# Create project directories
echo ""
echo "Creating project directories..."
mkdir -p data/{unified,entsoe,weather,commodities,gas_storage,outages,macro,sentiment,oil_fundamentals,transport,trade,hydrogen,epftoolbox,synthetic}
mkdir -p outputs
mkdir -p logs

# Set CARS_ROOT
echo ""
echo "Setting CARS_ROOT environment variable..."
if ! grep -q 'export CARS_ROOT=' ~/.bashrc 2>/dev/null; then
    echo "export CARS_ROOT=${SCRIPT_DIR}" >> ~/.bashrc
    echo "Added CARS_ROOT=${SCRIPT_DIR} to ~/.bashrc"
else
    echo "CARS_ROOT already set in ~/.bashrc"
fi

# Copy .env if needed
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
    echo ""
    echo "Note: Copy .env.example to .env and add your API keys:"
    echo "  cp .env.example .env"
    echo "  nano .env"
fi

# Print success message
echo ""
echo "=================================================="
echo "Setup completed successfully!"
echo "=================================================="
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To test:"
echo "  python -c \"import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())\""
echo "  python -c \"from electricity.country_config import COUNTRY_REGISTRY; print(len(COUNTRY_REGISTRY), 'countries')\""
echo ""
echo "To run an experiment:"
echo "  ./run_experiment.sh electricity/ds3m_electricity.py --country DE --train"
echo ""
