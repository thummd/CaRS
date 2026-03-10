#!/bin/bash
# Wrapper script for regime detection
# Usage: ./run_regime.sh <country> <n_regimes>

set -e

COUNTRY=$1
N_REGIMES=${2:-3}

echo "=============================================="
echo "Regime Detection: ${COUNTRY}"
echo "Number of regimes: ${N_REGIMES}"
echo "Start time: $(date)"
echo "=============================================="

# Change to electricity directory
cd /lustre/home/dthumm/CASTOR/electricity

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "GPU detected:"
    nvidia-smi
else
    DEVICE="cpu"
    echo "No GPU detected, using CPU"
fi

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/lustre/home/dthumm/CASTOR/electricity/regime_results/${COUNTRY}_${N_REGIMES}_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

echo "Output directory: ${OUTPUT_DIR}"
echo "Device: ${DEVICE}"

# Run regime detection with checkpointing
python3 fantom_regime.py \
    --country ${COUNTRY} \
    --n_regimes ${N_REGIMES} \
    --window_size 200 \
    --min_regime_size 80 \
    --max_iterations 3 \
    --device ${DEVICE} \
    --output_dir ${OUTPUT_DIR} \
    --checkpoint_dir ${OUTPUT_DIR}

echo "=============================================="
echo "Regime Detection Complete"
echo "End time: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
