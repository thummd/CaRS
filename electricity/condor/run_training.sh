#!/bin/bash
# Wrapper script for full model training
# Usage: ./run_training.sh <experiment>

set -e

EXPERIMENT=$1

echo "=============================================="
echo "Full Training: ${EXPERIMENT}"
echo "Start time: $(date)"
echo "=============================================="

# Change to electricity directory
cd /lustre/home/dthumm/CASTOR/electricity

# Run training
python3 train_fantom.py \
    --experiment ${EXPERIMENT} \
    --device cpu

echo "=============================================="
echo "Training Complete"
echo "End time: $(date)"
echo "=============================================="
