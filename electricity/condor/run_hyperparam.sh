#!/bin/bash
# Wrapper script for hyperparameter search
# Usage: ./run_hyperparam.sh <country> <n_trials>

set -e

COUNTRY=$1
N_TRIALS=${2:-30}

echo "=============================================="
echo "Hyperparameter Search: ${COUNTRY}"
echo "Trials: ${N_TRIALS}"
echo "Start time: $(date)"
echo "=============================================="

# Change to electricity directory
cd /lustre/home/dthumm/CASTOR/electricity

# Activate environment if needed
# source /path/to/venv/bin/activate

# Run hyperparameter search
python3 hyperparam_search.py \
    --country ${COUNTRY} \
    --n_trials ${N_TRIALS} \
    --n_folds 3 \
    --device cpu

echo "=============================================="
echo "Hyperparameter Search Complete"
echo "End time: $(date)"
echo "=============================================="
