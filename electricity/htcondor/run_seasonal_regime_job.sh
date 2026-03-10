#!/bin/bash
# Run a single seasonal regime detection experiment

set -e

COUNTRY=$1
N_REGIMES=$2
INIT_MODE=$3
MIN_REGIME_SIZE=$4
SEED=$5

echo "==========================================="
echo "Running seasonal regime detection experiment"
echo "Country: $COUNTRY"
echo "N_regimes: $N_REGIMES"
echo "Init mode: $INIT_MODE"
echo "Min regime size: $MIN_REGIME_SIZE"
echo "Seed: $SEED"
echo "==========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment (matching CASTOR cluster setup)
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the experiment
python3 fantom_regime_seasonal.py \
    --country ${COUNTRY} \
    --n_regimes ${N_REGIMES} \
    --init_mode ${INIT_MODE} \
    --min_regime_size ${MIN_REGIME_SIZE} \
    --seed ${SEED} \
    --device cuda

echo "==========================================="
echo "Experiment completed"
echo "==========================================="
