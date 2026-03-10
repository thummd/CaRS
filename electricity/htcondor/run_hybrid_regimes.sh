#!/bin/bash
# Run DS3M-FANTOM Hybrid with variable number of regimes

set -e

COUNTRY=$1
D_DIM=$2
SEED=$3

echo "=========================================="
echo "Running DS3MCausal Hybrid Regime Experiment"
echo "Country: $COUNTRY"
echo "D_dim (regimes): $D_DIM"
echo "Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Memory optimization for large models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the hybrid experiment with variable d_dim
python3 ds3m_fantom/run_experiments.py \
    --country ${COUNTRY} \
    --data_source unified \
    --sharing_mode independent \
    --training_mode end_to_end \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --train

echo "=========================================="
echo "Hybrid regime experiment completed"
echo "=========================================="
