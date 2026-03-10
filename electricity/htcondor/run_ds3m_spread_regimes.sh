#!/bin/bash
# Run DS3M spread prediction (DE-FR) with variable number of regimes

set -e

D_DIM=$1
SEED=$2

echo "=========================================="
echo "Running DS3M Spread Prediction Experiment"
echo "D_dim (regimes): $D_DIM"
echo "Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the spread prediction test
python3 test_ds3m_spread.py \
    --timestep 14 \
    --d_dim ${D_DIM} \
    --h_dim 30 \
    --z_dim 8 \
    --n_epochs 150 \
    --batch_size 64 \
    --lr 0.001 \
    --seed ${SEED} \
    --device cuda \
    --feature_groups spread calendar

echo "=========================================="
echo "DS3M spread experiment completed"
echo "=========================================="
