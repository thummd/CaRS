#!/bin/bash
# Run DS3M spread prediction test

set -e

SEED=$1
D_DIM=${2:-2}

echo "=========================================="
echo "Running DS3M on DE-FR Spread Prediction"
echo "Seed: $SEED"
echo "D_dim: $D_DIM"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the spread test
python3 test_ds3m_spread.py \
    --timestep 14 \
    --d_dim ${D_DIM} \
    --h_dim 30 \
    --z_dim 8 \
    --n_epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --seed ${SEED} \
    --device cuda \
    --feature_groups spread price_de price_fr load_de load_fr calendar

echo "=========================================="
echo "DS3M Spread test completed"
echo "=========================================="
