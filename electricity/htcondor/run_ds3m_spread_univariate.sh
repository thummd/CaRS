#!/bin/bash
# Run DS3M UNIVARIATE on DE-FR spread prediction
# Uses only spread history (no multivariate features)

set -e

D_DIM=$1
SEED=$2

echo "=========================================="
echo "Running DS3M UNIVARIATE Spread Prediction"
echo "D_dim (regimes): $D_DIM"
echo "Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Run univariate spread prediction
python3 test_ds3m_unified_univariate.py \
    --dataset DE_FR \
    --timestep 14 \
    --d_dim ${D_DIM} \
    --h_dim 30 \
    --z_dim 8 \
    --n_epochs 150 \
    --batch_size 64 \
    --lr 0.001 \
    --seed ${SEED} \
    --device cuda

echo "=========================================="
echo "DS3M Univariate spread experiment completed"
echo "=========================================="
