#!/bin/bash
# Run DS3M multivariate with variable number of regimes

set -e

COUNTRY=$1
D_DIM=$2
SEED=$3

echo "=========================================="
echo "Running DS3M Multivariate Regime Experiment"
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

# Run the test with all feature groups
python3 test_ds3m_unified.py \
    --dataset ${COUNTRY} \
    --feature_groups price load weather calendar outage commodity \
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
echo "DS3M regime experiment completed"
echo "=========================================="
