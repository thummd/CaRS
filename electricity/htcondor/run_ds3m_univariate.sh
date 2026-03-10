#!/bin/bash
# Run DS3M Univariate test on unified electricity data

set -e

COUNTRY=$1
SEED=$2

echo "=========================================="
echo "Running DS3M UNIVARIATE on unified electricity data"
echo "Country: $COUNTRY"
echo "Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the univariate test
python3 test_ds3m_unified_univariate.py \
    --dataset ${COUNTRY} \
    --timestep 14 \
    --d_dim 2 \
    --h_dim 30 \
    --z_dim 8 \
    --n_epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --seed ${SEED} \
    --device cuda

echo "=========================================="
echo "DS3M Univariate test completed"
echo "=========================================="
