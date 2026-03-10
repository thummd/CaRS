#!/bin/bash
# Run FANTOM test on unified electricity data

set -e

COUNTRY=$1
FEATURE_GROUPS_RAW=$2
SEED=$3

# Convert underscores to spaces for multiple arguments
FEATURE_GROUPS=$(echo "$FEATURE_GROUPS_RAW" | tr '_' ' ')

echo "=========================================="
echo "Running FANTOM on unified electricity data"
echo "Country: $COUNTRY"
echo "Feature groups: $FEATURE_GROUPS"
echo "Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the test
python3 test_fantom_unified.py \
    --dataset ${COUNTRY} \
    --feature_groups ${FEATURE_GROUPS} \
    --lag 1 \
    --max_features 15 \
    --n_epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --lambda_sparse 1.0 \
    --seed ${SEED} \
    --device cuda

echo "=========================================="
echo "FANTOM test completed"
echo "=========================================="
