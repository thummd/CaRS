#!/bin/bash
# Run FANTOM on DE-FR spread prediction

set -e

SEED=$1

echo "=========================================="
echo "Running FANTOM Spread Prediction"
echo "Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Run FANTOM spread prediction
python3 test_fantom_unified.py \
    --dataset DE_FR \
    --lag 1 \
    --max_features 15 \
    --n_epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --lambda_sparse 1.0 \
    --seed ${SEED} \
    --device cuda \
    --feature_groups spread calendar

echo "=========================================="
echo "FANTOM spread experiment completed"
echo "=========================================="
