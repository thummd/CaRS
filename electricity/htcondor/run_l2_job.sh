#!/bin/bash
# Run a single electricity L2 sparsity sweep experiment

set -e

COUNTRY=$1
LAMBDA_SPARSE=$2
LAMBDA_SPARSE_L2=$3
L2_GROUP_MODE=$4
SEED=$5

echo "=========================================="
echo "Running electricity L2 sparsity experiment"
echo "Country: $COUNTRY"
echo "Lambda sparse (L1): $LAMBDA_SPARSE"
echo "Lambda sparse L2: $LAMBDA_SPARSE_L2"
echo "L2 group mode: $L2_GROUP_MODE"
echo "Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment (matching CASTOR cluster setup)
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH

# Set CUDA device (use GPU if available)
export CUDA_VISIBLE_DEVICES=0

# Run the experiment
python3 train_sparsity_sweep.py \
    --country ${COUNTRY} \
    --lambda_sparse ${LAMBDA_SPARSE} \
    --lambda_sparse_l2 ${LAMBDA_SPARSE_L2} \
    --l2_group_mode ${L2_GROUP_MODE} \
    --seed ${SEED} \
    --output_dir sparsity_results

echo "=========================================="
echo "Experiment completed"
echo "=========================================="
