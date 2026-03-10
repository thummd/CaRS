#!/bin/bash
# Run a single electricity sparsity experiment

set -e

COUNTRY=$1
LAMBDA_SPARSE=$2
SEED=$3

echo "=========================================="
echo "Running electricity sparsity experiment"
echo "Country: $COUNTRY"
echo "Lambda sparse: $LAMBDA_SPARSE"
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
    --seed ${SEED} \
    --output_dir sparsity_results

echo "=========================================="
echo "Experiment completed"
echo "=========================================="
