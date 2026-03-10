#!/bin/bash
# Run DS3M-FANTOM Hybrid experiments on unified electricity data

set -e

COUNTRY=$1
DATA_SOURCE=$2
SHARING_MODE=$3
SEED=$4

echo "=========================================="
echo "Running DS3M-FANTOM Hybrid on unified electricity data"
echo "Country: $COUNTRY"
echo "Data source: $DATA_SOURCE"
echo "Sharing mode: $SHARING_MODE"
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

# Run the hybrid experiment
python3 ds3m_fantom/run_experiments.py \
    --country ${COUNTRY} \
    --data_source ${DATA_SOURCE} \
    --sharing_mode ${SHARING_MODE} \
    --training_mode end_to_end \
    --d_dim 2 \
    --seed ${SEED} \
    --train

echo "=========================================="
echo "Hybrid experiment completed"
echo "=========================================="
