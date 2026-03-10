#!/bin/bash
# Run multi-seed experiments for electricity price prediction
# Usage: ./run_multiseed_job.sh <model> <dataset> <d_dim> [task_type]
# task_type: 'prediction' (default) or 'estimation'

set -e

MODEL=$1
DATASET=$2
D_DIM=$3
TASK_TYPE=${4:-prediction}  # Default to 'prediction' if not specified

echo "=========================================="
echo "Running Multi-Seed Experiment"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "D_dim (regimes): $D_DIM"
echo "Task type: $TASK_TYPE"
echo "Seeds: 5"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment (matching CASTOR cluster setup)
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the experiment with 5 seeds
python3 run_multiseed_experiments.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --d_dim ${D_DIM} \
    --seeds 5 \
    --device cuda \
    --task_type ${TASK_TYPE}

echo "=========================================="
echo "Multi-seed experiment completed"
echo "=========================================="
