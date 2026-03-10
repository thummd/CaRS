#!/bin/bash
# Wrapper script for multiseed experiments
# Usage: ./run_multiseed.sh <model> <dataset> <d_dim> <task_type> <seeds>

set -e

MODEL=$1
DATASET=$2
D_DIM=$3
TASK_TYPE=$4
SEEDS=${5:-5}

echo "=============================================="
echo "Multiseed Experiment"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "d_dim: ${D_DIM}"
echo "Task type: ${TASK_TYPE}"
echo "Seeds: ${SEEDS}"
echo "Start time: $(date)"
echo "=============================================="

# Change to electricity directory
cd /lustre/home/dthumm/CASTOR/electricity

# Use system python (has torch 2.9.1+cu128)
# Note: venv_fantom does not have torch installed
export PYTHONPATH="/lustre/home/dthumm/CASTOR/venv_fantom/lib/python3.12/site-packages:$PYTHONPATH"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi
    echo ""
    echo "GPU Memory:"
    nvidia-smi --query-gpu=memory.total,memory.free --format=csv
else
    echo "WARNING: No GPU detected!"
fi

echo ""
echo "Running experiment..."
echo ""

# Run the experiment (use system python which has torch)
/usr/bin/python3 run_multiseed_experiments.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --d_dim ${D_DIM} \
    --task_type ${TASK_TYPE} \
    --seeds ${SEEDS}

echo ""
echo "=============================================="
echo "Experiment Complete"
echo "End time: $(date)"
echo "=============================================="
