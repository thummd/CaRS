#!/bin/bash
# Run DS3M training on temporal epftoolbox data
# This is for proper temporal forecasting (not cross-sectional)

set -e

DATASET=$1
MODE=$2
D_DIM=$3
SEED=$4
TIMESTEP=${5:-14}

echo "=========================================="
echo "DS3M Temporal Electricity Forecasting"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Mode: $MODE"
echo "D_DIM: $D_DIM"
echo "Seed: $SEED"
echo "Timestep: $TIMESTEP"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    DEVICE="cuda"
else
    echo "No GPU detected, using CPU"
    DEVICE="cpu"
fi

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/temporal/${DATASET}_${MODE}_d${D_DIM}_t${TIMESTEP}_seed${SEED}_${TIMESTAMP}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python3 train_ds3m_temporal.py \
    --dataset ${DATASET} \
    --mode ${MODE} \
    --d_dim ${D_DIM} \
    --timestep ${TIMESTEP} \
    --seed ${SEED} \
    --n_epochs 200 \
    --batch_size 64 \
    --device ${DEVICE} \
    --save_dir ${OUTPUT_DIR}

echo "=========================================="
echo "Training completed"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
