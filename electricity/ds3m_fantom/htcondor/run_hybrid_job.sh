#!/bin/bash
# Run a single DS3M-Causal hybrid experiment

set -e

COUNTRY=$1
SHARING_MODE=$2
D_DIM=$3
SEED=$4
TRAINING_MODE=${5:-end_to_end}  # Default to end_to_end if not specified

echo "=========================================="
echo "Running DS3M-Causal Hybrid Experiment"
echo "Country: $COUNTRY"
echo "Sharing mode: $SHARING_MODE"
echo "Training mode: $TRAINING_MODE"
echo "d_dim: $D_DIM"
echo "Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH

# Add DS3M to path
export PYTHONPATH=/lustre/home/dthumm/Deep-Switching-State-Space-Model/src:$PYTHONPATH
export PYTHONPATH=/lustre/home/dthumm/Deep-Switching-State-Space-Model:$PYTHONPATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ "$SHARING_MODE" == "shared_backbone" ]; then
    MODE_SHORT="shared"
else
    MODE_SHORT="ind"
fi
if [ "$TRAINING_MODE" == "two_stage" ]; then
    TRAIN_SHORT="2stg"
else
    TRAIN_SHORT="e2e"
fi
OUTPUT_DIR="outputs/ds3m_causal/${COUNTRY}_${MODE_SHORT}_${TRAIN_SHORT}_d${D_DIM}_seed${SEED}_${TIMESTAMP}"

# Create temporary config with d_dim override
CONFIG_FILE="ds3m_fantom/config/ds3m_causal.yaml"

# Run the experiment
python3 -m ds3m_fantom.run_experiments \
    --country ${COUNTRY} \
    --sharing_mode ${SHARING_MODE} \
    --training_mode ${TRAINING_MODE} \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --train \
    --config ${CONFIG_FILE} \
    --output_dir ${OUTPUT_DIR}

echo "=========================================="
echo "Experiment completed"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
