#!/bin/bash
# Run a single DS3M multivariate electricity experiment

set -e

COUNTRY=$1
MODE=$2
D_DIM=$3
SEED=$4

echo "=========================================="
echo "Running DS3M electricity experiment"
echo "Country: $COUNTRY"
echo "Mode: $MODE"
echo "d_dim: $D_DIM"
echo "Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment (matching CASTOR cluster setup)
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH

# Set CUDA device (use GPU if available)
export CUDA_VISIBLE_DEVICES=0

# Create output directory name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODE_SHORT="uv"
if [ "$MODE" == "multivariate" ]; then
    MODE_SHORT="mv"
fi
OUTPUT_DIR="outputs/ds3m/${COUNTRY}_TARGET_${MODE_SHORT}_d${D_DIM}_seed${SEED}_${TIMESTAMP}"

# Run the experiment
python3 ds3m_electricity.py \
    --country ${COUNTRY} \
    --mode ${MODE} \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --train \
    --output_dir ${OUTPUT_DIR}

echo "=========================================="
echo "Experiment completed"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
