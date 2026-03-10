#!/bin/bash
# Run a single DS3M electricity experiment

set -e

COUNTRY=$1
TARGET=$2
SEED=$3

echo "=========================================="
echo "Running DS3M electricity experiment"
echo "Country: $COUNTRY"
echo "Target: $TARGET"
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
OUTPUT_DIR="outputs/ds3m/${COUNTRY}_${TARGET//\//_}_seed${SEED}_${TIMESTAMP}"

# Run the experiment
python3 ds3m_electricity.py \
    --country ${COUNTRY} \
    --target ${TARGET} \
    --seed ${SEED} \
    --train \
    --output_dir ${OUTPUT_DIR}

echo "=========================================="
echo "Experiment completed"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
