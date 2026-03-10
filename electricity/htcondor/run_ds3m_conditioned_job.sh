#!/bin/bash
# Run a single conditioned DS3M experiment

set -e

COUNTRY=$1
D_DIM=$2
SEED=$3

echo "=========================================="
echo "Running CONDITIONED DS3M experiment"
echo "Country: $COUNTRY"
echo "d_dim: $D_DIM"
echo "Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Create output directory name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/ds3m_conditioned/${COUNTRY}_d${D_DIM}_seed${SEED}_${TIMESTAMP}"

# Run the experiment
python3 ds3m_conditioned.py \
    --country ${COUNTRY} \
    --mode multivariate \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --train \
    --output_dir ${OUTPUT_DIR}

echo "=========================================="
echo "Experiment completed"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
