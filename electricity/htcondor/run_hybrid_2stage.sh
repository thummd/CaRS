#!/bin/bash
# Run DS3M-FANTOM Hybrid with TWO-STAGE training (Option B)

set -e

COUNTRY=$1
D_DIM=$2
SEED=$3

echo "=========================================="
echo "Option B: Two-Stage Training"
echo "Country: $COUNTRY"
echo "D_dim (regimes): $D_DIM"
echo "Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run with two-stage training mode
python3 ds3m_fantom/run_experiments.py \
    --country ${COUNTRY} \
    --data_source unified \
    --sharing_mode independent \
    --training_mode two_stage \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --train

echo "=========================================="
echo "Option B experiment completed"
echo "=========================================="
