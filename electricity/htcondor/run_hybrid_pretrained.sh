#!/bin/bash
# Run DS3M-FANTOM Hybrid with PRE-TRAINED DS3M regimes (Option B variant)

set -e

COUNTRY=$1
D_DIM=$2
SEED=$3

echo "=========================================="
echo "Option B: Pre-trained DS3M Regimes"
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

# Run pre-trained hybrid
python3 test_hybrid_pretrained.py \
    --country ${COUNTRY} \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --device cuda \
    --feature_groups price load weather calendar

echo "=========================================="
echo "Option B (pretrained) experiment completed"
echo "=========================================="
