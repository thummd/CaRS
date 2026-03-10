#!/bin/bash
# Run DS3M-FANTOM Hybrid with REDUCED COMPLEXITY (Option C)

set -e

COUNTRY=$1
D_DIM=$2
SEED=$3

echo "=========================================="
echo "Option C: Reduced Complexity Model"
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

# Run with lite config (shared backbone)
python3 ds3m_fantom/run_experiments.py \
    --country ${COUNTRY} \
    --data_source unified \
    --sharing_mode shared_backbone \
    --training_mode end_to_end \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --config ds3m_fantom/config/ds3m_causal_lite.yaml \
    --train

echo "=========================================="
echo "Option C experiment completed"
echo "=========================================="
