#!/bin/bash
# Run DS3M-FANTOM Hybrid with TUNED hyperparameters (Option A)

set -e

COUNTRY=$1
D_DIM=$2
SEED=$3

echo "=========================================="
echo "Option A: Hybrid with Tuned Hyperparameters"
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

# Run with tuned config
python3 ds3m_fantom/run_experiments.py \
    --country ${COUNTRY} \
    --data_source unified \
    --sharing_mode independent \
    --training_mode end_to_end \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --config ds3m_fantom/config/ds3m_causal_tuned.yaml \
    --train

echo "=========================================="
echo "Option A experiment completed"
echo "=========================================="
