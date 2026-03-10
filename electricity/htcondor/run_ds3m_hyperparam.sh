#!/bin/bash
# Run DS3M hyperparameter sweep

set -e

COUNTRY=$1
D_DIM=$2
H_DIM=$3
Z_DIM=$4
TIMESTEP=$5
LR=$6
SEED=$7

echo "=========================================="
echo "Running DS3M Hyperparameter Sweep"
echo "Country: $COUNTRY"
echo "D_dim: $D_DIM, H_dim: $H_DIM, Z_dim: $Z_DIM"
echo "Timestep: $TIMESTEP, LR: $LR, Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the test with hyperparameters
python3 test_ds3m_unified.py \
    --dataset ${COUNTRY} \
    --timestep ${TIMESTEP} \
    --d_dim ${D_DIM} \
    --h_dim ${H_DIM} \
    --z_dim ${Z_DIM} \
    --n_epochs 150 \
    --batch_size 64 \
    --lr ${LR} \
    --seed ${SEED} \
    --device cuda \
    --feature_groups price load weather calendar

echo "=========================================="
echo "Hyperparameter sweep job completed"
echo "=========================================="
