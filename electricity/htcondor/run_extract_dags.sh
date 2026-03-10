#!/bin/bash
# Extract per-regime DAGs from DS3M Causal model
# Usage: ./run_extract_dags.sh <dataset> <d_dim> <seed> [lambda_sparse] [init_logits] [tau_gumbel] [edge_threshold]

set -e

DATASET=$1
D_DIM=$2
SEED=$3
LAMBDA_SPARSE=${4:-10.0}      # Default sparsity penalty
INIT_LOGITS=${5:--0.5}        # Default init logits (more negative = sparser)
TAU_GUMBEL=${6:-1.0}          # Default Gumbel temperature (lower = sharper)
EDGE_THRESHOLD=${7:-0.3}      # Default edge threshold

echo "=========================================="
echo "Extracting Per-Regime DAGs"
echo "Dataset: $DATASET"
echo "D_dim (regimes): $D_DIM"
echo "Seed: $SEED"
echo "Lambda_sparse: $LAMBDA_SPARSE"
echo "Init_logits: $INIT_LOGITS"
echo "Tau_gumbel: $TAU_GUMBEL"
echo "Edge_threshold: $EDGE_THRESHOLD"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Run extraction
python3 extract_regime_dags.py \
    --dataset ${DATASET} \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --lambda_sparse ${LAMBDA_SPARSE} \
    --init_logits ${INIT_LOGITS} \
    --tau_gumbel ${TAU_GUMBEL} \
    --edge_threshold ${EDGE_THRESHOLD} \
    --device cuda

echo "=========================================="
echo "DAG extraction completed"
echo "=========================================="
