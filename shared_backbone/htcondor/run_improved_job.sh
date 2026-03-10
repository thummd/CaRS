#!/bin/bash
# HTCondor wrapper script for Improved Training experiments
# Arguments: market d_dim seed
#
# Improvements over target_constrained:
# 1. Temperature annealing (tau=1.0 -> 0.1) for sparse binary edges
# 2. Regime differentiation constraint for distinct DAGs per regime
# 3. Early stopping based on directional accuracy
#
# Example: ./run_improved_job.sh DE 2 42

set -e

# Parse arguments
MARKET=$1
D_DIM=${2:-2}
SEED=${3:-42}

# Improved training parameters
LAMBDA_TARGET=10.0
LAMBDA_SPARSE=0.0001
LAMBDA_REGIME_DIFF=1.0
MAX_AUGLAG_STEPS=200
EARLY_STOPPING_PATIENCE=30
EARLY_STOPPING_METRIC=directional_accuracy
TAU_INIT=1.0
TAU_FINAL=0.1
TAU_ANNEAL_STEPS=100

# Project directory
PROJECT_DIR=/lustre/home/dthumm/FANTOM/shared_backbone

# Set up environment (matching CASTOR cluster setup)
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH
export PATH=/lustre/home/dthumm/.local/bin:$PATH
export TMPDIR=/lustre/home/dthumm/tmp
export PIP_CACHE_DIR=/lustre/home/dthumm/.cache/pip

# Ensure temp directory exists
mkdir -p $TMPDIR

# Navigate to project directory
cd $PROJECT_DIR

# Log job info
echo "======================================================================"
echo "Improved Training Experiment Job"
echo "======================================================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Parameters: market=$MARKET, d_dim=$D_DIM, seed=$SEED"
echo ""
echo "Loss weights:"
echo "  lambda_target=$LAMBDA_TARGET"
echo "  lambda_sparse=$LAMBDA_SPARSE"
echo "  lambda_regime_diff=$LAMBDA_REGIME_DIFF"
echo ""
echo "Temperature annealing:"
echo "  tau: $TAU_INIT -> $TAU_FINAL over $TAU_ANNEAL_STEPS steps"
echo ""
echo "Training:"
echo "  max_auglag_steps=$MAX_AUGLAG_STEPS"
echo "  early_stopping_patience=$EARLY_STOPPING_PATIENCE"
echo "  early_stopping_metric=$EARLY_STOPPING_METRIC"
echo "======================================================================"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU detected (running on CPU)"
echo ""

# Output directory
OUTPUT_DIR="${PROJECT_DIR}/results/${MARKET}/improved_d${D_DIM}_seed${SEED}"

# Build and run experiment command
CMD="python3 -u ${PROJECT_DIR}/run_shared_backbone.py \
    --market ${MARKET} \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --sharing_mode shared_backbone \
    --lambda_target ${LAMBDA_TARGET} \
    --lambda_sparse ${LAMBDA_SPARSE} \
    --lambda_regime_diff ${LAMBDA_REGIME_DIFF} \
    --max_auglag_steps ${MAX_AUGLAG_STEPS} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --early_stopping_metric ${EARLY_STOPPING_METRIC} \
    --tau_init ${TAU_INIT} \
    --tau_final ${TAU_FINAL} \
    --tau_anneal_steps ${TAU_ANNEAL_STEPS} \
    --output_dir ${OUTPUT_DIR}"

echo "Running: $CMD"
echo ""
eval $CMD

EXIT_CODE=$?

# Log completion
echo "======================================================================"
echo "Job completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================================================"

exit $EXIT_CODE
