#!/bin/bash
# HTCondor wrapper script for Target-Constrained experiments
# Arguments: market d_dim seed
#
# This script runs training with the target constraint enabled, which
# encourages edges TO the Price node to be learned.
#
# Example: ./run_target_constrained_job.sh DE 2 42

set -e

# Parse arguments
MARKET=$1
D_DIM=${2:-2}
SEED=${3:-42}

# Target constraint parameters
LAMBDA_TARGET=10.0
LAMBDA_SPARSE=0.001
MAX_AUGLAG_STEPS=200
EARLY_STOPPING_PATIENCE=25

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
echo "Target-Constrained Experiment Job"
echo "======================================================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Parameters: market=$MARKET, d_dim=$D_DIM, seed=$SEED"
echo "Target constraint: lambda_target=$LAMBDA_TARGET, lambda_sparse=$LAMBDA_SPARSE"
echo "Training: max_auglag_steps=$MAX_AUGLAG_STEPS, patience=$EARLY_STOPPING_PATIENCE"
echo "======================================================================"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU detected (running on CPU)"
echo ""

# Output directory
OUTPUT_DIR="${PROJECT_DIR}/results/${MARKET}/target_constrained_d${D_DIM}_seed${SEED}"

# Build and run experiment command
CMD="python3 -u ${PROJECT_DIR}/run_shared_backbone.py \
    --market ${MARKET} \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --sharing_mode shared_backbone \
    --lambda_target ${LAMBDA_TARGET} \
    --lambda_sparse ${LAMBDA_SPARSE} \
    --max_auglag_steps ${MAX_AUGLAG_STEPS} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
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
