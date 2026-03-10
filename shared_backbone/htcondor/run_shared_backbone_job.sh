#!/bin/bash
# HTCondor wrapper script for Shared Backbone experiments
# Arguments: market d_dim seed sharing_mode
#
# Example: ./run_shared_backbone_job.sh DE 2 42 shared_backbone

set -e

# Parse arguments
MARKET=$1
D_DIM=$2
SEED=$3
SHARING_MODE=${4:-shared_backbone}

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
echo "Shared Backbone Experiment Job"
echo "======================================================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Parameters: market=$MARKET, d_dim=$D_DIM, seed=$SEED, sharing_mode=$SHARING_MODE"
echo "======================================================================"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU detected (running on CPU)"
echo ""

# Output directory
OUTPUT_DIR="${PROJECT_DIR}/results/${MARKET}/${SHARING_MODE}_d${D_DIM}_seed${SEED}"

# Build and run experiment command
CMD="python3 -u ${PROJECT_DIR}/run_shared_backbone.py \
    --market ${MARKET} \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --sharing_mode ${SHARING_MODE} \
    --max_auglag_steps 50 \
    --max_inner_epochs 50 \
    --output_dir ${OUTPUT_DIR}"

echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?

# Log completion
echo "======================================================================"
echo "Job completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================================================"

exit $EXIT_CODE
