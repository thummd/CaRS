#!/bin/bash
# HTCondor wrapper script for Ablation experiments
# Arguments: market d_dim seed experiment_type
#
# Experiment types:
# - independent: Independent DAGs per regime (no sharing)
# - high_lambda: Higher lambda_regime_diff (10.0)
# - noise_init: Noise initialization for regime deviations (0.5)
# - combined: High lambda + noise init
#
# Example: ./run_ablation_job.sh DE 2 42 independent

set -e

# Parse arguments
MARKET=$1
D_DIM=${2:-2}
SEED=${3:-42}
EXPERIMENT_TYPE=${4:-independent}

# Project directory
PROJECT_DIR=/lustre/home/dthumm/FANTOM/shared_backbone

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH
export PATH=/lustre/home/dthumm/.local/bin:$PATH
export TMPDIR=/lustre/home/dthumm/tmp
export PIP_CACHE_DIR=/lustre/home/dthumm/.cache/pip

mkdir -p $TMPDIR
cd $PROJECT_DIR

# Set experiment-specific parameters
case $EXPERIMENT_TYPE in
    independent)
        SHARING_MODE="independent"
        LAMBDA_REGIME_DIFF=0.0
        REGIME_NOISE_STD=0.0
        ;;
    high_lambda)
        SHARING_MODE="shared_backbone"
        LAMBDA_REGIME_DIFF=10.0
        REGIME_NOISE_STD=0.0
        ;;
    noise_init)
        SHARING_MODE="shared_backbone"
        LAMBDA_REGIME_DIFF=1.0
        REGIME_NOISE_STD=0.5
        ;;
    combined)
        SHARING_MODE="shared_backbone"
        LAMBDA_REGIME_DIFF=10.0
        REGIME_NOISE_STD=0.5
        ;;
    *)
        echo "Unknown experiment type: $EXPERIMENT_TYPE"
        exit 1
        ;;
esac

# Common parameters (from improved experiments)
LAMBDA_TARGET=10.0
LAMBDA_SPARSE=0.0001
MAX_AUGLAG_STEPS=200
EARLY_STOPPING_PATIENCE=30
EARLY_STOPPING_METRIC=directional_accuracy
TAU_INIT=1.0
TAU_FINAL=0.1
TAU_ANNEAL_STEPS=100

# Log job info
echo "======================================================================"
echo "Ablation Experiment: $EXPERIMENT_TYPE"
echo "======================================================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Parameters: market=$MARKET, d_dim=$D_DIM, seed=$SEED"
echo ""
echo "Experiment config:"
echo "  sharing_mode=$SHARING_MODE"
echo "  lambda_regime_diff=$LAMBDA_REGIME_DIFF"
echo "  regime_noise_std=$REGIME_NOISE_STD"
echo ""
echo "Other parameters:"
echo "  lambda_target=$LAMBDA_TARGET"
echo "  lambda_sparse=$LAMBDA_SPARSE"
echo "  tau: $TAU_INIT -> $TAU_FINAL over $TAU_ANNEAL_STEPS steps"
echo "======================================================================"

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU detected"
echo ""

# Output directory
OUTPUT_DIR="${PROJECT_DIR}/results/${MARKET}/${EXPERIMENT_TYPE}_d${D_DIM}_seed${SEED}"

# Build command
CMD="python3 -u ${PROJECT_DIR}/run_shared_backbone.py \
    --market ${MARKET} \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --sharing_mode ${SHARING_MODE} \
    --lambda_target ${LAMBDA_TARGET} \
    --lambda_sparse ${LAMBDA_SPARSE} \
    --lambda_regime_diff ${LAMBDA_REGIME_DIFF} \
    --regime_noise_std ${REGIME_NOISE_STD} \
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

echo "======================================================================"
echo "Job completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================================================"

exit $EXIT_CODE
