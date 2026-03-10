#!/bin/bash
#
# Job wrapper script for FANTOM_BEM with returns target
# Usage: run_fantom_bem_returns.sh <market> <d_dim> <seed> <task_type>
#

set -e

# Arguments
MARKET=$1
D_DIM=$2
SEED=$3
TASK_TYPE=$4  # 'estimation' or 'prediction'

echo "======================================================================"
echo "FANTOM_BEM with Returns Target Experiment Job"
echo "======================================================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Parameters: market=$MARKET, d_dim=$D_DIM, seed=$SEED, task_type=$TASK_TYPE"
echo "======================================================================"

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Set up environment (matching CASTOR cluster setup)
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH
export PATH=/lustre/home/dthumm/.local/bin:$PATH
export TMPDIR=/lustre/home/dthumm/tmp
export PIP_CACHE_DIR=/lustre/home/dthumm/.cache/pip

# Ensure temp directory exists
mkdir -p $TMPDIR

# Set paths
FANTOM_DIR="/lustre/home/dthumm/FANTOM"
SCRIPT_DIR="$FANTOM_DIR/presentation/scripts"
OUTPUT_DIR="$FANTOM_DIR/presentation/results/cars"

# Determine lag based on task type
# For estimation (concurrent), lag=1 is sufficient
# For prediction (forecasting), lag=1 is also used for comparison
LAG=1

# Run the experiment with returns mode
echo ""
echo "Running: python3 -u $SCRIPT_DIR/run_cars_experiment.py \\"
echo "    --model fantom \\"
echo "    --regime_method bem \\"
echo "    --dataset $MARKET \\"
echo "    --n_regimes $D_DIM \\"
echo "    --lag $LAG \\"
echo "    --lambda_sparse 50.0 \\"
echo "    --seed $SEED \\"
echo "    --use_returns \\"
echo "    --output_dir $OUTPUT_DIR"
echo ""

python3 -u "$SCRIPT_DIR/run_cars_experiment.py" \
    --model fantom \
    --regime_method bem \
    --dataset "$MARKET" \
    --n_regimes "$D_DIM" \
    --lag "$LAG" \
    --lambda_sparse 50.0 \
    --seed "$SEED" \
    --use_returns \
    --output_dir "$OUTPUT_DIR"

echo "======================================================================"
echo "Job completed: $(date)"
echo "Exit code: $?"
echo "======================================================================"
