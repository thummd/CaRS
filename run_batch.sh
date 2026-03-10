#!/bin/bash
# =============================================================================
# CaRS Batch Experiment Runner for Shared GPU Server (AIDF)
#
# Runs multiple experiments from a parameter file, one at a time on a given GPU.
# For parallel runs across GPUs, launch multiple instances with different --gpu.
#
# Usage:
#   ./run_batch.sh <params_file> <script.py> [--gpu N]
#
# Parameter file format (CSV, one experiment per line):
#   # Comments start with #
#   arg1,arg2,arg3
#
# Examples:
#   ./run_batch.sh electricity/params/multiseed.txt electricity/ds3m_electricity.py
#   ./run_batch.sh electricity/params/multiseed.txt electricity/ds3m_electricity.py --gpu 1
#
# For parallel execution across 2 GPUs, use two terminals:
#   Terminal 1: ./run_batch.sh params_gpu0.txt script.py --gpu 0
#   Terminal 2: ./run_batch.sh params_gpu1.txt script.py --gpu 1
# =============================================================================

set -e

PARAMS_FILE=""
SCRIPT=""
GPU_ID=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            if [ -z "$PARAMS_FILE" ]; then
                PARAMS_FILE="$1"
            elif [ -z "$SCRIPT" ]; then
                SCRIPT="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$PARAMS_FILE" ] || [ -z "$SCRIPT" ]; then
    echo "Usage: ./run_batch.sh <params_file> <script.py> [--gpu N]"
    exit 1
fi

export CARS_ROOT="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES=$GPU_ID

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "${CARS_ROOT}/logs"

# Count total jobs
TOTAL=$(grep -v '^#' "$PARAMS_FILE" | grep -v '^$' | wc -l)
CURRENT=0

echo "=== CaRS Batch Runner ==="
echo "Params:  $PARAMS_FILE ($TOTAL jobs)"
echo "Script:  $SCRIPT"
echo "GPU:     $GPU_ID"
echo "==========================="

while IFS=',' read -r line || [ -n "$line" ]; do
    # Skip comments and empty lines
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue

    CURRENT=$((CURRENT + 1))
    ARGS=$(echo "$line" | tr ',' ' ')
    LOG_FILE="${CARS_ROOT}/logs/batch_${TIMESTAMP}_job${CURRENT}.log"

    echo ""
    echo "[$CURRENT/$TOTAL] Running: python3 $SCRIPT $ARGS"
    echo "  Log: $LOG_FILE"

    python3 "${CARS_ROOT}/${SCRIPT}" $ARGS > "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  Status: COMPLETED"
    else
        echo "  Status: FAILED (exit code $EXIT_CODE)"
        echo "  Check: tail -50 $LOG_FILE"
    fi
done < "$PARAMS_FILE"

echo ""
echo "=== Batch complete: $CURRENT/$TOTAL jobs processed ==="
