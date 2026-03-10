#!/bin/bash
# =============================================================================
# CaRS Experiment Runner for Shared GPU Server (AIDF)
#
# Usage:
#   ./run_experiment.sh <script.py> [args...]
#   ./run_experiment.sh --gpu 1 electricity/ds3m_electricity.py --country DE --train
#   ./run_experiment.sh --bg electricity/ds3m_electricity.py --country DE --train
#
# Options:
#   --gpu N     Use GPU N (default: 0). Check availability with: nvidia-smi
#   --bg        Run in background with nohup (logs to logs/<script>_<timestamp>.log)
#   --dry-run   Print the command without executing
# =============================================================================

set -e

# Default settings
GPU_ID=0
BACKGROUND=false
DRY_RUN=false
SCRIPT=""
SCRIPT_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --bg)
            BACKGROUND=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            if [ -z "$SCRIPT" ]; then
                SCRIPT="$1"
            else
                SCRIPT_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -z "$SCRIPT" ]; then
    echo "Usage: ./run_experiment.sh [--gpu N] [--bg] <script.py> [args...]"
    echo ""
    echo "Examples:"
    echo "  ./run_experiment.sh electricity/ds3m_electricity.py --country DE --train"
    echo "  ./run_experiment.sh --gpu 1 --bg electricity/ds3m_electricity.py --country FR --train"
    echo ""
    echo "Check GPU availability: nvidia-smi"
    exit 1
fi

# Set up environment
export CARS_ROOT="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES=$GPU_ID

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_NAME=$(basename "$SCRIPT" .py)
LOG_FILE="${CARS_ROOT}/logs/${SCRIPT_NAME}_${TIMESTAMP}.log"

# Ensure log directory exists
mkdir -p "${CARS_ROOT}/logs"

# Build command
CMD="python3 ${CARS_ROOT}/${SCRIPT} ${SCRIPT_ARGS[*]}"

echo "=== CaRS Experiment Runner ==="
echo "Script:  $SCRIPT"
echo "GPU:     $GPU_ID"
echo "Command: CUDA_VISIBLE_DEVICES=$GPU_ID $CMD"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would execute the above command"
    exit 0
fi

if [ "$BACKGROUND" = true ]; then
    echo "Log:     $LOG_FILE"
    echo "Running in background..."
    nohup $CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "PID:     $PID"
    echo "Monitor: tail -f $LOG_FILE"
    echo "Stop:    kill $PID"
else
    echo "Running in foreground (Ctrl+C to stop)..."
    echo "==========================="
    $CMD
fi
