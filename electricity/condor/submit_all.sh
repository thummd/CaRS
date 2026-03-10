#!/bin/bash
# Master script to submit all FANTOM electricity experiments to HTCondor
#
# Usage:
#   ./submit_all.sh              # Submit all experiments
#   ./submit_all.sh hyperparam   # Submit only hyperparameter search
#   ./submit_all.sh regime       # Submit only regime detection
#   ./submit_all.sh training     # Submit only full training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=============================================="
echo "FANTOM Electricity - HTCondor Job Submission"
echo "=============================================="
echo "Time: $(date)"
echo "Directory: ${SCRIPT_DIR}"
echo ""

submit_hyperparam() {
    echo "Submitting hyperparameter search jobs..."
    condor_submit_bid 100 hyperparam_search.sub
    echo "  -> DE and FR hyperparameter search submitted"
}

submit_regime() {
    echo "Submitting regime detection jobs..."
    condor_submit_bid 100 regime_detection.sub
    echo "  -> 4 regime detection jobs submitted (DE/FR × 2/3 regimes)"
}

submit_training() {
    echo "Submitting full training jobs..."
    condor_submit_bid 100 full_training.sub
    echo "  -> 3 training jobs submitted (germany, france, joint)"
}

case "${1:-all}" in
    hyperparam)
        submit_hyperparam
        ;;
    regime)
        submit_regime
        ;;
    training)
        submit_training
        ;;
    all)
        submit_hyperparam
        echo ""
        submit_regime
        echo ""
        submit_training
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 [hyperparam|regime|training|all]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Jobs submitted! Check status with:"
echo "  condor_q"
echo ""
echo "View logs in:"
echo "  ${SCRIPT_DIR}/logs/"
echo "=============================================="
