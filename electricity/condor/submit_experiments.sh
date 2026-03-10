#!/bin/bash
# Submit DS3M-Causal 80GB experiments and FANTOM_BEM verification
#
# Usage: ./submit_experiments.sh [--ds3m-only | --fantom-only | --all]
#
# Created: 2026-01-27
# Purpose:
#   1. DS3M-Causal d=4 experiments (require 80GB GPU)
#   2. FANTOM_BEM verification (after double-normalization fix)

set -e

cd /lustre/home/dthumm/CASTOR/electricity/condor

# Create logs directory if it doesn't exist
mkdir -p logs

# Make run script executable
chmod +x run_multiseed.sh

echo "=============================================="
echo "Submitting Electricity Experiments"
echo "Timestamp: $(date)"
echo "=============================================="

MODE=${1:-"--all"}

if [[ "$MODE" == "--ds3m-only" ]] || [[ "$MODE" == "--all" ]]; then
    echo ""
    echo "Submitting DS3M-Causal d=4 experiments (80GB GPU required)..."
    echo "  - DE d=4 (prediction, estimation)"
    echo "  - FR d=4 (prediction, estimation)"
    echo "  - DE_FR d=4 (prediction, estimation)"
    echo "  - DE_FR d=3 (prediction, estimation)"
    echo ""
    condor_submit_bid 100 ds3m_causal_80gb.sub
    echo "  DS3M-Causal jobs submitted (8 jobs)"
fi

if [[ "$MODE" == "--fantom-only" ]] || [[ "$MODE" == "--all" ]]; then
    echo ""
    echo "Submitting FANTOM_BEM verification experiments..."
    echo "  Testing calibration fix (removed double-normalization)"
    echo "  - DE d=2,3,4 (prediction, estimation)"
    echo "  - FR d=2,3,4 (prediction, estimation)"
    echo "  - DE_FR d=2,3,4 (prediction, estimation)"
    echo ""
    condor_submit_bid 100 fantom_bem_verify.sub
    echo "  FANTOM_BEM jobs submitted (18 jobs)"
fi

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor jobs with:"
echo "  condor_q                    # View your jobs"
echo "  condor_q -analyze <job_id>  # Analyze job status"
echo ""
echo "View logs with:"
echo "  tail -f logs/ds3m_causal_DE_d4_prediction.out"
echo "  tail -f logs/fantom_bem_DE_d2_prediction_verify.out"
echo ""
echo "Results will be saved to:"
echo "  /lustre/home/dthumm/CASTOR/electricity/outputs/multiseed/"
echo ""
echo "Expected verification:"
echo "  - DS3M-Causal d=4: Should complete without OOM"
echo "  - FANTOM_BEM: RMSE std should be much lower (was varying 65x)"
echo ""
