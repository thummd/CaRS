#!/bin/bash
# Submit all fixed electricity experiments to HTCondor
#
# This script submits:
# 1. Hyperparameter search jobs
# 2. Regime detection jobs with stability fixes
#
# Usage: ./submit_fixed.sh

set -e

cd /lustre/home/dthumm/CASTOR/electricity/condor

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=============================================="
echo "Submitting FANTOM Electricity Experiments"
echo "Timestamp: $(date)"
echo "=============================================="

# Submit hyperparameter search jobs
echo ""
echo "Submitting hyperparameter search jobs..."
condor_submit_bid 100 hyperparam_search.sub
echo "  Hyperparameter search jobs submitted"

# Submit regime detection jobs
echo ""
echo "Submitting regime detection jobs..."
condor_submit_bid 100 regime_detection.sub
echo "  Regime detection jobs submitted"

echo ""
echo "=============================================="
echo "All jobs submitted successfully!"
echo "=============================================="
echo ""
echo "Monitor jobs with:"
echo "  condor_q                    # View your jobs"
echo "  condor_q -analyze <job_id>  # Analyze job status"
echo ""
echo "View logs with:"
echo "  tail -f logs/regime_DE_2.out"
echo "  tail -f logs/regime_FR_3.out"
echo ""
echo "Results will be saved to:"
echo "  /lustre/home/dthumm/CASTOR/electricity/regime_results/"
echo "  /lustre/home/dthumm/CASTOR/electricity/hyperparam_results/"
echo ""
