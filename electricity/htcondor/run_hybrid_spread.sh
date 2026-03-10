#!/bin/bash
# Run DS3MCausal (Hybrid) on DE-FR spread prediction
# Options: tuned, 2stage, lite

set -e

OPTION=$1
D_DIM=$2
SEED=$3

echo "=========================================="
echo "Running Hybrid Spread Prediction Experiment"
echo "Option: $OPTION"
echo "D_dim (regimes): $D_DIM"
echo "Seed: $SEED"
echo "=========================================="

cd /lustre/home/dthumm/CASTOR/electricity

# Set up environment
export PYTHONUSERBASE=/lustre/home/dthumm/.local
export PYTHONPATH=/lustre/home/dthumm/.local/lib/python3.10/site-packages:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Choose config based on option
case $OPTION in
    "tuned")
        CONFIG="ds3m_fantom/config/ds3m_causal_tuned.yaml"
        TRAINING_MODE="end_to_end"
        ;;
    "2stage")
        CONFIG="ds3m_fantom/config/ds3m_causal.yaml"
        TRAINING_MODE="two_stage"
        ;;
    "lite")
        CONFIG="ds3m_fantom/config/ds3m_causal_lite.yaml"
        TRAINING_MODE="end_to_end"
        ;;
    *)
        echo "Unknown option: $OPTION"
        exit 1
        ;;
esac

echo "Config: $CONFIG"
echo "Training mode: $TRAINING_MODE"

# Run the hybrid spread prediction
python3 ds3m_fantom/run_experiments.py \
    --country DE_FR \
    --data_source unified \
    --sharing_mode independent \
    --training_mode $TRAINING_MODE \
    --d_dim ${D_DIM} \
    --seed ${SEED} \
    --config $CONFIG \
    --train

echo "=========================================="
echo "Hybrid spread experiment completed"
echo "=========================================="
