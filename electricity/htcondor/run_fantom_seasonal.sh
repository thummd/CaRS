#!/bin/bash
# Runner script for FANTOM seasonal experiments

N_REGIMES=$1

cd /lustre/home/dthumm/CASTOR/electricity
source /lustre/home/dthumm/.bashrc

echo "=========================================="
echo "FANTOM Seasonal Experiment"
echo "Country: ALL"
echo "N_Regimes: ${N_REGIMES}"
echo "=========================================="

python3 fantom_regime_seasonal.py --country ALL --n_regimes ${N_REGIMES} --init_mode seasonal --seed 42

echo "=========================================="
echo "Experiment completed"
echo "=========================================="
