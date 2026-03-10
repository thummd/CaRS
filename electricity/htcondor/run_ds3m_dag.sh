#!/bin/bash
# Runner script for DS3M + Per-Regime DAG experiments

COUNTRY=$1
D_DIM=$2
SEED=$3

cd /lustre/home/dthumm/CASTOR/electricity
source /lustre/home/dthumm/.bashrc

echo "=========================================="
echo "DS3M + Per-Regime DAG Experiment"
echo "Country: ${COUNTRY}"
echo "d_dim: ${D_DIM}"
echo "Seed: ${SEED}"
echo "=========================================="

python3 ds3m_with_dag.py --country ${COUNTRY} --d_dim ${D_DIM} --seed ${SEED}

echo "=========================================="
echo "Experiment completed"
echo "=========================================="
