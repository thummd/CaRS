#!/bin/bash
source ~/.bashrc
conda activate castor
cd /lustre/home/dthumm/CASTOR/electricity
python run_multiseed_experiments.py --model ds3m_causal --dataset FR --d_dim 2 --n_seeds 1 --task_type estimation --n_epochs 30
