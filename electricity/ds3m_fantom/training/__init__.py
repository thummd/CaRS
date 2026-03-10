"""
DS3M-FANTOM Training Scripts

Contains training utilities for the hybrid model:
- train_e2e: End-to-end training with joint optimization
- train_2stage: Two-stage training (regime detection then causal)
"""

from .train_e2e import train_end_to_end, AugmentedLagrangianTrainer

__all__ = ['train_end_to_end', 'AugmentedLagrangianTrainer']
