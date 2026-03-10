"""
DS3M-FANTOM Module Components

Contains building blocks for the hybrid model:
- causal_emission: FANTOM SEM adapted as DS3M emission network
- shared_dag: Variational distribution for regime-specific DAGs
"""

from .causal_emission import CausalEmission
from .shared_dag import SharedRegimeDAG, VarDistA_Temporal

__all__ = ['CausalEmission', 'SharedRegimeDAG', 'VarDistA_Temporal']
