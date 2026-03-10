"""
DS3M-FANTOM Hybrid: Combining Regime Switching with Causal Discovery

This package implements hybrid models that combine:
- DS3M's regime switching (d_t) and continuous latent dynamics (z_t)
- FANTOM's causal graph discovery via augmented Lagrangian optimization

Two approaches are implemented:
1. DS3M-Causal: Replace DS3M's emission networks with FANTOM's causal SEM
2. FANTOM-Dynamics: Augment FANTOM with DS3M's latent dynamics as features

Both approaches support:
- End-to-end training (joint optimization)
- Two-stage training (regime detection first, then causal discovery)
- Independent or shared-backbone DAG per regime
"""

__version__ = "0.1.0"
