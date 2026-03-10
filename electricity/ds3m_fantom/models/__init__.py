"""
DS3M-FANTOM Hybrid Models

Contains the main model architectures:
- DS3MCausal: DS3M with FANTOM causal emission per regime
"""

from .ds3m_causal import DS3MCausal

__all__ = ['DS3MCausal']
