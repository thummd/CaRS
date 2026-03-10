"""
Regime Regularization to Prevent Collapse

This module provides regularization techniques to prevent regime collapse
in DS3M and hybrid models.

The key issues with regime collapse:
1. KL divergence to uniform prior is too weak
2. Model finds it easier to use single regime
3. Entropy of regime posteriors drops to near-zero

Solutions implemented:
1. Entropy regularization: Encourage diverse regime assignments
2. Minimum regime usage: Penalize if any regime has < threshold samples
3. Temporal smoothness: Regimes should be temporally coherent
4. Annealing: Gradually increase regime diversity pressure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


def regime_entropy_loss(
    regime_posteriors: torch.Tensor,
    target_entropy: float = None
) -> torch.Tensor:
    """
    Compute entropy regularization loss for regime posteriors.

    Encourages regime assignments to be diverse (high entropy) rather than
    collapsing to a single regime.

    Args:
        regime_posteriors: [timestep, batch, d_dim] regime posterior probabilities
        target_entropy: Target entropy (default: max entropy = log(d_dim))

    Returns:
        Entropy loss (negative if we want to maximize entropy)
    """
    d_dim = regime_posteriors.shape[-1]

    # Average regime probabilities across time and batch
    avg_probs = regime_posteriors.mean(dim=(0, 1))  # [d_dim]

    # Entropy of average distribution (should be high for diverse regimes)
    entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))

    # Maximum entropy for d_dim categories
    max_entropy = np.log(d_dim)
    target = target_entropy if target_entropy else max_entropy

    # Loss: penalize low entropy (encourage diversity)
    loss = target - entropy

    return loss


def minimum_regime_usage_loss(
    regime_posteriors: torch.Tensor,
    min_usage_ratio: float = 0.1
) -> torch.Tensor:
    """
    Penalize regimes that are used less than a minimum threshold.

    Args:
        regime_posteriors: [timestep, batch, d_dim]
        min_usage_ratio: Minimum fraction of samples per regime

    Returns:
        Penalty for underused regimes
    """
    d_dim = regime_posteriors.shape[-1]
    total_samples = regime_posteriors.shape[0] * regime_posteriors.shape[1]

    # Expected usage per regime
    avg_probs = regime_posteriors.mean(dim=(0, 1))  # [d_dim]
    expected_per_regime = avg_probs * total_samples

    # Minimum samples per regime
    min_samples = min_usage_ratio * total_samples / d_dim

    # Soft penalty for regimes below minimum
    # Using ReLU to only penalize when below threshold
    underuse_penalty = F.relu(min_samples - expected_per_regime).sum()

    return underuse_penalty


def temporal_smoothness_loss(
    regime_posteriors: torch.Tensor,
    smoothness_weight: float = 1.0
) -> torch.Tensor:
    """
    Encourage temporal smoothness in regime assignments.

    Regimes should be sticky - frequent switching is penalized.

    Args:
        regime_posteriors: [timestep, batch, d_dim]
        smoothness_weight: Weight for smoothness penalty

    Returns:
        Smoothness loss
    """
    # Difference between consecutive timesteps
    diff = regime_posteriors[1:] - regime_posteriors[:-1]  # [T-1, B, d]

    # L2 norm of differences
    smoothness_loss = (diff ** 2).sum(dim=-1).mean()

    return smoothness_weight * smoothness_loss


def balanced_kl_loss(
    regime_posteriors: torch.Tensor,
    prior_type: str = 'uniform'
) -> torch.Tensor:
    """
    KL divergence to a balanced prior with stronger enforcement.

    Args:
        regime_posteriors: [timestep, batch, d_dim]
        prior_type: 'uniform' or 'temporal' (time-varying)

    Returns:
        KL divergence loss
    """
    d_dim = regime_posteriors.shape[-1]

    # Average regime probabilities
    avg_probs = regime_posteriors.mean(dim=(0, 1))  # [d_dim]

    if prior_type == 'uniform':
        # Uniform prior: each regime equally likely
        prior = torch.ones(d_dim, device=avg_probs.device) / d_dim
    else:
        prior = torch.ones(d_dim, device=avg_probs.device) / d_dim

    # KL(avg_probs || prior)
    kl = F.kl_div(
        torch.log(avg_probs + 1e-10),
        prior,
        reduction='sum'
    )

    return kl


class RegimeRegularizer(nn.Module):
    """
    Combined regime regularization module.

    Applies multiple regularization techniques to prevent regime collapse.
    """

    def __init__(
        self,
        d_dim: int,
        entropy_weight: float = 1.0,
        min_usage_weight: float = 0.5,
        smoothness_weight: float = 0.1,
        kl_weight: float = 1.0,
        min_usage_ratio: float = 0.1,
        annealing_start: int = 0,
        annealing_end: int = 100
    ):
        """
        Args:
            d_dim: Number of regimes
            entropy_weight: Weight for entropy regularization
            min_usage_weight: Weight for minimum usage penalty
            smoothness_weight: Weight for temporal smoothness
            kl_weight: Weight for balanced KL
            min_usage_ratio: Minimum fraction per regime
            annealing_start: Epoch to start annealing
            annealing_end: Epoch to reach full regularization
        """
        super().__init__()
        self.d_dim = d_dim
        self.entropy_weight = entropy_weight
        self.min_usage_weight = min_usage_weight
        self.smoothness_weight = smoothness_weight
        self.kl_weight = kl_weight
        self.min_usage_ratio = min_usage_ratio
        self.annealing_start = annealing_start
        self.annealing_end = annealing_end

        self.current_epoch = 0

    def get_annealing_factor(self) -> float:
        """Get current annealing factor (0 to 1)."""
        if self.current_epoch < self.annealing_start:
            return 0.0
        elif self.current_epoch >= self.annealing_end:
            return 1.0
        else:
            progress = (self.current_epoch - self.annealing_start) / (self.annealing_end - self.annealing_start)
            return progress

    def forward(
        self,
        regime_posteriors: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined regularization loss.

        Args:
            regime_posteriors: [timestep, batch, d_dim]
            return_components: If True, return individual loss components

        Returns:
            Total regularization loss (and optionally components dict)
        """
        anneal = self.get_annealing_factor()

        # Compute individual losses
        entropy_loss = regime_entropy_loss(regime_posteriors)
        min_usage_loss = minimum_regime_usage_loss(regime_posteriors, self.min_usage_ratio)
        smoothness_loss = temporal_smoothness_loss(regime_posteriors, 1.0)
        kl_loss = balanced_kl_loss(regime_posteriors)

        # Weighted sum with annealing
        total_loss = anneal * (
            self.entropy_weight * entropy_loss +
            self.min_usage_weight * min_usage_loss +
            self.smoothness_weight * smoothness_loss +
            self.kl_weight * kl_loss
        )

        if return_components:
            components = {
                'entropy_loss': entropy_loss.item(),
                'min_usage_loss': min_usage_loss.item(),
                'smoothness_loss': smoothness_loss.item(),
                'kl_loss': kl_loss.item(),
                'anneal_factor': anneal
            }
            return total_loss, components

        return total_loss

    def step(self):
        """Increment epoch counter for annealing."""
        self.current_epoch += 1


def diagnose_regime_collapse(
    regime_posteriors: torch.Tensor,
    threshold: float = 0.9
) -> Dict:
    """
    Diagnose whether regime collapse has occurred.

    Args:
        regime_posteriors: [timestep, batch, d_dim]
        threshold: If one regime has > threshold probability, it's collapsed

    Returns:
        Diagnostic information
    """
    d_dim = regime_posteriors.shape[-1]

    # Average probabilities
    avg_probs = regime_posteriors.mean(dim=(0, 1)).cpu().numpy()

    # Check for collapse
    max_prob = avg_probs.max()
    dominant_regime = avg_probs.argmax()
    collapsed = max_prob > threshold

    # Entropy
    entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-10))
    max_entropy = np.log(d_dim)
    relative_entropy = entropy / max_entropy

    # Per-regime statistics
    hard_assignments = regime_posteriors.argmax(dim=-1)  # [T, B]
    regime_counts = {}
    for r in range(d_dim):
        regime_counts[r] = int((hard_assignments == r).sum().item())

    return {
        'collapsed': collapsed,
        'dominant_regime': int(dominant_regime),
        'max_prob': float(max_prob),
        'avg_probs': avg_probs.tolist(),
        'entropy': float(entropy),
        'relative_entropy': float(relative_entropy),
        'regime_counts': regime_counts,
        'effective_regimes': sum(1 for p in avg_probs if p > 0.05)
    }


def redistribute_collapsed_regimes(
    regime_posteriors: torch.Tensor,
    data: torch.Tensor,
    method: str = 'kmeans'
) -> torch.Tensor:
    """
    Redistribute samples when regime collapse is detected.

    This is a recovery mechanism when training fails to learn diverse regimes.

    Args:
        regime_posteriors: [timestep, batch, d_dim]
        data: Input data [timestep, batch, features]
        method: 'kmeans' or 'temporal' redistribution

    Returns:
        New regime posteriors with balanced assignments
    """
    from sklearn.cluster import KMeans

    d_dim = regime_posteriors.shape[-1]
    T, B, _ = regime_posteriors.shape

    if method == 'kmeans':
        # Use feature clustering to assign regimes
        # Flatten temporal dimension for clustering
        data_flat = data.reshape(T * B, -1).cpu().numpy()

        kmeans = KMeans(n_clusters=d_dim, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_flat)
        labels = labels.reshape(T, B)

        # Convert to one-hot posteriors
        new_posteriors = torch.zeros_like(regime_posteriors)
        for t in range(T):
            for b in range(B):
                new_posteriors[t, b, labels[t, b]] = 1.0

    elif method == 'temporal':
        # Assign regimes based on temporal position
        segment_size = T // d_dim

        new_posteriors = torch.zeros_like(regime_posteriors)
        for r in range(d_dim):
            start = r * segment_size
            end = (r + 1) * segment_size if r < d_dim - 1 else T
            new_posteriors[start:end, :, r] = 1.0

    return new_posteriors
