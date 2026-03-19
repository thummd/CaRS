"""
DS3M-Causal: Hybrid Model combining DS3M with FANTOM Causal Discovery

This model replaces DS3M's black-box emission networks with FANTOM's
causal SEM (ICGNN), enabling interpretable causal structure discovery
per market regime.

Architecture:
    Input X_t → GRU Encoder → h_t (hidden state)
                    ↓
            d_t posterior → Regime assignment (Markov)
                    ↓
            z_t posterior → Continuous latent (per regime)
                    ↓
        ┌───────────────────────────────────────┐
        │  FANTOM Causal Emission (per regime)  │
        │  - Regime-specific DAG: A^(d)         │
        │  - ICGNN prediction                   │
        └───────────────────────────────────────┘
                    ↓
            Y_t prediction (TARGET)
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
from pathlib import Path

# Add DS3M to path via central paths.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from paths import DS3M_DIR

DS3M_PATH = DS3M_DIR
if str(DS3M_PATH) not in sys.path:
    sys.path.insert(0, str(DS3M_PATH))
    sys.path.insert(0, str(DS3M_PATH / "src"))

# Import DS3M components
try:
    from DSSSMCode import DSSSM
except ImportError:
    DSSSM = None
    print("Warning: DS3M not found. Install from Deep-Switching-State-Space-Model repo.")

# Import our modules - use absolute imports for direct script execution
try:
    from modules.causal_emission import CausalEmission
    from modules.shared_dag import SharedRegimeDAG
except ImportError:
    # Fallback to relative imports when used as package
    from ..modules.causal_emission import CausalEmission
    from ..modules.shared_dag import SharedRegimeDAG


class DS3MCausal(nn.Module):
    """
    DS3M with FANTOM Causal Emission per Regime.

    Combines:
    - DS3M's discrete regime switching (d_t via Markov transition)
    - DS3M's continuous latent dynamics (z_t)
    - DS3M's GRU encoder for temporal patterns
    - FANTOM's causal graph discovery (A^(d) per regime)
    - FANTOM's ICGNN-based structural equations

    Key features:
    - Learns interpretable causal structure per market regime
    - Supports shared backbone + regime-specific edges
    - Full probabilistic inference over regimes AND graphs
    - Augmented Lagrangian for DAG constraint enforcement
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        h_dim: int,
        z_dim: int,
        d_dim: int,
        device: torch.device,
        n_layers: int = 1,
        bidirection: bool = False,
        # FANTOM causal parameters
        num_nodes: int = None,
        lag: int = 1,
        sharing_mode: str = "independent",
        tau_gumbel: float = 1.0,
        init_logits: Optional[List[float]] = None,
        # Loss weights
        lambda_dag: float = 100.0,
        lambda_sparse: float = 0.01,  # Light sparsity to allow edge learning
        lambda_kl: float = 1.0,
        lambda_recon: float = 1.0,  # Reconstruction loss for all nodes (prevents W gradient starvation)
        # Target index
        target_idx: int = 0,  # Target variable index (Price = index 0)
        # Regime differentiation
        regime_noise_std: float = 0.0,  # Noise for regime deviation initialization
    ):
        """
        Initialize DS3M-Causal hybrid model.

        Args:
            x_dim: Input dimension (number of features)
            y_dim: Output dimension (target, typically 1)
            h_dim: GRU hidden dimension
            z_dim: Continuous latent dimension
            d_dim: Number of discrete regimes
            device: Torch device
            n_layers: Number of GRU layers
            bidirection: Use bidirectional GRU
            num_nodes: Number of nodes for causal graph (default: x_dim)
            lag: Temporal lag for causal structure
            sharing_mode: "independent" or "shared_backbone"
            tau_gumbel: Gumbel-Softmax temperature
            init_logits: Initial logits for graph sparsity
            lambda_dag: Weight for DAG constraint penalty
            lambda_sparse: Weight for sparsity penalty
            lambda_kl: Weight for KL divergence
        """
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.d_dim = d_dim
        self.device = device
        self.n_layers = n_layers
        self.bidirection = bidirection

        # Causal parameters
        self.num_nodes = num_nodes or x_dim
        self.lag = lag
        self.sharing_mode = sharing_mode
        self.lambda_dag = lambda_dag
        self.lambda_sparse = lambda_sparse
        self.lambda_kl = lambda_kl
        self.lambda_recon = lambda_recon
        self.target_idx = target_idx

        # ======== DS3M Encoder Components ========
        # GRU encoder (forward and backward)
        self.rnn_forward = nn.GRU(
            input_size=x_dim + y_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            batch_first=False
        ).to(device)

        self.rnn_backward = nn.GRU(
            input_size=x_dim + y_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            batch_first=False
        ).to(device)

        # Bidirectional hidden dim
        h_dim_enc = h_dim * 2 if bidirection else h_dim

        # ======== DS3M Discrete State (d_t) ========
        # Transition matrix: P(d_t | d_{t-1})
        self.d_transition = nn.Parameter(
            torch.ones(d_dim, d_dim, device=device) / d_dim
        )

        # d_t posterior networks (one per regime)
        self.d_posterior_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim_enc + y_dim, 64),
                nn.ReLU(),
                nn.Linear(64, d_dim)
            ).to(device)
            for _ in range(d_dim)
        ])

        # ======== DS3M Continuous Latent (z_t) ========
        # z_t prior networks (one per regime)
        self.z_prior_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z_dim, 64),
                nn.ReLU(),
                nn.Linear(64, z_dim * 2)  # mean and log_var
            ).to(device)
            for _ in range(d_dim)
        ])

        # z_t posterior networks (one per regime)
        self.z_posterior_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim_enc + y_dim + z_dim, 64),
                nn.ReLU(),
                nn.Linear(64, z_dim * 2)  # mean and log_var
            ).to(device)
            for _ in range(d_dim)
        ])

        # ======== FANTOM Causal Components ========
        # Shared/regime-specific DAG distribution
        self.dag_dist = SharedRegimeDAG(
            device=device,
            num_nodes=self.num_nodes,
            lag=lag,
            d_dim=d_dim,
            sharing_mode=sharing_mode,
            tau_gumbel=tau_gumbel,
            init_logits=init_logits or [1.0, 1.0],  # Sparsity bias: no-edge logit=1.0 → ~42% inst / ~73% lagged no-edge
            regime_noise_std=regime_noise_std,  # Noise for regime differentiation
        )

        # Causal emission networks (one per regime)
        self.causal_emissions = nn.ModuleList([
            CausalEmission(
                num_nodes=self.num_nodes,
                device=device,
                target_idx=self.target_idx,  # Target variable (Price = index 0)
                lag=lag,
                h_dim=h_dim,
                z_dim=z_dim,
                embedding_size=32,
                encoder_layer_sizes=[64, 64],
                decoder_layer_sizes=[64, 64],
                norm_layers=True,
                heteroscedastic=True,
            )
            for _ in range(d_dim)
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequences using bidirectional GRU.

        Args:
            x: Input features [timestep, batch, x_dim]
            y: Target values [timestep, batch, y_dim]

        Returns:
            h_forward: Forward hidden states [timestep, batch, h_dim]
            h_backward: Backward hidden states [timestep, batch, h_dim]
        """
        # Concatenate input and target
        xy = torch.cat([x, y], dim=-1)  # [timestep, batch, x_dim + y_dim]

        # Forward pass
        h_forward, _ = self.rnn_forward(xy)  # [timestep, batch, h_dim]

        # Backward pass
        xy_reversed = torch.flip(xy, dims=[0])
        h_backward_rev, _ = self.rnn_backward(xy_reversed)
        h_backward = torch.flip(h_backward_rev, dims=[0])

        return h_forward, h_backward

    def get_d_posterior(
        self,
        h_t: torch.Tensor,
        y_t: torch.Tensor,
        d_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute posterior over discrete state d_t.

        Args:
            h_t: Hidden state [batch, h_dim]
            y_t: Target value [batch, y_dim]
            d_prev: Previous discrete state distribution [batch, d_dim]

        Returns:
            d_posterior: [batch, d_dim]
        """
        batch_size = h_t.shape[0]
        hy = torch.cat([h_t, y_t], dim=-1)  # [batch, h_dim + y_dim]

        # Compute emission probabilities from each regime
        logits = torch.zeros(batch_size, self.d_dim, device=self.device)
        for d in range(self.d_dim):
            logits[:, d] = self.d_posterior_nets[d](hy)[:, d]

        # Incorporate transition prior
        trans_prob = F.softmax(self.d_transition, dim=-1)  # [d_dim, d_dim]
        prior = torch.matmul(d_prev, trans_prob)  # [batch, d_dim]

        # Posterior: prior * likelihood
        posterior = F.softmax(logits + torch.log(prior + 1e-10), dim=-1)

        return posterior

    def get_z_posterior(
        self,
        h_t: torch.Tensor,
        y_t: torch.Tensor,
        z_prev: torch.Tensor,
        d_t: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior over continuous latent z_t for a specific regime.

        Args:
            h_t: Hidden state [batch, h_dim]
            y_t: Target value [batch, y_dim]
            z_prev: Previous latent state [batch, z_dim]
            d_t: Discrete regime index

        Returns:
            mean: [batch, z_dim]
            log_var: [batch, z_dim]
        """
        hyz = torch.cat([h_t, y_t, z_prev], dim=-1)
        output = self.z_posterior_nets[d_t](hyz)
        mean, log_var = output.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=-10, max=2)  # Numerical stability
        return mean, log_var

    def get_z_prior(
        self,
        z_prev: torch.Tensor,
        d_t: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prior over continuous latent z_t for a specific regime.

        Args:
            z_prev: Previous latent state [batch, z_dim]
            d_t: Discrete regime index

        Returns:
            mean: [batch, z_dim]
            log_var: [batch, z_dim]
        """
        output = self.z_prior_nets[d_t](z_prev)
        mean, log_var = output.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=-10, max=2)
        return mean, log_var

    def reparameterize(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick for sampling z."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass computing ELBO terms.

        Args:
            x: Input features [timestep, batch, x_dim]
            y: Target values [timestep, batch, y_dim]
            return_details: Return detailed outputs for analysis

        Returns:
            Dictionary containing:
                - nll: Negative log-likelihood
                - kl_z: KL divergence for continuous latent
                - kl_d: KL divergence for discrete state
                - dag_penalty: DAG constraint violation
                - sparse_penalty: Sparsity penalty
                - predictions: Predicted means [timestep, batch, y_dim]
                - regimes: Regime assignments [timestep, batch]
        """
        timestep, batch_size, _ = x.shape

        # Encode sequences
        h_forward, h_backward = self.encode(x, y)

        if self.bidirection:
            h = torch.cat([h_forward, h_backward], dim=-1)
        else:
            h = h_forward

        # Initialize tracking variables
        total_nll = torch.tensor(0.0, device=self.device)
        total_kl_z = torch.tensor(0.0, device=self.device)
        total_kl_d = torch.tensor(0.0, device=self.device)
        total_dag_penalty = torch.tensor(0.0, device=self.device)
        total_recon = torch.tensor(0.0, device=self.device)

        predictions = []
        pred_stds = []
        regime_posteriors = []
        z_samples = []

        # Initialize d_prev as uniform
        d_prev = torch.ones(batch_size, self.d_dim, device=self.device) / self.d_dim

        # Initialize z_prev as zeros
        z_prev = torch.zeros(batch_size, self.z_dim, device=self.device)

        # Process each timestep
        for t in range(timestep):
            h_t = h[t]  # [batch, h_dim]
            y_t = y[t]  # [batch, y_dim]
            x_t = x[t]  # [batch, x_dim]

            # Get discrete state posterior
            d_posterior = self.get_d_posterior(h_t, y_t, d_prev)
            regime_posteriors.append(d_posterior)

            # KL for discrete state
            trans_prob = F.softmax(self.d_transition, dim=-1)
            d_prior = torch.matmul(d_prev, trans_prob)
            kl_d_t = (d_posterior * (torch.log(d_posterior + 1e-10) -
                                     torch.log(d_prior + 1e-10))).sum(dim=-1)
            total_kl_d += kl_d_t.mean()

            # Compute per-regime contributions
            y_pred_mixture = torch.zeros(batch_size, self.y_dim, device=self.device)
            y_std_mixture = torch.zeros(batch_size, self.y_dim, device=self.device)

            for d in range(self.d_dim):
                # Get z posterior and prior
                z_post_mean, z_post_logvar = self.get_z_posterior(h_t, y_t, z_prev, d)
                z_prior_mean, z_prior_logvar = self.get_z_prior(z_prev, d)

                # Sample z
                z_t = self.reparameterize(z_post_mean, z_post_logvar)

                # KL for continuous latent
                kl_z_d = -0.5 * torch.sum(
                    1 + z_post_logvar - z_prior_logvar -
                    (z_post_logvar.exp() + (z_post_mean - z_prior_mean).pow(2)) /
                    z_prior_logvar.exp(),
                    dim=-1
                )

                # Get DAG for this regime with hierarchical penalty
                # This uses get_dag_with_penalty() which applies DAG constraint
                # to each component separately for shared_backbone mode
                A_d, dag_penalty_d = self.dag_dist.get_dag_with_penalty(d, sample=self.training)

                # Compute causal emission
                # Need to prepare input for causal emission with lag history
                # x has shape [timestep, batch, x_dim], we need [batch, lag+1, x_dim]
                # Gather x from timestep (t - lag) to t (inclusive)
                start_idx = max(0, t - self.lag)
                end_idx = t + 1
                x_history = x[start_idx:end_idx]  # [lag+1 or less, batch, x_dim]
                x_history = x_history.permute(1, 0, 2)  # [batch, lag+1 or less, x_dim]

                # Pad if we don't have enough history (at early timesteps)
                if x_history.shape[1] < self.lag + 1:
                    pad_size = self.lag + 1 - x_history.shape[1]
                    padding = x_history[:, :1, :].expand(-1, pad_size, -1)  # Repeat first timestep
                    x_history = torch.cat([padding, x_history], dim=1)

                y_pred_d, y_std_d, all_preds_d = self.causal_emissions[d](
                    x_history, A_d, h_t, z_t,
                    edge_mask=getattr(self, '_edge_mask', None)
                )

                # Weight by regime probability
                d_prob = d_posterior[:, d:d+1]  # [batch, 1]
                y_pred_mixture += d_prob * y_pred_d
                y_std_mixture += d_prob * y_std_d
                total_kl_z += (d_prob.squeeze() * kl_z_d).mean()

                # Reconstruction loss: MSE over all nodes to give gradients to all W columns
                recon_loss_d = F.mse_loss(all_preds_d, x_t, reduction='none').sum(dim=-1)  # [batch]
                total_recon += (d_prob.squeeze() * recon_loss_d).mean()

                # Accumulate DAG penalty (hierarchical for shared_backbone)
                total_dag_penalty += dag_penalty_d

            predictions.append(y_pred_mixture)
            pred_stds.append(y_std_mixture)

            # Compute NLL
            dist = td.Normal(y_pred_mixture, y_std_mixture + 1e-6)
            nll_t = -dist.log_prob(y_t).sum(dim=-1)
            total_nll += nll_t.mean()

            # Sample dominant regime for tracking
            z_samples.append(z_t)

            # Update for next timestep
            d_prev = d_posterior.detach()
            z_prev = z_t.detach()

        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [timestep, batch, y_dim]
        pred_stds = torch.stack(pred_stds, dim=0)
        regime_posteriors = torch.stack(regime_posteriors, dim=0)  # [timestep, batch, d_dim]

        # Sparsity penalty (L1 on edge probabilities) - following FANTOM convention
        sparse_penalty = self.dag_dist.sparsity_penalty()

        # Compute total edge probability for monitoring
        total_edge_prob = torch.tensor(0.0, device=self.device)
        for d in range(self.d_dim):
            A_d = self.dag_dist.get_dag(d, sample=False)
            total_edge_prob = total_edge_prob + A_d.sum()
        avg_edges_per_regime = total_edge_prob / self.d_dim

        # Compute total loss (matching FANTOM structure)
        # Note: sparse_penalty encourages sparsity via L1 norm on edge probabilities
        # recon loss ensures all W columns get gradients (prevents star/sink topology)
        loss = (
            total_nll +
            self.lambda_kl * (total_kl_z + total_kl_d) +
            self.lambda_dag * total_dag_penalty +
            self.lambda_sparse * sparse_penalty +
            self.lambda_recon * total_recon
        )

        result = {
            'loss': loss,
            'nll': total_nll,
            'kl_z': total_kl_z,
            'kl_d': total_kl_d,
            'dag_penalty': total_dag_penalty,
            'sparse_penalty': sparse_penalty,
            'recon_loss': total_recon,
            'avg_edges_per_regime': avg_edges_per_regime,
            'predictions': predictions,
            'pred_stds': pred_stds,
            'regime_posteriors': regime_posteriors,
            # Return adjacency matrices for target constraint in training
            'adj_matrices': [self.dag_dist.get_dag(d, sample=False) for d in range(self.d_dim)],
        }

        if return_details:
            result['regime_assignments'] = regime_posteriors.argmax(dim=-1)
            result['dags'] = [self.dag_dist.get_dag(d, sample=False) for d in range(self.d_dim)]

        return result

    def predict(
        self,
        x: torch.Tensor,
        y_context: Optional[torch.Tensor] = None,
        n_samples: int = 100,
        edge_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions (inference mode).

        Args:
            x: Input features [timestep, batch, x_dim]
            y_context: Known target values for conditioning (optional)
            n_samples: Number of Monte Carlo samples
            edge_mask: Optional binary mask [lag+1, num_nodes, num_nodes] to
                zero out specific edges for ablation. 1 = keep, 0 = ablate.

        Returns:
            predictions: Mean predictions [timestep, batch, y_dim]
            predictions_std: Standard deviation [timestep, batch, y_dim]
            regimes: Most likely regime [timestep, batch]
        """
        self.eval()
        timestep, batch_size, _ = x.shape

        # Temporarily store edge mask for forward pass
        self._edge_mask = edge_mask

        with torch.no_grad():
            all_preds = []

            for _ in range(n_samples):
                # Use zeros for y if not provided
                if y_context is None:
                    y_dummy = torch.zeros(timestep, batch_size, self.y_dim, device=self.device)
                else:
                    y_dummy = y_context

                result = self.forward(x, y_dummy, return_details=False)
                all_preds.append(result['predictions'])

            all_preds = torch.stack(all_preds, dim=0)  # [n_samples, timestep, batch, y_dim]

            predictions_mean = all_preds.mean(dim=0)
            predictions_std = all_preds.std(dim=0)

            # Get regime assignments from last forward pass
            regimes = result['regime_posteriors'].argmax(dim=-1)

        self._edge_mask = None  # Clear after use

        return {
            'predictions': predictions_mean,
            'predictions_std': predictions_std,
            'regimes': regimes
        }

    def get_causal_graphs(self) -> List[np.ndarray]:
        """
        Get learned causal graphs for all regimes.

        Returns:
            List of adjacency matrices [lag+1, num_nodes, num_nodes]
        """
        with torch.no_grad():
            graphs = []
            for d in range(self.d_dim):
                A = self.dag_dist.get_dag(d, sample=False)
                graphs.append(A.cpu().numpy())
        return graphs

    def get_weighted_causal_graphs(self) -> List[np.ndarray]:
        """
        Get learned weighted causal graphs (W * A) for all regimes.

        Returns the actual causal coefficients used in the SEM predictions,
        computed as the element-wise product of the ICGNN weight matrix W
        and the structural adjacency mask A.

        Returns:
            List of weighted adjacency matrices [lag+1, num_nodes, num_nodes]
        """
        with torch.no_grad():
            graphs = []
            for d in range(self.d_dim):
                A = self.dag_dist.get_dag(d, sample=False)
                W = self.causal_emissions[d].icgnn.get_weighted_adjacency()
                weighted_A = (W * A).cpu().numpy()
                graphs.append(weighted_A)
        return graphs

    def get_regime_assignments(self, X: np.ndarray) -> np.ndarray:
        """
        Get regime assignments for each timestep using native Markov inference.

        Args:
            X: Input data of shape (T, lag+1, n_features) where last feature is target

        Returns:
            assignments: Array of shape (T,) with regime indices
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

            # X has shape (T, lag+1, n_features)
            # We need to extract x (features) and y (target)
            T = X_tensor.shape[0]

            # Transpose to (lag+1, T, n_features) for sequential processing
            X_seq = X_tensor.permute(1, 0, 2)  # [lag+1, T, n_features]

            # All features as x, target (Price) extracted as y
            x = X_seq  # [lag+1, T, x_dim]
            y = X_seq[:, :, self.target_idx:self.target_idx+1]  # [lag+1, T, 1]

            # Encode sequences
            h_fwd, h_bwd = self.encode(x, y)

            if self.bidirection:
                h = torch.cat([h_fwd, h_bwd], dim=-1)
            else:
                h = h_fwd

            # Get d_t posterior for each timestep
            d_prev = torch.ones(T, self.d_dim, device=self.device) / self.d_dim
            assignments = []

            for t in range(X_seq.shape[0]):  # Iterate over lag+1 timesteps
                h_t = h[t]  # [T, h_dim]
                y_t = y[t]  # [T, 1]

                d_post = self.get_d_posterior(h_t, y_t, d_prev)
                assignments.append(d_post.argmax(dim=-1).cpu().numpy())
                d_prev = d_post

            # Return assignments for the last timestep (current time)
            # Since we process lag+1 timesteps, the last one is the "current" prediction
            return assignments[-1]

    def get_regime_posteriors(self, X: np.ndarray) -> np.ndarray:
        """
        Get soft regime posterior probabilities for each timestep.

        Args:
            X: Input data of shape (T, lag+1, n_features)

        Returns:
            posteriors: Array of shape (T, d_dim) with regime probabilities
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

            T = X_tensor.shape[0]
            X_seq = X_tensor.permute(1, 0, 2)
            x = X_seq  # All features
            y = X_seq[:, :, self.target_idx:self.target_idx+1]  # Price

            h_fwd, h_bwd = self.encode(x, y)

            if self.bidirection:
                h = torch.cat([h_fwd, h_bwd], dim=-1)
            else:
                h = h_fwd

            d_prev = torch.ones(T, self.d_dim, device=self.device) / self.d_dim
            posteriors = []

            for t in range(X_seq.shape[0]):
                h_t = h[t]
                y_t = y[t]

                d_post = self.get_d_posterior(h_t, y_t, d_prev)
                posteriors.append(d_post.cpu().numpy())
                d_prev = d_post

            return posteriors[-1]


if __name__ == "__main__":
    # Test the DS3M-Causal model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Model parameters (matching QRT data)
    x_dim = 32  # Number of features
    y_dim = 1   # TARGET
    h_dim = 32
    z_dim = 8
    d_dim = 2   # Number of regimes
    timestep = 14
    batch_size = 16

    # Create model
    model = DS3MCausal(
        x_dim=x_dim,
        y_dim=y_dim,
        h_dim=h_dim,
        z_dim=z_dim,
        d_dim=d_dim,
        device=device,
        num_nodes=x_dim,
        lag=1,
        sharing_mode="independent",
        lambda_dag=100.0,
        lambda_sparse=1.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # Create test data
    x = torch.randn(timestep, batch_size, x_dim, device=device)
    y = torch.randn(timestep, batch_size, y_dim, device=device)

    # Forward pass
    print("\n--- Forward Pass ---")
    model.train()
    result = model(x, y, return_details=True)

    print(f"Loss: {result['loss'].item():.4f}")
    print(f"  NLL: {result['nll'].item():.4f}")
    print(f"  KL(z): {result['kl_z'].item():.4f}")
    print(f"  KL(d): {result['kl_d'].item():.4f}")
    print(f"  DAG penalty: {result['dag_penalty'].item():.4f}")
    print(f"  Sparsity: {result['sparse_penalty'].item():.4f}")
    print(f"Predictions shape: {result['predictions'].shape}")
    print(f"Regime posteriors shape: {result['regime_posteriors'].shape}")

    # Test prediction mode
    print("\n--- Prediction Mode ---")
    pred_result = model.predict(x, n_samples=10)
    print(f"Predictions shape: {pred_result['predictions'].shape}")
    print(f"Predictions std shape: {pred_result['predictions_std'].shape}")
    print(f"Regimes shape: {pred_result['regimes'].shape}")

    # Get causal graphs
    print("\n--- Causal Graphs ---")
    graphs = model.get_causal_graphs()
    for d, g in enumerate(graphs):
        print(f"Regime {d}: shape={g.shape}, edges={g.sum():.2f}")

    # Test gradient flow
    print("\n--- Gradient Test ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(3):
        optimizer.zero_grad()
        result = model(x, y)
        result['loss'].backward()
        optimizer.step()
        print(f"Step {step}: loss={result['loss'].item():.4f}")

    print("\nDS3M-Causal model test passed!")
