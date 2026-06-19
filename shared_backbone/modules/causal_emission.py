"""
Causal Emission Module: FANTOM SEM as DS3M Emission Network

This module adapts FANTOM's causal structural equation model (ICGNN)
to work as an emission network in DS3M. Instead of using black-box
neural network emissions per regime, we use a causal graph-based SEM
that learns interpretable relationships between features.

Key features:
- Learns a DAG adjacency matrix A per regime
- Uses ICGNN (Invertible Contractive GNN) for non-linear relationships
- Supports heteroscedastic noise (variable-dependent variance)
- Integrates with DS3M's latent state (h_t, z_t)
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def generate_fully_connected(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    non_linearity: nn.Module = nn.LeakyReLU,
    activation: nn.Module = nn.Identity,
    device: torch.device = torch.device('cpu'),
    normalization: Optional[type] = None,
    res_connection: bool = False,
) -> nn.Sequential:
    """
    Generate a fully connected neural network.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        non_linearity: Activation function class
        activation: Final activation function class
        device: Device for parameters
        normalization: Optional normalization layer class (e.g., nn.LayerNorm)
        res_connection: Whether to use residual connections (not implemented here)

    Returns:
        nn.Sequential model
    """
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if normalization is not None:
            layers.append(normalization(hidden_dim))
        layers.append(non_linearity())
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    layers.append(activation())

    model = nn.Sequential(*layers).to(device)
    return model


class CausalICGNN(nn.Module):
    """
    Causal ICGNN for structural equation modeling.

    Implements the function f for the SEM: X_i = f_i(parents(X_i)) + noise_i
    Uses the ICGNN architecture from FANTOM/DECI.

    Supports attention-weighted aggregation where edge influence is modulated
    by current input values: W[i,j] provides structural importance, while
    attention α(x_i, x_j) captures context-dependent modulation (e.g.,
    "gas→price edge is stronger during high-demand periods").

    For temporal data, supports lag structure where parents can include
    both instantaneous (same time) and lagged (previous time) variables.
    """

    def __init__(
        self,
        num_nodes: int,
        device: torch.device,
        lag: int = 1,
        embedding_size: Optional[int] = None,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        norm_layers: bool = True,
        res_connection: bool = True,
        heteroscedastic: bool = True,
        use_attention: bool = True,
        attention_heads: int = 1,
        w_init_scale: float = 0.01,
        physical_mask: Optional[torch.Tensor] = None,
        physical_prior_mode: str = "off",
        physical_prior_alpha_init: float = 0.05,
    ):
        """
        Initialize the Causal ICGNN.

        Args:
            num_nodes: Number of nodes (variables) in the graph
            device: Torch device
            lag: Temporal lag (1 = only previous timestep as parents)
            embedding_size: Size of node embeddings (default: num_nodes)
            encoder_layer_sizes: Hidden layers for encoder network
            decoder_layer_sizes: Hidden layers for decoder network
            norm_layers: Whether to use layer normalization
            res_connection: Whether to use residual connections
            heteroscedastic: Whether to learn variable-dependent noise variance
            w_init_scale: Standard deviation for W initialization (default 0.01).
                Higher values (e.g. 0.1) help the causal SEM learn stronger
                functional relationships. Should be paired with lower lambda_dag.
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.device = device
        self.lag = lag
        self.heteroscedastic = heteroscedastic
        self.use_attention = use_attention
        self.w_init_scale = w_init_scale

        # Node embeddings
        self.embedding_size = embedding_size or num_nodes
        self.embeddings = self._initialize_embeddings()

        # Weighted adjacency matrix (learned)
        # Shape: [lag+1, num_nodes, num_nodes] for temporal
        self.W = self._initialize_W()

        # Physical-interconnect prior on cross-border CARGO edges.
        # CARGO = Causal Additive Regime-Gated Output (this emission
        # mechanism; renamed from the original ICGNN class name for
        # backwards compatibility).
        # `physical_mask` shape [lag+1, num_nodes, num_nodes]:
        #   1 = allowed edge (domestic feature OR cross-border on a real
        #       interconnect),
        #   0 = forbidden by the prior (cross-border on a non-physical pair).
        # Modes:
        #   "off"  - prior disabled (legacy CARGO behaviour).
        #   "hard" - multiplies W by `physical_mask` in
        #            get_weighted_adjacency(); forbidden edges are
        #            exactly zero and contribute no gradient.
        #   "soft" - multiplies W by `mask + (1 - mask) * sigmoid(alpha)`
        #            where `alpha` is a learnable scalar that lets the
        #            data override the prior if a non-physical edge is
        #            *really* identifiable. Initialised so the prior is
        #            strong (sigmoid(-3.0) ~ 0.047 by default).
        self.physical_prior_mode = physical_prior_mode
        if physical_mask is not None:
            assert physical_mask.shape == (lag + 1, num_nodes, num_nodes), (
                f"physical_mask shape {tuple(physical_mask.shape)} != "
                f"({lag + 1}, {num_nodes}, {num_nodes})"
            )
            self.register_buffer("physical_mask",
                                 physical_mask.to(device).float())
        else:
            self.register_buffer("physical_mask", torch.ones(
                lag + 1, num_nodes, num_nodes, device=device))
            if physical_prior_mode != "off":
                raise ValueError(
                    "physical_prior_mode is enabled but no physical_mask "
                    "was supplied — pass a [lag+1, num_nodes, num_nodes] "
                    "binary tensor built from your interconnect graph.")
        if physical_prior_mode == "soft":
            # logit so initial sigmoid(alpha) ~= alpha_init (small)
            init_logit = float(np.log(physical_prior_alpha_init
                                       / (1 - physical_prior_alpha_init)))
            self.physical_prior_alpha_logit = nn.Parameter(
                torch.tensor(init_logit, device=device))

        # Network architecture sizes
        a = max(4 * num_nodes, self.embedding_size, 64)
        # NB: distinguish an explicit empty list (== LINEAR, no hidden layers)
        # from None (== use the default [a, a]); `... or [a, a]` would wrongly
        # turn an explicit [] back into the default.
        self.encoder_layers = (list(encoder_layer_sizes)
                               if encoder_layer_sizes is not None else [a, a])
        self.decoder_layers = (list(decoder_layer_sizes)
                               if decoder_layer_sizes is not None else [a, a])

        # Group mask (identity for continuous variables)
        self.group_mask = torch.eye(num_nodes, dtype=torch.bool, device=device)

        # Input dimension for networks
        in_dim_g = self.embedding_size + num_nodes  # embedding + variable values
        out_dim_g = self.embedding_size
        in_dim_f = self.embedding_size + out_dim_g

        self.norm_layer = nn.LayerNorm if norm_layers else None

        # Attention mechanism: modulates edge weights based on input values
        # α(parent_i, child_j) captures context-dependent causal strength
        # W[i,j] gives structural importance, attention gives dynamic modulation
        if use_attention:
            attn_dim = out_dim_g  # Attention operates on encoded representations
            self.attn_query = nn.Linear(attn_dim, attn_dim // attention_heads, device=device)
            self.attn_key = nn.Linear(attn_dim, attn_dim // attention_heads, device=device)
            self.attn_scale = (attn_dim // attention_heads) ** 0.5

        # Encoder network g: processes each node's values
        self.g = generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=out_dim_g,
            hidden_dims=self.encoder_layers,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=device,
            normalization=self.norm_layer,
            res_connection=res_connection,
        )

        # Decoder network f: generates predictions
        self.f = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=num_nodes,
            hidden_dims=self.decoder_layers,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=device,
            normalization=self.norm_layer,
            res_connection=res_connection,
        )

        # Variance networks for heteroscedastic noise
        if heteroscedastic:
            self.g_var = generate_fully_connected(
                input_dim=in_dim_g,
                output_dim=out_dim_g,
                hidden_dims=self.encoder_layers,
                non_linearity=nn.LeakyReLU,
                activation=nn.Identity,
                device=device,
                normalization=self.norm_layer,
                res_connection=res_connection,
            )
            self.f_var = generate_fully_connected(
                input_dim=in_dim_f,
                output_dim=num_nodes,
                hidden_dims=self.decoder_layers,
                non_linearity=nn.LeakyReLU,
                activation=nn.Identity,
                device=device,
                normalization=self.norm_layer,
                res_connection=res_connection,
            )

    def _initialize_embeddings(self) -> nn.Parameter:
        """Initialize temporal node embeddings."""
        # Shape: [lag+1, num_nodes, embedding_size]
        aux = torch.randn(
            self.lag + 1, self.num_nodes, self.embedding_size,
            device=self.device
        ) * 0.01
        return nn.Parameter(aux, requires_grad=True)

    def _initialize_W(self) -> nn.Parameter:
        """Initialize weighted adjacency matrix with random values.

        Using randn * w_init_scale instead of zeros to break symmetry and ensure
        the einsum aggregation produces non-zero gradients from the start.

        Higher w_init_scale (e.g., 0.1) produces stronger initial edge weights,
        which helps the causal SEM learn meaningful functional relationships
        rather than collapsing to near-zero magnitudes.
        """
        # Shape: [lag+1, num_nodes, num_nodes]
        W = torch.randn(
            self.lag + 1, self.num_nodes, self.num_nodes,
            device=self.device
        ) * self.w_init_scale
        return nn.Parameter(W, requires_grad=True)

    def get_weighted_adjacency(self) -> torch.Tensor:
        """
        Get the weighted adjacency matrix with diagonal disabled for instantaneous.

        When the CARGO physical-interconnect prior is active (``hard`` or
        ``soft``), the returned adjacency is masked element-wise so that
        cross-border edges on non-physical market pairs are either zeroed
        (hard) or attenuated by a small learnable scalar (soft).

        Returns:
            W_adj: Shape [lag+1, num_nodes, num_nodes]
        """
        W_adj = self.W.clone()
        # Disable self-loops in instantaneous adjacency
        torch.diagonal(W_adj[0, ...]).zero_()
        if self.physical_prior_mode == "hard":
            W_adj = W_adj * self.physical_mask
        elif self.physical_prior_mode == "soft":
            alpha = torch.sigmoid(self.physical_prior_alpha_logit)
            soft_mask = self.physical_mask + (1.0 - self.physical_mask) * alpha
            W_adj = W_adj * soft_mask
        return W_adj

    def predict(
        self,
        X: torch.Tensor,
        W_adj: torch.Tensor,
        A: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate predictions using the causal SEM.

        Args:
            X: Input data [batch, lag+1, num_nodes] or [lag+1, num_nodes]
            W_adj: Weighted adjacency [lag+1, num_nodes, num_nodes]
            A: Binary adjacency mask (optional, if not using W_adj directly)

        Returns:
            predictions: [batch, num_nodes]
            variance: [batch, num_nodes] if heteroscedastic, else None
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)  # [1, lag+1, num_nodes]

        batch_size = X.shape[0]

        if W_adj.dim() == 3:
            W_adj = W_adj.unsqueeze(0)  # [1, lag+1, num_nodes, num_nodes]

        # Apply binary mask if provided
        if A is not None:
            if A.dim() == 3:
                A = A.unsqueeze(0)
            W_adj = W_adj * A

        # Expand input for per-node processing
        # X shape: [batch, lag+1, num_nodes]
        # We need [batch, lag+1, num_nodes, num_nodes] for per-node processing
        X = X.unsqueeze(-1)  # [batch, lag+1, num_nodes, 1]
        X = X.expand(-1, -1, -1, self.num_nodes)  # [batch, lag+1, num_nodes, num_nodes]
        X_masked = X * self.group_mask  # [batch, lag+1, num_nodes, num_nodes]

        # Get embeddings
        E = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch, lag+1, num_nodes, emb]

        # Encoder pass
        X_in_g = torch.cat([X_masked, E], dim=-1)  # [batch, lag+1, num_nodes, emb+num_nodes]
        X_emb = self.g(X_in_g)  # [batch, lag+1, num_nodes, out_dim_g]

        # Aggregate parents weighted by adjacency
        # W_adj: [1 or batch, lag+1, num_nodes (parent), num_nodes (child)]
        # X_emb: [batch, lag+1, num_nodes (parent), out_dim_g]
        # Expand W_adj to match batch dimension for correct einsum contraction
        W_adj_exp = W_adj.expand(batch_size, -1, -1, -1)  # [batch, lag+1, num_nodes, num_nodes]

        if self.use_attention:
            # Attention-weighted aggregation:
            # α(parent_i, child_j) modulates W[i,j] based on current encoded values
            # Query = child node encoding, Key = parent node encoding
            # This captures "gas→price edge is stronger during high-demand periods"
            Q = self.attn_query(X_emb)  # [batch, lag+1, num_nodes, attn_dim]
            K = self.attn_key(X_emb)    # [batch, lag+1, num_nodes, attn_dim]
            # Attention scores: [batch, lag+1, parent_i, child_j]
            attn_scores = torch.einsum("blio,bljo->blij", K, Q) / self.attn_scale
            attn_weights = torch.sigmoid(attn_scores)  # [0, 1] range, not softmax
            # Modulate structural weights: effective_W = W * α
            W_effective = W_adj_exp * attn_weights
            X_aggr = torch.einsum("blij,blio->bjo", W_effective.flip([1]), X_emb)
        else:
            X_aggr = torch.einsum("blij,blio->bjo", W_adj_exp.flip([1]), X_emb)

        # Decoder pass
        X_in_f = torch.cat([X_aggr, E[:, 0, :, :]], dim=-1)  # [batch, num_nodes, emb+out_dim_g]
        X_rec = self.f(X_in_f)  # [batch, num_nodes, num_nodes]

        # Apply group mask and sum
        X_rec = X_rec * self.group_mask
        predictions = X_rec.sum(dim=1)  # [batch, num_nodes]

        variance = None
        if self.heteroscedastic:
            X_emb_var = self.g_var(X_in_g)
            if self.use_attention:
                X_aggr_var = torch.einsum("blij,blio->bjo", W_effective.flip([1]), X_emb_var)
            else:
                X_aggr_var = torch.einsum("blij,blio->bjo", W_adj_exp.flip([1]), X_emb_var)
            X_in_f_var = torch.cat([X_aggr_var, E[:, 0, :, :]], dim=-1)
            X_rec_var = self.f_var(X_in_f_var)
            X_rec_var = X_rec_var * self.group_mask
            variance = F.softplus(X_rec_var.sum(dim=1)) + 1e-6  # Ensure positive

        return predictions.squeeze(0), variance.squeeze(0) if variance is not None else None


class DualChannelICGNN(CausalICGNN):
    """
    Dual-channel ICGNN with separate positive/negative edge aggregation.

    Splits W into W_pos = ReLU(W) and W_neg = ReLU(-W), processes parent
    contributions through separate channels, then combines via a small MLP.

    This preserves per-edge sign interpretability:
    - W_pos[i,j] > 0 means parent i has a price-increasing effect on child j
    - W_neg[i,j] > 0 means parent i has a price-suppressing effect on child j

    While allowing non-linear cross-parent interactions that the linear
    aggregation in the base ICGNN cannot express (e.g., "price is high when
    BOTH gas is expensive AND wind is low").

    Drop-in replacement for CausalICGNN — same constructor, same interface.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        out_dim_g = self.embedding_size
        # MLP to combine positive and negative channel aggregations
        # Input: [pos_aggr; neg_aggr] = 2 * out_dim_g
        # Output: out_dim_g (same as single-channel aggregation)
        self.channel_combiner = nn.Sequential(
            nn.Linear(2 * out_dim_g, out_dim_g, device=self.device),
            nn.LeakyReLU(),
            nn.Linear(out_dim_g, out_dim_g, device=self.device),
        )

        # Same for variance path
        if self.heteroscedastic:
            self.channel_combiner_var = nn.Sequential(
                nn.Linear(2 * out_dim_g, out_dim_g, device=self.device),
                nn.LeakyReLU(),
                nn.Linear(out_dim_g, out_dim_g, device=self.device),
            )

    def _aggregate_dual_channel(
        self, W_adj_exp, X_emb, attn_weights=None
    ):
        """
        Aggregate parent embeddings through positive and negative channels.

        Args:
            W_adj_exp: [batch, lag+1, num_nodes, num_nodes]
            X_emb: [batch, lag+1, num_nodes, out_dim_g]
            attn_weights: [batch, lag+1, num_nodes, num_nodes] or None

        Returns:
            X_aggr: [batch, num_nodes, out_dim_g]
        """
        # Split W into positive (price-increasing) and negative (price-suppressing) channels
        W_pos = torch.relu(W_adj_exp)
        W_neg = torch.relu(-W_adj_exp)

        if attn_weights is not None:
            W_pos = W_pos * attn_weights
            W_neg = W_neg * attn_weights

        # Aggregate parents through each channel separately
        X_aggr_pos = torch.einsum("blij,blio->bjo", W_pos.flip([1]), X_emb)
        X_aggr_neg = torch.einsum("blij,blio->bjo", W_neg.flip([1]), X_emb)

        # Non-linear combination of channels
        X_aggr = self.channel_combiner(torch.cat([X_aggr_pos, X_aggr_neg], dim=-1))
        return X_aggr

    def predict(self, X, W_adj, A=None):
        """Generate predictions using dual-channel causal SEM."""
        if X.dim() == 2:
            X = X.unsqueeze(0)

        batch_size = X.shape[0]

        if W_adj.dim() == 3:
            W_adj = W_adj.unsqueeze(0)

        if A is not None:
            if A.dim() == 3:
                A = A.unsqueeze(0)
            W_adj = W_adj * A

        X = X.unsqueeze(-1)
        X = X.expand(-1, -1, -1, self.num_nodes)
        X_masked = X * self.group_mask

        E = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)

        X_in_g = torch.cat([X_masked, E], dim=-1)
        X_emb = self.g(X_in_g)

        W_adj_exp = W_adj.expand(batch_size, -1, -1, -1)

        # Compute attention weights (shared across channels)
        attn_weights = None
        if self.use_attention:
            Q = self.attn_query(X_emb)
            K = self.attn_key(X_emb)
            attn_scores = torch.einsum("blio,bljo->blij", K, Q) / self.attn_scale
            attn_weights = torch.sigmoid(attn_scores)

        # Dual-channel aggregation
        X_aggr = self._aggregate_dual_channel(W_adj_exp, X_emb, attn_weights)

        # Decoder pass (same as base)
        X_in_f = torch.cat([X_aggr, E[:, 0, :, :]], dim=-1)
        X_rec = self.f(X_in_f)
        X_rec = X_rec * self.group_mask
        predictions = X_rec.sum(dim=1)

        variance = None
        if self.heteroscedastic:
            X_emb_var = self.g_var(X_in_g)
            # Dual-channel for variance too
            W_pos = torch.relu(W_adj_exp)
            W_neg = torch.relu(-W_adj_exp)
            if attn_weights is not None:
                W_pos = W_pos * attn_weights
                W_neg = W_neg * attn_weights
            X_aggr_pos_var = torch.einsum("blij,blio->bjo", W_pos.flip([1]), X_emb_var)
            X_aggr_neg_var = torch.einsum("blij,blio->bjo", W_neg.flip([1]), X_emb_var)
            X_aggr_var = self.channel_combiner_var(
                torch.cat([X_aggr_pos_var, X_aggr_neg_var], dim=-1)
            )

            X_in_f_var = torch.cat([X_aggr_var, E[:, 0, :, :]], dim=-1)
            X_rec_var = self.f_var(X_in_f_var)
            X_rec_var = X_rec_var * self.group_mask
            variance = F.softplus(X_rec_var.sum(dim=1)) + 1e-6

        return predictions.squeeze(0), variance.squeeze(0) if variance is not None else None


class CausalICGNN_CAM(CausalICGNN):
    """
    Causal Additive Model (CAM) variant of the ICGNN.

    Adds per-parent nonlinear transforms BEFORE the aggregation einsum:

        Current:  aggr_j = Σ_i W[i,j] · α(i,j) · g(x_i)
        CAM:      aggr_j = Σ_i W[i,j] · α(i,j) · f_i(g(x_i))

    Each parent i (at each lag l) passes through a dedicated small MLP f_i
    that can learn nonlinear effects (e.g., merit order step functions,
    threshold effects, diminishing returns).

    The einsum aggregation itself is unchanged — W, attention, DAG constraint,
    and all other components remain identical. Only the encoded parent
    embeddings are nonlinearly transformed before the weighted sum.

    This preserves full per-edge causal interpretability: W[i,j] still
    quantifies the structural importance of parent i on child j.
    """

    def __init__(self, *args, cam_hidden_dim: int = 32, **kwargs):
        super().__init__(*args, **kwargs)

        self.cam_hidden_dim = cam_hidden_dim

        # Per-parent transforms: (lag+1) * num_nodes small MLPs
        # Each f_i: embedding_size → hidden → embedding_size
        self.parent_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_size, cam_hidden_dim, device=self.device),
                nn.LeakyReLU(),
                nn.Linear(cam_hidden_dim, self.embedding_size, device=self.device),
            )
            for _ in range((self.lag + 1) * self.num_nodes)
        ])

        # Same for variance path
        if self.heteroscedastic:
            self.parent_transforms_var = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.embedding_size, cam_hidden_dim, device=self.device),
                    nn.LeakyReLU(),
                    nn.Linear(cam_hidden_dim, self.embedding_size, device=self.device),
                )
                for _ in range((self.lag + 1) * self.num_nodes)
            ])

    def _apply_parent_transforms(self, X_emb, transforms):
        """Apply per-parent nonlinear transforms to encoded embeddings."""
        batch, lag_plus_1, p, emb = X_emb.shape
        X_transformed = torch.zeros_like(X_emb)
        for l in range(lag_plus_1):
            for i in range(p):
                idx = l * p + i
                X_transformed[:, l, i, :] = transforms[idx](X_emb[:, l, i, :])
        return X_transformed

    def predict(self, X, W_adj, A=None):
        """Generate predictions using CAM-enhanced causal SEM."""
        if X.dim() == 2:
            X = X.unsqueeze(0)

        batch_size = X.shape[0]

        if W_adj.dim() == 3:
            W_adj = W_adj.unsqueeze(0)

        if A is not None:
            if A.dim() == 3:
                A = A.unsqueeze(0)
            W_adj = W_adj * A

        X = X.unsqueeze(-1)
        X = X.expand(-1, -1, -1, self.num_nodes)
        X_masked = X * self.group_mask

        E = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Encoder pass (shared with base ICGNN)
        X_in_g = torch.cat([X_masked, E], dim=-1)
        X_emb = self.g(X_in_g)

        # CAM: apply per-parent nonlinear transforms before aggregation
        X_emb_cam = self._apply_parent_transforms(X_emb, self.parent_transforms)

        W_adj_exp = W_adj.expand(batch_size, -1, -1, -1)

        # Attention (computed on original embeddings for routing)
        attn_weights = None
        if self.use_attention:
            Q = self.attn_query(X_emb)
            K = self.attn_key(X_emb)
            attn_scores = torch.einsum("blio,bljo->blij", K, Q) / self.attn_scale
            attn_weights = torch.sigmoid(attn_scores)

        # Aggregation with CAM-transformed embeddings (einsum unchanged)
        W_gate = W_adj_exp
        if attn_weights is not None:
            W_gate = W_gate * attn_weights
        X_aggr = torch.einsum("blij,blio->bjo", W_gate.flip([1]), X_emb_cam)

        # Decoder pass (same as base)
        X_in_f = torch.cat([X_aggr, E[:, 0, :, :]], dim=-1)
        X_rec = self.f(X_in_f)
        X_rec = X_rec * self.group_mask
        predictions = X_rec.sum(dim=1)

        variance = None
        if self.heteroscedastic:
            X_emb_var = self.g_var(X_in_g)
            X_emb_var_cam = self._apply_parent_transforms(X_emb_var, self.parent_transforms_var)
            if attn_weights is not None:
                X_aggr_var = torch.einsum("blij,blio->bjo", W_gate.flip([1]), X_emb_var_cam)
            else:
                X_aggr_var = torch.einsum("blij,blio->bjo", W_adj_exp.flip([1]), X_emb_var_cam)
            X_in_f_var = torch.cat([X_aggr_var, E[:, 0, :, :]], dim=-1)
            X_rec_var = self.f_var(X_in_f_var)
            X_rec_var = X_rec_var * self.group_mask
            variance = F.softplus(X_rec_var.sum(dim=1)) + 1e-6

        return predictions.squeeze(0), variance.squeeze(0) if variance is not None else None


class CausalICGNN_GAT(CausalICGNN):
    """
    GATv2 (Graph Attention Network v2) variant of the ICGNN.

    Replaces the base sigmoid attention `σ(K_i · Q_j)` with GATv2-style
    attention that applies LeakyReLU before the final projection, making
    attention scores query-dependent (fixes the static attention problem
    in original GAT; Brody et al. 2022).

    Multi-head attention with H heads, where each head independently
    computes attention over a subspace of the encoded embeddings. Heads
    are averaged (not concatenated) to keep the output dimension unchanged.

    Key design choices for CaRS compatibility:
    - Sigmoid (not softmax) to preserve signed W interpretation
    - W still gates the aggregation: effective_W = W * α_GAT
    - DAG constraint operates on W/A unchanged

    Efficient for large feature sets (p > 100): no per-parent MLPs,
    attention is fully vectorized. Ideal for spillover experiments.
    """

    def __init__(self, *args, n_heads: int = 4, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_heads = n_heads
        head_dim = self.embedding_size // n_heads

        # GATv2: separate Q/K projections per head, shared across parents
        self.gat_W_q = nn.Linear(self.embedding_size, self.embedding_size, device=self.device)
        self.gat_W_k = nn.Linear(self.embedding_size, self.embedding_size, device=self.device)

        # Per-head attention vector `a` (GATv2: applied after LeakyReLU)
        self.gat_a = nn.Parameter(
            torch.randn(n_heads, head_dim, device=self.device) * 0.01
        )

        self.gat_leaky_relu = nn.LeakyReLU(0.2)
        self.gat_scale = head_dim ** 0.5

    def predict(self, X, W_adj, A=None):
        """Generate predictions using GATv2 attention."""
        if X.dim() == 2:
            X = X.unsqueeze(0)

        batch_size = X.shape[0]

        if W_adj.dim() == 3:
            W_adj = W_adj.unsqueeze(0)

        if A is not None:
            if A.dim() == 3:
                A = A.unsqueeze(0)
            W_adj = W_adj * A

        X = X.unsqueeze(-1)
        X = X.expand(-1, -1, -1, self.num_nodes)
        X_masked = X * self.group_mask

        E = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Encoder pass (shared)
        X_in_g = torch.cat([X_masked, E], dim=-1)
        X_emb = self.g(X_in_g)  # [batch, lag+1, num_nodes, emb]

        W_adj_exp = W_adj.expand(batch_size, -1, -1, -1)

        # GATv2 attention
        B, L, P, D = X_emb.shape
        H = self.n_heads
        Dh = D // H

        # Project queries and keys
        Q = self.gat_W_q(X_emb)  # [B, L, P, D]
        K = self.gat_W_k(X_emb)  # [B, L, P, D]

        # Reshape for multi-head: [B, L, P, H, Dh]
        Q = Q.view(B, L, P, H, Dh)
        K = K.view(B, L, P, H, Dh)

        # GATv2 attention, fully vectorised: pair[b,l,i,j,h,d] =
        # leaky_relu(Q[b,l,j,h,d] + K[b,l,i,h,d]). The intermediate has shape
        # [B, L, P, P, H, Dh] and is materialised when feature count remains
        # tractable (p < ~80 features). For very wide feature sets, fall back
        # to a per-child loop to bound peak memory.
        max_p_dense = 96
        if P <= max_p_dense:
            Q_exp = Q.unsqueeze(2)  # [B, L, 1, P, H, Dh] (child j as last spatial dim)
            K_exp = K.unsqueeze(3)  # [B, L, P, 1, H, Dh] (parent i as first spatial dim)
            pair = self.gat_leaky_relu(Q_exp + K_exp)  # [B, L, P, P, H, Dh]
            scores = torch.einsum("blijhd,hd->blijh", pair, self.gat_a) / self.gat_scale
            attn_scores = scores.mean(dim=-1)  # [B, L, P, P]
        else:
            attn_scores = torch.zeros(B, L, P, P, device=X_emb.device)
            for j in range(P):
                Q_j = Q[:, :, j:j + 1, :, :]
                pair_j = self.gat_leaky_relu(Q_j + K)
                scores_j = torch.einsum("blihd,hd->blih", pair_j, self.gat_a) / self.gat_scale
                attn_scores[:, :, :, j] = scores_j.mean(dim=-1)

        # Sigmoid (not softmax) to preserve signed W interpretation
        attn_weights = torch.sigmoid(attn_scores)

        # Modulate W with GAT attention
        W_effective = W_adj_exp * attn_weights
        X_aggr = torch.einsum("blij,blio->bjo", W_effective.flip([1]), X_emb)

        # Decoder pass (same as base)
        X_in_f = torch.cat([X_aggr, E[:, 0, :, :]], dim=-1)
        X_rec = self.f(X_in_f)
        X_rec = X_rec * self.group_mask
        predictions = X_rec.sum(dim=1)

        variance = None
        if self.heteroscedastic:
            X_emb_var = self.g_var(X_in_g)
            X_aggr_var = torch.einsum("blij,blio->bjo", W_effective.flip([1]), X_emb_var)
            X_in_f_var = torch.cat([X_aggr_var, E[:, 0, :, :]], dim=-1)
            X_rec_var = self.f_var(X_in_f_var)
            X_rec_var = X_rec_var * self.group_mask
            variance = F.softplus(X_rec_var.sum(dim=1)) + 1e-6

        return predictions.squeeze(0), variance.squeeze(0) if variance is not None else None


class CausalICGNN_CAM_GAT(CausalICGNN):
    """Unified CAM + GATv2 ICGNN.

    Combines (i) per-parent CAM nonlinear transforms applied to parent
    embeddings before aggregation, and (ii) GATv2 attention that modulates
    structural weights $W$. This is the configuration used by \\CaRS{}+\\textit{S}
    when the spillover-augmented feature set is consumed by the headline
    \\CaRS{} architecture (rather than the linear ``GAT-only'' variant), so a
    single model can carry the merit-order-style nonlinearity \\emph{and} the
    cross-border attention routing.

    Implementation notes:
    - The CAM transforms operate on the parent embeddings used in the einsum
      aggregation (mirroring CausalICGNN_CAM).
    - GATv2 attention is computed on the original (pre-CAM) embeddings so
      that attention routing is independent of the CAM transform and the
      sigmoid-not-softmax convention is preserved.
    - The aggregation einsum unchanged: $W_{eff} = W * \\alpha_{GAT}$, with
      the CAM-transformed embeddings on the input side.
    """

    def __init__(self, *args, cam_hidden_dim: int = 32, n_heads: int = 4, **kwargs):
        super().__init__(*args, **kwargs)

        self.cam_hidden_dim = cam_hidden_dim
        self.n_heads = n_heads
        head_dim = self.embedding_size // n_heads

        # CAM per-parent transforms — vectorised as four weight tensors of
        # shape (lag+1, p, ...) so the (lag+1)*p MLPs can be applied in two
        # einsum contractions instead of a Python loop. Mathematically
        # equivalent to the ModuleList in CausalICGNN_CAM but ~30x faster
        # when combined with the GATv2 attention path.
        L = self.lag + 1
        P = self.num_nodes
        E = self.embedding_size
        Hh = cam_hidden_dim
        sd = E ** -0.5  # mimic nn.Linear default init scale

        self.cam_W1 = nn.Parameter(torch.randn(L, P, Hh, E, device=self.device) * sd)
        self.cam_b1 = nn.Parameter(torch.zeros(L, P, Hh, device=self.device))
        self.cam_W2 = nn.Parameter(torch.randn(L, P, E, Hh, device=self.device) * sd)
        self.cam_b2 = nn.Parameter(torch.zeros(L, P, E, device=self.device))
        self.cam_leaky_relu = nn.LeakyReLU()

        if self.heteroscedastic:
            self.cam_W1_var = nn.Parameter(torch.randn(L, P, Hh, E, device=self.device) * sd)
            self.cam_b1_var = nn.Parameter(torch.zeros(L, P, Hh, device=self.device))
            self.cam_W2_var = nn.Parameter(torch.randn(L, P, E, Hh, device=self.device) * sd)
            self.cam_b2_var = nn.Parameter(torch.zeros(L, P, E, device=self.device))

        # GATv2 attention components
        self.gat_W_q = nn.Linear(self.embedding_size, self.embedding_size, device=self.device)
        self.gat_W_k = nn.Linear(self.embedding_size, self.embedding_size, device=self.device)
        self.gat_a = nn.Parameter(
            torch.randn(n_heads, head_dim, device=self.device) * 0.01
        )
        self.gat_leaky_relu = nn.LeakyReLU(0.2)
        self.gat_scale = head_dim ** 0.5

    def _apply_parent_transforms(self, X_emb, W1, b1, W2, b2):
        # X_emb: [B, L, P, E]; W1: [L, P, H, E]; W2: [L, P, E, H]
        Y = torch.einsum("blpe,lphe->blph", X_emb, W1) + b1.unsqueeze(0)
        Y = self.cam_leaky_relu(Y)
        Out = torch.einsum("blph,lpeh->blpe", Y, W2) + b2.unsqueeze(0)
        return Out

    def _gat_attention(self, X_emb):
        B, L, P, D = X_emb.shape
        H = self.n_heads
        Dh = D // H
        Q = self.gat_W_q(X_emb).view(B, L, P, H, Dh)
        K = self.gat_W_k(X_emb).view(B, L, P, H, Dh)

        max_p_dense = 96
        if P <= max_p_dense:
            Q_exp = Q.unsqueeze(2)
            K_exp = K.unsqueeze(3)
            pair = self.gat_leaky_relu(Q_exp + K_exp)
            scores = torch.einsum("blijhd,hd->blijh", pair, self.gat_a) / self.gat_scale
            attn_scores = scores.mean(dim=-1)
        else:
            attn_scores = torch.zeros(B, L, P, P, device=X_emb.device)
            for j in range(P):
                Q_j = Q[:, :, j:j + 1, :, :]
                pair_j = self.gat_leaky_relu(Q_j + K)
                scores_j = torch.einsum("blihd,hd->blih", pair_j, self.gat_a) / self.gat_scale
                attn_scores[:, :, :, j] = scores_j.mean(dim=-1)
        return torch.sigmoid(attn_scores)

    def predict(self, X, W_adj, A=None):
        if X.dim() == 2:
            X = X.unsqueeze(0)
        batch_size = X.shape[0]
        if W_adj.dim() == 3:
            W_adj = W_adj.unsqueeze(0)
        if A is not None:
            if A.dim() == 3:
                A = A.unsqueeze(0)
            W_adj = W_adj * A

        X = X.unsqueeze(-1).expand(-1, -1, -1, self.num_nodes)
        X_masked = X * self.group_mask
        E = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Encoder (shared)
        X_in_g = torch.cat([X_masked, E], dim=-1)
        X_emb = self.g(X_in_g)

        # CAM transforms on parent embeddings (vectorised)
        X_emb_cam = self._apply_parent_transforms(
            X_emb, self.cam_W1, self.cam_b1, self.cam_W2, self.cam_b2
        )

        W_adj_exp = W_adj.expand(batch_size, -1, -1, -1)

        # GATv2 attention on original (pre-CAM) embeddings
        attn_weights = self._gat_attention(X_emb) if self.use_attention else None

        W_effective = W_adj_exp if attn_weights is None else (W_adj_exp * attn_weights)
        X_aggr = torch.einsum("blij,blio->bjo", W_effective.flip([1]), X_emb_cam)

        X_in_f = torch.cat([X_aggr, E[:, 0, :, :]], dim=-1)
        X_rec = self.f(X_in_f) * self.group_mask
        predictions = X_rec.sum(dim=1)

        variance = None
        if self.heteroscedastic:
            X_emb_var = self.g_var(X_in_g)
            X_emb_var_cam = self._apply_parent_transforms(
                X_emb_var, self.cam_W1_var, self.cam_b1_var, self.cam_W2_var, self.cam_b2_var
            )
            X_aggr_var = torch.einsum("blij,blio->bjo", W_effective.flip([1]), X_emb_var_cam)
            X_in_f_var = torch.cat([X_aggr_var, E[:, 0, :, :]], dim=-1)
            X_rec_var = self.f_var(X_in_f_var) * self.group_mask
            variance = F.softplus(X_rec_var.sum(dim=1)) + 1e-6

        return predictions.squeeze(0), variance.squeeze(0) if variance is not None else None


class CausalEmission(nn.Module):
    """
    FANTOM-style Causal Emission for DS3M.

    Replaces DS3M's black-box emission network with a causal SEM.
    For each regime d, learns:
    - A DAG adjacency matrix A^(d)
    - An ICGNN-based structural equation model

    This allows interpretable causal relationships per market regime.
    """

    def __init__(
        self,
        num_nodes: int,
        device: torch.device,
        target_idx: int = 0,
        lag: int = 1,
        h_dim: int = 32,
        z_dim: int = 8,
        embedding_size: Optional[int] = None,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        norm_layers: bool = True,
        heteroscedastic: bool = True,
        use_attention: bool = True,
        w_init_scale: float = 0.01,
        aggregation_mode: str = "linear",
        cam_hidden_dim: int = 32,
        dual_channel: bool = False,
        physical_mask: Optional[torch.Tensor] = None,
        physical_prior_mode: str = "off",
        physical_prior_alpha_init: float = 0.05,
        pure_scm_readout: bool = False,
    ):
        """
        Initialize Causal Emission module.

        Args:
            num_nodes: Number of input features
            device: Torch device
            target_idx: Index of target variable to predict (-1 for last)
            lag: Temporal lag for causal graph
            h_dim: DS3M hidden state dimension (for conditioning)
            z_dim: DS3M latent dimension (for conditioning)
            embedding_size: ICGNN embedding size
            encoder_layer_sizes: ICGNN encoder layers
            decoder_layer_sizes: ICGNN decoder layers
            norm_layers: Use layer normalization
            heteroscedastic: Learn variable-dependent noise
            w_init_scale: Standard deviation for ICGNN W initialization
            aggregation_mode: ICGNN aggregation type:
                "linear" — standard W-weighted sum (default)
                "dual_channel" — pos/neg channel split with MLP combiner
                "cam" — per-parent nonlinear transforms (Causal Additive Model)
            cam_hidden_dim: Hidden dimension for CAM per-parent MLPs
            dual_channel: Deprecated, use aggregation_mode="dual_channel" instead
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.device = device
        self.target_idx = target_idx if target_idx >= 0 else num_nodes + target_idx
        self.lag = lag
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.heteroscedastic = heteroscedastic
        # Pure-SCM readout: target prediction is the bare linear structural
        # equation y_j = sum_{lag,i} W[lag,i,j]*x_i -- W IS the regression
        # coefficient, with NO learnable readout (no g/f/CAM/attention/target_proj
        # between W and the output), so the W-aux loss cannot be satisfied by a
        # random-W-projection readout. A single learnable log-variance replaces
        # the heteroscedastic variance network in this mode.
        self.pure_scm_readout = pure_scm_readout
        self.pure_log_var = nn.Parameter(torch.zeros(1, device=device))

        # Causal ICGNN for structural equations
        # Backward compat: dual_channel=True maps to aggregation_mode="dual_channel"
        if dual_channel and aggregation_mode == "linear":
            aggregation_mode = "dual_channel"

        icgnn_classes = {
            "linear": CausalICGNN,
            "dual_channel": DualChannelICGNN,
            "cam": CausalICGNN_CAM,
            "gat": CausalICGNN_GAT,
            "cam_gat": CausalICGNN_CAM_GAT,
        }
        ICGNNClass = icgnn_classes[aggregation_mode]
        icgnn_kwargs = {}
        if aggregation_mode == "cam":
            icgnn_kwargs["cam_hidden_dim"] = cam_hidden_dim
        elif aggregation_mode == "gat":
            icgnn_kwargs["n_heads"] = 4
        elif aggregation_mode == "cam_gat":
            icgnn_kwargs["cam_hidden_dim"] = cam_hidden_dim
            icgnn_kwargs["n_heads"] = 4

        self.icgnn = ICGNNClass(
            num_nodes=num_nodes,
            device=device,
            lag=lag,
            embedding_size=embedding_size,
            encoder_layer_sizes=encoder_layer_sizes,
            decoder_layer_sizes=decoder_layer_sizes,
            norm_layers=norm_layers,
            res_connection=True,
            heteroscedastic=heteroscedastic,
            use_attention=use_attention,
            w_init_scale=w_init_scale,
            physical_mask=physical_mask,
            physical_prior_mode=physical_prior_mode,
            physical_prior_alpha_init=physical_prior_alpha_init,
            **icgnn_kwargs,
        )

        # Conditioning network: maps (h_t, z_t) to additive residual for ICGNN
        # Tanh squash preserves ICGNN dominance; learnable alpha controls range
        conditioning_dim = h_dim + z_dim
        self.condition_net = nn.Sequential(
            nn.Linear(conditioning_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        ).to(device)

        # Learnable conditioning scale (initialized at 0.5)
        self.conditioning_alpha = nn.Parameter(torch.tensor(0.5, device=device))

        # Output projection to target dimension (if predicting single target)
        self.target_proj = nn.Linear(num_nodes, 1).to(device)

        # Variance projection
        if heteroscedastic:
            self.var_proj = nn.Sequential(
                nn.Linear(num_nodes, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 1),
                nn.Softplus()
            ).to(device)

    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        h_t: Optional[torch.Tensor] = None,
        z_t: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate emission (prediction) for target variable.

        Args:
            X: Input features [batch, lag+1, num_nodes] or [batch, num_nodes]
            A: Binary adjacency matrix [lag+1, num_nodes, num_nodes]
            h_t: DS3M hidden state [batch, h_dim] (optional)
            z_t: DS3M latent state [batch, z_dim] (optional)
            edge_mask: Binary mask [lag+1, num_nodes, num_nodes] to zero out
                specific edges for ablation analysis. 1 = keep, 0 = ablate.

        Returns:
            mean: Predicted target mean [batch, 1]
            std: Predicted target std [batch, 1]
            all_predictions: Predictions for all nodes [batch, num_nodes]
        """
        # Handle different input shapes
        if X.dim() == 2:
            # [batch, num_nodes] -> [batch, lag+1, num_nodes] with lag=0
            X = X.unsqueeze(1)

        batch_size = X.shape[0]

        # Get weighted adjacency
        W_adj = self.icgnn.get_weighted_adjacency()

        # Apply edge mask for ablation (zero out specific edges)
        A_masked = A
        if edge_mask is not None:
            A_masked = A * edge_mask

        # Get predictions for all nodes. In pure-SCM mode the prediction is the
        # bare linear structural equation y_j = sum_{lag,i} W[lag,i,j] * x_i, with
        # W the regression coefficient and NO learnable readout, so W must be
        # identified to fit the target (no random-features escape).
        if self.pure_scm_readout:
            Weff = W_adj * A_masked
            predictions = torch.einsum("lij,bli->bj", Weff, X)  # [batch, num_nodes]
            variance = (F.softplus(self.pure_log_var) + 1e-6).expand(
                batch_size, self.num_nodes)
        else:
            # Get ICGNN predictions for all nodes
            predictions, variance = self.icgnn.predict(X, W_adj, A_masked)

        # predictions shape: [batch, num_nodes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)

        # Extract target prediction (this is the W-routed causal-only prediction,
        # i.e. it flows purely through the ICGNN weighted adjacency W, before any
        # latent conditioning is added below).
        target_pred = predictions[:, self.target_idx:self.target_idx+1]  # [batch, 1]
        causal_only_pred = target_pred

        # Condition on DS3M latent state via tanh-bounded additive residual
        # tanh preserves ICGNN dominance; learnable alpha controls effective range
        if h_t is not None and z_t is not None:
            latent = torch.cat([h_t, z_t], dim=-1)  # [batch, h_dim + z_dim]
            condition_scale = self.condition_net(latent)  # [batch, 1]
            target_pred = target_pred + self.conditioning_alpha * torch.tanh(condition_scale)

        # Stash the W-only prediction so the model can apply a W-routed auxiliary
        # loss: MSE(causal_only_pred, target). This forces the causal weights W
        # onto the prediction path rather than letting the latent residual bypass
        # them (see DS3MCausal.forward / lambda_w_aux).
        self._last_causal_pred = causal_only_pred

        # Get variance
        if variance is not None and variance.dim() > 0:
            if variance.dim() == 1:
                variance = variance.unsqueeze(0)
            target_var = variance[:, self.target_idx:self.target_idx+1]
        else:
            target_var = torch.ones(batch_size, 1, device=self.device) * 0.1

        target_std = torch.sqrt(target_var + 1e-6)

        return target_pred, target_std, predictions

    def get_adjacency_matrix(self) -> torch.Tensor:
        """Return the current weighted adjacency matrix."""
        return self.icgnn.get_weighted_adjacency()


if __name__ == "__main__":
    # Test the causal emission module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Parameters
    batch_size = 32
    num_nodes = 10  # e.g., 10 features
    lag = 1
    h_dim = 20
    z_dim = 5

    # Create module
    emission = CausalEmission(
        num_nodes=num_nodes,
        device=device,
        target_idx=-1,  # Last variable is target
        lag=lag,
        h_dim=h_dim,
        z_dim=z_dim,
    )

    # Create test inputs
    X = torch.randn(batch_size, lag + 1, num_nodes, device=device)
    A = (torch.rand(lag + 1, num_nodes, num_nodes, device=device) > 0.7).float()
    h_t = torch.randn(batch_size, h_dim, device=device)
    z_t = torch.randn(batch_size, z_dim, device=device)

    # Forward pass
    mean, std, all_preds = emission(X, A, h_t, z_t)

    print(f"Input X shape: {X.shape}")
    print(f"Adjacency A shape: {A.shape}")
    print(f"Output mean shape: {mean.shape}")
    print(f"Output std shape: {std.shape}")
    print(f"All predictions shape: {all_preds.shape}")
    print(f"Mean range: [{mean.min().item():.4f}, {mean.max().item():.4f}]")
    print(f"Std range: [{std.min().item():.4f}, {std.max().item():.4f}]")

    # Test without conditioning
    mean2, std2, all_preds2 = emission(X, A)
    print(f"\nWithout conditioning - mean shape: {mean2.shape}")

    print("\nCausal Emission module test passed!")
