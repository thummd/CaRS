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
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.device = device
        self.lag = lag
        self.heteroscedastic = heteroscedastic

        # Node embeddings
        self.embedding_size = embedding_size or num_nodes
        self.embeddings = self._initialize_embeddings()

        # Weighted adjacency matrix (learned)
        # Shape: [lag+1, num_nodes, num_nodes] for temporal
        self.W = self._initialize_W()

        # Network architecture sizes
        a = max(4 * num_nodes, self.embedding_size, 64)
        self.encoder_layers = encoder_layer_sizes or [a, a]
        self.decoder_layers = decoder_layer_sizes or [a, a]

        # Group mask (identity for continuous variables)
        self.group_mask = torch.eye(num_nodes, dtype=torch.bool, device=device)

        # Input dimension for networks
        in_dim_g = self.embedding_size + num_nodes  # embedding + variable values
        out_dim_g = self.embedding_size
        in_dim_f = self.embedding_size + out_dim_g

        self.norm_layer = nn.LayerNorm if norm_layers else None

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
        """Initialize weighted adjacency matrix."""
        # Shape: [lag+1, num_nodes, num_nodes]
        W = torch.zeros(
            self.lag + 1, self.num_nodes, self.num_nodes,
            device=self.device
        )
        return nn.Parameter(W, requires_grad=True)

    def get_weighted_adjacency(self) -> torch.Tensor:
        """
        Get the weighted adjacency matrix with diagonal disabled for instantaneous.

        Returns:
            W_adj: Shape [lag+1, num_nodes, num_nodes]
        """
        W_adj = self.W.clone()
        # Disable self-loops in instantaneous adjacency
        torch.diagonal(W_adj[0, ...]).zero_()
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
        # W_adj: [batch, lag+1, num_nodes (parent), num_nodes (child)]
        # We need to sum over lag and parent dimensions
        X_aggr = torch.einsum("blij,klio->kjo", W_adj.flip([1]), X_emb)  # [batch, num_nodes, out_dim_g]

        # Decoder pass
        X_in_f = torch.cat([X_aggr, E[:, 0, :, :]], dim=-1)  # [batch, num_nodes, emb+out_dim_g]
        X_rec = self.f(X_in_f)  # [batch, num_nodes, num_nodes]

        # Apply group mask and sum
        X_rec = X_rec * self.group_mask
        predictions = X_rec.sum(dim=1)  # [batch, num_nodes]

        variance = None
        if self.heteroscedastic:
            X_emb_var = self.g_var(X_in_g)
            X_aggr_var = torch.einsum("blij,klio->kjo", W_adj.flip([1]), X_emb_var)
            X_in_f_var = torch.cat([X_aggr_var, E[:, 0, :, :]], dim=-1)
            X_rec_var = self.f_var(X_in_f_var)
            X_rec_var = X_rec_var * self.group_mask
            variance = F.softplus(X_rec_var.sum(dim=1)) + 1e-6  # Ensure positive

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
        target_idx: int = -1,
        lag: int = 1,
        h_dim: int = 32,
        z_dim: int = 8,
        embedding_size: Optional[int] = None,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        norm_layers: bool = True,
        heteroscedastic: bool = True,
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
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.device = device
        self.target_idx = target_idx if target_idx >= 0 else num_nodes + target_idx
        self.lag = lag
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.heteroscedastic = heteroscedastic

        # Causal ICGNN for structural equations
        self.icgnn = CausalICGNN(
            num_nodes=num_nodes,
            device=device,
            lag=lag,
            embedding_size=embedding_size,
            encoder_layer_sizes=encoder_layer_sizes,
            decoder_layer_sizes=decoder_layer_sizes,
            norm_layers=norm_layers,
            res_connection=True,
            heteroscedastic=heteroscedastic,
        )

        # Conditioning network: maps (h_t, z_t) to modulation of ICGNN
        # This allows DS3M's latent state to influence the causal prediction
        conditioning_dim = h_dim + z_dim
        self.condition_net = nn.Sequential(
            nn.Linear(conditioning_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)  # Scale factor for target prediction
        ).to(device)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate emission (prediction) for target variable.

        Args:
            X: Input features [batch, lag+1, num_nodes] or [batch, num_nodes]
            A: Binary adjacency matrix [lag+1, num_nodes, num_nodes]
            h_t: DS3M hidden state [batch, h_dim] (optional)
            z_t: DS3M latent state [batch, z_dim] (optional)

        Returns:
            mean: Predicted target mean [batch, 1]
            std: Predicted target std [batch, 1]
        """
        # Handle different input shapes
        if X.dim() == 2:
            # [batch, num_nodes] -> [batch, lag+1, num_nodes] with lag=0
            X = X.unsqueeze(1)

        batch_size = X.shape[0]

        # Get weighted adjacency
        W_adj = self.icgnn.get_weighted_adjacency()

        # Get ICGNN predictions for all nodes
        predictions, variance = self.icgnn.predict(X, W_adj, A)

        # predictions shape: [batch, num_nodes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)

        # Extract target prediction
        target_pred = predictions[:, self.target_idx:self.target_idx+1]  # [batch, 1]

        # Condition on DS3M latent state if provided
        if h_t is not None and z_t is not None:
            # Concatenate latent states
            latent = torch.cat([h_t, z_t], dim=-1)  # [batch, h_dim + z_dim]
            condition_scale = self.condition_net(latent)  # [batch, 1]
            # Additive conditioning: ICGNN provides base prediction, condition_net adds residual
            # This ensures the DAG-based ICGNN must learn to contribute
            target_pred = target_pred + 0.1 * torch.tanh(condition_scale)

        # Get variance
        if variance is not None and variance.dim() > 0:
            if variance.dim() == 1:
                variance = variance.unsqueeze(0)
            target_var = variance[:, self.target_idx:self.target_idx+1]
        else:
            target_var = torch.ones(batch_size, 1, device=self.device) * 0.1

        target_std = torch.sqrt(target_var + 1e-6)

        return target_pred, target_std

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
    mean, std = emission(X, A, h_t, z_t)

    print(f"Input X shape: {X.shape}")
    print(f"Adjacency A shape: {A.shape}")
    print(f"Output mean shape: {mean.shape}")
    print(f"Output std shape: {std.shape}")
    print(f"Mean range: [{mean.min().item():.4f}, {mean.max().item():.4f}]")
    print(f"Std range: [{std.min().item():.4f}, {std.max().item():.4f}]")

    # Test without conditioning
    mean2, std2 = emission(X, A)
    print(f"\nWithout conditioning - mean shape: {mean2.shape}")

    print("\nCausal Emission module test passed!")
