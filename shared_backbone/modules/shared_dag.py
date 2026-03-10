"""
Shared DAG Module: Variational Distribution for Regime-Specific DAGs

This module implements variational distributions over adjacency matrices
that support two modes:
1. Independent: Completely separate DAG per regime
2. Shared backbone: Common edges + regime-specific edges

The shared backbone approach allows regimes to share structural
similarities while having regime-specific causal relationships.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np


class VarDistA_Temporal(nn.Module):
    """
    Variational distribution for temporal adjacency matrix.

    Extends the ThreeWayGraphDist approach from FANTOM to support:
    - Instantaneous effects: Uses 3-way categorical (A->B, B->A, no edge)
    - Lagged effects: Uses Bernoulli distribution (edges from past to present)

    The temporal adjacency matrix has shape [lag+1, num_nodes, num_nodes]:
    - A[0, ...]: Instantaneous adjacency (DAG constraint required)
    - A[1:, ...]: Lagged adjacency (no DAG constraint, allows cycles across time)
    """

    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        lag: int,
        tau_gumbel: float = 1.0,
        init_logits: Optional[List[float]] = None,
    ):
        """
        Initialize the temporal variational distribution.

        Args:
            device: Torch device
            num_nodes: Number of nodes in the graph
            lag: Number of time lags (temporal depth)
            tau_gumbel: Temperature for Gumbel-Softmax
            init_logits: Initial logits [instantaneous_no_edge, lagged_no_edge]
                        Negative values encourage sparser graphs
        """
        super().__init__()

        self.device = device
        self.num_nodes = num_nodes
        self.lag = lag
        self.tau_gumbel = tau_gumbel
        self.init_logits = init_logits

        assert lag > 0, "Lag must be > 0"

        # Instantaneous adjacency: 3-way categorical for each node pair
        # For n nodes, we have n(n-1)/2 unique pairs
        # Each pair has 3 options: i->j, j->i, no edge
        n_pairs = (num_nodes * (num_nodes - 1)) // 2
        self.logits_inst = nn.Parameter(
            torch.zeros(3, n_pairs, device=device),
            requires_grad=True
        )

        # Lagged adjacency: Bernoulli for each edge
        # Shape: [2, lag, num_nodes, num_nodes]
        # logits_lag[0]: no edge, logits_lag[1]: edge exists
        self.logits_lag = nn.Parameter(
            torch.zeros(2, lag, num_nodes, num_nodes, device=device),
            requires_grad=True
        )

        # Set initial logits if provided (matching FANTOM convention)
        # For instantaneous: set no-edge class logit (higher = more sparse)
        # For lagged: set no-edge class logit (higher = more sparse)
        if init_logits is not None:
            self.logits_inst.data[2, ...] = init_logits[0]  # no-edge class for instantaneous
            self.logits_lag.data[0, ...] = init_logits[1]   # no-edge class for lagged (FIXED: was [1])

        # Pre-compute triangular indices for efficient conversion
        self.lower_idxs = torch.unbind(
            torch.tril_indices(num_nodes, num_nodes, offset=-1, device=device), 0
        )

    def _triangular_vec_to_matrix(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Convert triangular vector to adjacency matrix.

        Args:
            vec: Shape [k, n(n-1)/2] where k in {2, 3}

        Returns:
            matrix: Shape [num_nodes, num_nodes]
        """
        output = torch.zeros(
            self.num_nodes, self.num_nodes, device=self.device
        )
        output[self.lower_idxs[0], self.lower_idxs[1]] = vec[0, ...]
        output[self.lower_idxs[1], self.lower_idxs[0]] = vec[1, ...]
        return output

    def get_adj_matrix(self, do_round: bool = False) -> torch.Tensor:
        """
        Get the expected adjacency matrix (edge probabilities).

        Args:
            do_round: If True, round probabilities to 0/1

        Returns:
            A: Shape [lag+1, num_nodes, num_nodes]
        """
        A = torch.zeros(
            self.lag + 1, self.num_nodes, self.num_nodes,
            device=self.device
        )

        # Instantaneous: 3-way softmax
        probs_inst = F.softmax(self.logits_inst, dim=0)  # [3, n_pairs]
        A[0, ...] = self._triangular_vec_to_matrix(probs_inst)  # i->j and j->i probs

        # Lagged: Bernoulli probabilities
        probs_lag = F.softmax(self.logits_lag, dim=0)[1, ...]  # [lag, n, n]
        A[1:, ...] = probs_lag

        if do_round:
            return A.round()
        return A

    def sample_A(self) -> torch.Tensor:
        """
        Sample adjacency matrix using Gumbel-Softmax (hard samples).

        Returns:
            A: Shape [lag+1, num_nodes, num_nodes]
        """
        A = torch.zeros(
            self.lag + 1, self.num_nodes, self.num_nodes,
            device=self.device
        )

        # Instantaneous: Gumbel-Softmax
        sample_inst = F.gumbel_softmax(
            self.logits_inst, tau=self.tau_gumbel, hard=True, dim=0
        )  # [3, n_pairs]
        A[0, ...] = self._triangular_vec_to_matrix(sample_inst)

        # Lagged: Gumbel-Softmax
        sample_lag = F.gumbel_softmax(
            self.logits_lag, tau=self.tau_gumbel, hard=True, dim=0
        )[1, ...]  # [lag, n, n]
        A[1:, ...] = sample_lag

        return A

    def entropy(self) -> torch.Tensor:
        """
        Compute entropy of the variational distribution.

        Returns:
            Total entropy (scalar)
        """
        # Instantaneous: 3-way categorical entropy
        dist_inst = td.Categorical(logits=self.logits_inst.transpose(0, -1))
        entropy_inst = dist_inst.entropy().sum()

        # Lagged: Bernoulli entropy
        logits_diff = self.logits_lag[1, ...] - self.logits_lag[0, ...]
        dist_lag = td.Independent(td.Bernoulli(logits=logits_diff), 2)
        entropy_lag = dist_lag.entropy().sum()

        return entropy_inst + entropy_lag

    def kl_divergence(self, prior_prob: float = 0.5) -> torch.Tensor:
        """
        Compute KL divergence from uniform prior.

        Args:
            prior_prob: Prior edge probability

        Returns:
            KL divergence (scalar)
        """
        # Instantaneous KL
        probs_inst = F.softmax(self.logits_inst, dim=0)
        prior_inst = torch.ones_like(probs_inst) / 3.0
        kl_inst = (probs_inst * (torch.log(probs_inst + 1e-10) - torch.log(prior_inst))).sum()

        # Lagged KL
        probs_lag = F.softmax(self.logits_lag, dim=0)[1, ...]
        prior_lag = torch.ones_like(probs_lag) * prior_prob
        kl_lag = (probs_lag * torch.log(probs_lag / prior_lag + 1e-10) +
                  (1 - probs_lag) * torch.log((1 - probs_lag) / (1 - prior_lag) + 1e-10)).sum()

        return kl_inst + kl_lag


class SharedRegimeDAG(nn.Module):
    """
    DAG distribution with shared backbone and regime-specific edges.

    Supports two modes:
    1. "independent": Completely separate DAG per regime
    2. "shared_backbone": A_total = A_shared + A_regime[d]

    The shared backbone allows structural similarities across regimes
    while permitting regime-specific causal relationships.
    """

    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        lag: int,
        d_dim: int,
        sharing_mode: str = "independent",
        tau_gumbel: float = 1.0,
        init_logits: Optional[List[float]] = None,
        regime_noise_std: float = 0.0,  # Noise std for regime deviation initialization
    ):
        """
        Initialize shared regime DAG.

        Args:
            device: Torch device
            num_nodes: Number of nodes
            lag: Temporal lag
            d_dim: Number of regimes
            sharing_mode: "independent" or "shared_backbone"
            tau_gumbel: Gumbel-Softmax temperature
            init_logits: Initial logits for sparsity
        """
        super().__init__()

        self.device = device
        self.num_nodes = num_nodes
        self.lag = lag
        self.d_dim = d_dim
        self.sharing_mode = sharing_mode
        self.tau_gumbel = tau_gumbel

        self.regime_noise_std = regime_noise_std

        if sharing_mode == "shared_backbone":
            # Shared component
            self.var_dist_A_shared = VarDistA_Temporal(
                device=device,
                num_nodes=num_nodes,
                lag=lag,
                tau_gumbel=tau_gumbel,
                init_logits=init_logits,
            )
            # Regime-specific residuals (VERY sparse - only deviations from backbone)
            # Add 1.5 to make regime-specific much sparser than shared component
            regime_init = [i + 1.5 for i in init_logits] if init_logits else None
            self.var_dist_A_regime = nn.ModuleList([
                VarDistA_Temporal(
                    device=device,
                    num_nodes=num_nodes,
                    lag=lag,
                    tau_gumbel=tau_gumbel,
                    init_logits=regime_init,
                )
                for _ in range(d_dim)
            ])
            # Add noise to regime deviations for differentiation
            if regime_noise_std > 0:
                for d, var_dist in enumerate(self.var_dist_A_regime):
                    # Different noise for each regime to break symmetry
                    noise_inst = torch.randn_like(var_dist.logits_inst) * regime_noise_std * (d + 1)
                    noise_lag = torch.randn_like(var_dist.logits_lag) * regime_noise_std * (d + 1)
                    var_dist.logits_inst.data += noise_inst
                    var_dist.logits_lag.data += noise_lag
        else:  # independent
            self.var_dist_A_regime = nn.ModuleList([
                VarDistA_Temporal(
                    device=device,
                    num_nodes=num_nodes,
                    lag=lag,
                    tau_gumbel=tau_gumbel,
                    init_logits=init_logits,
                )
                for _ in range(d_dim)
            ])
            self.var_dist_A_shared = None

    def get_dag(
        self,
        regime_id: int,
        sample: bool = True
    ) -> torch.Tensor:
        """
        Get adjacency matrix for a specific regime.

        Args:
            regime_id: Regime index
            sample: If True, use Gumbel-Softmax sampling; else use expected value

        Returns:
            A: Shape [lag+1, num_nodes, num_nodes]
        """
        if self.sharing_mode == "shared_backbone":
            # Get shared component
            if sample:
                A_shared = self.var_dist_A_shared.sample_A()
                A_regime = self.var_dist_A_regime[regime_id].sample_A()
            else:
                A_shared = self.var_dist_A_shared.get_adj_matrix(do_round=False)
                A_regime = self.var_dist_A_regime[regime_id].get_adj_matrix(do_round=False)

            # Combine using probabilistic OR: P(edge) = 1 - (1-P_shared)(1-P_regime)
            # This properly combines two probability distributions
            # Edge exists if it's in shared OR regime-specific (or both)
            A = 1.0 - (1.0 - A_shared) * (1.0 - A_regime)
        else:
            if sample:
                A = self.var_dist_A_regime[regime_id].sample_A()
            else:
                A = self.var_dist_A_regime[regime_id].get_adj_matrix(do_round=False)

        return A

    def get_dag_with_penalty(
        self,
        regime_id: int,
        sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get adjacency matrix and compute DAG penalty.

        For shared_backbone mode, applies DAG constraint to the COMBINED
        matrix (probabilistic OR of shared and regime-specific components).
        This ensures the union is acyclic — constraining components separately
        does NOT guarantee the union is a DAG.

        Args:
            regime_id: Regime index
            sample: If True, use Gumbel-Softmax sampling; else use expected value

        Returns:
            A: Combined adjacency matrix [lag+1, num_nodes, num_nodes]
            dag_penalty: DAG penalty on the combined matrix
        """
        if self.sharing_mode == "shared_backbone":
            # Get both components
            if sample:
                A_shared = self.var_dist_A_shared.sample_A()
                A_regime = self.var_dist_A_regime[regime_id].sample_A()
            else:
                A_shared = self.var_dist_A_shared.get_adj_matrix(do_round=False)
                A_regime = self.var_dist_A_regime[regime_id].get_adj_matrix(do_round=False)

            # Combined DAG for prediction (probabilistic OR)
            A = 1.0 - (1.0 - A_shared) * (1.0 - A_regime)

            # DAG penalty on the COMBINED matrix to ensure the union is acyclic
            dag_penalty = dagness_factor(A)

            return A, dag_penalty
        else:
            # Independent mode: single DAG per regime
            if sample:
                A = self.var_dist_A_regime[regime_id].sample_A()
            else:
                A = self.var_dist_A_regime[regime_id].get_adj_matrix(do_round=False)
            return A, dagness_factor(A)

    def get_all_dags(
        self,
        sample: bool = True
    ) -> List[torch.Tensor]:
        """
        Get adjacency matrices for all regimes.

        Args:
            sample: Whether to sample or use expected value

        Returns:
            List of adjacency matrices, one per regime
        """
        return [self.get_dag(d, sample=sample) for d in range(self.d_dim)]

    def entropy(self) -> torch.Tensor:
        """
        Compute total entropy of the distribution.

        Returns:
            Total entropy
        """
        total_entropy = torch.tensor(0.0, device=self.device)

        if self.sharing_mode == "shared_backbone":
            total_entropy += self.var_dist_A_shared.entropy()

        for var_dist in self.var_dist_A_regime:
            total_entropy += var_dist.entropy()

        return total_entropy

    def sparsity_penalty(self, norm: str = "l1") -> torch.Tensor:
        """
        Compute sparsity penalty on all DAGs.

        Args:
            norm: "l1" or "l2"

        Returns:
            Total sparsity penalty
        """
        penalty = torch.tensor(0.0, device=self.device)

        for d in range(self.d_dim):
            A = self.get_dag(d, sample=False)
            if norm == "l1":
                penalty += A.abs().sum()
            else:
                penalty += (A ** 2).sum()

        return penalty

    def similarity_penalty(self) -> torch.Tensor:
        """
        Compute penalty encouraging regime DAGs to be similar.

        Useful when we expect regimes to share structure.

        Returns:
            Similarity penalty (lower when DAGs are more similar)
        """
        if self.d_dim < 2:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        dags = self.get_all_dags(sample=False)

        for i in range(self.d_dim):
            for j in range(i + 1, self.d_dim):
                # L2 distance between DAGs
                diff = (dags[i] - dags[j]) ** 2
                penalty += diff.sum()

        return penalty

    def get_shared_edges(self) -> Optional[torch.Tensor]:
        """
        Get edges that are shared across all regimes.

        Returns:
            Shared adjacency matrix or None if not shared_backbone mode
        """
        if self.sharing_mode != "shared_backbone":
            return None

        return self.var_dist_A_shared.get_adj_matrix(do_round=False)

    def get_regime_specific_edges(self, regime_id: int) -> Optional[torch.Tensor]:
        """
        Get regime-specific edges (excluding shared).

        Returns:
            Regime-specific adjacency or full DAG if independent mode
        """
        return self.var_dist_A_regime[regime_id].get_adj_matrix(do_round=False)


def dagness_factor(A: torch.Tensor) -> torch.Tensor:
    """
    Compute DAGness penalty: tr(exp(A)) - d.

    This is 0 if and only if A is a DAG.
    For temporal adjacency, only instantaneous effects need DAG constraint.

    Args:
        A: Adjacency matrix [lag+1, n, n] or [n, n]

    Returns:
        DAGness penalty (scalar)
    """
    if A.dim() == 3:
        A_inst = A[0, ...]  # Instantaneous
    else:
        A_inst = A

    d = A_inst.shape[0]
    return torch.trace(torch.matrix_exp(A_inst)) - d


if __name__ == "__main__":
    # Test the shared DAG module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    num_nodes = 10
    lag = 1
    d_dim = 2

    # Test independent mode
    print("\n--- Testing Independent Mode ---")
    shared_dag = SharedRegimeDAG(
        device=device,
        num_nodes=num_nodes,
        lag=lag,
        d_dim=d_dim,
        sharing_mode="independent",
    )

    for d in range(d_dim):
        A = shared_dag.get_dag(d, sample=True)
        print(f"Regime {d} - A shape: {A.shape}")
        print(f"  Instantaneous edges: {A[0].sum().item():.0f}")
        print(f"  Lagged edges: {A[1:].sum().item():.0f}")
        print(f"  DAGness: {dagness_factor(A).item():.4f}")

    entropy = shared_dag.entropy()
    sparsity = shared_dag.sparsity_penalty()
    print(f"\nTotal entropy: {entropy.item():.4f}")
    print(f"Sparsity penalty: {sparsity.item():.4f}")

    # Test shared_backbone mode
    print("\n--- Testing Shared Backbone Mode ---")
    shared_dag_bb = SharedRegimeDAG(
        device=device,
        num_nodes=num_nodes,
        lag=lag,
        d_dim=d_dim,
        sharing_mode="shared_backbone",
        init_logits=[-0.5, -0.5],  # Encourage sparsity
    )

    shared_A = shared_dag_bb.get_shared_edges()
    print(f"Shared edges: {shared_A.sum().item():.2f}")

    for d in range(d_dim):
        A = shared_dag_bb.get_dag(d, sample=True)
        regime_specific = shared_dag_bb.get_regime_specific_edges(d)
        print(f"Regime {d}:")
        print(f"  Total edges: {A.sum().item():.2f}")
        print(f"  Regime-specific edges: {regime_specific.sum().item():.2f}")

    similarity = shared_dag_bb.similarity_penalty()
    print(f"\nSimilarity penalty: {similarity.item():.4f}")

    # Test gradient flow
    print("\n--- Testing Gradient Flow ---")
    optimizer = torch.optim.Adam(shared_dag_bb.parameters(), lr=0.01)

    for step in range(5):
        optimizer.zero_grad()
        loss = shared_dag_bb.sparsity_penalty() + 0.1 * shared_dag_bb.similarity_penalty()
        loss.backward()
        optimizer.step()
        print(f"Step {step}: loss = {loss.item():.4f}")

    print("\nShared DAG module test passed!")
