"""
Hierarchical Multi-Market DAG Module for Cross-Border Price Transmission.

Implements a three-level hierarchical causal structure:
1. Global shared DAG: Common causal relationships across all European markets
   (e.g., gas price → electricity price universally)
2. Regional cluster DAGs: Cluster-specific edges capturing regional dynamics
   (e.g., Nordic hydro effects, Central European interconnection)
3. Country-regime DAGs: Per-country, per-regime deviations

The market-level DAG operates on compressed representations:
- Each country's features are compressed to a latent vector via a GRU encoder
- The ICGNN learns edges between these 12 market nodes + shared fundamentals
- W[DE, FR] directly quantifies Germany→France price transmission

This answers RQ3 (cross-border price transmission) by making cross-border
causal edges explicit and interpretable in the learned DAG.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from shared_backbone.modules.shared_dag import (
    VarDistA_Temporal, SharedRegimeDAG, dagness_factor,
)


# Default regional clusters based on market coupling and geography
DEFAULT_CLUSTERS = {
    'central': ['DE', 'AT', 'CZ', 'PL'],
    'western': ['FR', 'BE', 'NL'],
    'nordic': ['DK', 'SE'],
    'southern': ['ES', 'IT'],
    'eastern': ['HU'],
}

# Physical interconnections (for edge prior initialization)
CROSS_BORDER_PAIRS = [
    ('DE', 'FR'), ('DE', 'NL'), ('DE', 'BE'), ('DE', 'AT'),
    ('DE', 'CZ'), ('DE', 'PL'), ('DE', 'DK'), ('DE', 'SE'),
    ('FR', 'BE'), ('FR', 'ES'), ('FR', 'IT'),
    ('NL', 'BE'), ('NL', 'DK'),
    ('AT', 'CZ'), ('AT', 'HU'), ('AT', 'IT'),
    ('CZ', 'PL'), ('PL', 'SE'), ('DK', 'SE'),
]


class MarketEncoder(nn.Module):
    """
    Compresses per-country features into a fixed-size market representation.

    Each country has a different number of features (128-206). This module
    maps them all to the same latent dimension for the market-level DAG.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        device: torch.device = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64, device=device),
            nn.LayerNorm(64, device=device),
            nn.LeakyReLU(),
            nn.Linear(64, latent_dim, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, features] country-specific features

        Returns:
            z: [batch, latent_dim] compressed market representation
        """
        return self.encoder(x)


class HierarchicalMarketDAG(nn.Module):
    """
    Hierarchical DAG over 12 European electricity markets.

    Operates on market-level nodes (not individual features) to keep the
    DAG tractable. Each market is represented by its compressed latent
    vector from MarketEncoder.

    The DAG has three levels:
    - Global: edges shared across all regimes and clusters
    - Cluster: edges shared within a regional cluster
    - Country-regime: per-country, per-regime deviations

    The combined DAG for country c in regime d is:
        A(c,d) = 1 - (1 - A_global) * (1 - A_cluster[c]) * (1 - A_regime[d])
    """

    def __init__(
        self,
        device: torch.device,
        countries: List[str],
        n_shared_fundamentals: int = 3,
        lag: int = 1,
        d_dim: int = 2,
        tau_gumbel: float = 0.5,
        clusters: Optional[Dict[str, List[str]]] = None,
        init_logits: Optional[List[float]] = None,
    ):
        """
        Args:
            device: Torch device
            countries: List of country codes (e.g., ['DE', 'FR', ...])
            n_shared_fundamentals: Number of shared fundamental nodes
                (e.g., gas price, carbon price, oil price) appended after
                market nodes. These are common inputs to all markets.
            lag: Temporal lag for causal structure
            d_dim: Number of regimes
            tau_gumbel: Gumbel-Softmax temperature
            clusters: Dict mapping cluster name to list of country codes.
                Countries not in any cluster get their own singleton cluster.
            init_logits: Initial logits for edge sparsity
        """
        super().__init__()

        self.device = device
        self.countries = countries
        self.n_countries = len(countries)
        self.n_shared = n_shared_fundamentals
        self.num_nodes = self.n_countries + self.n_shared
        self.lag = lag
        self.d_dim = d_dim

        # Country index mapping
        self.country_to_idx = {c: i for i, c in enumerate(countries)}

        # Cluster assignments
        if clusters is None:
            clusters = DEFAULT_CLUSTERS
        self.clusters = clusters
        self.cluster_names = sorted(clusters.keys())
        self.country_to_cluster = {}
        for cluster_name, members in clusters.items():
            for c in members:
                if c in self.country_to_idx:
                    self.country_to_cluster[c] = cluster_name
        # Assign unclustered countries to singleton clusters
        for c in countries:
            if c not in self.country_to_cluster:
                self.country_to_cluster[c] = f'singleton_{c}'
                self.cluster_names.append(f'singleton_{c}')

        self.n_clusters = len(self.cluster_names)
        self.cluster_to_idx = {c: i for i, c in enumerate(self.cluster_names)}

        if init_logits is None:
            init_logits = [-0.5, -0.5]

        # Level 1: Global shared DAG (common across all markets and regimes)
        self.global_dag = VarDistA_Temporal(
            device=device,
            num_nodes=self.num_nodes,
            lag=lag,
            tau_gumbel=tau_gumbel,
            init_logits=init_logits,
        )

        # Level 2: Per-cluster DAGs (regional dynamics)
        cluster_init = [i + 1.0 for i in init_logits]  # Sparser than global
        self.cluster_dags = nn.ModuleDict({
            name: VarDistA_Temporal(
                device=device,
                num_nodes=self.num_nodes,
                lag=lag,
                tau_gumbel=tau_gumbel,
                init_logits=cluster_init,
            )
            for name in self.cluster_names
        })

        # Level 3: Per-regime DAGs (regime-specific deviations)
        regime_init = [i + 1.5 for i in init_logits]  # Sparsest level
        self.regime_dags = nn.ModuleList([
            VarDistA_Temporal(
                device=device,
                num_nodes=self.num_nodes,
                lag=lag,
                tau_gumbel=tau_gumbel,
                init_logits=regime_init,
            )
            for _ in range(d_dim)
        ])

        # Initialize cross-border edge priors: slightly encourage edges
        # between physically connected markets
        self._init_cross_border_priors()

    def _init_cross_border_priors(self):
        """
        Nudge initial logits to favor edges between physically connected markets.

        Adds a small positive bias to the edge logits for known interconnections,
        making the model more likely to discover real cross-border relationships.
        """
        bias = 0.3  # Small encouragement, not hard constraint
        for c_from, c_to in CROSS_BORDER_PAIRS:
            if c_from in self.country_to_idx and c_to in self.country_to_idx:
                i = self.country_to_idx[c_from]
                j = self.country_to_idx[c_to]
                # Bias the lagged edge logits (edge-exists class)
                self.global_dag.logits_lag.data[1, :, i, j] += bias
                self.global_dag.logits_lag.data[1, :, j, i] += bias

    def get_dag(
        self,
        country: str,
        regime_id: int,
        sample: bool = True,
    ) -> torch.Tensor:
        """
        Get the combined hierarchical DAG for a specific country and regime.

        A(c,d) = 1 - (1 - A_global) * (1 - A_cluster[c]) * (1 - A_regime[d])

        Args:
            country: Country code
            regime_id: Regime index
            sample: Use Gumbel-Softmax sampling

        Returns:
            A: [lag+1, num_nodes, num_nodes] combined adjacency
        """
        if sample:
            A_global = self.global_dag.sample_A()
            cluster_name = self.country_to_cluster[country]
            A_cluster = self.cluster_dags[cluster_name].sample_A()
            A_regime = self.regime_dags[regime_id].sample_A()
        else:
            A_global = self.global_dag.get_adj_matrix()
            cluster_name = self.country_to_cluster[country]
            A_cluster = self.cluster_dags[cluster_name].get_adj_matrix()
            A_regime = self.regime_dags[regime_id].get_adj_matrix()

        # Three-way probabilistic OR
        A = 1.0 - (1.0 - A_global) * (1.0 - A_cluster) * (1.0 - A_regime)
        return A

    def get_dag_with_penalty(
        self,
        country: str,
        regime_id: int,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get combined DAG and its acyclicity penalty.

        Returns:
            A: Combined adjacency matrix
            dag_penalty: DAGness penalty on the combined instantaneous graph
        """
        A = self.get_dag(country, regime_id, sample)
        penalty = dagness_factor(A)
        return A, penalty

    def get_cross_border_edges(
        self,
        regime_id: int = None,
        threshold: float = 0.3,
    ) -> List[Tuple[str, str, float]]:
        """
        Extract learned cross-border edges as interpretable (from, to, weight) triples.

        Args:
            regime_id: If None, use global DAG only. If specified, combine all levels.
            threshold: Minimum edge probability to report

        Returns:
            List of (country_from, country_to, edge_prob) tuples
        """
        if regime_id is not None:
            # Average over countries in each cluster
            edges = []
            for country in self.countries:
                A = self.get_dag(country, regime_id, sample=False)
                for c_from in self.countries:
                    for c_to in self.countries:
                        if c_from == c_to:
                            continue
                        i = self.country_to_idx[c_from]
                        j = self.country_to_idx[c_to]
                        # Sum instantaneous + lagged edge probabilities
                        prob = A[:, i, j].sum().item()
                        if prob > threshold:
                            edges.append((c_from, c_to, prob))
            # Deduplicate by averaging
            edge_dict = {}
            for fr, to, p in edges:
                key = (fr, to)
                if key not in edge_dict:
                    edge_dict[key] = []
                edge_dict[key].append(p)
            return [(fr, to, np.mean(ps)) for (fr, to), ps in edge_dict.items()
                    if np.mean(ps) > threshold]
        else:
            A = self.global_dag.get_adj_matrix()
            edges = []
            for c_from in self.countries:
                for c_to in self.countries:
                    if c_from == c_to:
                        continue
                    i = self.country_to_idx[c_from]
                    j = self.country_to_idx[c_to]
                    prob = A[:, i, j].sum().item()
                    if prob > threshold:
                        edges.append((c_from, c_to, prob))
            return edges

    def sparsity_penalty(self) -> torch.Tensor:
        """L1 sparsity penalty across all hierarchy levels."""
        penalty = torch.tensor(0.0, device=self.device)

        A_global = self.global_dag.get_adj_matrix()
        penalty += A_global.abs().sum()

        for dag in self.cluster_dags.values():
            penalty += dag.get_adj_matrix().abs().sum()

        for dag in self.regime_dags:
            penalty += dag.get_adj_matrix().abs().sum()

        return penalty

    def entropy(self) -> torch.Tensor:
        """Total entropy across all hierarchy levels."""
        total = self.global_dag.entropy()
        for dag in self.cluster_dags.values():
            total += dag.entropy()
        for dag in self.regime_dags:
            total += dag.entropy()
        return total


if __name__ == "__main__":
    device = torch.device("cpu")
    countries = ['DE', 'FR', 'NL', 'BE', 'AT', 'IT', 'ES', 'PL', 'DK', 'SE', 'HU', 'CZ']

    print("Testing HierarchicalMarketDAG")
    print(f"Countries: {countries}")
    print(f"Nodes: {len(countries)} markets + 3 shared fundamentals = {len(countries) + 3}")

    hdag = HierarchicalMarketDAG(
        device=device,
        countries=countries,
        n_shared_fundamentals=3,
        lag=1,
        d_dim=2,
    )

    total_params = sum(p.numel() for p in hdag.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Clusters: {hdag.cluster_names}")

    for d in range(2):
        A, penalty = hdag.get_dag_with_penalty('DE', regime_id=d)
        print(f"\nRegime {d} DAG for DE:")
        print(f"  Shape: {A.shape}")
        print(f"  Instantaneous edges: {A[0].sum():.1f}")
        print(f"  Lagged edges: {A[1:].sum():.1f}")
        print(f"  DAG penalty: {penalty:.4f}")

    # Cross-border edges
    print("\nGlobal cross-border edges (>0.3):")
    edges = hdag.get_cross_border_edges(regime_id=None, threshold=0.3)
    for fr, to, prob in sorted(edges, key=lambda x: -x[2])[:10]:
        print(f"  {fr} -> {to}: {prob:.3f}")

    print("\nRegime 0 cross-border edges (>0.3):")
    edges_r0 = hdag.get_cross_border_edges(regime_id=0, threshold=0.3)
    for fr, to, prob in sorted(edges_r0, key=lambda x: -x[2])[:10]:
        print(f"  {fr} -> {to}: {prob:.3f}")

    # Gradient test
    print("\nGradient flow test...")
    opt = torch.optim.Adam(hdag.parameters(), lr=0.01)
    for step in range(3):
        opt.zero_grad()
        loss = hdag.sparsity_penalty()
        for d in range(2):
            _, pen = hdag.get_dag_with_penalty('DE', d)
            loss += pen
        loss.backward()
        opt.step()
        print(f"  Step {step}: loss={loss.item():.4f}")

    print("\nHierarchicalMarketDAG test passed!")
