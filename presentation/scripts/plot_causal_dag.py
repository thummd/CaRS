#!/usr/bin/env python3
"""
Generate Causal DAG visualizations for CaRS results.

Creates directed graph visualizations showing learned causal structure:
- Nodes = features
- Edges = causal relationships with weights
- Blue = instantaneous edges
- Orange = lagged edges (t-1 → t)

Usage:
    python plot_causal_dag.py
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple

# Add CASTOR electricity to path for unified data loading
CASTOR_ELECTRICITY_DIR = '/lustre/home/dthumm/CASTOR/electricity'
sys.path.insert(0, CASTOR_ELECTRICITY_DIR)

# Directories
RESULTS_DIR = Path(__file__).parent.parent / "results" / "cars"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# Style settings
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'figure.dpi': 150,
})

# Colors
INSTANTANEOUS_COLOR = '#1f77b4'  # Blue
LAGGED_COLOR = '#ff7f0e'  # Orange
TARGET_COLOR = '#d62728'  # Red for price node

# Feature name shortening for readability
NAME_MAPPING = {
    'Day_Ahead_Price': 'Price',
    'DE_Day_Ahead_Price': 'DE Price',
    'FR_Day_Ahead_Price': 'FR Price',
    'Fossil Brown coal/Lignite_Actual Aggregated': 'Lignite',
    'Fossil Gas_Actual Aggregated': 'Gas',
    'Fossil Hard coal_Actual Aggregated': 'Hard Coal',
    'Hydro Pumped Storage_Actual Aggregated': 'Hydro Pumped',
    'Nuclear_Actual Aggregated': 'Nuclear',
    'Solar_Actual Aggregated': 'Solar',
    'Wind Onshore_Actual Aggregated': 'Wind Onshore',
    'Wind Offshore_Actual Aggregated': 'Wind Offshore',
    'Actual Load': 'Load',
    'DE_temperature_2m': 'Temperature',
    'DE_wind_speed_10m': 'Wind 10m',
    'DE_wind_speed_100m': 'Wind 100m',
    'DE_shortwave_radiation': 'Radiation',
    'DE_cloud_cover': 'Cloud Cover',
    'DE_precipitation': 'Precipitation',
    'Flow_to_FR': 'Flow to FR',
    'Flow_from_FR': 'Flow from FR',
    'Net_Flow_FR': 'Net Flow FR',
    'commodity_natural_gas': 'Nat Gas Price',
    'commodity_brent_oil': 'Brent Oil',
    # FR variants
    'FR_temperature_2m': 'Temperature',
    'FR_wind_speed_10m': 'Wind 10m',
    'FR_wind_speed_100m': 'Wind 100m',
    'FR_shortwave_radiation': 'Radiation',
    'FR_cloud_cover': 'Cloud Cover',
    'FR_precipitation': 'Precipitation',
    # DE_FR dataset variants
    'DE_Fossil Brown coal/Lignite_Actual Aggregated': 'DE Lignite',
    'DE_Fossil Gas_Actual Aggregated': 'DE Gas',
    'DE_Fossil Hard coal_Actual Aggregated': 'DE Hard Coal',
    'DE_Hydro Pumped Storage_Actual Aggregated': 'DE Hydro Pumped',
    'DE_Nuclear_Actual Aggregated': 'DE Nuclear',
    'DE_Solar_Actual Aggregated': 'DE Solar',
    'DE_Wind Onshore_Actual Aggregated': 'DE Wind Onshore',
    'DE_Wind Offshore_Actual Aggregated': 'DE Wind Offshore',
    'DE_Actual Load': 'DE Load',
    'FR_Fossil Gas_Actual Aggregated': 'FR Gas',
    'FR_Fossil Hard coal_Actual Aggregated': 'FR Hard Coal',
    'FR_Hydro Pumped Storage_Actual Aggregated': 'FR Hydro Pumped',
    'FR_Nuclear_Actual Aggregated': 'FR Nuclear',
    'FR_Solar_Actual Aggregated': 'FR Solar',
    'FR_Wind Onshore_Actual Aggregated': 'FR Wind Onshore',
    'FR_Wind Offshore_Actual Aggregated': 'FR Wind Offshore',
    'FR_Actual Load': 'FR Load',
    'DE_DE_temperature_2m': 'DE Temp',
    'DE_DE_wind_speed_10m': 'DE Wind 10m',
    'DE_DE_wind_speed_100m': 'DE Wind 100m',
    'DE_DE_shortwave_radiation': 'DE Radiation',
    'DE_DE_cloud_cover': 'DE Cloud',
    'DE_DE_precipitation': 'DE Precip',
    'FR_FR_temperature_2m': 'FR Temp',
    'FR_FR_wind_speed_10m': 'FR Wind 10m',
    'FR_FR_wind_speed_100m': 'FR Wind 100m',
    'FR_FR_shortwave_radiation': 'FR Radiation',
    'FR_FR_cloud_cover': 'FR Cloud',
    'FR_FR_precipitation': 'FR Precip',
}


def prune_instantaneous_cycles(adj_matrix: np.ndarray) -> np.ndarray:
    """Remove cycles from instantaneous adjacency by keeping the stronger direction.

    Compares edges by absolute value to handle signed causal coefficients.
    """
    A = adj_matrix.copy()
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > 0 and abs(A[j, i]) > 0:
                if abs(A[i, j]) >= abs(A[j, i]):
                    A[j, i] = 0.0
                else:
                    A[i, j] = 0.0
    return A


def shorten_name(name: str) -> str:
    """Shorten feature name for display."""
    return NAME_MAPPING.get(name, name.replace('_Actual Aggregated', '').replace('_', ' '))


def load_result(dataset: str, d: int = 2, lag: int = 1, ls: float = 5.0, seed: int = 42) -> Dict:
    """Load result JSON for specified configuration."""
    filename = f"cars_{dataset.lower()}_d{d}_lag{lag}_ls{ls}_seed{seed}_ds3m_causal_native_markov.json"
    filepath = RESULTS_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Result not found: {filepath}")

    with open(filepath) as f:
        return json.load(f)


def should_swap_regimes(result: Dict, dataset: str) -> bool:
    """
    Determine if regime labels should be swapped for consistency.

    Uses same logic as normalize_regime_labels() in generate_appendix_plots.py:
    Regime 0 should have lower variance (stable), Regime 1 higher variance (crisis).
    Returns True if regime labels need to be swapped.
    """
    try:
        from unified_data_loader import load_unified_dataset
        df = load_unified_dataset(dataset, clean=True)
    except ImportError:
        print(f"Warning: Could not load unified dataset for {dataset}")
        return False

    assignments = result.get('regime_assignments', [])
    if not assignments:
        return False

    n_regimes = len(result.get('adjacency_matrices', []))
    if n_regimes < 2:
        return False

    # Determine price column
    if 'Day_Ahead_Price' in df.columns:
        price_col = 'Day_Ahead_Price'
    elif 'DE_Day_Ahead_Price' in df.columns:
        price_col = 'DE_Day_Ahead_Price'
    else:
        price_col = df.columns[0]

    lag = result.get('config', {}).get('lag', 1)
    prices = df[price_col].values[lag:lag + len(assignments)]

    # Calculate variance per regime
    variances = {}
    for regime in range(n_regimes):
        mask = np.array(assignments) == regime
        valid_mask = mask[:len(prices)]
        if valid_mask.sum() > 0:
            variances[regime] = np.var(prices[valid_mask])

    # Swap if regime 0 has higher variance than regime 1
    if n_regimes == 2 and len(variances) == 2:
        if variances.get(0, 0) > variances.get(1, 0):
            print(f"    Regime swap needed for {dataset} (var0={variances[0]:.2f} > var1={variances[1]:.2f})")
            return True

    return False


def extract_edges(
    adj_matrix: np.ndarray,
    feature_names: List[str],
    top_k: int = 15,
    min_weight: float = 0.0
) -> List[Tuple[str, str, float]]:
    """
    Extract top-K edges from adjacency matrix by absolute weight.

    Handles signed causal coefficients (W * A). Filters by absolute value
    and sorts by absolute value descending.

    Returns list of (source, target, weight) tuples.
    """
    n = adj_matrix.shape[0]
    edges = []

    for i in range(n):
        for j in range(n):
            if i != j and abs(adj_matrix[i, j]) > min_weight:
                edges.append((feature_names[i], feature_names[j], float(adj_matrix[i, j])))

    # Sort by absolute weight and take top-K
    edges.sort(key=lambda x: -abs(x[2]))
    return edges[:top_k]


def extract_edges_to_target(
    adj_matrix: np.ndarray,
    feature_names: List[str],
    target_idx: int = 0,
    top_k: int = 10,
    min_weight: float = 0.0
) -> List[Tuple[str, str, float]]:
    """
    Extract edges TO the target node (column target_idx of adjacency matrix).

    Handles signed causal coefficients. Sorts by absolute weight.

    Returns list of (source, target, weight) tuples sorted by absolute weight.
    """
    n = adj_matrix.shape[0]
    edges = []
    target_name = feature_names[target_idx]

    for i in range(n):
        if i != target_idx and abs(adj_matrix[i, target_idx]) > min_weight:
            edges.append((feature_names[i], target_name, float(adj_matrix[i, target_idx])))

    # Sort by absolute weight and take top-K
    edges.sort(key=lambda x: -abs(x[2]))
    return edges[:top_k]


def create_dag_figure(
    result: Dict,
    regime: int,
    output_path: Path,
    dataset: str = '',
    top_k_instant: int = 15,
    top_k_lagged: int = 10,
    target_constrained: bool = True,
    show_edge_weights: bool = True,
    display_regime: int = None
):
    """
    Create DAG visualization for a specific regime.

    Args:
        result: Loaded JSON result
        regime: Regime index (0 or 1)
        output_path: Where to save the figure
        top_k_instant: Number of top instantaneous edges to show
        top_k_lagged: Number of top lagged edges to show
        target_constrained: If True, show 1-hop neighborhood around Price
        show_edge_weights: If True, display edge weight labels
    """
    adj_matrices = result['adjacency_matrices']
    feature_names = result['feature_names']

    # Backward compatibility: old results have (n-1)-node adj but n feature_names
    # (the target variable was excluded from the adj matrix). Trim feature_names to match.
    n_adj = len(adj_matrices[regime][0])
    if n_adj < len(feature_names):
        feature_names = feature_names[:n_adj]

    # Get adjacency matrices for this regime
    # Structure: adj_matrices[regime][lag_level][from][to]
    instant_adj = prune_instantaneous_cycles(np.array(adj_matrices[regime][0]))
    lagged_adj = np.array(adj_matrices[regime][1]) if len(adj_matrices[regime]) > 1 else None

    # Compute adaptive min_weight threshold using absolute values for signed coefficients.
    # Only apply when graph is near-complete (density > 50%), i.e. most edges are noise.
    # For already-sparse graphs, the learned weights are meaningful — use top-K only.
    n_features = len(feature_names)
    max_possible = n_features * (n_features - 1)

    abs_instant = np.abs(instant_adj)
    nonzero_instant = abs_instant[abs_instant > 1e-10]
    instant_density = len(nonzero_instant) / max_possible if max_possible > 0 else 0
    if instant_density > 0.5:
        min_weight_instant = float(np.mean(nonzero_instant) + np.std(nonzero_instant))
    else:
        min_weight_instant = 0.0

    if lagged_adj is not None:
        abs_lagged = np.abs(lagged_adj)
        nonzero_lagged = abs_lagged[abs_lagged > 1e-10]
        max_possible_lagged = n_features * n_features  # lagged allows self-edges
        lagged_density = len(nonzero_lagged) / max_possible_lagged if max_possible_lagged > 0 else 0
        min_weight_lagged = float(np.mean(nonzero_lagged) + np.std(nonzero_lagged)) if lagged_density > 0.5 else 0.0
    else:
        min_weight_lagged = 0.0

    total_instant = int((abs_instant > 1e-10).sum())
    surviving_instant = int((abs_instant > min_weight_instant).sum())
    print(f"    Regime {regime}: instant density={instant_density:.1%}, "
          f"threshold={'on' if instant_density > 0.5 else 'off'} (mean+1std={min_weight_instant:.6f}), "
          f"{surviving_instant}/{total_instant} edges survive")
    if lagged_adj is not None:
        total_lagged = int((abs_lagged > 1e-10).sum())
        surviving_lagged = int((abs_lagged > min_weight_lagged).sum())
        print(f"    Regime {regime}: lagged density={lagged_density:.1%}, "
              f"threshold={'on' if lagged_density > 0.5 else 'off'}, "
              f"{surviving_lagged}/{total_lagged} edges survive")

    # Find all price target indices (DE_FR has two: DE_Day_Ahead_Price and FR_Day_Ahead_Price)
    target_indices = []
    target_nodes = []  # Shortened names for the targets
    for i, name in enumerate(feature_names):
        if 'Price' in name or 'price' in name:
            target_indices.append(i)
            target_nodes.append(shorten_name(name))

    # Fallback if no price found
    if not target_indices:
        target_indices = [0]
        target_nodes = [shorten_name(feature_names[0])]

    # Extract edges: use top-K for structural control, threshold for noise filtering
    # For dense graphs (threshold on), pass all surviving edges; for sparse, top-K limits
    effective_top_k_instant = top_k_instant if min_weight_instant == 0.0 else max_possible
    effective_top_k_lagged = top_k_lagged if min_weight_lagged == 0.0 else n_features * n_features
    instant_edges = extract_edges(instant_adj, feature_names, top_k=effective_top_k_instant, min_weight=min_weight_instant)
    lagged_edges = extract_edges(lagged_adj, feature_names, top_k=effective_top_k_lagged, min_weight=min_weight_lagged) if lagged_adj is not None else []

    # Also extract edges specifically TO all price targets
    effective_to_price_k = 10 if min_weight_instant == 0.0 else n_features
    instant_edges_to_price = []
    lagged_edges_to_price = []
    for target_idx in target_indices:
        instant_edges_to_price.extend(extract_edges_to_target(instant_adj, feature_names, target_idx=target_idx, top_k=effective_to_price_k, min_weight=min_weight_instant))
        if lagged_adj is not None:
            effective_to_price_lagged_k = 5 if min_weight_lagged == 0.0 else n_features
            lagged_edges_to_price.extend(extract_edges_to_target(lagged_adj, feature_names, target_idx=target_idx, top_k=effective_to_price_lagged_k, min_weight=min_weight_lagged))

    # Build full graph first
    G_full = nx.DiGraph()

    # Add all nodes
    for name in feature_names:
        G_full.add_node(shorten_name(name))

    # Add instantaneous edges
    for src, tgt, weight in instant_edges:
        src_short = shorten_name(src)
        tgt_short = shorten_name(tgt)
        if src_short != tgt_short:
            G_full.add_edge(src_short, tgt_short, weight=weight, edge_type='instant')

    # Add edges TO Price (instantaneous) - explicitly include these
    for src, tgt, weight in instant_edges_to_price:
        src_short = shorten_name(src)
        tgt_short = shorten_name(tgt)
        if src_short != tgt_short and not G_full.has_edge(src_short, tgt_short):
            G_full.add_edge(src_short, tgt_short, weight=weight, edge_type='instant')

    # Add lagged edges
    for src, tgt, weight in lagged_edges:
        src_short = shorten_name(src)
        tgt_short = shorten_name(tgt)
        G_full.add_edge(src_short, tgt_short, weight=weight, edge_type='lagged')

    # Add edges TO Price (lagged) - explicitly include these
    for src, tgt, weight in lagged_edges_to_price:
        src_short = shorten_name(src)
        tgt_short = shorten_name(tgt)
        if src_short != tgt_short and not G_full.has_edge(src_short, tgt_short):
            G_full.add_edge(src_short, tgt_short, weight=weight, edge_type='lagged')

    # If target-constrained and graph is dense, extract 1-hop ego graph around Price.
    # For sparse graphs (already interpretable), show the full graph instead.
    graph_is_sparse = G_full.number_of_edges() <= 50
    if target_constrained and not graph_is_sparse:
        neighbors = set()
        for target_node in target_nodes:
            if target_node in G_full:
                neighbors |= set(G_full.predecessors(target_node)) | set(G_full.successors(target_node))
                neighbors.add(target_node)

        if neighbors:
            # Create subgraph with Price targets + neighbors + edges between them
            G = G_full.subgraph(neighbors).copy()
        else:
            G = G_full.copy()
    else:
        # Use full graph (already sparse enough to be readable)
        G = G_full.copy()

    filter_label = "1-hop filter" if (target_constrained and not graph_is_sparse) else "full (sparse)"
    print(f"    Regime {regime}: {G_full.number_of_edges()} edges in full graph, "
          f"{G.number_of_edges()} after {filter_label} ({G.number_of_nodes()} nodes)")

    # Always ensure Price nodes exist
    for target_node in target_nodes:
        if target_node not in G:
            G.add_node(target_node)

    # Remove isolated nodes EXCEPT Price targets
    isolated = [n for n in nx.isolates(G) if n not in target_nodes]
    G.remove_nodes_from(isolated)

    # Calculate node importance (weighted degree using absolute weights)
    node_importance = {}
    for node in G.nodes():
        in_weight = sum(abs(d['weight']) for _, _, d in G.in_edges(node, data=True))
        out_weight = sum(abs(d['weight']) for _, _, d in G.out_edges(node, data=True))
        # Weight incoming edges more (what affects this node)
        node_importance[node] = in_weight + 0.5 * out_weight

    # Scale node sizes by importance (increased range for better differentiation)
    max_imp = max(node_importance.values()) if node_importance.values() and max(node_importance.values()) > 0 else 1
    node_sizes = []
    for n in G.nodes():
        if n in target_nodes:
            # Price nodes always prominent
            size = max(1000, 300 + 1200 * (node_importance.get(n, 0) / max_imp))
        else:
            size = 300 + 1200 * (node_importance.get(n, 0) / max_imp)
        node_sizes.append(size)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use spring layout with first price node as center
    primary_target = target_nodes[0] if target_nodes else None
    if primary_target in G.nodes():
        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42, center=(0, 0))
        # Center on primary price node
        if primary_target in pos:
            offset = pos[primary_target]
            pos = {k: (v[0] - offset[0], v[1] - offset[1]) for k, v in pos.items()}
    else:
        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Draw nodes with importance-based sizes (price targets in red)
    node_colors = [TARGET_COLOR if n in target_nodes else '#a8d5ba' for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9, edgecolors='#333333', linewidths=1)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

    # Draw edges by type
    instant_edgelist = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'instant']
    lagged_edgelist = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'lagged']

    # Get edge weights for width scaling (use absolute values for signed coefficients)
    all_weights = [abs(d['weight']) for _, _, d in G.edges(data=True)]
    max_w = max(all_weights) if all_weights else 1

    # Color scheme: positive edges = blue/orange, negative edges = red tint
    NEGATIVE_INSTANT_COLOR = '#c44e52'   # Muted red for negative instantaneous
    NEGATIVE_LAGGED_COLOR = '#c44e52'    # Muted red for negative lagged

    if instant_edgelist:
        instant_weights = [G[u][v]['weight'] for u, v in instant_edgelist]
        instant_widths = [1 + 3 * (abs(w) / max_w) for w in instant_weights]
        instant_colors = [INSTANTANEOUS_COLOR if w >= 0 else NEGATIVE_INSTANT_COLOR for w in instant_weights]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=instant_edgelist,
                               edge_color=instant_colors, width=instant_widths,
                               alpha=0.7, arrows=True, arrowsize=15,
                               connectionstyle='arc3,rad=0.1')

    if lagged_edgelist:
        lagged_weights = [G[u][v]['weight'] for u, v in lagged_edgelist]
        lagged_widths = [1 + 3 * (abs(w) / max_w) for w in lagged_weights]
        lagged_colors = [LAGGED_COLOR if w >= 0 else NEGATIVE_LAGGED_COLOR for w in lagged_weights]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=lagged_edgelist,
                               edge_color=lagged_colors, width=lagged_widths,
                               alpha=0.7, arrows=True, arrowsize=15,
                               style='dashed', connectionstyle='arc3,rad=-0.1')

    # Add edge weight labels
    if show_edge_weights and G.number_of_edges() > 0:
        edge_labels = {(u, v): f'{d["weight"]:.4f}' for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                      font_size=6, font_color='#444444',
                                      bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8),
                                      ax=ax)

    # Add legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color=INSTANTANEOUS_COLOR, linewidth=2, label='Instant. (+)'),
        Line2D([0], [0], color=LAGGED_COLOR, linewidth=2, linestyle='--', label='Lagged (+)'),
        Line2D([0], [0], color='#c44e52', linewidth=2, label='Negative (-)'),
        Patch(facecolor=TARGET_COLOR, edgecolor='#333333', label='Price (Target)'),
        Patch(facecolor='#a8d5ba', edgecolor='#333333', label='Features'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    # Add info about node sizes
    ax.text(0.02, 0.02, 'Node size ∝ weighted degree',
            transform=ax.transAxes, fontsize=7, color='gray', style='italic')

    # Use display_regime for title if provided (for label normalization)
    title_regime = display_regime if display_regime is not None else regime
    regime_labels = ['Regime 0 (Stable)', 'Regime 1 (Crisis)']
    label = regime_labels[title_regime] if title_regime < len(regime_labels) else f'Regime {title_regime}'
    title_prefix = f'{dataset} ' if dataset else ''
    ax.set_title(f'{title_prefix}{label}', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 70)
    print("Generating Causal DAG Visualizations")
    print("=" * 70)

    # Ensure output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate DAGs for DE, FR, and DE_FR, both regimes
    # DE_FR uses higher lambda_sparse (50.0) for meaningful sparsity on 31-node graph
    dataset_ls = {'DE': 5.0, 'FR': 5.0, 'DE_FR': 50.0}
    for dataset in ['DE', 'FR', 'DE_FR']:
        ls = dataset_ls[dataset]
        print(f"\n--- {dataset} (ls={ls}) ---")
        try:
            result = load_result(dataset, d=2, lag=1, ls=ls, seed=42)
            n_regimes = len(result['adjacency_matrices'])
            print(f"  Loaded: {n_regimes} regimes, {len(result['feature_names'])} features")

            # Determine if regime labels need swapping for consistency with time series plots
            # - DE: Force swap (adjacency_matrices indices are inverted vs regime_assignments)
            # - DE_FR: Use variance check (var0 > var1 means swap needed)
            # - FR: No swap needed (already aligned)
            if dataset == 'DE':
                swap_regimes = True  # Force swap for DE
            elif dataset == 'DE_FR':
                swap_regimes = should_swap_regimes(result, dataset)  # Variance-based check
            else:
                swap_regimes = False  # FR is already aligned

            if swap_regimes:
                print(f"    Swapping regime labels for {dataset} to match time series plot")

            for regime in range(n_regimes):
                # Apply regime normalization for consistency with time series plots
                if swap_regimes and n_regimes == 2:
                    display_regime = 1 - regime  # Swap: 0→1, 1→0
                else:
                    display_regime = regime

                output_path = FIGURES_DIR / f'causal_dag_{dataset.lower()}_regime{display_regime}.svg'
                create_dag_figure(result, regime, output_path, dataset=dataset, display_regime=display_regime)

        except FileNotFoundError as e:
            print(f"  Error: {e}")
        except Exception as e:
            print(f"  Error generating DAG: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"DAG figures saved to {FIGURES_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
