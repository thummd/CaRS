"""
Visualize DS3M regime switching with per-regime causal DAGs.

Generates:
1. Time series with regime background coloring
2. Regime probability/assignment timeline
3. Per-regime DAG visualization (side by side)

Output format: SVG (vector, scalable)

Usage:
    python plot_ds3m_regime_dag.py --results_dir outputs/ds3m_dag/ALL_d2_20260119_...
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Try to import networkx, fall back gracefully
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not available, DAG visualization will be simplified")

# Style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

REGIME_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def load_results(results_dir: Path) -> Dict:
    """Load results from ds3m_with_dag output."""
    with open(results_dir / 'results.json', 'r') as f:
        results = json.load(f)

    results['test_regimes'] = np.load(results_dir / 'test_regimes.npy')
    results['train_regimes'] = np.load(results_dir / 'train_regimes.npy')
    results['full_regimes'] = np.load(results_dir / 'full_regimes.npy')

    # Load regime sequences if available
    if (results_dir / 'test_regime_sequence.npy').exists():
        results['test_regime_sequence'] = np.load(results_dir / 'test_regime_sequence.npy')
    if (results_dir / 'train_regime_sequence.npy').exists():
        results['train_regime_sequence'] = np.load(results_dir / 'train_regime_sequence.npy')

    # Load DAGs if present
    results['dags'] = {}
    for f in results_dir.glob('dag_regime_*.npy'):
        regime_id = int(f.stem.split('_')[-1])
        results['dags'][regime_id] = np.load(f)

    return results


def load_raw_data(data_path: str = str(DATA_DIR / "qrt")) -> pd.DataFrame:
    """Load raw QRT data."""
    data_path = Path(data_path)
    X_train = pd.read_csv(data_path / 'X_train_NHkHMNU.csv')
    Y_train = pd.read_csv(data_path / 'y_train_ZAN5mwg.csv')
    df = X_train.merge(Y_train, on='ID').sort_values('DAY_ID').reset_index(drop=True)
    return df


def get_regime_segments(regime_assignments: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Find contiguous segments of the same regime.

    Returns list of (start_idx, end_idx, regime_id) tuples.
    """
    segments = []
    if len(regime_assignments) == 0:
        return segments

    current_regime = regime_assignments[0]
    start_idx = 0

    for i in range(1, len(regime_assignments)):
        if regime_assignments[i] != current_regime:
            segments.append((start_idx, i - 1, current_regime))
            current_regime = regime_assignments[i]
            start_idx = i

    segments.append((start_idx, len(regime_assignments) - 1, current_regime))
    return segments


def plot_regime_timeseries(
    df: pd.DataFrame,
    regime_assignments: np.ndarray,
    output_path: Path,
    variables: List[str] = None,
    title: str = "Regime Switching Time Series",
    country: str = None
):
    """
    Plot time series with regime background coloring.

    Creates a figure with:
    - Top panels: Variables over time with regime background
    - Bottom panel: Regime assignment timeline
    """
    n_regimes = len(np.unique(regime_assignments[regime_assignments >= 0]))
    colors = REGIME_COLORS[:max(n_regimes, 2)]

    # Filter data if country specified
    if country and country != 'ALL':
        df = df[df['COUNTRY'] == country].reset_index(drop=True)

    if variables is None:
        variables = ['TARGET']

    n_samples = min(len(df), len(regime_assignments))
    n_vars = len(variables)

    # Create figure
    fig = plt.figure(figsize=(14, 3 * (n_vars + 1)))
    gs = GridSpec(n_vars + 1, 1, height_ratios=[3] * n_vars + [1], hspace=0.1)

    x = np.arange(n_samples)

    # Plot each variable with regime background
    for i, var in enumerate(variables):
        ax = fig.add_subplot(gs[i])

        if var not in df.columns:
            ax.text(0.5, 0.5, f"Variable '{var}' not found",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(var)
            continue

        values = df[var].values[:n_samples]

        # Add regime background coloring
        segments = get_regime_segments(regime_assignments[:n_samples])
        for start, end, regime in segments:
            if regime >= 0 and regime < len(colors):
                ax.axvspan(start, end + 1, alpha=0.3, color=colors[regime], linewidth=0)

        # Plot the time series
        ax.plot(x, values, 'k-', linewidth=0.8, alpha=0.9)
        ax.set_ylabel(var)
        ax.set_xlim(0, n_samples - 1)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        if i == 0:
            ax.set_title(title, fontweight='bold')

        if i < n_vars - 1:
            ax.set_xticklabels([])

    # Regime timeline (bottom)
    ax_regime = fig.add_subplot(gs[-1])
    ax_regime.step(x, regime_assignments[:n_samples], where='post',
                   color='navy', linewidth=1.5)
    ax_regime.set_ylabel('Regime')
    ax_regime.set_xlabel('Sample Index')
    ax_regime.set_ylim(-0.5, n_regimes - 0.5)
    ax_regime.set_yticks(range(n_regimes))
    ax_regime.set_xlim(0, n_samples - 1)
    ax_regime.grid(True, alpha=0.3, linewidth=0.5)

    # Fill regime background in timeline too
    segments = get_regime_segments(regime_assignments[:n_samples])
    for start, end, regime in segments:
        if regime >= 0 and regime < len(colors):
            ax_regime.axvspan(start, end + 1, alpha=0.3, color=colors[regime], linewidth=0)

    # Legend
    patches = [mpatches.Patch(color=colors[i], alpha=0.3, label=f'Regime {i}')
               for i in range(n_regimes)]
    fig.legend(handles=patches, loc='upper right', bbox_to_anchor=(0.99, 0.99))

    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_regime_distribution(
    results: Dict,
    output_path: Path
):
    """Plot bar chart of regime distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    datasets = ['train', 'test', 'full']
    titles = ['Train Set', 'Test Set', 'Full Dataset']

    for ax, dataset, title in zip(axes, datasets, titles):
        dist = results.get(f'regime_distribution_{dataset}', {})
        if not dist:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        regimes = sorted([int(k) for k in dist.keys()])
        counts = [dist[str(r)] for r in regimes]
        total = sum(counts)
        percentages = [100 * c / total for c in counts]

        colors = [REGIME_COLORS[r] for r in regimes]
        bars = ax.bar(regimes, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add count labels
        for bar, count, pct in zip(bars, counts, percentages):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Regime')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.set_xticks(regimes)

    plt.suptitle(f"Regime Distribution: {results.get('country', 'ALL')}", fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def draw_dag_networkx(
    A: np.ndarray,
    feature_names: List[str],
    ax: plt.Axes,
    target_idx: int = -1,
    threshold: float = 0.3,
    title: str = "",
    max_edges: int = 15
):
    """Draw a causal DAG using networkx focusing on edges to TARGET."""
    if not HAS_NETWORKX:
        ax.text(0.5, 0.5, "networkx not available", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        return

    if target_idx < 0:
        target_idx = len(feature_names) + target_idx

    target_name = feature_names[target_idx]
    G = nx.DiGraph()
    G.add_node(target_name)

    # Find edges TO target (sorted by weight)
    edges_to_target = []
    for lag in range(A.shape[0]):
        for i in range(A.shape[1]):
            if i != target_idx and abs(A[lag, i, target_idx]) > threshold:
                weight = A[lag, i, target_idx]
                edges_to_target.append((feature_names[i], weight, lag))

    # Sort by absolute weight and take top edges
    edges_to_target.sort(key=lambda x: abs(x[1]), reverse=True)
    edges_to_target = edges_to_target[:max_edges]

    if not edges_to_target:
        ax.text(0.5, 0.5, "No significant edges\nabove threshold",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        return

    # Add nodes and edges
    for name, weight, lag in edges_to_target:
        G.add_node(name)
        edge_label = f"({lag})" if lag > 0 else ""
        G.add_edge(name, target_name, weight=weight, lag=lag, label=edge_label)

    # Layout: TARGET at bottom center, parents arranged in arc above
    pos = {}
    pos[target_name] = (0.5, 0.15)

    parents = [n for n in G.nodes() if n != target_name]
    n_parents = len(parents)
    for i, parent in enumerate(parents):
        # Arrange in an arc
        if n_parents == 1:
            angle = np.pi / 2
        else:
            angle = np.pi * (0.15 + 0.7 * i / (n_parents - 1))
        radius = 0.35
        pos[parent] = (0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle) * 0.6)

    # Node colors
    node_colors = ['lightcoral' if n == target_name else 'lightblue' for n in G.nodes()]

    # Edge widths based on absolute weight
    weights = [abs(G[u][v]['weight']) for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    edge_widths = [1 + 2 * w / max_weight for w in weights]

    # Draw
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1200, node_color=node_colors,
                          edgecolors='black', linewidths=0.5)

    # Truncate long names for display
    labels = {n: n[:12] + '...' if len(n) > 15 else n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, edge_color='gray',
                          width=edge_widths, connectionstyle="arc3,rad=0.1",
                          arrowsize=12, alpha=0.8)

    ax.set_title(title)
    ax.axis('off')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)


def draw_correlation_box(
    info: Dict,
    feature_names: List[str],
    ax: plt.Axes,
    title: str = ""
):
    """Draw a text box showing correlation analysis results."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    text_lines = [title, f"({info.get('n_samples', 0)} samples)", ""]

    if info.get('type') == 'correlation' or info.get('type') == 'correlation_fallback':
        text_lines.append("Top correlations with TARGET:")
        text_lines.append("")
        for name, corr in info.get('top_correlations', [])[:8]:
            # Truncate long names
            short_name = name[:20] + '...' if len(name) > 23 else name
            text_lines.append(f"  {short_name}: {corr:+.3f}")
    elif info.get('type') == 'error':
        text_lines.append(f"Error: {info.get('error', 'Unknown')}")
    elif info.get('type') == 'skipped':
        text_lines.append("Analysis skipped")
    else:
        text_lines.append("No analysis available")

    text = '\n'.join(text_lines)
    ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes,
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')


def plot_regime_dags(
    results: Dict,
    output_path: Path,
    threshold: float = 0.3
):
    """Plot per-regime DAGs or correlation analysis side by side."""
    dags = results.get('dags', {})
    regime_analysis = results.get('regime_analysis', {})
    feature_names = results.get('fantom_feature_cols', results.get('feature_cols', []))
    d_dim = results.get('d_dim', 2)

    fig, axes = plt.subplots(1, d_dim, figsize=(7 * d_dim, 6))
    if d_dim == 1:
        axes = [axes]

    for regime_id in range(d_dim):
        ax = axes[regime_id]
        info = regime_analysis.get(str(regime_id), {})
        n_samples = info.get('n_samples', 0)
        title = f"Regime {regime_id}\n({n_samples} samples)"

        if regime_id in dags:
            A = dags[regime_id]
            draw_dag_networkx(A, feature_names, ax, target_idx=-1,
                             threshold=threshold, title=title)
        else:
            draw_correlation_box(info, feature_names, ax, title=title)

    plt.suptitle(f"Per-Regime Causal Analysis: {results.get('country', 'ALL')}",
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_regime_transitions(
    regime_assignments: np.ndarray,
    output_path: Path,
    title: str = "Regime Transitions"
):
    """Plot regime transition matrix and change points."""
    n_regimes = len(np.unique(regime_assignments[regime_assignments >= 0]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Transition matrix
    ax1 = axes[0]
    transitions = np.zeros((n_regimes, n_regimes))
    for i in range(len(regime_assignments) - 1):
        r_from = regime_assignments[i]
        r_to = regime_assignments[i + 1]
        if r_from >= 0 and r_to >= 0:
            transitions[r_from, r_to] += 1

    # Normalize rows
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = transitions / row_sums

    im = ax1.imshow(trans_prob, cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks(range(n_regimes))
    ax1.set_yticks(range(n_regimes))
    ax1.set_xticklabels([f'Regime {i}' for i in range(n_regimes)])
    ax1.set_yticklabels([f'Regime {i}' for i in range(n_regimes)])
    ax1.set_xlabel('To')
    ax1.set_ylabel('From')
    ax1.set_title('Transition Probabilities')

    # Add text annotations
    for i in range(n_regimes):
        for j in range(n_regimes):
            ax1.text(j, i, f'{trans_prob[i, j]:.2f}', ha='center', va='center',
                     color='white' if trans_prob[i, j] > 0.5 else 'black')

    fig.colorbar(im, ax=ax1, shrink=0.8)

    # Regime change histogram
    ax2 = axes[1]
    change_points = []
    for i in range(1, len(regime_assignments)):
        if regime_assignments[i] != regime_assignments[i - 1]:
            change_points.append(i)

    if change_points:
        ax2.hist(change_points, bins=min(50, len(change_points)), color='navy', alpha=0.7)
        ax2.axvline(np.mean(change_points), color='red', linestyle='--', label=f'Mean: {np.mean(change_points):.0f}')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Regime Change Points ({len(change_points)} transitions)')
    ax2.legend()

    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DS3M regime + DAG results")
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to ds3m_with_dag output directory')
    parser.add_argument('--data_path', type=str,
                        default=str(DATA_DIR / "qrt"),
                        help='Path to raw data')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Edge threshold for DAG visualization')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)

    country = results.get('country', 'ALL')
    print(f"Country: {country}")
    print(f"Regimes: {results.get('d_dim', 2)}")

    # Load raw data for plotting
    print("\nLoading raw data...")
    df = load_raw_data(args.data_path)

    # Filter by country if needed
    if country != 'ALL':
        df = df[df['COUNTRY'] == country].reset_index(drop=True)

    print(f"Data samples: {len(df)}")

    # Create output directory
    output_dir = results_dir / 'figures'
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\nGenerating visualizations...")

    # 1. Time series with regimes
    print("  - Regime time series...")
    plot_regime_timeseries(
        df, results['full_regimes'],
        output_dir / 'regime_timeseries.svg',
        variables=['TARGET'],
        title=f'{country} Electricity Market - Regime Switching',
        country=country
    )

    # 2. Regime distribution
    print("  - Regime distribution...")
    plot_regime_distribution(results, output_dir / 'regime_distribution.svg')

    # 3. Per-regime DAGs
    print("  - Per-regime DAGs...")
    plot_regime_dags(results, output_dir / 'regime_dags.svg', threshold=args.threshold)

    # 4. Transition analysis
    print("  - Regime transitions...")
    plot_regime_transitions(
        results['full_regimes'],
        output_dir / 'regime_transitions.svg',
        title=f'{country} Regime Transitions'
    )

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
