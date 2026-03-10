#!/usr/bin/env python3
"""
DAG Visualization Script for CaRS Ablation Study

Generates comparison heatmaps showing:
- Independent mode: Regime DAGs (should be identical)
- Noise_init mode: Regime DAGs (should be different)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_dags(results_dir: Path, experiment: str, market: str, seed: int):
    """Load DAG adjacency matrices from results."""
    graphs_file = results_dir / market / f"{experiment}_d2_seed{seed}" / "graphs.npz"
    if not graphs_file.exists():
        raise FileNotFoundError(f"Graph file not found: {graphs_file}")

    graphs = np.load(graphs_file)
    dag0 = graphs['regime_0']
    dag1 = graphs['regime_1']
    return dag0, dag1


def compute_dag_stats(dag0, dag1):
    """Compute comparison statistics between two DAGs."""
    l2_diff = np.sqrt(np.sum((dag0 - dag1)**2))
    max_diff = np.max(np.abs(dag0 - dag1))

    d0_flat = dag0.flatten()
    d1_flat = dag1.flatten()
    cos_sim = np.dot(d0_flat, d1_flat) / (np.linalg.norm(d0_flat) * np.linalg.norm(d1_flat) + 1e-8)

    return {
        'l2_diff': l2_diff,
        'max_diff': max_diff,
        'cos_sim': cos_sim
    }


def plot_dag_comparison(results_dir: Path, market: str, seed: int, output_dir: Path):
    """
    Create comparison plot of DAGs from independent vs noise_init modes.

    Layout:
    Row 1: Independent mode - Regime 0, Regime 1, Difference
    Row 2: Noise_init mode - Regime 0, Regime 1, Difference
    """
    # Load DAGs
    ind_dag0, ind_dag1 = load_dags(results_dir, 'independent', market, seed)
    noise_dag0, noise_dag1 = load_dags(results_dir, 'noise_init', market, seed)

    # Use instantaneous edges only (lag index 0)
    ind_dag0_inst = ind_dag0[0]
    ind_dag1_inst = ind_dag1[0]
    noise_dag0_inst = noise_dag0[0]
    noise_dag1_inst = noise_dag1[0]

    # Compute stats
    ind_stats = compute_dag_stats(ind_dag0_inst, ind_dag1_inst)
    noise_stats = compute_dag_stats(noise_dag0_inst, noise_dag1_inst)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Common colormap settings
    vmin, vmax = 0, 1
    diff_vmax = max(np.abs(ind_dag1_inst - ind_dag0_inst).max(),
                   np.abs(noise_dag1_inst - noise_dag0_inst).max())
    diff_vmax = max(diff_vmax, 0.1)  # Minimum scale for visibility

    # Row 1: Independent mode
    sns.heatmap(ind_dag0_inst, ax=axes[0, 0], vmin=vmin, vmax=vmax,
                cmap='Blues', cbar_kws={'label': 'Edge Weight'})
    axes[0, 0].set_title(f'Independent: Regime 0')
    axes[0, 0].set_xlabel('To Node')
    axes[0, 0].set_ylabel('From Node')

    sns.heatmap(ind_dag1_inst, ax=axes[0, 1], vmin=vmin, vmax=vmax,
                cmap='Blues', cbar_kws={'label': 'Edge Weight'})
    axes[0, 1].set_title(f'Independent: Regime 1')
    axes[0, 1].set_xlabel('To Node')
    axes[0, 1].set_ylabel('From Node')

    diff_ind = ind_dag1_inst - ind_dag0_inst
    sns.heatmap(diff_ind, ax=axes[0, 2], vmin=-diff_vmax, vmax=diff_vmax,
                cmap='RdBu_r', center=0, cbar_kws={'label': 'Difference'})
    axes[0, 2].set_title(f'Difference (R1-R0)\nL2={ind_stats["l2_diff"]:.3f}, MaxDiff={ind_stats["max_diff"]:.4f}')
    axes[0, 2].set_xlabel('To Node')
    axes[0, 2].set_ylabel('From Node')

    # Row 2: Noise_init mode
    sns.heatmap(noise_dag0_inst, ax=axes[1, 0], vmin=vmin, vmax=vmax,
                cmap='Blues', cbar_kws={'label': 'Edge Weight'})
    axes[1, 0].set_title(f'Noise Init: Regime 0')
    axes[1, 0].set_xlabel('To Node')
    axes[1, 0].set_ylabel('From Node')

    sns.heatmap(noise_dag1_inst, ax=axes[1, 1], vmin=vmin, vmax=vmax,
                cmap='Blues', cbar_kws={'label': 'Edge Weight'})
    axes[1, 1].set_title(f'Noise Init: Regime 1')
    axes[1, 1].set_xlabel('To Node')
    axes[1, 1].set_ylabel('From Node')

    diff_noise = noise_dag1_inst - noise_dag0_inst
    sns.heatmap(diff_noise, ax=axes[1, 2], vmin=-diff_vmax, vmax=diff_vmax,
                cmap='RdBu_r', center=0, cbar_kws={'label': 'Difference'})
    axes[1, 2].set_title(f'Difference (R1-R0)\nL2={noise_stats["l2_diff"]:.3f}, MaxDiff={noise_stats["max_diff"]:.4f}')
    axes[1, 2].set_xlabel('To Node')
    axes[1, 2].set_ylabel('From Node')

    # Overall title
    fig.suptitle(f'DAG Comparison: {market} Market (Seed {seed})\n'
                 f'Independent: Identical DAGs (cos={ind_stats["cos_sim"]:.4f}) | '
                 f'Noise Init: Different DAGs (cos={noise_stats["cos_sim"]:.4f})',
                 fontsize=14)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'dag_comparison_{market}_seed{seed}.pdf'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")

    # Also save PNG for quick viewing
    output_file_png = output_dir / f'dag_comparison_{market}_seed{seed}.png'
    plt.savefig(output_file_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file_png}")

    plt.close()

    return ind_stats, noise_stats


def plot_target_constrained_comparison(results_dir: Path, market: str, seed: int,
                                        output_dir: Path, feature_names: list = None):
    """
    Create comparison plot showing only edges TO the target variable.

    Layout:
    Row 1: Independent mode - Regime 0 bars, Regime 1 bars, Difference
    Row 2: Noise_init mode - Regime 0 bars, Regime 1 bars, Difference
    """
    # Load DAGs
    ind_dag0, ind_dag1 = load_dags(results_dir, 'independent', market, seed)
    noise_dag0, noise_dag1 = load_dags(results_dir, 'noise_init', market, seed)

    # Extract edges TO target (last column, excluding self-loop)
    # Instantaneous edges only (lag index 0)
    target_idx = -1
    ind_to_target_0 = ind_dag0[0, :-1, target_idx]  # All nodes -> target, regime 0
    ind_to_target_1 = ind_dag1[0, :-1, target_idx]  # All nodes -> target, regime 1
    noise_to_target_0 = noise_dag0[0, :-1, target_idx]
    noise_to_target_1 = noise_dag1[0, :-1, target_idx]

    # Also include lagged edges to target
    ind_to_target_0_lag = ind_dag0[1, :, target_idx]  # Lagged -> target
    ind_to_target_1_lag = ind_dag1[1, :, target_idx]
    noise_to_target_0_lag = noise_dag0[1, :, target_idx]
    noise_to_target_1_lag = noise_dag1[1, :, target_idx]

    n_features = len(ind_to_target_0)
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(n_features)]

    # Compute stats for target edges only
    ind_diff = ind_to_target_1 - ind_to_target_0
    noise_diff = noise_to_target_1 - noise_to_target_0

    ind_l2 = np.sqrt(np.sum(ind_diff**2))
    noise_l2 = np.sqrt(np.sum(noise_diff**2))

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    x = np.arange(n_features)
    width = 0.8

    # Row 1: Independent mode
    axes[0, 0].bar(x, ind_to_target_0, width, color='steelblue', alpha=0.8)
    axes[0, 0].set_title('Independent: Regime 0\nEdges TO Target')
    axes[0, 0].set_ylabel('Edge Weight')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')

    axes[0, 1].bar(x, ind_to_target_1, width, color='steelblue', alpha=0.8)
    axes[0, 1].set_title('Independent: Regime 1\nEdges TO Target')
    axes[0, 1].set_ylabel('Edge Weight')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    colors = ['green' if d > 0 else 'red' for d in ind_diff]
    axes[0, 2].bar(x, ind_diff, width, color=colors, alpha=0.8)
    axes[0, 2].set_title(f'Difference (R1-R0)\nL2={ind_l2:.4f}')
    axes[0, 2].set_ylabel('Difference')
    axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 2].set_ylim(-0.8, 0.8)

    # Row 2: Noise_init mode
    axes[1, 0].bar(x, noise_to_target_0, width, color='darkorange', alpha=0.8)
    axes[1, 0].set_title('Noise Init: Regime 0\nEdges TO Target')
    axes[1, 0].set_ylabel('Edge Weight')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    axes[1, 1].bar(x, noise_to_target_1, width, color='darkorange', alpha=0.8)
    axes[1, 1].set_title('Noise Init: Regime 1\nEdges TO Target')
    axes[1, 1].set_ylabel('Edge Weight')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    colors = ['green' if d > 0 else 'red' for d in noise_diff]
    axes[1, 2].bar(x, noise_diff, width, color=colors, alpha=0.8)
    axes[1, 2].set_title(f'Difference (R1-R0)\nL2={noise_l2:.4f}')
    axes[1, 2].set_ylabel('Difference')
    axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 2].set_ylim(-0.8, 0.8)

    # Set x-axis labels (only show every Nth for readability if many features)
    for ax in axes.flat:
        if n_features <= 15:
            ax.set_xticks(x)
            ax.set_xticklabels([f'{i}' for i in range(n_features)], rotation=45, ha='right')
        else:
            ax.set_xticks(x[::5])
            ax.set_xticklabels([f'{i}' for i in range(0, n_features, 5)])
        ax.set_xlabel('Feature Index')

    fig.suptitle(f'Target-Constrained DAG: {market} Market (Seed {seed})\n'
                 f'Edges TO Target Variable Only\n'
                 f'Independent L2={ind_l2:.4f} | Noise Init L2={noise_l2:.4f}',
                 fontsize=14)

    plt.tight_layout()

    # Save
    output_file = output_dir / f'dag_target_constrained_{market}_seed{seed}.pdf'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")

    output_file_png = output_dir / f'dag_target_constrained_{market}_seed{seed}.png'
    plt.savefig(output_file_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file_png}")

    plt.close()

    return {'ind_l2': ind_l2, 'noise_l2': noise_l2}


def main():
    parser = argparse.ArgumentParser(description='Generate DAG comparison visualizations')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory')
    parser.add_argument('--output_dir', type=str, default='results/figures',
                        help='Output directory for figures')
    parser.add_argument('--markets', type=str, nargs='+', default=['DE', 'FR'],
                        help='Markets to visualize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("DAG Comparison Visualization")
    print("=" * 60)

    for market in args.markets:
        print(f"\n### {market} Market ###")
        try:
            # Full DAG comparison
            print("  Generating full DAG comparison...")
            ind_stats, noise_stats = plot_dag_comparison(
                results_dir, market, args.seed, output_dir
            )
            print(f"  Full DAG - Independent: L2={ind_stats['l2_diff']:.4f}, "
                  f"CosSim={ind_stats['cos_sim']:.6f}")
            print(f"  Full DAG - Noise Init:  L2={noise_stats['l2_diff']:.4f}, "
                  f"CosSim={noise_stats['cos_sim']:.6f}")

            # Target-constrained DAG comparison
            print("  Generating target-constrained DAG comparison...")
            target_stats = plot_target_constrained_comparison(
                results_dir, market, args.seed, output_dir
            )
            print(f"  Target-Constrained - Independent: L2={target_stats['ind_l2']:.4f}")
            print(f"  Target-Constrained - Noise Init:  L2={target_stats['noise_l2']:.4f}")

        except FileNotFoundError as e:
            print(f"  Skipped: {e}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == '__main__':
    main()
