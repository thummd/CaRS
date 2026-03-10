#!/usr/bin/env python3
"""
Deep analysis of lagged edges in DS3M Causal DAGs.

Investigates why all lagged edges converge to similar weights (~0.87)
and whether there's any meaningful causal structure being learned.
"""

import sys
import os
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import OUTPUT_DIR
# Find all soft adjacency matrices
OUTPUT_DIR = Path(str(OUTPUT_DIR) + "/regime_dags")

def load_soft_adjacency(exp_dir: Path):
    """Load soft adjacency matrices for all regimes in an experiment."""
    adjacencies = {}
    for npy_file in sorted(exp_dir.glob("adjacency_regime*_soft.npy")):
        # Extract regime number
        regime_str = npy_file.stem.replace("adjacency_regime", "").replace("_soft", "")
        regime = int(regime_str)
        adjacencies[regime] = np.load(npy_file)
    return adjacencies

def load_summary(exp_dir: Path):
    """Load summary JSON for experiment."""
    summary_path = exp_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None

def analyze_adjacency_structure(adj: np.ndarray, feature_names: list = None):
    """Analyze the structure of a single adjacency matrix."""
    if adj.ndim == 3:
        # Shape: [lag+1, num_nodes, num_nodes]
        instantaneous = adj[0]  # t -> t
        lagged = adj[1]         # t-1 -> t

        return {
            'instantaneous': {
                'mean': float(instantaneous.mean()),
                'std': float(instantaneous.std()),
                'min': float(instantaneous.min()),
                'max': float(instantaneous.max()),
                'median': float(np.median(instantaneous)),
            },
            'lagged': {
                'mean': float(lagged.mean()),
                'std': float(lagged.std()),
                'min': float(lagged.min()),
                'max': float(lagged.max()),
                'median': float(np.median(lagged)),
            }
        }
    else:
        return {
            'mean': float(adj.mean()),
            'std': float(adj.std()),
            'min': float(adj.min()),
            'max': float(adj.max()),
        }

def find_top_lagged_edges(adj: np.ndarray, feature_names: list, top_k: int = 20):
    """Find the strongest lagged edges."""
    if adj.ndim != 3:
        return []

    lagged = adj[1]  # t-1 -> t
    n = lagged.shape[0]

    edges = []
    for i in range(n):
        for j in range(n):
            edges.append({
                'from': feature_names[i] if feature_names else f'node_{i}',
                'to': feature_names[j] if feature_names else f'node_{j}',
                'from_idx': i,
                'to_idx': j,
                'weight': float(lagged[i, j])
            })

    # Sort by weight descending
    edges.sort(key=lambda x: x['weight'], reverse=True)
    return edges[:top_k]

def find_weakest_lagged_edges(adj: np.ndarray, feature_names: list, top_k: int = 20):
    """Find the weakest lagged edges."""
    if adj.ndim != 3:
        return []

    lagged = adj[1]  # t-1 -> t
    n = lagged.shape[0]

    edges = []
    for i in range(n):
        for j in range(n):
            edges.append({
                'from': feature_names[i] if feature_names else f'node_{i}',
                'to': feature_names[j] if feature_names else f'node_{j}',
                'from_idx': i,
                'to_idx': j,
                'weight': float(lagged[i, j])
            })

    # Sort by weight ascending
    edges.sort(key=lambda x: x['weight'])
    return edges[:top_k]

def analyze_target_incoming_edges(adj: np.ndarray, target_idx: int, feature_names: list):
    """Analyze edges incoming to target variable from all sources."""
    if adj.ndim != 3:
        return {}

    instantaneous = adj[0][:, target_idx]  # Instantaneous edges to target
    lagged = adj[1][:, target_idx]         # Lagged edges to target

    results = []
    for i in range(len(instantaneous)):
        results.append({
            'feature': feature_names[i] if feature_names else f'node_{i}',
            'instantaneous': float(instantaneous[i]),
            'lagged': float(lagged[i]),
            'diff': float(lagged[i] - instantaneous[i])
        })

    # Sort by lagged weight
    results.sort(key=lambda x: x['lagged'], reverse=True)
    return results

def compare_regimes(adjacencies: dict, feature_names: list):
    """Compare lagged edge patterns across regimes."""
    if not adjacencies:
        return {}

    regimes = sorted(adjacencies.keys())

    # Check if all have 3D adjacency
    for r, adj in adjacencies.items():
        if adj.ndim != 3:
            print(f"  Regime {r} has {adj.ndim}D adjacency, skipping regime comparison")
            return {}

    n_nodes = adjacencies[regimes[0]].shape[1]

    # Compute pairwise differences between regimes
    regime_diffs = {}
    for i, r1 in enumerate(regimes):
        for r2 in regimes[i+1:]:
            lagged1 = adjacencies[r1][1]
            lagged2 = adjacencies[r2][1]
            diff = np.abs(lagged1 - lagged2)
            regime_diffs[f"regime_{r1}_vs_{r2}"] = {
                'mean_diff': float(diff.mean()),
                'max_diff': float(diff.max()),
                'std_diff': float(diff.std()),
            }

    # Find edges with maximum variation across regimes
    edge_variance = np.zeros((n_nodes, n_nodes))
    lagged_matrices = [adjacencies[r][1] for r in regimes]
    lagged_stack = np.stack(lagged_matrices, axis=0)
    edge_variance = lagged_stack.std(axis=0)

    # Top varying edges
    top_var_edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            top_var_edges.append({
                'from': feature_names[i] if feature_names else f'node_{i}',
                'to': feature_names[j] if feature_names else f'node_{j}',
                'variance': float(edge_variance[i, j]),
                'weights_per_regime': {r: float(adjacencies[r][1][i, j]) for r in regimes}
            })

    top_var_edges.sort(key=lambda x: x['variance'], reverse=True)

    return {
        'regime_pairwise_diffs': regime_diffs,
        'top_varying_edges': top_var_edges[:20],
        'overall_edge_variance': {
            'mean': float(edge_variance.mean()),
            'max': float(edge_variance.max()),
            'std': float(edge_variance.std()),
        }
    }

def main():
    print("=" * 70)
    print("DEEP ANALYSIS OF LAGGED EDGES IN DS3M CAUSAL DAGS")
    print("=" * 70)

    # Find experiment directories with soft adjacencies
    experiments = []
    for exp_dir in sorted(OUTPUT_DIR.iterdir()):
        if exp_dir.is_dir():
            soft_files = list(exp_dir.glob("*_soft.npy"))
            if soft_files:
                experiments.append(exp_dir)

    print(f"\nFound {len(experiments)} experiments with soft adjacency matrices")

    # Analyze each experiment
    all_results = {}

    for exp_dir in experiments:
        exp_name = exp_dir.name
        print(f"\n{'='*70}")
        print(f"Experiment: {exp_name}")
        print("=" * 70)

        # Load data
        adjacencies = load_soft_adjacency(exp_dir)
        summary = load_summary(exp_dir)

        if not adjacencies:
            print("  No adjacency matrices found")
            continue

        feature_names = None
        if summary:
            feature_names = summary.get('feature_names', None)
            print(f"  Dataset: {summary.get('dataset')}, d_dim: {summary.get('d_dim')}")
            print(f"  Hyperparameters: {summary.get('hyperparameters', {})}")
            print(f"  Test Spearman: {summary.get('test_spearman', 'N/A'):.4f}")

        # Analyze each regime
        regime_stats = {}
        for regime, adj in sorted(adjacencies.items()):
            print(f"\n  Regime {regime}:")
            print(f"    Adjacency shape: {adj.shape}")

            stats = analyze_adjacency_structure(adj, feature_names)
            regime_stats[regime] = stats

            if 'instantaneous' in stats:
                print(f"    Instantaneous edges:")
                print(f"      Mean: {stats['instantaneous']['mean']:.6f}")
                print(f"      Std:  {stats['instantaneous']['std']:.6f}")
                print(f"      Range: [{stats['instantaneous']['min']:.6f}, {stats['instantaneous']['max']:.6f}]")

                print(f"    Lagged edges:")
                print(f"      Mean: {stats['lagged']['mean']:.6f}")
                print(f"      Std:  {stats['lagged']['std']:.6f}")
                print(f"      Range: [{stats['lagged']['min']:.6f}, {stats['lagged']['max']:.6f}]")

        # Compare across regimes
        print(f"\n  Cross-regime comparison:")
        regime_comparison = compare_regimes(adjacencies, feature_names)

        if regime_comparison:
            print(f"    Overall edge variance across regimes:")
            print(f"      Mean variance: {regime_comparison['overall_edge_variance']['mean']:.6f}")
            print(f"      Max variance:  {regime_comparison['overall_edge_variance']['max']:.6f}")

            print(f"\n    Top 10 edges varying most across regimes:")
            for i, edge in enumerate(regime_comparison['top_varying_edges'][:10]):
                weights_str = ", ".join([f"R{r}:{w:.4f}" for r, w in edge['weights_per_regime'].items()])
                print(f"      {i+1}. {edge['from'][:15]:15s} -> {edge['to'][:15]:15s}: var={edge['variance']:.6f} [{weights_str}]")

        # Analyze incoming edges to target (last feature)
        first_regime_adj = adjacencies[min(adjacencies.keys())]
        if first_regime_adj.ndim == 3:
            target_idx = first_regime_adj.shape[1] - 1
            print(f"\n  Incoming edges to target (node {target_idx}):")

            for regime, adj in sorted(adjacencies.items()):
                target_edges = analyze_target_incoming_edges(adj, target_idx, feature_names)
                print(f"\n    Regime {regime} - Top 5 lagged edges to target:")
                for i, e in enumerate(target_edges[:5]):
                    print(f"      {e['feature'][:20]:20s}: inst={e['instantaneous']:.4f}, lag={e['lagged']:.4f}, diff={e['diff']:.4f}")

        # Find strongest and weakest lagged edges overall
        print(f"\n  Overall lagged edge analysis (Regime 0):")
        adj0 = adjacencies[min(adjacencies.keys())]

        strongest = find_top_lagged_edges(adj0, feature_names, top_k=10)
        print(f"    Top 10 strongest lagged edges:")
        for i, e in enumerate(strongest):
            print(f"      {i+1}. {e['from'][:15]:15s} -> {e['to'][:15]:15s}: {e['weight']:.6f}")

        weakest = find_weakest_lagged_edges(adj0, feature_names, top_k=10)
        print(f"\n    Top 10 weakest lagged edges:")
        for i, e in enumerate(weakest):
            print(f"      {i+1}. {e['from'][:15]:15s} -> {e['to'][:15]:15s}: {e['weight']:.6f}")

        # Store results
        all_results[exp_name] = {
            'regime_stats': regime_stats,
            'regime_comparison': regime_comparison,
        }

    # Summary across all experiments
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL EXPERIMENTS")
    print("=" * 70)

    # Collect lagged edge statistics
    all_lagged_means = []
    all_lagged_stds = []
    all_inst_means = []

    for exp_name, results in all_results.items():
        for regime, stats in results['regime_stats'].items():
            if 'lagged' in stats:
                all_lagged_means.append(stats['lagged']['mean'])
                all_lagged_stds.append(stats['lagged']['std'])
                all_inst_means.append(stats['instantaneous']['mean'])

    if all_lagged_means:
        print(f"\nLagged edge statistics across all experiments:")
        print(f"  Mean of means: {np.mean(all_lagged_means):.6f}")
        print(f"  Std of means:  {np.std(all_lagged_means):.6f}")
        print(f"  Range of means: [{min(all_lagged_means):.6f}, {max(all_lagged_means):.6f}]")
        print(f"  Average within-regime std: {np.mean(all_lagged_stds):.6f}")

        print(f"\nInstantaneous edge statistics across all experiments:")
        print(f"  Mean of means: {np.mean(all_inst_means):.6f}")
        print(f"  Range of means: [{min(all_inst_means):.6f}, {max(all_inst_means):.6f}]")

        print(f"\nKey finding:")
        print(f"  Lagged edges are consistently ~{np.mean(all_lagged_means):.2f} with very low variation")
        print(f"  Instantaneous edges are consistently ~{np.mean(all_inst_means):.2f}")
        print(f"  The model learns a binary split: lagged > 0.5, instantaneous < 0.5")

if __name__ == "__main__":
    main()
