#!/usr/bin/env python3
"""
Analyze Shared Backbone Experiment Results.

Generates comparison tables and visualizations for shared_backbone vs independent
mode experiments.

Usage:
    python analyze_results.py --results_dir results/experiment_20240101_120000
    python analyze_results.py --results_dir results/ --aggregate  # Aggregate all
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Experiment Results')

    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures (default: results_dir/analysis)')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate results from multiple experiment runs')

    return parser.parse_args()


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load results from a results directory."""
    results = []

    # Look for results files
    for results_file in results_dir.rglob('results.json'):
        try:
            with open(results_file) as f:
                result = json.load(f)

            # Get config from parent directory name or config file
            config_file = results_file.parent / 'config.json'
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                    result.update(config.get('args', {}))
                    result.update(config.get('model', {}))
                    result['market'] = config.get('data', {}).get('market')

            result['experiment_dir'] = str(results_file.parent)
            results.append(result)

        except Exception as e:
            print(f"Error loading {results_file}: {e}")

    if not results:
        # Try loading from aggregated file
        agg_file = results_dir / 'results_final.json'
        if agg_file.exists():
            with open(agg_file) as f:
                results = json.load(f)

    return pd.DataFrame(results)


def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create comparison table for sharing modes."""
    # Group by configuration
    group_cols = ['market', 'd_dim']
    metric_cols = ['spearman', 'rmse']

    # Filter to successful runs
    success_df = df[df.get('status', 'success') == 'success'].copy() if 'status' in df.columns else df.copy()

    if len(success_df) == 0:
        return pd.DataFrame()

    # Create pivot table
    results = []
    for (market, d_dim), group in success_df.groupby(group_cols):
        row = {'market': market, 'd_dim': d_dim}

        for mode in ['shared_backbone', 'independent']:
            mode_data = group[group['sharing_mode'] == mode]
            if len(mode_data) > 0:
                for metric in metric_cols:
                    if metric in mode_data.columns:
                        mean_val = mode_data[metric].mean()
                        std_val = mode_data[metric].std()
                        row[f'{mode}_{metric}_mean'] = mean_val
                        row[f'{mode}_{metric}_std'] = std_val
                        row[f'{mode}_{metric}'] = f'{mean_val:.4f} ± {std_val:.4f}'

        results.append(row)

    return pd.DataFrame(results)


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Create comparison plots."""
    if len(df) == 0:
        print("No data to plot")
        return

    success_df = df[df.get('status', 'success') == 'success'].copy() if 'status' in df.columns else df.copy()

    if len(success_df) == 0:
        print("No successful runs to plot")
        return

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Spearman comparison by market and sharing mode
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, market in enumerate(['DE', 'FR', 'DE_FR']):
        ax = axes[i]
        market_data = success_df[success_df['market'] == market]

        if len(market_data) > 0 and 'spearman' in market_data.columns:
            sns.boxplot(data=market_data, x='d_dim', y='spearman', hue='sharing_mode', ax=ax)
            ax.set_title(f'{market} Market')
            ax.set_xlabel('Number of Regimes')
            ax.set_ylabel('Spearman Correlation')
            if i > 0:
                ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig(output_dir / 'spearman_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'spearman_comparison.svg', bbox_inches='tight')
    plt.close()

    # 2. RMSE comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, market in enumerate(['DE', 'FR', 'DE_FR']):
        ax = axes[i]
        market_data = success_df[success_df['market'] == market]

        if len(market_data) > 0 and 'rmse' in market_data.columns:
            sns.boxplot(data=market_data, x='d_dim', y='rmse', hue='sharing_mode', ax=ax)
            ax.set_title(f'{market} Market')
            ax.set_xlabel('Number of Regimes')
            ax.set_ylabel('RMSE')
            if i > 0:
                ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'rmse_comparison.svg', bbox_inches='tight')
    plt.close()

    # 3. Overall summary heatmap
    pivot_data = success_df.pivot_table(
        values='spearman',
        index=['market', 'd_dim'],
        columns='sharing_mode',
        aggfunc='mean'
    )

    if len(pivot_data) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
        ax.set_title('Spearman Correlation by Configuration')
        plt.tight_layout()
        plt.savefig(output_dir / 'spearman_heatmap.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'spearman_heatmap.svg', bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {output_dir}")


def analyze_learned_graphs(results_dir: Path, output_dir: Path):
    """Analyze learned causal graphs."""
    # Find all graph files
    graph_files = list(results_dir.rglob('graphs.npz'))

    if not graph_files:
        print("No graph files found")
        return

    print(f"\nFound {len(graph_files)} graph files")

    # Analyze graph statistics
    stats = []
    for graph_file in graph_files:
        try:
            data = np.load(graph_file)
            parent_dir = graph_file.parent

            # Load config to get experiment details
            config_file = parent_dir / 'config.json'
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                    args = config.get('args', {})
            else:
                args = {}

            for key in data.files:
                graph = data[key]
                n_edges = (np.abs(graph) > 0.5).sum()
                mean_weight = np.abs(graph).mean()
                max_weight = np.abs(graph).max()

                stats.append({
                    'experiment': str(parent_dir.name),
                    'market': args.get('market', 'unknown'),
                    'sharing_mode': args.get('sharing_mode', 'unknown'),
                    'd_dim': args.get('d_dim', 0),
                    'seed': args.get('seed', 0),
                    'regime': key,
                    'n_edges_0.5': n_edges,
                    'mean_weight': mean_weight,
                    'max_weight': max_weight,
                })

        except Exception as e:
            print(f"Error loading {graph_file}: {e}")

    if stats:
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(output_dir / 'graph_statistics.csv', index=False)

        # Summary by sharing mode
        summary = stats_df.groupby(['market', 'sharing_mode', 'd_dim']).agg({
            'n_edges_0.5': ['mean', 'std'],
            'mean_weight': ['mean', 'std'],
        }).round(3)

        print("\nGraph Statistics Summary:")
        print(summary)

        summary.to_csv(output_dir / 'graph_statistics_summary.csv')


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Analyzing Shared Backbone Experiment Results")
    print(f"{'='*60}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Load results
    df = load_results(results_dir)
    print(f"\nLoaded {len(df)} experiment results")

    if len(df) == 0:
        print("No results found!")
        return

    # Create comparison table
    comparison_df = create_comparison_table(df)
    if len(comparison_df) > 0:
        print("\nComparison Table:")
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(output_dir / 'comparison_table.csv', index=False)

    # Create plots
    plot_comparison(df, output_dir)

    # Analyze graphs
    analyze_learned_graphs(results_dir, output_dir)

    # Save full results
    df.to_csv(output_dir / 'all_results.csv', index=False)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
