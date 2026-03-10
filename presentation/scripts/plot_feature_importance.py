#!/usr/bin/env python3
"""
Generate feature importance bar charts from CaRS experiment results.

Creates visualization showing:
- Average edge weight magnitude by feature
- Grouped by regime (side-by-side bars)
- Separate panels for instantaneous vs lagged effects
- Color coding by feature category

Usage:
    python plot_feature_importance.py --results_file cars_de_d2_lag1_ls5.0_seed42.json --output de_feature_importance.svg
    python plot_feature_importance.py --results_dir presentation/results/cars/ --output_dir presentation/figures/
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

# Style settings for publication-quality figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Feature category colors (matching presentation DAG legend)
FEATURE_CATEGORIES = {
    # Renewable
    'wind': 'renewable',
    'solar': 'renewable',
    'hydro': 'renewable',
    # Fossil/Commodity
    'gas': 'fossil',
    'coal': 'fossil',
    'oil': 'fossil',
    'carbon': 'fossil',
    # Nuclear
    'nuclear': 'nuclear',
    # Load/Demand
    'load': 'demand',
    'demand': 'demand',
    'residual': 'demand',
    # Price (Target)
    'price': 'target',
    # Weather
    'temperature': 'weather',
    'temp': 'weather',
    'rain': 'weather',
    'humidity': 'weather',
    # Temporal
    'hour': 'temporal',
    'day': 'temporal',
    'week': 'temporal',
    'month': 'temporal',
}

CATEGORY_COLORS = {
    'renewable': '#4CAF50',    # Green
    'fossil': '#FF9800',       # Orange
    'nuclear': '#9C27B0',      # Purple
    'demand': '#2196F3',       # Blue
    'target': '#FFC107',       # Yellow/Gold
    'weather': '#00BCD4',      # Cyan
    'temporal': '#795548',     # Brown
    'other': '#9E9E9E',        # Gray
}

REGIME_COLORS = [
    '#1f77b4',  # Blue - Regime 0
    '#ff7f0e',  # Orange - Regime 1
    '#2ca02c',  # Green - Regime 2
    '#d62728',  # Red - Regime 3
]

REGIME_LABELS = ['Stable', 'Crisis', 'Regime 2', 'Regime 3']


def get_feature_category(feature_name: str) -> str:
    """Determine the category of a feature based on its name."""
    feature_lower = feature_name.lower()
    for keyword, category in FEATURE_CATEGORIES.items():
        if keyword in feature_lower:
            return category
    return 'other'


def load_results(results_file: str) -> dict:
    """Load CaRS experiment results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def extract_feature_importance(results: dict) -> Dict[str, np.ndarray]:
    """Extract feature importance from adjacency matrices.

    Args:
        results: CaRS results dictionary

    Returns:
        Dictionary with feature importance arrays
    """
    adj_matrices = results['adjacency_matrices']
    feature_names = results['feature_names']
    n_regimes = len(adj_matrices)
    lag = results['config']['lag']
    n_features = len(feature_names)

    # Initialize importance arrays
    # Shape: (n_features, n_regimes)
    incoming_importance = np.zeros((n_features, n_regimes))
    outgoing_importance = np.zeros((n_features, n_regimes))
    total_importance = np.zeros((n_features, n_regimes))

    # Separate by lag
    instant_importance = np.zeros((n_features, n_regimes))
    lagged_importance = np.zeros((n_features, n_regimes))

    # Direct causal edge weight from each feature TO Price (target = index 0)
    target_idx = 0
    for i, name in enumerate(feature_names):
        if 'Price' in name or 'price' in name:
            target_idx = i
            break
    to_target_importance = np.zeros((n_features, n_regimes))

    for r, adj in enumerate(adj_matrices):
        adj = np.array(adj)  # Shape: (lag+1, n_features, n_features)

        for i in range(n_features):
            # Incoming edges (column sum) - how much this feature is influenced
            incoming_importance[i, r] = np.sum(np.abs(adj[:, :, i]))

            # Outgoing edges (row sum) - how much this feature influences others
            outgoing_importance[i, r] = np.sum(np.abs(adj[:, i, :]))

            # Total importance
            total_importance[i, r] = incoming_importance[i, r] + outgoing_importance[i, r]

            # Instantaneous (lag=0)
            instant_importance[i, r] = np.sum(np.abs(adj[0, :, i])) + np.sum(np.abs(adj[0, i, :]))

            # Lagged (lag>0)
            if lag > 0:
                lagged_importance[i, r] = np.sum(np.abs(adj[1:, :, i])) + np.sum(np.abs(adj[1:, i, :]))

            # Direct edge weight: feature i → Price
            if i != target_idx:
                to_target_importance[i, r] = np.sum(np.abs(adj[:, i, target_idx]))

    return {
        'incoming': incoming_importance,
        'outgoing': outgoing_importance,
        'total': total_importance,
        'instantaneous': instant_importance,
        'lagged': lagged_importance,
        'to_target': to_target_importance,
        'feature_names': feature_names,
        'n_regimes': n_regimes,
        'lag': lag,
    }


def plot_feature_importance_bars(
    importance_data: dict,
    output_path: str,
    title: str = "CaRS Feature Importance",
    importance_type: str = 'to_target',
    show_categories: bool = True,
    figsize: tuple = (12, 6),
):
    """Create grouped bar chart of feature importance.

    Args:
        importance_data: Dictionary from extract_feature_importance
        output_path: Path to save figure
        title: Plot title
        importance_type: 'total', 'incoming', 'outgoing', 'instantaneous', or 'lagged'
        show_categories: Whether to color bars by feature category
        figsize: Figure size
    """
    feature_names = importance_data['feature_names']
    n_regimes = importance_data['n_regimes']
    importance = importance_data[importance_type]

    n_features = len(feature_names)

    # Sort features by total importance across all regimes
    total_per_feature = np.sum(importance, axis=1)
    sort_idx = np.argsort(total_per_feature)[::-1]

    sorted_names = [feature_names[i] for i in sort_idx]
    sorted_importance = importance[sort_idx, :]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Bar positions
    x = np.arange(n_features)
    bar_width = 0.8 / n_regimes

    # Plot bars for each regime
    for r in range(n_regimes):
        offset = (r - n_regimes / 2 + 0.5) * bar_width

        if show_categories:
            # Color by feature category
            colors = [CATEGORY_COLORS[get_feature_category(name)] for name in sorted_names]
            # Add regime-specific alpha
            alpha = 0.6 + 0.4 * (r / max(n_regimes - 1, 1))
            regime_label = REGIME_LABELS[r] if r < len(REGIME_LABELS) else f'Regime {r}'
            bars = ax.bar(
                x + offset,
                sorted_importance[:, r],
                bar_width,
                label=regime_label,
                color=colors,
                alpha=alpha,
                edgecolor=REGIME_COLORS[r % len(REGIME_COLORS)],
                linewidth=1.5,
            )
        else:
            regime_label = REGIME_LABELS[r] if r < len(REGIME_LABELS) else f'Regime {r}'
            bars = ax.bar(
                x + offset,
                sorted_importance[:, r],
                bar_width,
                label=regime_label,
                color=REGIME_COLORS[r % len(REGIME_COLORS)],
                alpha=0.8,
            )

    # Customize plot
    ax.set_xlabel('Feature')
    ylabel = 'Causal Edge Weight → Price' if importance_type == 'to_target' else f'{importance_type.capitalize()} Edge Weight Magnitude'
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.legend(title='Regime', loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add category legend if showing categories
    if show_categories:
        category_patches = [
            mpatches.Patch(color=color, label=cat.capitalize())
            for cat, color in CATEGORY_COLORS.items()
            if cat != 'other' and any(get_feature_category(name) == cat for name in sorted_names)
        ]
        if category_patches:
            ax2 = ax.twinx()
            ax2.set_yticks([])
            ax2.legend(
                handles=category_patches,
                title='Category',
                loc='upper left',
                framealpha=0.9,
            )

    plt.tight_layout()

    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")

        if output_path.endswith('.svg'):
            pdf_path = output_path.replace('.svg', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"Saved: {pdf_path}")

    plt.close()
    return fig


def plot_feature_importance_detailed(
    importance_data: dict,
    output_path: str,
    title: str = "CaRS Feature Importance",
    figsize: tuple = (14, 10),
):
    """Create detailed multi-panel feature importance visualization.

    Shows:
    - Top panel: Total importance by regime
    - Middle panel: Instantaneous vs Lagged comparison
    - Bottom panel: Incoming vs Outgoing edge weights
    """
    feature_names = importance_data['feature_names']
    n_regimes = importance_data['n_regimes']
    lag = importance_data['lag']
    n_features = len(feature_names)

    # Sort features by total importance
    total_importance = np.sum(importance_data['total'], axis=1)
    sort_idx = np.argsort(total_importance)[::-1]
    sorted_names = [feature_names[i] for i in sort_idx]

    # Create figure with 3 panels
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    x = np.arange(n_features)
    bar_width = 0.8 / n_regimes

    # Panel 1: Total importance by regime
    ax1 = axes[0]
    for r in range(n_regimes):
        offset = (r - n_regimes / 2 + 0.5) * bar_width
        sorted_values = importance_data['total'][sort_idx, r]
        ax1.bar(
            x + offset,
            sorted_values,
            bar_width,
            label=REGIME_LABELS[r] if r < len(REGIME_LABELS) else f'Regime {r}',
            color=REGIME_COLORS[r % len(REGIME_COLORS)],
            alpha=0.8,
        )
    ax1.set_ylabel('Total Weight')
    ax1.set_title(f'{title}\nTotal Edge Weight Magnitude')
    ax1.legend(title='Regime', loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Instantaneous vs Lagged
    ax2 = axes[1]
    if lag > 0:
        bar_width_2 = 0.35
        instant_total = np.sum(importance_data['instantaneous'][sort_idx, :], axis=1)
        lagged_total = np.sum(importance_data['lagged'][sort_idx, :], axis=1)

        ax2.bar(x - bar_width_2/2, instant_total, bar_width_2, label='Instantaneous', color='#2196F3', alpha=0.8)
        ax2.bar(x + bar_width_2/2, lagged_total, bar_width_2, label='Lagged', color='#FF5722', alpha=0.8)
        ax2.set_ylabel('Weight')
        ax2.set_title('Instantaneous vs Lagged Effects')
        ax2.legend(loc='upper right')
    else:
        instant_total = np.sum(importance_data['instantaneous'][sort_idx, :], axis=1)
        ax2.bar(x, instant_total, 0.7, label='Instantaneous', color='#2196F3', alpha=0.8)
        ax2.set_ylabel('Weight')
        ax2.set_title('Instantaneous Effects (lag=0 only)')
        ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Incoming vs Outgoing
    ax3 = axes[2]
    bar_width_3 = 0.35
    incoming_total = np.sum(importance_data['incoming'][sort_idx, :], axis=1)
    outgoing_total = np.sum(importance_data['outgoing'][sort_idx, :], axis=1)

    ax3.bar(x - bar_width_3/2, incoming_total, bar_width_3, label='Incoming (influenced by)', color='#4CAF50', alpha=0.8)
    ax3.bar(x + bar_width_3/2, outgoing_total, bar_width_3, label='Outgoing (influences)', color='#9C27B0', alpha=0.8)
    ax3.set_ylabel('Weight')
    ax3.set_xlabel('Feature')
    ax3.set_title('Incoming vs Outgoing Edge Weights')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")

        if output_path.endswith('.svg'):
            pdf_path = output_path.replace('.svg', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"Saved: {pdf_path}")

    plt.close()
    return fig


def process_results_file(results_file: str, output_path: str, detailed: bool = True):
    """Process a single results file and generate visualization."""
    results = load_results(results_file)

    dataset = results['config']['dataset']
    n_regimes = results['config']['n_regimes_final']
    lag = results['config']['lag']
    lambda_sparse = results['config']['lambda_sparse']
    seed = results['config']['seed']

    # Extract importance
    importance_data = extract_feature_importance(results)

    # Generate title
    title = f"CaRS Feature Importance: {dataset}\n(d={n_regimes}, lag={lag}, λ_sparse={lambda_sparse}, seed={seed})"

    if detailed:
        plot_feature_importance_detailed(importance_data, output_path, title)
    else:
        plot_feature_importance_bars(importance_data, output_path, title)


def aggregate_importance_across_seeds(results_files: List[str]) -> dict:
    """Aggregate feature importance across multiple seeds.

    Args:
        results_files: List of results file paths

    Returns:
        Aggregated importance dictionary with mean and std
    """
    all_importance = defaultdict(list)

    for results_file in results_files:
        results = load_results(results_file)
        importance = extract_feature_importance(results)

        for key in ['total', 'incoming', 'outgoing', 'instantaneous', 'lagged']:
            all_importance[key].append(importance[key])

        all_importance['feature_names'] = importance['feature_names']
        all_importance['n_regimes'] = importance['n_regimes']
        all_importance['lag'] = importance['lag']

    # Compute mean and std
    aggregated = {
        'feature_names': all_importance['feature_names'],
        'n_regimes': all_importance['n_regimes'],
        'lag': all_importance['lag'],
    }

    for key in ['total', 'incoming', 'outgoing', 'instantaneous', 'lagged']:
        stacked = np.stack(all_importance[key], axis=0)
        aggregated[key] = np.mean(stacked, axis=0)
        aggregated[f'{key}_std'] = np.std(stacked, axis=0)

    return aggregated


def process_all_results(results_dir: str, output_dir: str):
    """Process all results files in a directory."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for results_file in results_path.glob('cars_*.json'):
        output_name = results_file.stem.replace('cars_', 'feature_importance_') + '.svg'
        output_file = output_path / output_name

        print(f"Processing: {results_file.name}")
        try:
            process_results_file(str(results_file), str(output_file))
        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate feature importance plots")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Single results JSON file to process")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory containing results files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (for single file)")
    parser.add_argument("--output_dir", type=str, default="presentation/figures/",
                        help="Output directory for batch processing")
    parser.add_argument("--simple", action="store_true",
                        help="Generate simple single-panel plot instead of detailed")

    args = parser.parse_args()

    if args.results_file:
        output_path = args.output or args.results_file.replace('.json', '_feature_importance.svg')
        process_results_file(args.results_file, output_path, detailed=not args.simple)
    elif args.results_dir:
        process_all_results(args.results_dir, args.output_dir)
    else:
        print("Please specify either --results_file or --results_dir")
        parser.print_help()


if __name__ == "__main__":
    main()
