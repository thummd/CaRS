#!/usr/bin/env python3
"""
Generate combined regime visualization plots for CaRS experiments.

Creates a three-panel figure:
- Top: Price time series with regime coloring
- Middle: Regime probability curves
- Bottom: Detected changepoints

Usage:
    python plot_regime_timeseries.py --results_file cars_de_d2_lag1_ls5.0_seed42.json --output de_regime_plot.svg
    python plot_regime_timeseries.py --results_dir presentation/results/cars/ --dataset DE --output_dir presentation/figures/
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd

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

# Regime color palettes
REGIME_COLORS = [
    '#1f77b4',  # Blue - Regime 0 (Stable)
    '#ff7f0e',  # Orange - Regime 1 (Crisis)
    '#2ca02c',  # Green - Regime 2
    '#d62728',  # Red - Regime 3
    '#9467bd',  # Purple - Regime 4
]

REGIME_COLORS_LIGHT = [
    '#aec7e8',  # Light blue
    '#ffbb78',  # Light orange
    '#98df8a',  # Light green
    '#ff9896',  # Light red
    '#c5b0d5',  # Light purple
]

REGIME_LABELS = ['Stable', 'Crisis', 'Regime 2', 'Regime 3', 'Regime 4']

# Economic events for European electricity markets (2015-2024)
ECONOMIC_EVENTS = [
    (pd.Timestamp('2015-01-01'), pd.Timestamp('2016-06-30'),
     'Oil price collapse',       'Commodity oversupply'),
    (pd.Timestamp('2016-07-01'), pd.Timestamp('2018-06-30'),
     'FR nuclear safety crisis', 'Carbon market adjustment'),
    (pd.Timestamp('2018-07-01'), pd.Timestamp('2019-12-31'),
     'EU ETS carbon rally',      'Global LNG glut'),
    (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-30'),
     'COVID demand shock',       'COVID lockdown'),
    (pd.Timestamp('2020-07-01'), pd.Timestamp('2021-06-30'),
     'Post-COVID gas tightening','Demand recovery'),
    (pd.Timestamp('2021-07-01'), pd.Timestamp('2022-06-30'),
     'Russian gas curtailments', 'Storage replenishment'),
    (pd.Timestamp('2022-07-01'), pd.Timestamp('2023-06-30'),
     'TTF gas peak',             'Gas price normalization'),
    (pd.Timestamp('2023-07-01'), pd.Timestamp('2024-12-31'),
     'Renewable intermittency',  'Post-crisis stabilization'),
]


def map_changepoint_to_event(cp_date, from_regime, to_regime):
    """Map a detected changepoint date to an economic event label."""
    entering_crisis = (to_regime > from_regime)
    for range_start, range_end, crisis_label, normal_label in ECONOMIC_EVENTS:
        if range_start <= cp_date <= range_end:
            return crisis_label if entering_crisis else normal_label
    return cp_date.strftime('%b %Y')


def load_results(results_file: str) -> dict:
    """Load CaRS experiment results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def load_price_data(dataset: str, data_dir: str = None) -> tuple:
    """Load price data for visualization.

    Returns:
        Tuple of (prices, timestamps, feature_names)
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '../../data/unified')

    data_path = os.path.join(data_dir, f'{dataset.lower()}_unified.csv')

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Try to find price column
        price_cols = [c for c in df.columns if 'price' in c.lower()]
        if price_cols:
            prices = df[price_cols[0]].values
        else:
            # Use first numeric column as proxy
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            prices = df[numeric_cols[0]].values if len(numeric_cols) > 0 else None

        timestamps = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else None
        return prices, timestamps, list(df.columns)

    return None, None, None


def compute_regime_probabilities(regime_assignments: np.ndarray, window_size: int = 24) -> np.ndarray:
    """Compute smoothed regime probabilities using a sliding window.

    Args:
        regime_assignments: Array of regime assignments (0, 1, 2, ...)
        window_size: Size of the smoothing window (default: 24 hours)

    Returns:
        Array of shape (T, n_regimes) with smoothed probabilities
    """
    T = len(regime_assignments)
    n_regimes = int(np.max(regime_assignments)) + 1

    probabilities = np.zeros((T, n_regimes))

    # One-hot encode regime assignments
    for t in range(T):
        probabilities[t, regime_assignments[t]] = 1.0

    # Apply smoothing with sliding window
    smoothed = np.zeros_like(probabilities)
    half_window = window_size // 2

    for t in range(T):
        start = max(0, t - half_window)
        end = min(T, t + half_window + 1)
        smoothed[t] = np.mean(probabilities[start:end], axis=0)

    return smoothed


def detect_changepoints(regime_assignments: np.ndarray, min_duration: int = 6) -> List[int]:
    """Detect regime changepoints.

    Args:
        regime_assignments: Array of regime assignments
        min_duration: Minimum duration to consider a regime change (to filter noise)

    Returns:
        List of changepoint indices
    """
    changepoints = []
    current_regime = regime_assignments[0]
    regime_start = 0

    for t in range(1, len(regime_assignments)):
        if regime_assignments[t] != current_regime:
            # Check if previous regime lasted long enough
            if t - regime_start >= min_duration:
                changepoints.append(t)
            current_regime = regime_assignments[t]
            regime_start = t

    return changepoints


def plot_regime_timeseries(
    prices: np.ndarray,
    regime_assignments: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    n_regimes: int = 2,
    title: str = "CaRS Regime Analysis",
    output_path: str = None,
    figsize: tuple = (14, 8),
    window_size: int = 24,
):
    """Create combined regime visualization plot.

    Args:
        prices: Price time series array
        regime_assignments: Array of regime assignments
        timestamps: Optional array of timestamps
        n_regimes: Number of regimes
        title: Plot title
        output_path: Path to save the figure
        figsize: Figure size
        window_size: Window size for probability smoothing
    """
    T = len(regime_assignments)

    # Align data lengths (regime assignments may be shorter due to lag)
    if prices is not None and len(prices) > T:
        prices = prices[-T:]
    if timestamps is not None and len(timestamps) > T:
        timestamps = timestamps[-T:]

    # Create time index
    if timestamps is not None:
        time_index = timestamps
        x_label = "Date"
    else:
        time_index = np.arange(T)
        x_label = "Time (hours)"

    # Compute regime probabilities
    regime_probs = compute_regime_probabilities(regime_assignments, window_size)

    # Detect changepoints
    changepoints = detect_changepoints(regime_assignments)

    # Create figure with three panels
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                              gridspec_kw={'height_ratios': [2, 1, 0.5], 'hspace': 0.1})

    # Panel 1: Price time series with regime coloring
    ax1 = axes[0]

    if prices is not None:
        # Plot price as thin black line
        ax1.plot(time_index, prices, 'k-', linewidth=0.5, alpha=0.7, label='Price')

        # Color background by regime
        for r in range(n_regimes):
            regime_mask = regime_assignments == r
            if np.any(regime_mask):
                # Find continuous segments
                segments = []
                in_segment = False
                start = 0

                for t in range(T):
                    if regime_mask[t] and not in_segment:
                        start = t
                        in_segment = True
                    elif not regime_mask[t] and in_segment:
                        segments.append((start, t))
                        in_segment = False

                if in_segment:
                    segments.append((start, T))

                # Plot colored background for each segment
                for seg_start, seg_end in segments:
                    ax1.axvspan(
                        time_index[seg_start],
                        time_index[min(seg_end, T-1)],
                        alpha=0.3,
                        color=REGIME_COLORS[r % len(REGIME_COLORS)],
                        linewidth=0,
                    )

        ax1.set_ylabel('Price (EUR/MWh)')
    else:
        # If no prices, just show regime coloring
        ax1.set_ylabel('Regime')
        for r in range(n_regimes):
            regime_mask = regime_assignments == r
            ax1.fill_between(
                time_index,
                0, 1,
                where=regime_mask,
                alpha=0.5,
                color=REGIME_COLORS[r % len(REGIME_COLORS)],
                label=REGIME_LABELS[r] if r < len(REGIME_LABELS) else f'Regime {r}'
            )

    # Add legend for regimes
    legend_patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.5,
                       label=REGIME_LABELS[r] if r < len(REGIME_LABELS) else f'Regime {r}')
        for r in range(n_regimes)
    ]
    ax1.legend(handles=legend_patches, loc='upper right', framealpha=0.9)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Regime probabilities
    ax2 = axes[1]

    for r in range(n_regimes):
        ax2.plot(
            time_index,
            regime_probs[:, r],
            color=REGIME_COLORS[r % len(REGIME_COLORS)],
            linewidth=1.5,
            label=f'P({REGIME_LABELS[r]})' if r < len(REGIME_LABELS) else f'P(Regime {r})'
        )

    ax2.set_ylabel('Probability')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Changepoints
    ax3 = axes[2]

    # Draw timeline
    ax3.axhline(y=0.5, color='gray', linewidth=2)

    # Mark changepoints with economic event labels
    for cp in changepoints:
        cp_date = time_index[cp]
        from_r = regime_assignments[cp - 1] if cp > 0 else 0
        to_r = regime_assignments[cp]
        label = map_changepoint_to_event(pd.Timestamp(cp_date), int(from_r), int(to_r))

        ax3.axvline(x=cp_date, color='red', linewidth=1.5, alpha=0.7)
        ax3.plot(cp_date, 0.5, 'rv', markersize=8)

        # Add label to price panel (Panel 1)
        y_top = ax1.get_ylim()[1]
        ax1.axvline(x=cp_date, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax1.text(
            cp_date, y_top * 0.97,
            f' {label}',
            fontsize=10,
            rotation=90,
            va='top',
            ha='left',
            color='#444444',
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.7),
        )

    ax3.set_ylabel('Change\nPoints')
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.set_xlabel(x_label)

    # Add changepoint count annotation
    ax3.text(
        0.02, 0.8, f'{len(changepoints)} changepoints detected',
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment='top'
    )

    plt.tight_layout()

    # Save figure
    if output_path:
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Save in multiple formats
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")

        # Also save as PDF if SVG
        if output_path.endswith('.svg'):
            pdf_path = output_path.replace('.svg', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"Saved: {pdf_path}")

    plt.close()
    return fig


def process_results_file(results_file: str, output_path: str, data_dir: str = None):
    """Process a single results file and generate visualization."""
    results = load_results(results_file)

    dataset = results['config']['dataset']
    n_regimes = results['config']['n_regimes_final']
    lag = results['config']['lag']
    lambda_sparse = results['config']['lambda_sparse']
    seed = results['config']['seed']

    regime_assignments = np.array(results['regime_assignments'])

    # Try to load price data
    prices, timestamps, _ = load_price_data(dataset, data_dir)

    # Generate title
    title = f"CaRS Regime Analysis: {dataset}\n(d={n_regimes}, lag={lag}, λ_sparse={lambda_sparse}, seed={seed})"

    plot_regime_timeseries(
        prices=prices,
        regime_assignments=regime_assignments,
        timestamps=timestamps,
        n_regimes=n_regimes,
        title=title,
        output_path=output_path,
    )


def process_all_results(results_dir: str, output_dir: str, data_dir: str = None):
    """Process all results files in a directory."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for results_file in results_path.glob('cars_*.json'):
        # Generate output filename
        output_name = results_file.stem.replace('cars_', 'regime_plot_') + '.svg'
        output_file = output_path / output_name

        print(f"Processing: {results_file.name}")
        try:
            process_results_file(str(results_file), str(output_file), data_dir)
        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate regime visualization plots")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Single results JSON file to process")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory containing results files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (for single file) or directory (for batch)")
    parser.add_argument("--output_dir", type=str, default="presentation/figures/",
                        help="Output directory for batch processing")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing unified electricity data")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=['DE', 'FR', 'DE_FR'],
                        help="Filter results by dataset (for batch processing)")

    args = parser.parse_args()

    if args.results_file:
        # Process single file
        output_path = args.output or args.results_file.replace('.json', '_regime_plot.svg')
        process_results_file(args.results_file, output_path, args.data_dir)
    elif args.results_dir:
        # Process all files in directory
        process_all_results(args.results_dir, args.output_dir, args.data_dir)
    else:
        print("Please specify either --results_file or --results_dir")
        parser.print_help()


if __name__ == "__main__":
    main()
