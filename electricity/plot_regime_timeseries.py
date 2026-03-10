"""
Plot time series with regime assignments overlay.

Usage:
    python plot_regime_timeseries.py --results_dir seasonal_results/DE_4_seasonal_20260116_093238
    python plot_regime_timeseries.py --country DE --k 4  # Auto-find latest results
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CARS_ROOT, DATA_DIR, ELECTRICITY_DIR
# Data paths
DATA_DIR = DATA_DIR / "qrt"
RESULTS_BASE = ELECTRICITY_DIR
OUTPUT_DIR = Path(str(CARS_ROOT / "presentation") + "/figures")


def load_raw_data() -> pd.DataFrame:
    """Load and merge raw data."""
    X_train = pd.read_csv(DATA_DIR / "X_train_NHkHMNU.csv")
    Y_train = pd.read_csv(DATA_DIR / "y_train_ZAN5mwg.csv")
    df = X_train.merge(Y_train, on='ID')
    df = df.sort_values('DAY_ID').reset_index(drop=True)
    return df


def load_regime_results(results_dir: Path) -> dict:
    """Load regime model results."""
    with open(results_dir / "regime_model.json", 'r') as f:
        regime_model = json.load(f)
    with open(results_dir / "summary.json", 'r') as f:
        summary = json.load(f)
    return {
        'gamma_hat': np.array(regime_model['gamma_hat']),
        'n_regimes': regime_model['n_regimes'],
        'summary': summary
    }


def find_latest_results(country: str, k: int, init_mode: str = "seasonal") -> Optional[Path]:
    """Find the latest results directory for given country and K."""
    pattern = f"{country}_{k}_{init_mode}_*"
    results_dirs = list((RESULTS_BASE / f"{init_mode}_results").glob(pattern))
    if not results_dirs:
        # Try regime_results as fallback
        results_dirs = list((RESULTS_BASE / "regime_results").glob(f"{country}_{k}_*"))

    if not results_dirs:
        return None

    # Sort by timestamp (directory name contains timestamp)
    results_dirs.sort(key=lambda x: x.name, reverse=True)
    return results_dirs[0]


def get_regime_colors(n_regimes: int) -> List[str]:
    """Get distinct colors for regimes."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    return colors[:n_regimes]


def plot_regime_timeseries(
    df: pd.DataFrame,
    gamma_hat: np.ndarray,
    country: str,
    n_regimes: int,
    output_path: Optional[Path] = None,
    variables: Optional[List[str]] = None,
    title_suffix: str = ""
):
    """
    Plot time series with regime background coloring.

    Args:
        df: DataFrame with DAY_ID and feature columns
        gamma_hat: (N, K) array of regime probabilities
        country: 'DE' or 'FR'
        n_regimes: Number of regimes
        output_path: Path to save figure
        variables: List of variables to plot (default: TARGET + key features)
        title_suffix: Additional text for title
    """
    # Get regime assignments (hard assignment from gamma_hat)
    regime_assignments = np.argmax(gamma_hat, axis=1)

    # Handle pruned samples (all zeros in gamma_hat)
    pruned_mask = np.sum(gamma_hat, axis=1) == 0
    regime_assignments[pruned_mask] = -1  # Mark as pruned

    # Default variables to plot
    if variables is None:
        target_col = f'TARGET_{country}' if f'TARGET_{country}' in df.columns else 'TARGET'
        if country == 'DE':
            variables = [target_col, 'DE_WINDPOW', 'COAL_RET', 'CARBON_RET']
        else:
            variables = [target_col, 'FR_NUCLEAR', 'COAL_RET', 'CARBON_RET']

    # Filter to available variables
    variables = [v for v in variables if v in df.columns]
    n_vars = len(variables)

    if n_vars == 0:
        print("No valid variables to plot!")
        return

    # Create figure
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 3 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    colors = get_regime_colors(n_regimes)
    day_ids = df['DAY_ID'].values

    # Ensure alignment between data and regime assignments
    n_samples = min(len(df), len(regime_assignments))
    day_ids = day_ids[:n_samples]
    regime_assignments = regime_assignments[:n_samples]

    # Plot each variable
    for ax, var in zip(axes, variables):
        values = df[var].values[:n_samples]

        # Plot the time series
        ax.plot(day_ids, values, 'k-', linewidth=0.8, alpha=0.8)

        # Add regime background coloring
        # Find contiguous regime segments
        segments = []
        current_regime = regime_assignments[0]
        start_idx = 0

        for i in range(1, n_samples):
            if regime_assignments[i] != current_regime:
                segments.append((start_idx, i - 1, current_regime))
                current_regime = regime_assignments[i]
                start_idx = i
        segments.append((start_idx, n_samples - 1, current_regime))

        # Color each segment
        for start, end, regime in segments:
            if regime >= 0:  # Skip pruned samples
                ax.axvspan(
                    day_ids[start], day_ids[end],
                    alpha=0.3,
                    color=colors[regime],
                    linewidth=0
                )

        ax.set_ylabel(var, fontsize=10)
        ax.grid(True, alpha=0.3)

    # X-axis label
    axes[-1].set_xlabel('DAY_ID', fontsize=11)

    # Title
    title = f'{country} Electricity Market - Regime Detection (K={n_regimes})'
    if title_suffix:
        title += f' {title_suffix}'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Legend
    patches = [mpatches.Patch(color=colors[i], alpha=0.3, label=f'Regime {i}')
               for i in range(n_regimes)]
    fig.legend(handles=patches, loc='upper right', bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_regime_probability_heatmap(
    df: pd.DataFrame,
    gamma_hat: np.ndarray,
    country: str,
    n_regimes: int,
    output_path: Optional[Path] = None
):
    """
    Plot regime probability over time as a stacked area chart.
    """
    day_ids = df['DAY_ID'].values
    n_samples = min(len(day_ids), len(gamma_hat))
    day_ids = day_ids[:n_samples]
    gamma_hat = gamma_hat[:n_samples]

    fig, ax = plt.subplots(figsize=(14, 3))

    colors = get_regime_colors(n_regimes)

    # Stacked area chart
    ax.stackplot(day_ids, gamma_hat.T, colors=colors, alpha=0.7,
                 labels=[f'Regime {i}' for i in range(n_regimes)])

    ax.set_xlabel('DAY_ID', fontsize=11)
    ax.set_ylabel('P(Regime)', fontsize=11)
    ax.set_title(f'{country} Regime Probabilities Over Time', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_regime_changepoints(
    df: pd.DataFrame,
    gamma_hat: np.ndarray,
    country: str,
    n_regimes: int,
    output_path: Optional[Path] = None
):
    """
    Plot target variable with vertical lines at regime change points.
    """
    day_ids = df['DAY_ID'].values
    target_col = f'TARGET_{country}' if f'TARGET_{country}' in df.columns else 'TARGET'

    n_samples = min(len(df), len(gamma_hat))
    day_ids = day_ids[:n_samples]
    target = df[target_col].values[:n_samples]

    # Get hard regime assignments
    regime_assignments = np.argmax(gamma_hat, axis=1)[:n_samples]

    # Find change points
    change_points = []
    for i in range(1, n_samples):
        if regime_assignments[i] != regime_assignments[i-1]:
            change_points.append((i, day_ids[i], regime_assignments[i-1], regime_assignments[i]))

    fig, ax = plt.subplots(figsize=(14, 4))

    colors = get_regime_colors(n_regimes)

    # Plot target
    ax.plot(day_ids, target, 'k-', linewidth=1, label='TARGET')

    # Add vertical lines at change points
    for idx, day_id, from_regime, to_regime in change_points:
        ax.axvline(x=day_id, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        # Annotate
        y_pos = ax.get_ylim()[1] * 0.9
        ax.annotate(f'{from_regime}→{to_regime}',
                   xy=(day_id, y_pos), fontsize=8, color='red',
                   ha='center', va='bottom')

    ax.set_xlabel('DAY_ID', fontsize=11)
    ax.set_ylabel('TARGET', fontsize=11)
    ax.set_title(f'{country} Price Variation with Regime Change Points ({len(change_points)} transitions)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot regime time series visualization")
    parser.add_argument("--results_dir", type=str, help="Path to results directory")
    parser.add_argument("--country", type=str, choices=['DE', 'FR'], help="Country code")
    parser.add_argument("--k", type=int, help="Number of initial regimes")
    parser.add_argument("--init_mode", type=str, default="seasonal", help="Initialization mode")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--format", type=str, default="svg", choices=['svg', 'png', 'pdf'])

    args = parser.parse_args()

    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = RESULTS_BASE / results_dir
    elif args.country and args.k:
        results_dir = find_latest_results(args.country, args.k, args.init_mode)
        if results_dir is None:
            print(f"No results found for {args.country} K={args.k}")
            return
    else:
        print("Either --results_dir or both --country and --k required")
        return

    print(f"Loading results from: {results_dir}")

    # Extract country from results dir name
    dir_name = results_dir.name
    country = dir_name.split('_')[0]

    # Load data
    df = load_raw_data()

    # Filter to country
    target_col = f'TARGET_{country}'
    if target_col in df.columns:
        df = df[df[target_col].notna()].reset_index(drop=True)

    # Load regime results
    results = load_regime_results(results_dir)
    gamma_hat = results['gamma_hat']
    n_regimes = results['n_regimes']

    print(f"Country: {country}")
    print(f"N regimes: {n_regimes}")
    print(f"Data samples: {len(df)}, Gamma samples: {len(gamma_hat)}")

    # Output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_initial = dir_name.split('_')[1]
    base_name = f"{country.lower()}_regime_k{k_initial}"

    # Generate plots
    print("\nGenerating plots...")

    # 1. Time series with regime background
    plot_regime_timeseries(
        df, gamma_hat, country, n_regimes,
        output_path=output_dir / f"{base_name}_timeseries.{args.format}"
    )

    # 2. Regime probability heatmap
    plot_regime_probability_heatmap(
        df, gamma_hat, country, n_regimes,
        output_path=output_dir / f"{base_name}_probability.{args.format}"
    )

    # 3. Change points visualization
    plot_regime_changepoints(
        df, gamma_hat, country, n_regimes,
        output_path=output_dir / f"{base_name}_changepoints.{args.format}"
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
