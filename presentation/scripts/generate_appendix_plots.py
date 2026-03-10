#!/usr/bin/env python3
"""
Generate appendix plots for CaRS presentation.

Creates:
1. Regime Time Series Visualization (DE, FR, DE-FR)
2. Causal Feature Importance bar charts (DE, FR, DE-FR)
3. Lag Analysis comparison

Usage:
    python generate_appendix_plots.py
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

# Add CASTOR electricity to path for unified data loading
CASTOR_ELECTRICITY_DIR = '/lustre/home/dthumm/CASTOR/electricity'
sys.path.insert(0, CASTOR_ELECTRICITY_DIR)

# Results and output directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results/cars')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../figures')

# Style settings for publication
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
COLORS = {
    'regime0': '#1f77b4',  # Blue
    'regime1': '#ff7f0e',  # Orange
    'regime2': '#2ca02c',  # Green
    'regime3': '#d62728',  # Red
    'price': '#333333',
    'grid': '#cccccc',
}

FIGSIZE_WIDE = (14, 8)
FIGSIZE_SQUARE = (10, 8)

REGIME_LABELS = ['Stable', 'Crisis']

# Preferred lambda_sparse per dataset (matching DAG-fix retraining)
PREFERRED_LAMBDA_SPARSE = {'DE': 5.0, 'FR': 5.0, 'DE_FR': 50.0}

# Economic events for European electricity markets (2015-2024)
# Each entry: (range_start, range_end, label_when_entering_crisis, label_when_entering_stable)
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
     'Renewable intermittency','Post-crisis stabilization'),
]


def load_results(model_filter: str = 'ds3m_causal') -> list:
    """Load all result files with optional model filter."""
    pattern = os.path.join(RESULTS_DIR, '*.json')
    files = glob.glob(pattern)

    results = []
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                model = data.get('config', {}).get('model', 'fantom')
                if model_filter and model != model_filter:
                    continue
                data['_filepath'] = filepath
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")

    return results


def load_unified_data(dataset: str):
    """Load unified electricity data."""
    try:
        from unified_data_loader import load_unified_dataset
        df = load_unified_dataset(dataset, clean=True)
        return df
    except ImportError:
        print(f"Warning: Could not load unified dataset for {dataset}")
        return None


def get_best_result(results: list, dataset: str, n_regimes: int = 2, lag: int = 1) -> dict:
    """Get the best result for a given configuration (non-collapsed preferred).

    Priority order:
    1. Native Markov results with n_regimes_final >= n_regimes
    2. Prefer results with more samples (latest data)
    3. For DE_FR, prefer smoothed results to reduce rapid switching
    4. Any non-collapsed result
    5. Any matching result
    """
    matching = [
        r for r in results
        if r.get('config', {}).get('dataset', '').upper() == dataset.upper()
        and r.get('config', {}).get('n_regimes_requested') == n_regimes
        and r.get('config', {}).get('lag') == lag
    ]

    if not matching:
        return None

    # Prefer native_markov results (better regime detection)
    native_markov = [
        r for r in matching
        if r.get('config', {}).get('regime_method') == 'native_markov'
        and r.get('config', {}).get('n_regimes_final', 1) >= n_regimes
    ]

    if native_markov:
        # Prefer results matching the DAG-fix lambda_sparse (consistent across all plots)
        preferred_ls = PREFERRED_LAMBDA_SPARSE.get(dataset.upper())
        if preferred_ls is not None:
            preferred = [r for r in native_markov
                         if r.get('config', {}).get('lambda_sparse') == preferred_ls]
            if preferred:
                return max(preferred, key=lambda r: r.get('config', {}).get('n_samples', 0))

        # Fallback: prefer results with the most samples (latest data)
        return max(native_markov, key=lambda r: r.get('config', {}).get('n_samples', 0))

    # Next, prefer any non-collapsed result
    non_collapsed = [
        r for r in matching
        if not r.get('regime_statistics', {}).get('regime_collapsed', True)
        or r.get('config', {}).get('n_regimes_final', 1) >= n_regimes
    ]
    if non_collapsed:
        return max(non_collapsed, key=lambda r: r.get('config', {}).get('n_samples', 0))

    return matching[0]


def normalize_regime_labels(result: dict, df: pd.DataFrame) -> dict:
    """Normalize regime labels so Regime 0 = stable, Regime 1 = crisis.

    Uses price volatility as discriminator: higher variance = crisis regime.
    This ensures consistent regime labeling across different countries/datasets.
    """
    import copy
    result = copy.deepcopy(result)  # Don't modify original

    assignments = result.get('regime_assignments', [])
    if not assignments:
        return result

    n_regimes = result.get('config', {}).get('n_regimes_final', 1)
    if n_regimes < 2:
        return result

    # Determine price column - handle DE_FR spread case
    dataset = result.get('config', {}).get('dataset', '').upper()
    if dataset == 'DE_FR' and 'price_spread' in df.columns:
        price_col = 'price_spread'
    elif 'Day_Ahead_Price' in df.columns:
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

    # For 2 regimes: Regime 0 should be stable (lower variance), Regime 1 = crisis (higher variance)
    if n_regimes == 2 and len(variances) == 2:
        if variances.get(0, 0) > variances.get(1, 0):
            # Swap labels: 0->1, 1->0
            result['regime_assignments'] = [1 - r for r in assignments]
            print(f"    Swapped regime labels (var0={variances[0]:.2f} > var1={variances[1]:.2f})")

            # Also swap edge_statistics regime info
            edge_stats = result.get('edge_statistics', {})
            regimes_list = edge_stats.get('regimes', [])
            if len(regimes_list) == 2:
                # Swap regime indices
                for regime_info in regimes_list:
                    regime_info['regime'] = 1 - regime_info['regime']
                # Re-sort by regime index
                regimes_list.sort(key=lambda x: x.get('regime', 0))

    return result


def detect_major_changepoints(regime_assignments, dates, min_duration=60, max_points=6):
    """Detect major regime transition points from assignment data.

    Args:
        regime_assignments: Array of regime labels (0=stable, 1=crisis after normalization)
        dates: Corresponding datetime array (same length as regime_assignments)
        min_duration: Minimum segment duration (in samples) to qualify as a major switch
        max_points: Maximum number of changepoints to return
    Returns:
        List of dicts with index, date, from_regime, to_regime, prev_duration
    """
    changepoints = []
    current_regime = regime_assignments[0]
    regime_start = 0

    for t in range(1, len(regime_assignments)):
        if regime_assignments[t] != current_regime:
            duration = t - regime_start
            if duration >= min_duration:
                changepoints.append({
                    'index': t,
                    'date': pd.Timestamp(dates[t]),
                    'from_regime': int(current_regime),
                    'to_regime': int(regime_assignments[t]),
                    'prev_duration': duration,
                })
            current_regime = regime_assignments[t]
            regime_start = t

    # If too many, keep only those with longest preceding segments
    if len(changepoints) > max_points:
        changepoints.sort(key=lambda cp: cp['prev_duration'], reverse=True)
        changepoints = changepoints[:max_points]
        changepoints.sort(key=lambda cp: cp['index'])
    return changepoints


def map_changepoint_to_event(cp_date, from_regime, to_regime):
    """Map a detected changepoint date to an economic event label.

    Args:
        cp_date: Timestamp of the changepoint
        from_regime: Regime being exited (0=stable, 1=crisis)
        to_regime: Regime being entered
    Returns:
        String label for the event
    """
    entering_crisis = (to_regime > from_regime)
    for range_start, range_end, crisis_label, normal_label in ECONOMIC_EVENTS:
        if range_start <= cp_date <= range_end:
            return crisis_label if entering_crisis else normal_label
    return cp_date.strftime('%b %Y')


def add_regime_switch_annotations(ax1, ax2, changepoints, labels):
    """Add regime switch annotations (vertical lines + labels) to both axes.

    Args:
        ax1: Top axis (price time series)
        ax2: Bottom axis (regime indicator)
        changepoints: List of changepoint dicts from detect_major_changepoints()
        labels: List of string labels (same length as changepoints)
    """
    for cp, label in zip(changepoints, labels):
        cp_date = cp['date']
        for ax in [ax1, ax2]:
            ax.axvline(x=cp_date, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        y_top = ax1.get_ylim()[1]
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


def plot_regime_timeseries(results: list, dataset: str, output_path: str):
    """
    Create regime time series visualization.
    Shows price time series colored by detected regime with probability curves.
    """
    # Search for best result across all lags, preferring native_markov with multiple regimes
    result = None
    for lag in [1, 2, 3]:
        for n_reg in [2, 3]:
            candidate = get_best_result(results, dataset, n_regimes=n_reg, lag=lag)
            if candidate:
                # Strongly prefer native_markov results (they preserve regimes)
                is_native = candidate.get('config', {}).get('regime_method') == 'native_markov'
                n_final = candidate.get('config', {}).get('n_regimes_final', 1)
                is_non_collapsed = n_final >= n_reg

                if is_native and is_non_collapsed:
                    result = candidate
                    break
                elif is_non_collapsed and result is None:
                    result = candidate
        if result and result.get('config', {}).get('regime_method') == 'native_markov':
            break

    if not result:
        print(f"No non-collapsed results found for {dataset}")
        return

    print(f"  Using result: lag={result.get('config', {}).get('lag')}, "
          f"method={result.get('config', {}).get('regime_method', 'bem')}, "
          f"n_regimes_final={result.get('config', {}).get('n_regimes_final')}")

    # Load data
    df = load_unified_data(dataset)
    if df is None:
        print(f"Could not load data for {dataset}")
        return

    # Normalize regime labels for consistency across datasets
    result = normalize_regime_labels(result, df)

    # Get regime assignments
    regime_assignments = result.get('regime_assignments', [])
    n_regimes = result.get('config', {}).get('n_regimes_final', 1)

    # Align data with regime assignments (accounting for lag)
    lag = result.get('config', {}).get('lag', 1)
    n_samples = len(regime_assignments)

    # Get price column - handle DE_FR spread case
    if dataset.upper() == 'DE_FR' and 'price_spread' in df.columns:
        price_col = 'price_spread'
        ylabel = 'Price Spread DE-FR (EUR/MWh)'
        title = 'DE-FR Price Spread with Detected Regimes'
    elif 'Day_Ahead_Price' in df.columns:
        price_col = 'Day_Ahead_Price'
        ylabel = 'Price (EUR/MWh)'
        title = f'{dataset} Day-Ahead Price with Detected Regimes'
    elif 'DE_Day_Ahead_Price' in df.columns:
        price_col = 'DE_Day_Ahead_Price'
        ylabel = 'DE Price (EUR/MWh)'
        title = f'{dataset} Day-Ahead Price with Detected Regimes'
    else:
        price_col = df.columns[0]
        ylabel = 'Value'
        title = f'{dataset} with Detected Regimes'

    print(f"    Using price column: {price_col}")
    prices = df[price_col].values[lag:lag + n_samples]
    dates = df.index.to_numpy()[lag:lag + n_samples]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_WIDE, gridspec_kw={'height_ratios': [3, 1]})

    # Plot 1: Price time series with regime coloring
    ax1 = axes[0]

    # Color each segment by regime
    regime_colors = [COLORS[f'regime{r % 4}'] for r in range(n_regimes)]

    for i in range(len(prices) - 1):
        regime = regime_assignments[i] if i < len(regime_assignments) else 0
        ax1.plot(dates[i:i+2], prices[i:i+2], color=regime_colors[regime], linewidth=0.8)

    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # Add legend for regimes with economic interpretation
    from matplotlib.lines import Line2D
    regime_labels = ['Regime 0 (Stable)', 'Regime 1 (Crisis)'] + [f'Regime {r}' for r in range(2, n_regimes)]
    legend_elements = [
        Line2D([0], [0], color=regime_colors[r], linewidth=2, label=regime_labels[r])
        for r in range(n_regimes)
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # Detect major regime switches (used for annotations below)
    changepoints = detect_major_changepoints(
        regime_assignments, dates, min_duration=60, max_points=6)
    labels = [
        map_changepoint_to_event(cp['date'], cp['from_regime'], cp['to_regime'])
        for cp in changepoints
    ]

    # Plot 2: Regime assignment over time
    ax2 = axes[1]

    # Create binary regime indicators
    for r in range(n_regimes):
        regime_mask = np.array(regime_assignments) == r
        ax2.fill_between(dates[:len(regime_mask)], 0, regime_mask.astype(float),
                         alpha=0.7, color=regime_colors[r],
                         label=REGIME_LABELS[r] if r < len(REGIME_LABELS) else f'Regime {r}')

    ax2.set_ylabel('Regime', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # Add regime switch annotations to both panels
    add_regime_switch_annotations(ax1, ax2, changepoints, labels)

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_importance(results: list, dataset: str, output_path: str, top_n: int = 10,
                            xlim: float = None):
    """
    Create feature importance bar chart.
    Shows direct causal edge weight from each feature TO Price, grouped by regime.

    Args:
        results: List of CaRS result dictionaries
        dataset: Dataset name (DE, FR, DE_FR)
        output_path: Path to save SVG
        top_n: Number of top features to show
        xlim: Optional x-axis limit for consistent scale across plots
    """
    # Aggregate direct causal edge weights TO Price from adjacency matrices
    importance = defaultdict(lambda: defaultdict(list))
    results_used = 0

    for r in results:
        if r.get('config', {}).get('dataset', '').upper() != dataset.upper():
            continue

        # Only use d=2 results to match regime time series plots
        n_regimes_requested = r.get('config', {}).get('n_regimes_requested', 0)
        if n_regimes_requested != 2:
            continue

        # Skip collapsed results - only use results with multiple regimes
        n_regimes_final = r.get('config', {}).get('n_regimes_final', 1)
        regime_collapsed = r.get('regime_statistics', {}).get('regime_collapsed', True)
        if n_regimes_final < 2 or regime_collapsed:
            continue

        # Only use ds3m_causal_native_markov results for consistent scale across datasets
        model = r.get('config', {}).get('model', 'unknown')
        method = r.get('config', {}).get('regime_method', 'bem')
        if model != 'ds3m_causal' or method != 'native_markov':
            continue

        adj_matrices = r.get('adjacency_matrices', [])
        feature_names = r.get('feature_names', [])
        if not adj_matrices or not feature_names:
            continue

        # Backward compat: old results have n-1 node adj but n feature_names
        n_adj = len(adj_matrices[0][0])
        if n_adj < len(feature_names):
            feature_names = feature_names[:n_adj]

        # Find price index (first column = Day_Ahead_Price)
        price_idx = 0
        for idx, name in enumerate(feature_names):
            if 'Price' in name or 'price' in name:
                price_idx = idx
                break

        # Check if regime labels need swapping (normalize: 0=stable, 1=crisis)
        df = load_unified_data(dataset)
        swapped = False
        if df is not None:
            import copy
            r_copy = copy.deepcopy(r)
            r_normalized = normalize_regime_labels(r_copy, df)
            orig_assignments = r.get('regime_assignments', [])
            norm_assignments = r_normalized.get('regime_assignments', [])
            if orig_assignments and norm_assignments and orig_assignments[0] != norm_assignments[0]:
                swapped = True

        results_used += 1
        for regime_idx, adj in enumerate(adj_matrices):
            adj = np.array(adj)  # Shape: (lag+1, n_nodes, n_nodes)
            # Apply regime swap for consistent labeling
            display_regime = (1 - regime_idx) if swapped else regime_idx

            for i, fname in enumerate(feature_names):
                if i == price_idx:
                    continue  # Skip Price→Price self-edge
                # Direct causal weight: feature i → Price (summed over all lags)
                weight = float(np.sum(np.abs(adj[:, i, price_idx])))
                importance[display_regime][fname].append(weight)

    print(f"  Using {results_used} non-collapsed results for feature importance")

    if not importance:
        print(f"No feature importance data for {dataset}")
        return

    # Average across runs
    avg_importance = {}
    for regime, feat_dict in importance.items():
        avg_importance[regime] = {
            feat: np.mean(vals) for feat, vals in feat_dict.items()
        }

    # Get top features (union across regimes)
    all_features = set()
    for regime_data in avg_importance.values():
        sorted_feats = sorted(regime_data.items(), key=lambda x: -x[1])[:top_n]
        all_features.update([f for f, _ in sorted_feats])

    # Sort features by max importance across regimes
    feature_max = {}
    for feat in all_features:
        feature_max[feat] = max(
            avg_importance.get(r, {}).get(feat, 0)
            for r in avg_importance.keys()
        )
    sorted_features = sorted(feature_max.items(), key=lambda x: -x[1])[:top_n]
    top_features = [f for f, _ in sorted_features]

    # Create plot
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    n_regimes = len(avg_importance)
    x = np.arange(len(top_features))
    width = 0.8 / n_regimes

    for i, (regime, feat_data) in enumerate(sorted(avg_importance.items())):
        values = [feat_data.get(f, 0) for f in top_features]
        offset = (i - (n_regimes - 1) / 2) * width
        regime_label = REGIME_LABELS[regime] if regime < len(REGIME_LABELS) else f'Regime {regime}'
        bars = ax.barh(x + offset, values, width, label=regime_label,
                       color=COLORS[f'regime{regime % 4}'], alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels([f[:25] + '...' if len(f) > 25 else f for f in top_features], fontsize=10)
    ax.set_xlabel('Direct Causal Edge Weight → Price', fontsize=12)
    ax.set_title(f'{dataset}: Top Causal Features by Regime', fontsize=14)
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    # Apply consistent x-axis limit if provided
    if xlim:
        ax.set_xlim(0, xlim * 1.05)  # Add 5% padding

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_lag_analysis(results: list, output_path: str):
    """
    Create lag analysis comparison.
    Shows edge count across lag=1,2,3 settings by dataset.
    """
    # Aggregate by dataset and lag
    data = defaultdict(lambda: defaultdict(list))

    for r in results:
        config = r.get('config', {})
        dataset = config.get('dataset', 'unknown')
        lag = config.get('lag', 1)

        edge_stats = r.get('edge_statistics', {})
        for regime_info in edge_stats.get('regimes', []):
            total_edges = regime_info.get('total_edges', 0)
            data[(dataset, lag)]['total_edges'].append(total_edges)

    if not data:
        print("No data for lag analysis")
        return

    # Prepare plot data
    dataset_order = ['DE', 'FR', 'DE_FR']
    available = set(k[0] for k in data.keys())
    datasets = [d for d in dataset_order if d in available]
    lags = sorted(set(k[1] for k in data.keys()))

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    x = np.arange(len(lags))
    width = 0.25
    offsets = np.linspace(-width, width, len(datasets))

    for i, dataset in enumerate(datasets):
        means = []
        stds = []
        for lag in lags:
            values = data.get((dataset, lag), {}).get('total_edges', [0])
            means.append(np.mean(values))
            stds.append(np.std(values))

        ax.bar(x + offsets[i], means, width, label=dataset,
               yerr=stds, capsize=3, alpha=0.8)

    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Total Edges (avg across regimes)', fontsize=12)
    ax.set_title('Effect of Lag on Learned Graph Sparsity', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'lag={l}' for l in lags])
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_lag_analysis_by_model(results: list, output_dir: str):
    """Create separate lag analysis plots for each model type."""
    # Group results by model/method
    model_results = defaultdict(list)
    for r in results:
        config = r.get('config', {})
        model = config.get('model', 'unknown')
        method = config.get('regime_method', 'bem')
        key = f"{model}_{method}"
        model_results[key].append(r)

    for model_key, model_data in model_results.items():
        # Check if we have multiple lags
        lags = set(r.get('config', {}).get('lag', 1) for r in model_data)
        if len(lags) < 2:
            print(f"  Skipping {model_key}: only {len(lags)} lag value(s)")
            continue

        output_path = os.path.join(output_dir, f'lag_comparison_{model_key}.svg')
        print(f"  Generating lag plot for {model_key} ({len(model_data)} results)")
        plot_lag_analysis(model_data, output_path)


def compute_global_max_importance(results: list) -> float:
    """Compute global max of averaged direct causal edge weights TO Price.

    This mirrors the extraction logic in plot_feature_importance() to get the
    actual max value that will appear in the plots.
    """
    # Aggregate importance values by dataset, regime, and feature
    importance = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for r in results:
        dataset = r.get('config', {}).get('dataset', '').upper()
        n_regimes_requested = r.get('config', {}).get('n_regimes_requested', 0)
        if n_regimes_requested != 2:
            continue
        n_regimes_final = r.get('config', {}).get('n_regimes_final', 1)
        regime_collapsed = r.get('regime_statistics', {}).get('regime_collapsed', True)
        if n_regimes_final < 2 or regime_collapsed:
            continue
        model = r.get('config', {}).get('model', 'unknown')
        method = r.get('config', {}).get('regime_method', 'bem')
        if model != 'ds3m_causal' or method != 'native_markov':
            continue

        adj_matrices = r.get('adjacency_matrices', [])
        feature_names = r.get('feature_names', [])
        if not adj_matrices or not feature_names:
            continue

        # Backward compat: old results have n-1 node adj but n feature_names
        n_adj = len(adj_matrices[0][0])
        if n_adj < len(feature_names):
            feature_names = feature_names[:n_adj]

        # Find price index
        price_idx = 0
        for idx, name in enumerate(feature_names):
            if 'Price' in name or 'price' in name:
                price_idx = idx
                break

        for regime_idx, adj in enumerate(adj_matrices):
            adj = np.array(adj)
            for i, fname in enumerate(feature_names):
                if i == price_idx:
                    continue
                weight = float(np.sum(np.abs(adj[:, i, price_idx])))
                importance[dataset][regime_idx][fname].append(weight)

    # Find max of averaged values
    max_val = 0
    for dataset, regime_dict in importance.items():
        for regime, feat_dict in regime_dict.items():
            for feat_name, values in feat_dict.items():
                avg_val = np.mean(values)
                max_val = max(max_val, avg_val)

    return max_val


def main():
    print("=" * 80)
    print("Generating Appendix Plots for CaRS Presentation")
    print("=" * 80)

    # Ensure output directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load all results (includes FANTOM, DS3MCausal with native_markov and bem)
    # This ensures we get lag=2,3 data from all available sources
    results = load_results(model_filter=None)
    print(f"Loaded {len(results)} total result files")

    # Summarize by model type and regime method
    model_counts = {}
    for r in results:
        config = r.get('config', {})
        model = config.get('model', 'unknown')
        method = config.get('regime_method', 'bem')  # Default to bem for older results
        key = f"{model}_{method}"
        model_counts[key] = model_counts.get(key, 0) + 1
    print(f"Results by model/method: {model_counts}")

    # Compute global max for consistent x-axis across feature importance plots
    global_max_importance = compute_global_max_importance(results)
    print(f"Global max importance: {global_max_importance:.6f}")

    # Generate plots for each dataset
    datasets = ['DE', 'FR', 'DE_FR']

    for dataset in datasets:
        print(f"\n--- Generating plots for {dataset} ---")

        # 1. Regime time series
        output_path = os.path.join(FIGURES_DIR, f'regime_timeseries_{dataset.lower()}.svg')
        try:
            plot_regime_timeseries(results, dataset, output_path)
        except Exception as e:
            print(f"Error generating regime timeseries for {dataset}: {e}")

        # 2. Feature importance
        output_path = os.path.join(FIGURES_DIR, f'feature_importance_{dataset.lower()}.svg')
        try:
            plot_feature_importance(results, dataset, output_path, xlim=global_max_importance)
        except Exception as e:
            print(f"Error generating feature importance for {dataset}: {e}")

    # 3. Lag analysis (all datasets combined)
    print("\n--- Generating lag analysis plot ---")
    output_path = os.path.join(FIGURES_DIR, 'lag_comparison.svg')
    try:
        plot_lag_analysis(results, output_path)
    except Exception as e:
        print(f"Error generating lag analysis: {e}")

    # 4. Lag analysis per model type
    print("\n--- Generating lag analysis plots per model ---")
    try:
        plot_lag_analysis_by_model(results, FIGURES_DIR)
    except Exception as e:
        print(f"Error generating per-model lag analysis: {e}")

    print("\n" + "=" * 80)
    print(f"Plots saved to {FIGURES_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
