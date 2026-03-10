#!/usr/bin/env python3
"""
Aggregate DS3MCausal (CaRS) experiment results for presentation.

Parses JSON result files and extracts:
- Edge statistics per regime
- Feature importance rankings
- Sparsity by lag and lambda_sparse settings
- Regime collapse statistics

Usage:
    python aggregate_cars_results.py
"""

import os
import json
import glob
from collections import defaultdict
import numpy as np
import pandas as pd

# Results directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results/cars')


def load_all_results(results_dir: str, model_filter: str = None) -> list:
    """Load all JSON result files from the results directory."""
    pattern = os.path.join(results_dir, '*.json')
    files = glob.glob(pattern)

    results = []
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                data['_filepath'] = filepath
                data['_filename'] = os.path.basename(filepath)

                # Filter by model type if specified
                if model_filter:
                    model = data.get('config', {}).get('model', 'fantom')
                    if model != model_filter:
                        continue

                results.append(data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load {filepath}: {e}")

    return results


def aggregate_by_config(results: list) -> dict:
    """Aggregate results by dataset, n_regimes, lag, lambda_sparse."""
    aggregated = defaultdict(list)

    for r in results:
        config = r.get('config', {})
        key = (
            config.get('dataset', 'unknown'),
            config.get('n_regimes_requested', 0),
            config.get('lag', 1),
            config.get('lambda_sparse', 0.0),
        )
        aggregated[key].append(r)

    return dict(aggregated)


def compute_edge_statistics(results: list) -> pd.DataFrame:
    """Compute edge statistics across all results."""
    rows = []

    for r in results:
        config = r.get('config', {})
        edge_stats = r.get('edge_statistics', {})
        regime_stats = r.get('regime_statistics', {})

        for regime_info in edge_stats.get('regimes', []):
            row = {
                'dataset': config.get('dataset', 'unknown'),
                'model': config.get('model', 'fantom'),
                'n_regimes_requested': config.get('n_regimes_requested', 0),
                'n_regimes_final': config.get('n_regimes_final', 0),
                'lag': config.get('lag', 1),
                'lambda_sparse': config.get('lambda_sparse', 0.0),
                'seed': config.get('seed', 0),
                'regime': regime_info.get('regime', 0),
                'total_edges': regime_info.get('total_edges', 0),
                'inst_edges': regime_info.get('instantaneous', {}).get('n_edges', 0),
                'inst_avg_weight': regime_info.get('instantaneous', {}).get('avg_weight', 0),
                'regime_collapsed': regime_stats.get('regime_collapsed', False),
            }

            # Add lagged edge counts
            for lag_key, lag_info in regime_info.get('lagged', {}).items():
                lag_num = int(lag_key.split('_')[1])
                row[f'lag{lag_num}_edges'] = lag_info.get('n_edges', 0)
                row[f'lag{lag_num}_avg_weight'] = lag_info.get('avg_weight', 0)

            rows.append(row)

    return pd.DataFrame(rows)


def compute_feature_importance(results: list, top_n: int = 10) -> dict:
    """Compute top feature importance per dataset/regime."""
    importance = defaultdict(lambda: defaultdict(list))

    for r in results:
        config = r.get('config', {})
        dataset = config.get('dataset', 'unknown')
        edge_stats = r.get('edge_statistics', {})

        for regime_info in edge_stats.get('regimes', []):
            regime = regime_info.get('regime', 0)
            feat_imp = regime_info.get('feature_importance', {})

            for feat_name, imp_data in feat_imp.items():
                total = imp_data.get('total', 0) if isinstance(imp_data, dict) else imp_data
                importance[(dataset, regime)][feat_name].append(total)

    # Average across seeds and sort
    top_features = {}
    for (dataset, regime), feat_dict in importance.items():
        avg_importance = {
            feat: np.mean(vals) for feat, vals in feat_dict.items()
        }
        sorted_features = sorted(avg_importance.items(), key=lambda x: -x[1])
        top_features[(dataset, regime)] = sorted_features[:top_n]

    return top_features


def compute_collapse_rates(results: list) -> pd.DataFrame:
    """Compute regime collapse rates by configuration."""
    rows = []

    aggregated = aggregate_by_config(results)
    for (dataset, n_regimes, lag, lambda_sparse), run_results in aggregated.items():
        n_runs = len(run_results)
        n_collapsed = sum(
            1 for r in run_results
            if r.get('regime_statistics', {}).get('regime_collapsed', False)
        )

        rows.append({
            'dataset': dataset,
            'n_regimes_requested': n_regimes,
            'lag': lag,
            'lambda_sparse': lambda_sparse,
            'n_runs': n_runs,
            'n_collapsed': n_collapsed,
            'collapse_rate': n_collapsed / n_runs if n_runs > 0 else 0,
        })

    return pd.DataFrame(rows)


def summarize_sparsity_by_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize edge counts by lag setting."""
    summary = df.groupby(['dataset', 'model', 'lag', 'lambda_sparse']).agg({
        'total_edges': ['mean', 'std'],
        'inst_edges': ['mean', 'std'],
        'regime_collapsed': ['sum', 'count'],
    }).reset_index()

    # Flatten column names
    summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary


def main():
    print("=" * 80)
    print("Aggregating CaRS (DS3MCausal) Results")
    print("=" * 80)

    # Load all results
    all_results = load_all_results(RESULTS_DIR)
    print(f"Loaded {len(all_results)} total result files")

    # Filter to DS3MCausal only
    ds3m_results = load_all_results(RESULTS_DIR, model_filter='ds3m_causal')
    print(f"Found {len(ds3m_results)} DS3MCausal result files")

    # Also include FANTOM results for comparison
    fantom_results = load_all_results(RESULTS_DIR, model_filter='fantom')
    print(f"Found {len(fantom_results)} FANTOM result files")

    # Compute statistics
    print("\n--- Edge Statistics ---")
    edge_df = compute_edge_statistics(all_results)
    print(f"Edge statistics DataFrame shape: {edge_df.shape}")

    # Summary by model and dataset
    print("\nSummary by Model and Dataset:")
    summary = edge_df.groupby(['model', 'dataset']).agg({
        'total_edges': ['mean', 'std', 'min', 'max'],
        'regime_collapsed': ['sum', 'count'],
    })
    print(summary)

    # Sparsity by lag
    print("\n--- Sparsity by Lag ---")
    sparsity_df = summarize_sparsity_by_lag(edge_df)
    print(sparsity_df.head(20))

    # Collapse rates
    print("\n--- Regime Collapse Rates ---")
    collapse_df = compute_collapse_rates(all_results)
    print(collapse_df.sort_values(['dataset', 'n_regimes_requested', 'lag']))

    # Top features
    print("\n--- Top Feature Importance (DS3MCausal) ---")
    top_features = compute_feature_importance(ds3m_results, top_n=5)
    for (dataset, regime), features in sorted(top_features.items()):
        print(f"\n{dataset} Regime {regime}:")
        for feat, imp in features:
            print(f"  {feat}: {imp:.4f}")

    # Save outputs
    output_dir = os.path.join(os.path.dirname(__file__), '../results')

    edge_df.to_csv(os.path.join(output_dir, 'edge_statistics.csv'), index=False)
    sparsity_df.to_csv(os.path.join(output_dir, 'sparsity_by_lag.csv'), index=False)
    collapse_df.to_csv(os.path.join(output_dir, 'collapse_rates.csv'), index=False)

    # Save top features as JSON
    top_features_json = {
        f"{dataset}_regime{regime}": [
            {"feature": f, "importance": float(i)} for f, i in features
        ]
        for (dataset, regime), features in top_features.items()
    }
    with open(os.path.join(output_dir, 'top_features.json'), 'w') as f:
        json.dump(top_features_json, f, indent=2)

    print(f"\nOutputs saved to {output_dir}/")
    print("  - edge_statistics.csv")
    print("  - sparsity_by_lag.csv")
    print("  - collapse_rates.csv")
    print("  - top_features.json")

    # Generate LaTeX table for presentation
    print("\n--- LaTeX Table: Sparsity by Lag (DS3MCausal) ---")
    ds3m_edge_df = edge_df[edge_df['model'] == 'ds3m_causal']
    if len(ds3m_edge_df) > 0:
        latex_summary = ds3m_edge_df.groupby(['dataset', 'lag']).agg({
            'total_edges': ['mean', 'std'],
            'inst_edges': ['mean', 'std'],
        }).round(1)
        print(latex_summary.to_latex())
    else:
        print("No DS3MCausal results found for LaTeX table")


if __name__ == "__main__":
    main()
