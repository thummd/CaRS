"""
Run all baseline models on the same data splits used by CaRS.

Loads data via prepare_unified_ds3m_data() for identical splits,
runs MS-VAR, Lasso/regime, and XGBoost, and saves results as JSON
compatible with CaRS output format.

Usage:
    python -m electricity.baselines.run_baselines --country DE
    python -m electricity.baselines.run_baselines --country DE --baselines xgboost,lasso
    python -m electricity.baselines.run_baselines --countries DE,FR,NL --baselines all
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from paths import OUTPUT_DIR
from shared_backbone.data_loader import prepare_unified_ds3m_data
from electricity.evaluation.metrics import compute_all_metrics


RESULTS_DIR = OUTPUT_DIR / "baselines"


def flatten_ds3m_input(X: np.ndarray) -> np.ndarray:
    """
    Flatten DS3M-format temporal windows for non-sequential baselines.

    Takes the last timestep's features (most recent observation)
    as input for regression baselines, consistent with predicting
    the next-step target.

    Args:
        X: [timestep, batch, features]

    Returns:
        [batch, features] using last timestep
    """
    return X[-1]  # Last timestep: [batch, features]


def run_xgboost(data: dict, seed: int = 42) -> dict:
    """Run XGBoost baseline."""
    from electricity.baselines.xgboost_baseline import XGBoostBaseline

    model = XGBoostBaseline(random_state=seed)

    trainX = data['trainX'].numpy()
    trainY = data['trainY'].numpy()
    valX = data['valX'].numpy()
    valY = data['valY'].numpy()
    testX = data['testX'].numpy()
    testY = data['testY'].numpy()

    model.fit(trainX, trainY, X_val=valX, Y_val=valY)
    result = model.predict(testX)

    # Compute metrics
    y_true = testY[-1, :, 0]  # Last timestep target
    y_pred = result['predictions']
    y_prev = testY[-2, :, 0]  # Previous timestep for directional accuracy

    metrics = compute_all_metrics(y_true, y_pred, y_prev=y_prev)

    return {
        'model': 'XGBoost',
        'metrics': metrics,
        'predictions': y_pred,
        'y_true': y_true,
        'feature_importance': model.get_feature_importance().tolist()
    }


def run_lasso(data: dict, seed: int = 42) -> dict:
    """Run Lasso per Regime baseline."""
    from electricity.baselines.lasso_per_regime import LassoPerRegimeBaseline

    model = LassoPerRegimeBaseline(random_state=seed)

    trainX = flatten_ds3m_input(data['trainX'].numpy())
    trainY = data['trainY'].numpy()[-1, :, 0]
    testX = flatten_ds3m_input(data['testX'].numpy())
    testY = data['testY'].numpy()

    # Full Y history for regime assignment
    Y_all = np.concatenate([
        data['trainY'].numpy()[-1, :, 0],
        data['valY'].numpy()[-1, :, 0],
        data['testY'].numpy()[-1, :, 0]
    ])

    model.fit(trainX, trainY)
    n_test = len(testX)
    result = model.predict(testX, Y_history=Y_all[-n_test:])

    y_true = testY[-1, :, 0]
    y_pred = result['predictions']
    y_prev = testY[-2, :, 0]

    metrics = compute_all_metrics(y_true, y_pred, y_prev=y_prev)

    # Get selected features per regime
    selected = model.get_selected_features()
    selected_serializable = {}
    for r, info in selected.items():
        selected_serializable[str(r)] = {
            'n_nonzero': len(info['nonzero_idx']),
            'alpha': float(info['alpha'])
        }

    return {
        'model': 'Lasso_per_Regime',
        'metrics': metrics,
        'predictions': y_pred,
        'y_true': y_true,
        'regimes': result['regimes'].tolist(),
        'selected_features': selected_serializable
    }


def run_msvar(data: dict, seed: int = 42) -> dict:
    """Run MS-VAR baseline."""
    from electricity.baselines.ms_var import MSVARBaseline

    model = MSVARBaseline(n_regimes=2, order=4)

    trainX = flatten_ds3m_input(data['trainX'].numpy())
    trainY = data['trainY'].numpy()[-1, :, 0]
    testX = flatten_ds3m_input(data['testX'].numpy())
    testY = data['testY'].numpy()

    model.fit(trainX, trainY)
    result = model.predict(testX, Y_history=trainY)

    y_true = testY[-1, :, 0]
    y_pred = result['predictions']
    y_prev = testY[-2, :, 0]

    metrics = compute_all_metrics(y_true, y_pred, y_prev=y_prev)

    return {
        'model': 'MS-VAR',
        'metrics': metrics,
        'predictions': y_pred,
        'y_true': y_true,
        'regimes': result['regimes'].tolist()
    }


def _naive_metrics(y_true, y_pred, y_prev, model_name):
    metrics = compute_all_metrics(y_true, y_pred, y_prev=y_prev)
    return {'model': model_name, 'metrics': metrics, 'predictions': y_pred, 'y_true': y_true}


def run_naive_zero(data: dict, seed: int = 42) -> dict:
    """Naive-zero: predict zero return (i.e., 'no change')."""
    testY = data['testY'].numpy()
    y_true = testY[-1, :, 0]
    y_pred = np.zeros_like(y_true)
    y_prev = testY[-2, :, 0]
    return _naive_metrics(y_true, y_pred, y_prev, 'Naive-Zero')


def run_naive_mean(data: dict, seed: int = 42) -> dict:
    """Naive-mean: predict the training-set mean of the target."""
    trainY = data['trainY'].numpy()[-1, :, 0]
    testY = data['testY'].numpy()
    y_true = testY[-1, :, 0]
    y_pred = np.full_like(y_true, fill_value=float(trainY.mean()))
    y_prev = testY[-2, :, 0]
    return _naive_metrics(y_true, y_pred, y_prev, 'Naive-Mean')


def run_naive_persistence(data: dict, seed: int = 42) -> dict:
    """Naive-persistence: predict that the target equals its most recent observed value.

    For a `horizon`-step-ahead return target the most recent observed *target value* at
    prediction time is testY[-(1+horizon), :, 0] (the entry that was 'today' when the
    h-step-ahead label was constructed). For h=1 this collapses to testY[-2, :, 0],
    matching `y_prev`.
    """
    testY = data['testY'].numpy()
    horizon = data.get('horizon', 1)
    idx = -(1 + horizon)
    if abs(idx) >= testY.shape[0]:
        idx = 0  # fall back to oldest available
    y_pred = testY[idx, :, 0]
    y_true = testY[-1, :, 0]
    y_prev = testY[-2, :, 0]
    return _naive_metrics(y_true, y_pred, y_prev, 'Naive-Persistence')


BASELINE_RUNNERS = {
    'naive_zero':        run_naive_zero,
    'naive_mean':        run_naive_mean,
    'naive_persistence': run_naive_persistence,
    'xgboost': run_xgboost,
    'lasso': run_lasso,
    'msvar': run_msvar,
}


def run_baselines(
    country: str,
    baselines: list = None,
    feature_groups: list = None,
    seed: int = 42,
    save: bool = True,
    horizon: int = 1,
    target_col: str = None,
    spillover: bool = False,
    frequency: str = 'H',
) -> dict:
    """
    Run specified baselines on a country dataset.

    Args:
        country: Country code (DE, FR, etc.)
        baselines: List of baseline names. None = all.
        feature_groups: Feature groups to load. None = default.
        seed: Random seed
        save: Whether to save results to disk
        horizon: Forecast horizon in hours (1, 6, 12, 24)
        target_col: Target column (None = default price_return)
        spillover: Include cross-border spillover features

    Returns:
        Dict mapping baseline name to results
    """
    if baselines is None:
        baselines = list(BASELINE_RUNNERS.keys())

    print(f"\n{'=' * 60}")
    print(f"Running baselines for {country} (horizon={horizon})")
    print(f"Baselines: {baselines}")
    print(f"{'=' * 60}")

    # Load data using same pipeline as CaRS
    data = prepare_unified_ds3m_data(
        country=country,
        timestep=14,
        feature_groups=feature_groups,
        horizon=horizon,
        target_col=target_col,
        spillover=spillover,
        frequency=frequency,
    )
    data['horizon'] = horizon

    print(f"Data loaded: train={data['trainX'].shape}, test={data['testX'].shape}")
    print(f"Features: {len(data['feature_cols'])} columns")

    all_results = {}
    for name in baselines:
        if name not in BASELINE_RUNNERS:
            print(f"WARNING: Unknown baseline '{name}', skipping")
            continue

        print(f"\n--- Running {name} ---")
        t0 = time.time()
        try:
            result = BASELINE_RUNNERS[name](data, seed=seed)
            elapsed = time.time() - t0
            result['elapsed_seconds'] = elapsed
            all_results[name] = result
            print(f"  RMSE: {result['metrics']['rmse']:.4f}")
            print(f"  MAE:  {result['metrics']['mae']:.4f}")
            print(f"  Spearman: {result['metrics']['spearman']:.4f}")
            if 'directional_accuracy' in result['metrics']:
                print(f"  DirAcc: {result['metrics']['directional_accuracy']:.4f}")
            print(f"  Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[name] = {'model': name, 'error': str(e)}

    if save:
        if frequency == 'D':
            save_dir = RESULTS_DIR / 'daily' / country / f'h{horizon}'
        else:
            h_suffix = f"_h{horizon}" if horizon > 1 else ""
            save_dir = RESULTS_DIR / f"{country}{h_suffix}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics summary (JSON-serializable)
        summary = {}
        for name, result in all_results.items():
            if 'error' in result:
                summary[name] = {'error': result['error']}
            else:
                summary[name] = {
                    'metrics': result['metrics'],
                    'elapsed_seconds': result.get('elapsed_seconds'),
                }
        summary['_meta'] = {'horizon': horizon, 'target_col': data.get('target_col', 'unknown')}
        summary_path = save_dir / f"baseline_results_seed{seed}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {summary_path}")

        # Save predictions as .npy
        for name, result in all_results.items():
            if 'predictions' in result:
                np.save(
                    save_dir / f"{name}_predictions_seed{seed}.npy",
                    result['predictions']
                )

    return all_results


def generate_comparison_table(
    countries: list,
    baselines: list = None,
    seed: int = 42
) -> str:
    """
    Generate a LaTeX comparison table from saved baseline results.

    Returns LaTeX string for the forecasting comparison table.
    """
    if baselines is None:
        baselines = list(BASELINE_RUNNERS.keys())

    rows = []
    for country in countries:
        results_path = RESULTS_DIR / country / f"baseline_results_seed{seed}.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            results = json.load(f)

        row = [country]
        for bl in baselines:
            if bl in results and 'metrics' in results[bl]:
                m = results[bl]['metrics']
                row.append(f"{m['rmse']:.2f}")
                row.append(f"{m['mae']:.2f}")
                row.append(f"{m['spearman']:.3f}")
            else:
                row.extend(['--', '--', '--'])
        rows.append(row)

    # Build LaTeX table
    n_bl = len(baselines)
    col_spec = 'l' + 'rrr' * n_bl
    header_parts = ['Country']
    for bl in baselines:
        header_parts.extend([f'\\multicolumn{{3}}{{c}}{{{bl}}}'])

    subheader = [''] + ['RMSE', 'MAE', 'Spearman'] * n_bl

    lines = [
        '\\begin{tabular}{' + col_spec + '}',
        '\\toprule',
        ' & '.join(header_parts) + ' \\\\',
        ' & '.join(subheader) + ' \\\\',
        '\\midrule',
    ]
    for row in rows:
        lines.append(' & '.join(row) + ' \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular}'])

    return '\n'.join(lines)


def generate_multiseed_comparison_table(
    countries: list,
    seeds: list,
    baselines: list = None,
    metrics: list = None,
) -> str:
    """
    Generate a LaTeX comparison table aggregated across multiple seeds.

    Loads per-seed baseline results, computes mean±std for each metric,
    and formats a LaTeX table matching the CaRS reporting style.

    Args:
        countries: List of country codes
        seeds: List of random seeds (e.g., [42, 123, 456, 789, 1011])
        baselines: Baseline names (default: all)
        metrics: Metric names to include
            (default: all available — DirAcc, Spearman, RMSE, MAE, sMAPE)

    Returns:
        LaTeX table string with mean±std per baseline per market
    """
    if baselines is None:
        baselines = list(BASELINE_RUNNERS.keys())
    if metrics is None:
        metrics = ['directional_accuracy', 'spearman', 'rmse', 'mae', 'smape']

    # Collect per-seed per-country per-baseline metrics
    all_data = {}  # {country: {baseline: {metric: [values across seeds]}}}
    for country in countries:
        all_data[country] = {}
        for seed in seeds:
            path = RESULTS_DIR / country / f"baseline_results_seed{seed}.json"
            if not path.exists():
                continue
            with open(path) as f:
                seed_result = json.load(f)
            for bl in baselines:
                if bl not in seed_result or 'metrics' not in seed_result[bl]:
                    continue
                if bl not in all_data[country]:
                    all_data[country][bl] = {m: [] for m in metrics}
                m = seed_result[bl]['metrics']
                for metric in metrics:
                    val = m.get(metric)
                    if val is None and metric == 'directional_accuracy':
                        val = m.get('dir_accuracy')
                    if val is not None:
                        all_data[country][bl][metric].append(val)

    # Format table rows
    metric_labels = {
        'directional_accuracy': 'DirAcc',
        'spearman': 'Spearman',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'smape': 'sMAPE',
        'crps': 'CRPS',
    }
    # Precision per metric type
    metric_fmt = {
        'directional_accuracy': '.3f',
        'spearman': '.3f',
        'rmse': '.3f',
        'mae': '.3f',
        'smape': '.1f',
        'crps': '.4f',
    }

    rows = []
    for country in countries:
        if country not in all_data or not all_data[country]:
            continue
        row = [country]
        for bl in baselines:
            for metric in metrics:
                values = all_data.get(country, {}).get(bl, {}).get(metric, [])
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    fmt = metric_fmt.get(metric, '.3f')
                    if std < 0.001:
                        row.append(f"{mean:{fmt}}")
                    else:
                        row.append(f"{mean:{fmt}}$\\pm${std:{fmt}}")
                else:
                    row.append('--')
        rows.append(row)

    n_cols = len(baselines) * len(metrics)
    col_spec = 'l' + 'r' * n_cols

    header_parts = ['Market']
    for bl in baselines:
        bl_label = {'xgboost': 'XGB', 'lasso': 'Lasso', 'ms_var': 'MS-R'}.get(bl, bl)
        header_parts.append(
            f'\\multicolumn{{{len(metrics)}}}{{c}}{{\\textbf{{{bl_label}}}}}'
        )

    subheader = ['']
    for _ in baselines:
        for metric in metrics:
            subheader.append(f'\\textbf{{{metric_labels.get(metric, metric)}}}')

    lines = [
        '\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill} ' + col_spec + ' @{}}',
        '\\toprule',
        ' & '.join(header_parts) + ' \\\\',
        ' & '.join(subheader) + ' \\\\',
        '\\midrule',
    ]
    for row in rows:
        lines.append(' & '.join(row) + ' \\\\')

    # Average row
    if rows:
        avg_row = ['\\textbf{Avg.}']
        for col_idx in range(1, len(rows[0])):
            vals = []
            for row in rows:
                try:
                    val_str = row[col_idx].split('$')[0]
                    vals.append(float(val_str))
                except (ValueError, IndexError):
                    pass
            if vals:
                avg_row.append(f"{np.mean(vals):.3f}")
            else:
                avg_row.append('--')
        lines.append('\\midrule')
        lines.append(' & '.join(avg_row) + ' \\\\')

    lines.extend(['\\bottomrule', '\\end{tabular*}'])

    table = '\n'.join(lines)

    # Also print a human-readable summary
    print(f"\nMulti-seed baseline comparison ({len(seeds)} seeds)")
    print(f"{'Market':<6}", end='')
    for bl in baselines:
        bl_label = {'xgboost': 'XGB', 'lasso': 'Lasso', 'ms_var': 'MS-R'}.get(bl, bl)
        for metric in metrics:
            print(f"  {bl_label}/{metric_labels.get(metric, metric):>8}", end='')
    print()
    for row in rows:
        print(f"{row[0]:<6}" + ''.join(f"  {v:>14}" for v in row[1:]))

    return table


def main():
    parser = argparse.ArgumentParser(description='Run baseline models')
    parser.add_argument('--country', type=str, default='DE',
                        help='Country code (default: DE)')
    parser.add_argument('--countries', type=str, default=None,
                        help='Comma-separated country codes for batch run')
    parser.add_argument('--baselines', type=str, default='all',
                        help='Comma-separated baseline names or "all"')
    parser.add_argument('--feature-groups', type=str, default=None,
                        help='Comma-separated feature groups')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds for multi-seed run (e.g. 42,123,456,789,1011)')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Forecast horizon in hours (1, 6, 12, 24)')
    parser.add_argument('--target-col', type=str, default=None,
                        help='Target column (default: price_return)')
    parser.add_argument('--spillover', action='store_true',
                        help='Include cross-border spillover features')
    parser.add_argument('--frequency', type=str, default='H', choices=['H', 'D'],
                        help='Data frequency: H (hourly) or D (daily).')
    parser.add_argument('--no-save', action='store_true')

    args = parser.parse_args()

    if args.baselines == 'all':
        baselines = None
    else:
        baselines = args.baselines.split(',')

    feature_groups = args.feature_groups.split(',') if args.feature_groups else None

    countries = args.countries.split(',') if args.countries else [args.country]
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else [args.seed]

    for seed in seeds:
        for country in countries:
            run_baselines(
                country=country,
                baselines=baselines,
                feature_groups=feature_groups,
                seed=seed,
                save=not args.no_save,
                horizon=args.horizon,
                target_col=args.target_col,
                spillover=args.spillover,
                frequency=args.frequency,
            )

    # Generate comparison table if multiple countries
    if len(countries) > 1:
        if len(seeds) > 1:
            table = generate_multiseed_comparison_table(
                countries, seeds, baselines,
                metrics=['directional_accuracy', 'spearman']
            )
        else:
            table = generate_comparison_table(countries, baselines, seed=seeds[0])
        table_path = RESULTS_DIR / "comparison_table.tex"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, 'w') as f:
            f.write(table)
        print(f"\nComparison table saved to {table_path}")


if __name__ == '__main__':
    main()
