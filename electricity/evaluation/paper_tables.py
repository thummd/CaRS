"""
Paper Output Generation for CaRS Applied Energy Submission.

Aggregates results from all evaluation modules and generates
publication-ready LaTeX tables and figures.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from paths import OUTPUT_DIR


def _fmt(val, fmt='.2f'):
    """Format a numeric value, handling None."""
    if val is None:
        return '--'
    return f'{val:{fmt}}'


def _bold_best(values, fmt='.2f', lower_better=True):
    """Format values with the best one bolded in LaTeX."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return ['--'] * len(values)

    if lower_better:
        best_idx = min(valid, key=lambda x: x[1])[0]
    else:
        best_idx = max(valid, key=lambda x: x[1])[0]

    result = []
    for i, v in enumerate(values):
        s = _fmt(v, fmt)
        if i == best_idx and v is not None:
            s = f'\\textbf{{{s}}}'
        result.append(s)
    return result


# =============================================================================
# Table 1: Forecasting Comparison
# =============================================================================

def table_forecasting_comparison(
    cars_results: Dict[str, Dict],
    baseline_results: Dict[str, Dict],
    countries: List[str],
    metrics: List[str] = None,
) -> str:
    """
    Generate Table 1: CaRS vs baselines forecasting comparison.

    Args:
        cars_results: {country: {metric: value}} from CaRS
        baseline_results: {country: {baseline_name: {metric: value}}}
        countries: Countries to include
        metrics: Metrics to show (default: RMSE, MAE, Spearman, DirAcc)

    Returns:
        LaTeX table string
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'spearman', 'directional_accuracy']

    metric_labels = {
        'rmse': 'RMSE',
        'mae': 'MAE',
        'smape': 'sMAPE',
        'spearman': 'Spearman',
        'directional_accuracy': 'Dir.Acc.',
        'crps': 'CRPS',
    }

    # Determine baselines present
    all_baselines = set()
    for country_bl in baseline_results.values():
        all_baselines.update(country_bl.keys())
    baselines = sorted(all_baselines)

    model_names = baselines + ['CaRS']
    n_models = len(model_names)

    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Forecasting performance comparison across European electricity markets.}')
    lines.append('\\label{tab:forecasting_comparison}')
    lines.append('\\small')

    # Column spec: country + metrics for each model
    col_spec = 'l' + 'r' * (len(metrics) * n_models)
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')

    # Model header
    model_header = ['']
    for name in model_names:
        model_header.append(f'\\multicolumn{{{len(metrics)}}}{{c}}{{{name}}}')
    lines.append(' & '.join(model_header) + ' \\\\')

    # Metric subheader
    subheader = ['Country']
    for _ in model_names:
        for m in metrics:
            subheader.append(metric_labels.get(m, m))
    lines.append(' & '.join(subheader) + ' \\\\')
    lines.append('\\midrule')

    # Data rows
    for country in countries:
        row = [country]

        for m in metrics:
            values = []
            for bl in baselines:
                bl_data = baseline_results.get(country, {}).get(bl, {})
                values.append(bl_data.get(m))

            cars_data = cars_results.get(country, {})
            values.append(cars_data.get(m))

            lower_better = m in ['rmse', 'mae', 'smape', 'crps']
            fmt = '.3f' if m == 'spearman' else '.2f'
            formatted = _bold_best(values, fmt=fmt, lower_better=lower_better)

            # Distribute across model columns
            for i, f in enumerate(formatted):
                # This is metric m for model i
                pass

        # Simpler approach: iterate by model
        for model_name in model_names:
            if model_name == 'CaRS':
                data = cars_results.get(country, {})
            else:
                data = baseline_results.get(country, {}).get(model_name, {})

            for m in metrics:
                val = data.get(m)
                fmt = '.3f' if m == 'spearman' else '.2f'
                row.append(_fmt(val, fmt))

        lines.append(' & '.join(row) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


# =============================================================================
# Table 2: Regime-Stratified Metrics
# =============================================================================

def table_regime_stratified(
    stratified_results: Dict[str, Dict],
    countries: List[str],
) -> str:
    """
    Generate Table 2: Metrics stratified by normal vs crisis periods.

    Args:
        stratified_results: {country: output of regime_stratified_metrics()}
        countries: Countries to include

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{CaRS performance stratified by market period (normal vs.\\ crisis).}')
    lines.append('\\label{tab:regime_stratified}')
    lines.append('\\small')
    lines.append('\\begin{tabular}{l l rrrr r}')
    lines.append('\\toprule')
    lines.append('Country & Period & RMSE & MAE & Spearman & Dir.Acc. & $N$ \\\\')
    lines.append('\\midrule')

    for country in countries:
        data = stratified_results.get(country, {})
        first = True
        for period in ['normal', 'crisis', 'overall']:
            if period not in data:
                continue
            m = data[period]
            label = country if first else ''
            first = False
            n = m.get('n_samples', '--')
            lines.append(
                f'{label} & {period.capitalize()} & '
                f'{_fmt(m.get("rmse"))} & {_fmt(m.get("mae"))} & '
                f'{_fmt(m.get("spearman"), ".3f")} & '
                f'{_fmt(m.get("directional_accuracy"), ".3f")} & '
                f'{n} \\\\'
            )
        lines.append('\\midrule')

    # Remove last midrule
    lines[-1] = '\\bottomrule'
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


# =============================================================================
# Table 3: Edge Stability
# =============================================================================

def table_edge_stability(
    stability_results: Dict[str, Dict],
    countries: List[str],
) -> str:
    """
    Generate Table 3: Edge stability across random seeds.

    Args:
        stability_results: {country: output of compute_edge_stability()}
        countries: Countries to include

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Causal edge stability across 5 random seeds. '
                 'Stable edges appear in $\\geq$80\\% of runs.}')
    lines.append('\\label{tab:edge_stability}')
    lines.append('\\small')
    lines.append('\\begin{tabular}{l rr rr r}')
    lines.append('\\toprule')
    lines.append(' & \\multicolumn{2}{c}{Regime 0} & \\multicolumn{2}{c}{Regime 1} & \\\\')
    lines.append('\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}')
    lines.append('Country & Stable & Unstable & Stable & Unstable & Jaccard \\\\')
    lines.append('\\midrule')

    for country in countries:
        data = stability_results.get(country, {})
        summary = data.get('summary', {})
        jaccard = data.get('jaccard_similarity', {})

        s0 = summary.get('regime_0_stable', '--')
        u0 = summary.get('regime_0_unstable', '--')
        s1 = summary.get('regime_1_stable', '--')
        u1 = summary.get('regime_1_unstable', '--')
        j_mean = jaccard.get('regime_0', {}).get('mean', None)

        lines.append(
            f'{country} & {s0} & {u0} & {s1} & {u1} & {_fmt(j_mean, ".3f")} \\\\'
        )

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


# =============================================================================
# Table 4: Merit Order Alignment
# =============================================================================

def table_merit_order(
    merit_results: Dict[str, Dict],
    countries: List[str],
) -> str:
    """
    Generate Table 4: Merit order alignment by country and regime.

    Args:
        merit_results: {country: {regime_0: merit_order_alignment(), ...}}
        countries: Countries to include

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Merit order alignment: fraction of generation$\\rightarrow$price edges '
                 'with economically correct signs.}')
    lines.append('\\label{tab:merit_order}')
    lines.append('\\small')
    lines.append('\\begin{tabular}{l rr rr}')
    lines.append('\\toprule')
    lines.append(' & \\multicolumn{2}{c}{Regime 0} & \\multicolumn{2}{c}{Regime 1} \\\\')
    lines.append('\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}')
    lines.append('Country & Alignment & Rank $\\rho$ & Alignment & Rank $\\rho$ \\\\')
    lines.append('\\midrule')

    for country in countries:
        data = merit_results.get(country, {})
        r0 = data.get('regime_0', {})
        r1 = data.get('regime_1', {})

        a0 = r0.get('alignment_score')
        rc0 = r0.get('rank_correlation')
        a1 = r1.get('alignment_score')
        rc1 = r1.get('rank_correlation')

        a0_str = f'{a0:.0%}' if a0 is not None else '--'
        a1_str = f'{a1:.0%}' if a1 is not None else '--'

        lines.append(
            f'{country} & {a0_str} & {_fmt(rc0, ".3f")} & '
            f'{a1_str} & {_fmt(rc1, ".3f")} \\\\'
        )

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


# =============================================================================
# Table 5: Edge Ablation Impact
# =============================================================================

def table_ablation(
    ablation_results: Dict[str, Dict],
    countries: List[str],
    top_k: int = 5,
) -> str:
    """
    Generate Table 5: Top-k most impactful edge ablations.

    Args:
        ablation_results: {country: output of edge_ablation_analysis()}
        countries: Countries to include
        top_k: Number of top edges to show per country

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Edge ablation analysis: RMSE change when removing the '
                 f'top-{top_k} most impactful causal edges.}}')
    lines.append('\\label{tab:ablation}')
    lines.append('\\small')
    lines.append('\\begin{tabular}{l l rr r}')
    lines.append('\\toprule')
    lines.append('Country & Edge & Base RMSE & Ablated RMSE & $\\Delta$RMSE (\\%) \\\\')
    lines.append('\\midrule')

    for country in countries:
        data = ablation_results.get(country, {})
        abl_list = data.get('ablation_results', [])[:top_k]

        first = True
        for abl in abl_list:
            label = country if first else ''
            first = False

            tech = abl.get('technology', abl.get('source_name', '?'))
            edge_label = f'{tech} $\\rightarrow$ Price'

            lines.append(
                f'{label} & {edge_label} & '
                f'{_fmt(abl["baseline_rmse"])} & '
                f'{_fmt(abl["ablated_rmse"])} & '
                f'{_fmt(abl["delta_rmse_pct"], ".1f")} \\\\'
            )
        if abl_list:
            lines.append('\\midrule')

    # Remove last midrule
    if lines[-1] == '\\midrule':
        lines[-1] = '\\bottomrule'
    else:
        lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


# =============================================================================
# Table 6: Cross-Country Regime-Conditional Effects
# =============================================================================

def table_regime_effects(
    effects_df: pd.DataFrame,
) -> str:
    """
    Generate Table 6: Regime-conditional causal effects across countries.

    Args:
        effects_df: Output of cross_country_causal_table()

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Regime-conditional causal effects: ICGNN weight ($W$) '
                 'for generation$\\rightarrow$price edges by country and regime.}')
    lines.append('\\label{tab:regime_effects}')
    lines.append('\\small')
    lines.append('\\begin{tabular}{l r rrrr}')
    lines.append('\\toprule')
    lines.append('Country & Regime & Wind & Solar & Gas & Nuclear \\\\')
    lines.append('\\midrule')

    techs = ['wind_onshore', 'solar', 'fossil_gas', 'nuclear']

    prev_country = None
    for _, row in effects_df.iterrows():
        country = row['country']
        regime = int(row['regime'])

        label = country if country != prev_country else ''
        prev_country = country

        cells = [label, str(regime)]
        for tech in techs:
            w = row.get(f'{tech}_w')
            sig = row.get(f'{tech}_sig', '')
            if w is not None:
                cells.append(f'{w:.3f}{sig}')
            else:
                cells.append('--')

        lines.append(' & '.join(cells) + ' \\\\')

        if regime == effects_df[effects_df['country'] == country]['regime'].max():
            lines.append('\\midrule')

    # Remove last midrule
    if lines[-1] == '\\midrule':
        lines[-1] = '\\bottomrule'
    lines.append('\\end{tabular}')
    lines.append('\\vspace{0.1cm}')
    lines.append('{\\footnotesize Significance: *** $\\geq$80\\% seed stability, '
                 '** $\\geq$60\\%, * $\\geq$40\\%. '
                 'Negative $W$: price-suppressing. Positive $W$: price-increasing.}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


# =============================================================================
# Master Generator
# =============================================================================

def generate_all_tables(
    results_dir: Path,
    output_dir: Path,
    countries: List[str] = None,
) -> None:
    """
    Load all results and generate all paper tables.

    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to write LaTeX table files
        countries: Countries to include (default: all 12)
    """
    if countries is None:
        countries = ['DE', 'FR', 'NL', 'BE', 'AT', 'IT', 'ES', 'PL', 'DK', 'SE', 'HU', 'CZ']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)

    tables = {}

    # Load baseline results
    baseline_dir = results_dir / 'baselines'
    if baseline_dir.exists():
        baseline_results = {}
        for country in countries:
            path = baseline_dir / country / 'baseline_results_seed42.json'
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                baseline_results[country] = {
                    bl: info.get('metrics', {})
                    for bl, info in data.items()
                }
        if baseline_results:
            # Placeholder for CaRS results (loaded from CaRS output dir)
            cars_results = {}
            cars_dir = results_dir / 'cars'
            for country in countries:
                path = cars_dir / country / 'results.json'
                if path.exists():
                    with open(path) as f:
                        cars_results[country] = json.load(f).get('metrics', {})

            tables['table1_forecasting'] = table_forecasting_comparison(
                cars_results, baseline_results, countries
            )

    # Load and generate each table
    for table_name, content in tables.items():
        out_path = output_dir / f'{table_name}.tex'
        with open(out_path, 'w') as f:
            f.write(content)
        print(f"Generated {out_path}")

    print(f"\nGenerated {len(tables)} tables in {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate paper tables')
    parser.add_argument('--results-dir', type=str,
                        default=str(OUTPUT_DIR))
    parser.add_argument('--output-dir', type=str,
                        default=str(OUTPUT_DIR / 'paper_tables'))
    parser.add_argument('--countries', type=str, default=None)

    args = parser.parse_args()
    countries = args.countries.split(',') if args.countries else None

    generate_all_tables(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        countries=countries,
    )
