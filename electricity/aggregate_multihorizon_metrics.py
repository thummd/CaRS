"""Aggregate multi-horizon forecasting metrics across the 5-seed daily CARGO sweep.

Reads `outputs/experiments/12market_cargo_controls_daily/{market}/h{H}/seed{S}/results.json`
(12 markets x 3 horizons x 5 seeds = 180 files) and produces:

  (1) Wide CSV `outputs/forecast_quality_daily.csv` -- one row per (market, horizon),
      with columns {metric}_mean / {metric}_std / {metric}_seeds (semicolon-joined raw
      seed values for full reproducibility).

  (2) Long CSV `outputs/forecast_quality_daily_long.csv` -- one row per
      (market, horizon, seed).

  (3) LaTeX table `paper/tables/forecast_quality.tex` summarising RMSE (EUR/MWh),
      MAE (EUR/MWh), CRPS, Spearman, and directional accuracy per market at each
      horizon, with an across-market mean row.

  (4) A stdout summary: per-horizon mean across markets for each metric.

Usage:
    python3 electricity/aggregate_multihorizon_metrics.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MARKETS = ['AT', 'BE', 'CZ', 'DE', 'DK', 'ES', 'FR', 'HU', 'IT', 'NL', 'PL', 'SE']
HORIZONS = [1, 7, 14]
SEEDS = [42, 123, 456, 789, 1011]

METRICS = [
    'rmse', 'rmse_eur_mwh',
    'mae', 'mae_eur_mwh',
    'smape',
    'crps',
    'spearman',
    'directional_accuracy',
]

LATEX_METRICS = ['rmse_eur_mwh', 'mae_eur_mwh', 'crps', 'spearman', 'directional_accuracy']

LATEX_HEADERS = {
    'rmse_eur_mwh':         r'RMSE',
    'mae_eur_mwh':          r'MAE',
    'crps':                 r'CRPS',
    'spearman':             r'$\rho_{s}$',
    'directional_accuracy': r'DA',
}


def load_results(results_dir):
    rows, missing = [], []
    for m in MARKETS:
        for h in HORIZONS:
            for s in SEEDS:
                p = Path(results_dir) / m / f'h{h}' / f'seed{s}' / 'results.json'
                if not p.exists():
                    missing.append((m, h, s))
                    continue
                with open(p) as f:
                    r = json.load(f)
                row = {'market': m, 'horizon': h, 'seed': s}
                for metric in METRICS:
                    row[metric] = float(r.get(metric, np.nan))
                rows.append(row)
    if missing:
        print(f'[warn] {len(missing)} results missing:')
        for m, h, s in missing[:10]:
            print(f'   - {m} h{h} seed{s}')
    return pd.DataFrame(rows)


def aggregate_wide(long_df):
    rows = []
    for (m, h), grp in long_df.groupby(['market', 'horizon']):
        row = {'market': m, 'horizon': h, 'n_seeds': len(grp)}
        for metric in METRICS:
            vals = grp[metric].dropna().values
            row[f'{metric}_mean'] = float(np.mean(vals)) if len(vals) else np.nan
            row[f'{metric}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f'{metric}_seeds'] = ';'.join(f'{v:.4f}' for v in vals)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['market', 'horizon']).reset_index(drop=True)


def render_latex(wide_df, output_path, decimals=3, da_decimals=1):
    n_metrics = len(LATEX_METRICS)
    col_spec = 'l' + ''.join('r' * n_metrics for _ in HORIZONS)

    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\footnotesize',
        r'\setlength{\tabcolsep}{4pt}',
        r'\caption{Multi-horizon forecast quality of CaRS-CARGO on the daily unified '
        r'dataset, mean (std) over 5 seeds (42, 123, 456, 789, 1011). '
        r'RMSE and MAE in EUR/MWh; CRPS in normalised units; '
        r'$\rho_s$ is Spearman rank correlation; DA is directional accuracy [\%].}',
        r'\label{tab:forecast_quality_multihorizon}',
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
    ]

    # Multi-column horizon headers
    horizon_hdr = ' & '.join(
        r'\multicolumn{' + str(n_metrics) + r'}{c}{$h=' + str(h) + r'$\,d}'
        for h in HORIZONS)
    lines.append('Market & ' + horizon_hdr + r' \\')
    cmid_parts = []
    for i, _ in enumerate(HORIZONS):
        a = 2 + i * n_metrics
        b = 1 + (i + 1) * n_metrics
        cmid_parts.append(r'\cmidrule(lr){' + f'{a}-{b}' + '}')
    lines.append(' '.join(cmid_parts))
    metric_hdr = ' & ' + ' & '.join(
        ' & '.join(LATEX_HEADERS[m] for m in LATEX_METRICS) for _ in HORIZONS) + r' \\'
    lines.append(metric_hdr)
    lines.append(r'\midrule')

    for m in MARKETS:
        cells = [m]
        for h in HORIZONS:
            sub = wide_df[(wide_df.market == m) & (wide_df.horizon == h)]
            if sub.empty:
                cells += ['---'] * n_metrics
                continue
            for met in LATEX_METRICS:
                mean = sub[f'{met}_mean'].iloc[0]
                std = sub[f'{met}_std'].iloc[0]
                if met == 'directional_accuracy':
                    cells.append(f'{100*mean:.{da_decimals}f}\\,({100*std:.{da_decimals}f})')
                else:
                    cells.append(f'{mean:.{decimals}f}\\,({std:.{decimals}f})')
        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\midrule')
    cells = [r'\textbf{Mean}']
    for h in HORIZONS:
        sub = wide_df[wide_df.horizon == h]
        for met in LATEX_METRICS:
            agg_mean = sub[f'{met}_mean'].mean()
            if met == 'directional_accuracy':
                cells.append(f'\\textbf{{{100*agg_mean:.{da_decimals}f}}}')
            else:
                cells.append(f'\\textbf{{{agg_mean:.{decimals}f}}}')
    lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n')
    print(f'[done] LaTeX table -> {output_path}')


def print_summary(wide_df):
    print('\n=== mean across 12 markets, per horizon ===')
    col_w = 14
    header = f'{"horizon":>8} ' + ' '.join(f'{m:>{col_w}}' for m in METRICS)
    print(header)
    print('-' * len(header))
    for h in HORIZONS:
        sub = wide_df[wide_df.horizon == h]
        vals = []
        for met in METRICS:
            v = sub[f'{met}_mean'].mean()
            if met == 'directional_accuracy':
                vals.append(f'{100*v:>{col_w-2}.1f} %')
            else:
                vals.append(f'{v:>{col_w}.4f}')
        print(f'{"h=" + str(h):>8} ' + ' '.join(vals))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', type=Path,
                        default=Path('outputs/experiments/12market_cargo_controls_daily'))
    parser.add_argument('--output_dir', type=Path, default=Path('outputs'))
    parser.add_argument('--latex_path', type=Path,
                        default=Path('paper/tables/forecast_quality.tex'))
    args = parser.parse_args()

    long_df = load_results(args.results_dir)
    if long_df.empty:
        print('No results found at', args.results_dir, file=sys.stderr)
        return 1

    wide_df = aggregate_wide(long_df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wide_path = args.output_dir / 'forecast_quality_daily.csv'
    long_path = args.output_dir / 'forecast_quality_daily_long.csv'
    wide_df.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)
    print(f'[done] wide CSV  -> {wide_path}  ({len(wide_df)} rows)')
    print(f'[done] long CSV  -> {long_path}  ({len(long_df)} rows)')

    render_latex(wide_df, args.latex_path)
    print_summary(wide_df)

    return 0


if __name__ == '__main__':
    sys.exit(main())
