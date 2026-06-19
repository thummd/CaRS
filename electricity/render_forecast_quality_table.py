"""Render the combined CARGO-vs-baselines forecast-quality table.

Reads:
  - CARGO seed-mean per (market, horizon) from
    outputs/experiments/12market_cargo_controls_daily/{m}/h{H}/seed{S}/results.json
  - Baseline per (market, horizon) from
    outputs/baselines/daily/{m}/h{H}/baseline_results_seed{S}.json

Produces:
  - paper/tables/forecast_quality_vs_baselines.tex (compact model x horizon table,
    across-market means)
  - outputs/forecast_quality_vs_baselines.csv (long-form CSV of the same)

Usage:
    python3 electricity/render_forecast_quality_table.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MARKETS = ['AT', 'BE', 'CZ', 'DE', 'DK', 'ES', 'FR', 'HU', 'IT', 'NL', 'PL', 'SE']
HORIZONS = [1, 7, 14]
CARGO_SEEDS = [42, 123, 456, 789, 1011]
BASELINE_SEED = 42

# Metrics common to both CARGO and baselines (normalised units; baselines don't compute EUR/MWh)
SHARED_METRICS = ['rmse', 'mae', 'spearman', 'directional_accuracy']

LATEX_HEADERS = {
    'rmse':                 r'RMSE',
    'mae':                  r'MAE',
    'spearman':             r'$\rho_{s}$',
    'directional_accuracy': r'DA',
}

# Display order of models in the final table
MODEL_ORDER = [
    'Naive-Zero',
    'Naive-Mean',
    'Naive-Persistence',
    'Lasso_per_Regime',
    'MS-VAR',
    'XGBoost',
    'CARGO',
]

MODEL_DISPLAY = {
    'Naive-Zero':        r'Naive-Zero',
    'Naive-Mean':        r'Naive-Mean',
    'Naive-Persistence': r'Naive-Persistence',
    'Lasso_per_Regime':  r'Lasso (per regime)',
    'MS-VAR':            r'MS-VAR',
    'XGBoost':           r'XGBoost',
    'CARGO':             r'\textbf{CaRS-CARGO}',
}


def load_cargo(cargo_dir):
    rows = []
    for m in MARKETS:
        for h in HORIZONS:
            for s in CARGO_SEEDS:
                p = Path(cargo_dir) / m / f'h{h}' / f'seed{s}' / 'results.json'
                if not p.exists():
                    continue
                r = json.load(open(p))
                rows.append({
                    'model': 'CARGO', 'market': m, 'horizon': h, 'seed': s,
                    'rmse': r.get('rmse'), 'mae': r.get('mae'),
                    'spearman': r.get('spearman'),
                    'directional_accuracy': r.get('directional_accuracy'),
                })
    return pd.DataFrame(rows)


def load_baselines(baseline_dir):
    rows = []
    for m in MARKETS:
        for h in HORIZONS:
            p = Path(baseline_dir) / m / f'h{h}' / f'baseline_results_seed{BASELINE_SEED}.json'
            if not p.exists():
                continue
            data = json.load(open(p))
            for name, payload in data.items():
                if name.startswith('_') or 'error' in payload:
                    continue
                mets = payload.get('metrics', {})
                # The runner's per-baseline returns store the model name in result['model'];
                # the JSON dump keys are the BASELINE_RUNNERS keys, not the display name.
                # Map back to a canonical display string.
                model = {
                    'naive_zero': 'Naive-Zero',
                    'naive_mean': 'Naive-Mean',
                    'naive_persistence': 'Naive-Persistence',
                    'lasso': 'Lasso_per_Regime',
                    'msvar': 'MS-VAR',
                    'xgboost': 'XGBoost',
                }.get(name, name)
                rows.append({
                    'model': model, 'market': m, 'horizon': h, 'seed': BASELINE_SEED,
                    'rmse': mets.get('rmse'), 'mae': mets.get('mae'),
                    'spearman': mets.get('spearman'),
                    'directional_accuracy': mets.get('directional_accuracy'),
                })
    return pd.DataFrame(rows)


def aggregate(df):
    """Per (model, horizon), mean across markets after seed-averaging within market."""
    seed_mean = df.groupby(['model', 'market', 'horizon'])[SHARED_METRICS].mean().reset_index()
    market_agg = (seed_mean.groupby(['model', 'horizon'])[SHARED_METRICS]
                  .agg(['mean', 'std'])
                  .reset_index())
    market_agg.columns = ['_'.join(c).rstrip('_') for c in market_agg.columns]
    return market_agg


def render_latex(agg, output_path, decimals=3, da_decimals=1):
    n_metrics = len(SHARED_METRICS)
    col_spec = 'l' + ''.join('r' * n_metrics for _ in HORIZONS)

    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\footnotesize',
        r'\setlength{\tabcolsep}{4pt}',
        r'\caption{Multi-horizon forecast quality: CaRS-CARGO vs. baselines on the daily '
        r'unified dataset. Values are mean across the 12 markets of the seed-mean per (market, horizon) '
        r'(CARGO: 5 seeds; baselines: 1 seed = 42). RMSE / MAE are in standardised-return units; '
        r'$\rho_s$ is Spearman rank correlation; DA is directional accuracy [\%]. '
        r'\textit{Bold} marks CaRS-CARGO; cells where CaRS-CARGO is the best in its column '
        r'are underlined.}',
        r'\label{tab:forecast_quality_vs_baselines}',
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
    ]

    horizon_hdr = ' & '.join(
        r'\multicolumn{' + str(n_metrics) + r'}{c}{$h=' + str(h) + r'$\,d}'
        for h in HORIZONS)
    lines.append('Model & ' + horizon_hdr + r' \\')
    cmid_parts = []
    for i in range(len(HORIZONS)):
        a = 2 + i * n_metrics
        b = 1 + (i + 1) * n_metrics
        cmid_parts.append(r'\cmidrule(lr){' + f'{a}-{b}' + '}')
    lines.append(' '.join(cmid_parts))
    metric_hdr = ' & ' + ' & '.join(
        ' & '.join(LATEX_HEADERS[m] for m in SHARED_METRICS) for _ in HORIZONS) + r' \\'
    lines.append(metric_hdr)
    lines.append(r'\midrule')

    # Compute per (horizon, metric) winner across models, separately for "lower-is-better"
    # (rmse, mae) vs "higher-is-better" (spearman, directional_accuracy).
    HIGHER_BETTER = {'spearman', 'directional_accuracy'}
    winners = {}
    for h in HORIZONS:
        sub = agg[agg.horizon == h]
        for met in SHARED_METRICS:
            col = f'{met}_mean'
            if sub.empty:
                continue
            if met in HIGHER_BETTER:
                winners[(h, met)] = sub.loc[sub[col].idxmax(), 'model']
            else:
                winners[(h, met)] = sub.loc[sub[col].idxmin(), 'model']

    for model in MODEL_ORDER:
        cells = [MODEL_DISPLAY[model]]
        for h in HORIZONS:
            for met in SHARED_METRICS:
                row = agg[(agg.model == model) & (agg.horizon == h)]
                if row.empty:
                    cells.append('---')
                    continue
                mean = row[f'{met}_mean'].iloc[0]
                if met == 'directional_accuracy':
                    txt = f'{100*mean:.{da_decimals}f}'
                else:
                    txt = f'{mean:.{decimals}f}'
                if winners.get((h, met)) == model:
                    txt = r'\underline{' + txt + '}'
                cells.append(txt)
        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n')
    print(f'[done] LaTeX table -> {output_path}')


def print_summary(agg):
    print('\n=== across-market mean per (model, horizon) ===')
    print(f'{"model":<22}{"horizon":>8}' + ' '.join(f'{m:>10}' for m in SHARED_METRICS))
    print('-' * (22 + 8 + 4 * 11))
    for model in MODEL_ORDER:
        for h in HORIZONS:
            row = agg[(agg.model == model) & (agg.horizon == h)]
            if row.empty:
                continue
            vals = []
            for met in SHARED_METRICS:
                v = row[f'{met}_mean'].iloc[0]
                if met == 'directional_accuracy':
                    vals.append(f'{100*v:>8.1f} %')
                else:
                    vals.append(f'{v:>10.4f}')
            print(f'{model:<22}{"h=" + str(h):>8} ' + ' '.join(vals))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cargo_dir', type=Path,
                        default=Path('outputs/experiments/12market_cargo_controls_daily'))
    parser.add_argument('--baseline_dir', type=Path,
                        default=Path('outputs/baselines/daily'))
    parser.add_argument('--output_dir', type=Path, default=Path('outputs'))
    parser.add_argument('--latex_path', type=Path,
                        default=Path('paper/tables/forecast_quality_vs_baselines.tex'))
    args = parser.parse_args()

    cargo = load_cargo(args.cargo_dir)
    base = load_baselines(args.baseline_dir)
    long_df = pd.concat([cargo, base], ignore_index=True)
    if long_df.empty:
        print('No results found.', file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.output_dir / 'forecast_quality_vs_baselines_long.csv', index=False)
    print(f'[done] long CSV -> {args.output_dir / "forecast_quality_vs_baselines_long.csv"}'
          f'  ({len(long_df)} rows; {long_df.model.nunique()} models)')

    agg = aggregate(long_df)
    agg.to_csv(args.output_dir / 'forecast_quality_vs_baselines.csv', index=False)
    print(f'[done] agg CSV  -> {args.output_dir / "forecast_quality_vs_baselines.csv"}'
          f'  ({len(agg)} rows)')

    render_latex(agg, args.latex_path)
    print_summary(agg)

    return 0


if __name__ == '__main__':
    sys.exit(main())
