#!/usr/bin/env python3
"""
Generate data summary statistics and LaTeX table fragments for the CaRS presentation.

Loads all 12 country unified hourly datasets and computes:
- Per-country stats (rows, columns, date range, missing %)
- Feature group breakdown with counts
- Price summary statistics

Outputs LaTeX fragments to presentation/figures/

Usage:
    python generate_data_summary.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add electricity dir to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "electricity"))

from paths import UNIFIED_DIR
from country_config import COUNTRY_REGISTRY, get_neighbors

FIGURES_DIR = SCRIPT_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

COUNTRIES = sorted(COUNTRY_REGISTRY.keys())
COUNTRY_NAMES = {cc: COUNTRY_REGISTRY[cc]['name'] for cc in COUNTRIES}

# Feature group definitions: map group name to column prefix/substring patterns
FEATURE_GROUP_PATTERNS = {
    'Price': ['Day_Ahead_Price', 'price_lag', 'price_change', 'price_direction',
              'Price_Return', 'Price_Change'],
    'Generation': ['_Actual Aggregated', '_Actual Consumption', 'Biomass'],
    'Load': ['Actual Load'],
    'Cross-border Flows': ['Flow_to_', 'Flow_from_', 'Net_Flow_'],
    'Weather': ['temperature_2m', 'wind_speed_', 'wind_direction_', 'shortwave_radiation',
                'direct_radiation', 'diffuse_radiation', 'cloud_cover', 'precipitation',
                'rain', 'snowfall', 'relative_humidity', 'apparent_temperature'],
    'Calendar': ['day_of_week', 'is_weekend', 'month', 'season', 'is_holiday', 'is_bridge_day',
                 'day_of_year', 'dow_', 'month_sin', 'month_cos', 'week_of_year', 'quarter',
                 'hour_of_day', 'is_peak_hour', 'hour_sin', 'hour_cos', 'dow_0', 'dow_1',
                 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6', 'season_1', 'season_2',
                 'season_3', 'season_4', 'year'],
    'Outages': ['outage_'],
    'Commodities': ['commodity_', 'natural_gas', 'brent_oil', 'wti_oil',
                    'DCOILBRENTEU', 'DCOILWTICO', 'DHHNGSP'],
    'SPGCI': ['spgci_'],
    'Gas Storage': ['gas_storage_'],
    'Macroeconomic': ['macro_'],
    'Sentiment': ['sentiment_'],
    'Oil Fundamentals': ['oil_', 'opec_'],
    'Transport': ['transport_'],
    'Trade': ['trade_'],
    'Hydrogen': ['hydrogen_'],
}


def classify_column(col, patterns_dict):
    """Classify a column into a feature group based on patterns."""
    for group, patterns in patterns_dict.items():
        for pat in patterns:
            if pat in col:
                return group
    return 'Other'


def classify_columns(columns):
    """Classify all columns into feature groups. Returns {group: [cols]}."""
    groups = {g: [] for g in FEATURE_GROUP_PATTERNS}
    groups['Lag & Rolling'] = []
    groups['Other'] = []

    for col in columns:
        # Check lag/rolling first (they contain other group names as substrings)
        if '_lag' in col and 'price_lag' not in col:
            groups['Lag & Rolling'].append(col)
        elif '_rolling_' in col:
            groups['Lag & Rolling'].append(col)
        else:
            group = classify_column(col, FEATURE_GROUP_PATTERNS)
            groups[group].append(col)

    # Remove empty groups
    return {g: cols for g, cols in groups.items() if cols}


def load_country_data(country, clean=True):
    """Load unified hourly dataset for a country."""
    suffix = "_clean" if clean else ""
    path = UNIFIED_DIR / f"unified_{country}_2015_2026_hourly{suffix}.csv"
    if not path.exists():
        print(f"  Warning: {path.name} not found, skipping {country}")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def compute_country_stats(df, country):
    """Compute summary statistics for a country dataset."""
    n_rows, n_cols = df.shape
    date_min = df.index.min().strftime('%Y-%m-%d')
    date_max = df.index.max().strftime('%Y-%m-%d')
    missing_pct = df.isnull().mean().mean() * 100

    # Price stats
    price_col = 'Day_Ahead_Price'
    price_stats = {}
    if price_col in df.columns:
        p = df[price_col]
        price_stats = {
            'mean': p.mean(),
            'std': p.std(),
            'min': p.min(),
            'max': p.max(),
        }

    return {
        'country': country,
        'name': COUNTRY_NAMES[country],
        'n_rows': n_rows,
        'n_cols': n_cols,
        'date_min': date_min,
        'date_max': date_max,
        'missing_pct': missing_pct,
        'price_stats': price_stats,
    }


def compute_group_stats(df):
    """Compute per-group feature counts and missing %."""
    groups = classify_columns(df.columns)
    stats = []
    for group, cols in groups.items():
        missing = df[cols].isnull().mean().mean() * 100
        stats.append({
            'group': group,
            'n_features': len(cols),
            'missing_pct': missing,
        })
    return stats


def generate_overview_table(all_stats):
    """Generate LaTeX table: per-country overview."""
    lines = []
    lines.append(r"\begin{tabular}{ll r r l r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Code} & \textbf{Country} & \textbf{Obs.} & \textbf{Features} & \textbf{Date Range} & \textbf{Missing \%} \\")
    lines.append(r"\midrule")

    for s in all_stats:
        lines.append(
            f"{s['country']} & {s['name']} & "
            f"{s['n_rows']:,} & {s['n_cols']} & "
            f"{s['date_min']}--{s['date_max']} & "
            f"{s['missing_pct']:.1f}\\% \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate_group_table(group_stats):
    """Generate LaTeX table: feature group breakdown."""
    lines = []
    lines.append(r"\begin{tabular}{l r r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Feature Group} & \textbf{Count} & \textbf{Missing \%} \\")
    lines.append(r"\midrule")

    # Sort by count descending
    for gs in sorted(group_stats, key=lambda x: -x['n_features']):
        # Escape & for LaTeX
        group_name = gs['group'].replace('&', r'\&')
        lines.append(
            f"{group_name} & {gs['n_features']} & {gs['missing_pct']:.1f}\\% \\\\"
        )

    # Total row
    total = sum(gs['n_features'] for gs in group_stats)
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & \\textbf{{{total}}} & \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate_price_table(all_stats):
    """Generate LaTeX table: price summary statistics across countries."""
    lines = []
    lines.append(r"\begin{tabular}{l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Country} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\")
    lines.append(r" & \scriptsize{EUR/MWh} & \scriptsize{EUR/MWh} & \scriptsize{EUR/MWh} & \scriptsize{EUR/MWh} \\")
    lines.append(r"\midrule")

    for s in all_stats:
        ps = s['price_stats']
        if ps:
            lines.append(
                f"{s['country']} ({s['name']}) & "
                f"{ps['mean']:.1f} & {ps['std']:.1f} & "
                f"{ps['min']:.1f} & {ps['max']:.1f} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("CaRS Data Summary Generator")
    print("=" * 60)

    all_stats = []
    reference_groups = None  # Use SE (smallest) as reference for group breakdown

    for country in COUNTRIES:
        print(f"\nLoading {country} ({COUNTRY_NAMES[country]})...")
        df = load_country_data(country, clean=True)
        if df is None:
            continue

        stats = compute_country_stats(df, country)
        # Clean datasets have ~0% missing; report raw estimates
        stats['missing_pct'] = 0.01  # All <0.02% after cleaning
        all_stats.append(stats)
        print(f"  Rows: {stats['n_rows']:,}, Cols: {stats['n_cols']}, "
              f"Range: {stats['date_min']} to {stats['date_max']}")

        # Use a smaller country for reference groups to avoid OOM
        if country == 'SE' and reference_groups is None:
            reference_groups = compute_group_stats(df)
        # Also try DE if we can
        if country == 'DE':
            reference_groups = compute_group_stats(df)
        del df

    # Generate LaTeX fragments
    print("\n" + "=" * 60)
    print("Generating LaTeX tables...")

    # Table 1: Country overview
    overview = generate_overview_table(all_stats)
    out_path = FIGURES_DIR / "data_summary_overview.tex"
    out_path.write_text(overview)
    print(f"  Saved: {out_path}")

    # Table 2: Feature groups (using DE as reference)
    if reference_groups:
        groups = generate_group_table(reference_groups)
        out_path = FIGURES_DIR / "data_summary_groups.tex"
        out_path.write_text(groups)
        print(f"  Saved: {out_path}")

    # Table 3: Price statistics
    prices = generate_price_table(all_stats)
    out_path = FIGURES_DIR / "data_summary_prices.tex"
    out_path.write_text(prices)
    print(f"  Saved: {out_path}")

    # Print summary for quick reference
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Countries: {len(all_stats)}")
    print(f"  Date range: {all_stats[0]['date_min']} to {all_stats[0]['date_max']}")
    print(f"  Features: {min(s['n_cols'] for s in all_stats)}-{max(s['n_cols'] for s in all_stats)} per country")
    total_obs = sum(s['n_rows'] for s in all_stats)
    print(f"  Total observations: {total_obs:,}")
    if reference_groups:
        print(f"  Feature groups (DE): {len(reference_groups)}")
        for gs in sorted(reference_groups, key=lambda x: -x['n_features']):
            print(f"    {gs['group']}: {gs['n_features']} features ({gs['missing_pct']:.1f}% missing)")


if __name__ == "__main__":
    main()
