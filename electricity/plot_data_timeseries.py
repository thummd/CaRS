"""
Generate time series visualization for electricity data presentation.

Creates a multi-panel figure showing:
- Day-Ahead Price over time
- Feature groups: Commodities, Weather, Production

Updated to use unified ENTSO-E data (2015-2024).

Usage:
    python plot_data_timeseries.py --country DE --output figures/de_timeseries.svg
    python plot_data_timeseries.py --country FR --output figures/fr_timeseries.svg
"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CARS_ROOT, UNIFIED_DIR
# Data directory - unified ENTSO-E data
DATA_DIR = UNIFIED_DIR

# Feature groups for visualization (unified data column names)
COMMODITY_FEATURES = ['natural_gas', 'brent_oil', 'wti_oil']
COMMODITY_LABELS = {'natural_gas': 'Natural Gas', 'brent_oil': 'Brent Oil', 'wti_oil': 'WTI Oil'}

WEATHER_FEATURES_DE = ['DE_temperature_2m', 'DE_precipitation', 'DE_wind_speed_10m']
WEATHER_FEATURES_FR = ['FR_temperature_2m', 'FR_precipitation', 'FR_wind_speed_10m']
WEATHER_LABELS = {'temperature_2m': 'Temp', 'precipitation': 'Precip', 'wind_speed_10m': 'Wind'}

PRODUCTION_FEATURES_DE = [
    'Solar_Actual Aggregated',
    'Wind Onshore_Actual Aggregated',
    'Nuclear_Actual Aggregated',
    'Fossil Gas_Actual Aggregated',
    'Fossil Hard coal_Actual Aggregated'
]
PRODUCTION_FEATURES_FR = [
    'Solar_Actual Aggregated',
    'Wind Onshore_Actual Aggregated',
    'Nuclear_Actual Aggregated',
    'Fossil Gas_Actual Aggregated',
    'Hydro Water Reservoir_Actual Aggregated'
]
PRODUCTION_LABELS = {
    'Solar_Actual Aggregated': 'Solar',
    'Wind Onshore_Actual Aggregated': 'Wind',
    'Nuclear_Actual Aggregated': 'Nuclear',
    'Fossil Gas_Actual Aggregated': 'Gas',
    'Fossil Hard coal_Actual Aggregated': 'Coal',
    'Hydro Water Reservoir_Actual Aggregated': 'Hydro'
}


def load_data(country: str = 'DE') -> pd.DataFrame:
    """Load unified data for a specific country."""
    file_path = DATA_DIR / f"unified_{country}_2015_2024_clean.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df


def plot_timeseries(
    df: pd.DataFrame,
    country: str,
    output_path: str = None,
    figsize: tuple = (14, 10)
):
    """
    Create multi-panel time series plot.

    Args:
        df: DataFrame with electricity data
        country: 'DE' or 'FR'
        output_path: Path to save figure
        figsize: Figure size
    """
    # Select feature groups based on country
    if country == 'DE':
        weather_features = WEATHER_FEATURES_DE
        production_features = PRODUCTION_FEATURES_DE
    else:
        weather_features = WEATHER_FEATURES_FR
        production_features = PRODUCTION_FEATURES_FR

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # X-axis: use datetime index (convert to numpy array for matplotlib compatibility)
    dates = df.index.to_numpy()

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Panel 1: Day-Ahead Price
    ax = axes[0]
    price = df['Day_Ahead_Price'].values
    ax.plot(dates, price, 'k-', linewidth=0.8, alpha=0.8)
    ax.axhline(y=price.mean(), color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Day-Ahead Price\n(EUR/MWh)')
    ax.set_title(f'{country} Electricity Market - Time Series Overview (2015-2024)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Highlight 2022 energy crisis period
    crisis_start = pd.Timestamp('2022-01-01')
    crisis_end = pd.Timestamp('2023-01-01')
    ax.axvspan(crisis_start, crisis_end, alpha=0.15, color='red', label='2022 Crisis')
    ax.legend(loc='upper left', fontsize=8)

    # Panel 2: Commodity prices
    ax = axes[1]
    for i, feat in enumerate(COMMODITY_FEATURES):
        if feat in df.columns:
            # Normalize for better visualization
            values = df[feat].values
            if np.std(values) > 0:
                values_norm = (values - np.mean(values)) / np.std(values)
            else:
                values_norm = values
            label = COMMODITY_LABELS.get(feat, feat)
            ax.plot(dates, values_norm, linewidth=0.7, alpha=0.8, label=label, color=colors[i])
    ax.set_ylabel('Commodities\n(z-score)')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # Panel 3: Weather features
    ax = axes[2]
    valid_weather = [f for f in weather_features if f in df.columns and not df[f].isna().all()]
    for i, feat in enumerate(valid_weather):
        # Normalize and fill NaN for plotting
        values = df[feat].fillna(df[feat].mean()).values
        if np.std(values) > 0:
            values_norm = (values - np.mean(values)) / np.std(values)
        else:
            values_norm = values
        # Extract label from feature name
        for key, label in WEATHER_LABELS.items():
            if key in feat:
                break
        else:
            label = feat.split('_')[-1]
        ax.plot(dates, values_norm, linewidth=0.7, alpha=0.8, label=label, color=colors[i+3])
    ax.set_ylabel('Weather\n(z-score)')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # Panel 4: Production features
    ax = axes[3]
    for i, feat in enumerate(production_features):
        if feat in df.columns:
            values = df[feat].fillna(0).values
            if np.std(values) > 0:
                values_norm = (values - np.mean(values)) / np.std(values)
            else:
                values_norm = values
            label = PRODUCTION_LABELS.get(feat, feat.split('_')[0])
            ax.plot(dates, values_norm, linewidth=0.7, alpha=0.8, label=label, color=colors[i])
    ax.set_ylabel('Generation\n(z-score)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right', fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Adjust layout
    plt.tight_layout()

    # Add text annotation
    date_range = f"{pd.Timestamp(dates.min()).strftime('%Y-%m-%d')} to {pd.Timestamp(dates.max()).strftime('%Y-%m-%d')}"
    fig.text(0.02, 0.02, f'Source: ENTSO-E Transparency Platform | {len(df)} daily observations | {date_range}',
             fontsize=8, style='italic', alpha=0.7)

    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")

    return fig


def plot_target_with_features(
    df: pd.DataFrame,
    country: str,
    output_path: str = None,
    figsize: tuple = (12, 6)
):
    """
    Create compact 2-panel plot: Price + Commodities.
    Better for slides.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})

    dates = df.index

    # Panel 1: Day-Ahead Price
    ax = axes[0]
    price = df['Day_Ahead_Price'].values
    ax.plot(dates, price, 'k-', linewidth=0.8, alpha=0.9)
    ax.axhline(y=price.mean(), color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.set_ylabel('Price (EUR/MWh)')
    ax.set_title(f'{country} Day-Ahead Electricity Price (2015-2024)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Highlight 2022 energy crisis
    crisis_start = pd.Timestamp('2022-01-01')
    crisis_end = pd.Timestamp('2023-01-01')
    ax.axvspan(crisis_start, crisis_end, alpha=0.15, color='coral', label='2022 Crisis')
    ax.legend(loc='upper left', fontsize=8)

    # Panel 2: Key features (Commodities)
    ax = axes[1]
    colors = ['#d62728', '#2ca02c', '#9467bd']  # Red, Green, Purple
    for i, feat in enumerate(COMMODITY_FEATURES):
        if feat in df.columns:
            values = df[feat].values
            if np.std(values) > 0:
                values_norm = (values - np.mean(values)) / np.std(values)
            else:
                values_norm = values
            label = COMMODITY_LABELS.get(feat, feat)
            ax.plot(dates, values_norm, linewidth=0.7, alpha=0.8, label=label, color=colors[i])
    ax.set_ylabel('Commodities (z-score)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved compact figure to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate electricity time series plot")
    parser.add_argument('--country', type=str, default='DE', choices=['DE', 'FR'])
    parser.add_argument('--output', type=str, default=None, help='Output path for figure')
    parser.add_argument('--compact', action='store_true', help='Generate compact 2-panel plot')

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = fstr(CARS_ROOT / "presentation") + "/figures/{args.country.lower()}_unified_timeseries.svg"

    print(f"Loading {args.country} unified data...")
    df = load_data(args.country)
    print(f"Loaded {len(df)} samples from {df.index.min()} to {df.index.max()}")

    print("Generating plot...")
    if args.compact:
        plot_target_with_features(df, args.country, args.output)
    else:
        plot_timeseries(df, args.country, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
