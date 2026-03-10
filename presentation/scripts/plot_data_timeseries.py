#!/usr/bin/env python3
"""
Generate data time series plots for presentation.

Creates:
1. DE: Price, Weather & Generation
2. FR: Price, Weather & Generation
3. DE-FR: Spread & Commodities

Usage:
    python plot_data_timeseries.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Data and output directories
DATA_DIR = Path("/lustre/home/dthumm/CASTOR/data/unified")
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# Style settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# Colors for different series
COLORS = {
    'price': '#1f77b4',
    'temperature': '#d62728',
    'wind': '#2ca02c',
    'precipitation': '#17becf',
    'solar': '#ff7f0e',
    'wind_on': '#2ca02c',
    'wind_off': '#98df8a',
    'nuclear': '#9467bd',
    'gas': '#ff7f0e',
    'hard_coal': '#8c564b',
    'lignite': '#7f7f7f',
    'spread': '#1f77b4',
    'nat_gas': '#ff7f0e',
    'brent': '#2ca02c',
    'wti': '#d62728',
    'de_price': '#1f77b4',   # Blue for DE
    'fr_price': '#e377c2',   # Pink for FR
    # Gas storage
    'fill_level': '#1f77b4',     # Blue
    'injection': '#2ca02c',      # Green
    'withdrawal': '#d62728',     # Red
    'net_flow': '#9467bd',       # Purple
    'working_vol': '#7f7f7f',    # Grey
    'trend': '#ff7f0e',          # Orange
}


def load_data(dataset: str) -> pd.DataFrame:
    """Load unified dataset."""
    path = DATA_DIR / f"unified_{dataset}_2015_2024_clean.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def compute_shared_ylim(df_de: pd.DataFrame, df_fr: pd.DataFrame, cols: list,
                        convert_gw: bool = False) -> tuple:
    """Compute shared y-axis limits across DE and FR datasets.

    Args:
        df_de: German dataset (daily resampled)
        df_fr: French dataset (daily resampled)
        cols: List of column names to consider
        convert_gw: If True, divide values by 1000 (MW to GW conversion)

    Returns:
        Tuple of (min, max) for y-axis limits with 5% padding
    """
    all_values = []
    for df in [df_de, df_fr]:
        for col in cols:
            if col in df.columns:
                vals = df[col].dropna().values
                if convert_gw:
                    vals = vals / 1000
                all_values.extend(vals)

    if not all_values:
        return None

    vmin, vmax = np.min(all_values), np.max(all_values)
    padding = (vmax - vmin) * 0.05
    return (max(0, vmin - padding), vmax + padding)


def resample_daily(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Resample to daily averages for cleaner visualization."""
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols].resample('D').mean()


def plot_de_timeseries(df: pd.DataFrame, output_path: Path,
                       ylim_renewable: tuple = None,
                       ylim_conventional: tuple = None,
                       ylim_precip: tuple = None):
    """Plot DE: price, weather, generation (4 panels).

    Args:
        df: German dataset
        output_path: Path to save SVG
        ylim_renewable: Optional (min, max) for renewable generation panel
        ylim_conventional: Optional (min, max) for conventional generation panel
        ylim_precip: Optional (min, max) for precipitation axis
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Resample to daily for cleaner plots
    df_daily = df.resample('D').mean()

    dates = df_daily.index.to_numpy()

    # Panel 1: Price
    ax = axes[0]
    ax.plot(dates, df_daily['Day_Ahead_Price'].values, color=COLORS['price'], linewidth=0.8)
    ax.set_ylabel('Price (EUR/MWh)')
    ax.set_title('DE Day-Ahead Price')
    ax.grid(True, alpha=0.3)

    # Panel 2: Weather (dual y-axis for temp vs precipitation)
    ax = axes[1]
    ax2 = ax.twinx()

    # Temperature and wind on left axis
    ax.plot(dates, df_daily['DE_temperature_2m'].values, color=COLORS['temperature'],
            linewidth=0.8, label='Temperature (°C)')
    ax.plot(dates, df_daily['DE_wind_speed_100m'].values, color=COLORS['wind'],
            linewidth=0.8, label='Wind Speed (m/s)')
    ax.set_ylabel('Temperature (°C) / Wind (m/s)')
    ax.legend(loc='upper left')

    # Precipitation on right axis
    ax2.plot(dates, df_daily['DE_precipitation'].values, color=COLORS['precipitation'],
             linewidth=0.8, alpha=0.7, label='Precipitation (mm)')
    ax2.set_ylabel('Precipitation (mm)')
    ax2.legend(loc='upper right')

    ax.set_title('DE Weather')
    ax.grid(True, alpha=0.3)
    if ylim_precip:
        ax2.set_ylim(ylim_precip)

    # Panel 3: Renewables
    ax = axes[2]
    # Convert MW to GW
    ax.plot(dates, df_daily['Solar_Actual Aggregated'].values / 1000,
            color=COLORS['solar'], linewidth=0.8, label='Solar')
    ax.plot(dates, df_daily['Wind Onshore_Actual Aggregated'].values / 1000,
            color=COLORS['wind_on'], linewidth=0.8, label='Wind Onshore')
    if 'Wind Offshore_Actual Aggregated' in df_daily.columns:
        ax.plot(dates, df_daily['Wind Offshore_Actual Aggregated'].values / 1000,
                color=COLORS['wind_off'], linewidth=0.8, label='Wind Offshore')
    ax.set_ylabel('Generation (GW)')
    ax.set_title('DE Renewable Generation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if ylim_renewable:
        ax.set_ylim(ylim_renewable)

    # Panel 4: Conventional
    ax = axes[3]
    ax.plot(dates, df_daily['Nuclear_Actual Aggregated'].values / 1000,
            color=COLORS['nuclear'], linewidth=0.8, label='Nuclear')
    ax.plot(dates, df_daily['Fossil Gas_Actual Aggregated'].values / 1000,
            color=COLORS['gas'], linewidth=0.8, label='Gas')
    ax.plot(dates, df_daily['Fossil Hard coal_Actual Aggregated'].values / 1000,
            color=COLORS['hard_coal'], linewidth=0.8, label='Hard Coal')
    if 'Fossil Brown coal/Lignite_Actual Aggregated' in df_daily.columns:
        ax.plot(dates, df_daily['Fossil Brown coal/Lignite_Actual Aggregated'].values / 1000,
                color=COLORS['lignite'], linewidth=0.8, label='Lignite')
    ax.set_ylabel('Generation (GW)')
    ax.set_xlabel('Date')
    ax.set_title('DE Conventional Generation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if ylim_conventional:
        ax.set_ylim(ylim_conventional)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fr_timeseries(df: pd.DataFrame, output_path: Path,
                       ylim_renewable: tuple = None,
                       ylim_conventional: tuple = None,
                       ylim_precip: tuple = None):
    """Plot FR: price, weather, generation (4 panels).

    Args:
        df: French dataset
        output_path: Path to save SVG
        ylim_renewable: Optional (min, max) for renewable generation panel
        ylim_conventional: Optional (min, max) for conventional generation panel
        ylim_precip: Optional (min, max) for precipitation axis
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Resample to daily for cleaner plots
    df_daily = df.resample('D').mean()

    dates = df_daily.index.to_numpy()

    # Panel 1: Price
    ax = axes[0]
    ax.plot(dates, df_daily['Day_Ahead_Price'].values, color=COLORS['price'], linewidth=0.8)
    ax.set_ylabel('Price (EUR/MWh)')
    ax.set_title('FR Day-Ahead Price')
    ax.grid(True, alpha=0.3)

    # Panel 2: Weather (dual y-axis for temp vs precipitation)
    ax = axes[1]
    ax2 = ax.twinx()

    # Temperature and wind on left axis
    ax.plot(dates, df_daily['FR_temperature_2m'].values, color=COLORS['temperature'],
            linewidth=0.8, label='Temperature (°C)')
    ax.plot(dates, df_daily['FR_wind_speed_100m'].values, color=COLORS['wind'],
            linewidth=0.8, label='Wind Speed (m/s)')
    ax.set_ylabel('Temperature (°C) / Wind (m/s)')
    ax.legend(loc='upper left')

    # Precipitation on right axis
    ax2.plot(dates, df_daily['FR_precipitation'].values, color=COLORS['precipitation'],
             linewidth=0.8, alpha=0.7, label='Precipitation (mm)')
    ax2.set_ylabel('Precipitation (mm)')
    ax2.legend(loc='upper right')

    ax.set_title('FR Weather')
    ax.grid(True, alpha=0.3)
    if ylim_precip:
        ax2.set_ylim(ylim_precip)

    # Panel 3: Renewables
    ax = axes[2]
    # Convert MW to GW
    ax.plot(dates, df_daily['Solar_Actual Aggregated'].values / 1000,
            color=COLORS['solar'], linewidth=0.8, label='Solar')
    ax.plot(dates, df_daily['Wind Onshore_Actual Aggregated'].values / 1000,
            color=COLORS['wind_on'], linewidth=0.8, label='Wind Onshore')
    ax.set_ylabel('Generation (GW)')
    ax.set_title('FR Renewable Generation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if ylim_renewable:
        ax.set_ylim(ylim_renewable)

    # Panel 4: Conventional
    ax = axes[3]
    ax.plot(dates, df_daily['Nuclear_Actual Aggregated'].values / 1000,
            color=COLORS['nuclear'], linewidth=0.8, label='Nuclear')
    ax.plot(dates, df_daily['Fossil Gas_Actual Aggregated'].values / 1000,
            color=COLORS['gas'], linewidth=0.8, label='Gas')
    if 'Fossil Hard coal_Actual Aggregated' in df_daily.columns:
        ax.plot(dates, df_daily['Fossil Hard coal_Actual Aggregated'].values / 1000,
                color=COLORS['hard_coal'], linewidth=0.8, label='Hard Coal')
    ax.set_ylabel('Generation (GW)')
    ax.set_xlabel('Date')
    ax.set_title('FR Conventional Generation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if ylim_conventional:
        ax.set_ylim(ylim_conventional)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_defr_timeseries(df: pd.DataFrame, output_path: Path):
    """Plot DE-FR: spread, commodities (2 panels)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Resample to daily for cleaner plots
    df_daily = df.resample('D').mean()

    dates = df_daily.index.to_numpy()

    # Panel 1: Price Spread
    ax = axes[0]
    ax.plot(dates, df_daily['price_spread'].values, color=COLORS['spread'], linewidth=0.8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylabel('Price Spread (EUR/MWh)')
    ax.set_title('DE-FR Price Spread (DE - FR)')
    ax.grid(True, alpha=0.3)

    # Panel 2: Commodities (dual y-axis for gas vs oil)
    ax = axes[1]
    ax2 = ax.twinx()

    # Natural gas on left axis
    gas_col = 'DE_commodity_natural_gas' if 'DE_commodity_natural_gas' in df_daily.columns else 'DE_natural_gas'
    ax.plot(dates, df_daily[gas_col].values, color=COLORS['nat_gas'],
            linewidth=0.8, label='Natural Gas (EUR/MWh)')
    ax.set_ylabel('Natural Gas (EUR/MWh)')
    ax.legend(loc='upper left')

    # Oil on right axis
    brent_col = 'DE_commodity_brent_oil' if 'DE_commodity_brent_oil' in df_daily.columns else 'DE_brent_oil'
    wti_col = 'DE_commodity_wti_oil' if 'DE_commodity_wti_oil' in df_daily.columns else 'DE_wti_oil'
    ax2.plot(dates, df_daily[brent_col].values, color=COLORS['brent'],
             linewidth=0.8, label='Brent Oil (USD/bbl)')
    ax2.plot(dates, df_daily[wti_col].values, color=COLORS['wti'],
             linewidth=0.8, label='WTI Oil (USD/bbl)')
    ax2.set_ylabel('Oil Price (USD/bbl)')
    ax2.legend(loc='upper right')

    ax.set_xlabel('Date')
    ax.set_title('Commodity Prices')
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_price_group(df_de: pd.DataFrame, df_fr: pd.DataFrame, df_defr: pd.DataFrame,
                     output_path: Path):
    """Plot price group: DE/FR prices, spread, commodities (3 panels).

    Args:
        df_de: German dataset
        df_fr: French dataset
        df_defr: DE-FR combined dataset (for spread and commodities)
        output_path: Path to save SVG
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Resample all datasets to daily
    df_de_daily = df_de.resample('D').mean()
    df_fr_daily = df_fr.resample('D').mean()
    df_defr_daily = df_defr.resample('D').mean()

    dates_de = df_de_daily.index.to_numpy()
    dates_fr = df_fr_daily.index.to_numpy()
    dates_defr = df_defr_daily.index.to_numpy()

    # Panel 1: DE and FR Day-Ahead Prices
    ax = axes[0]
    ax.plot(dates_de, df_de_daily['Day_Ahead_Price'].values,
            color=COLORS['de_price'], linewidth=0.8, label='DE')
    ax.plot(dates_fr, df_fr_daily['Day_Ahead_Price'].values,
            color=COLORS['fr_price'], linewidth=0.8, label='FR')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.set_title('Day-Ahead Prices: DE vs FR')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Price Spread
    ax = axes[1]
    ax.plot(dates_defr, df_defr_daily['price_spread'].values,
            color=COLORS['spread'], linewidth=0.8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylabel('Price Spread (EUR/MWh)')
    ax.set_title('DE-FR Price Spread (DE - FR)')
    ax.grid(True, alpha=0.3)

    # Panel 3: Gas & Oil (dual y-axis)
    ax = axes[2]
    ax2 = ax.twinx()

    # Natural gas on left axis
    gas_col = 'DE_commodity_natural_gas' if 'DE_commodity_natural_gas' in df_defr_daily.columns else 'DE_natural_gas'
    ax.plot(dates_defr, df_defr_daily[gas_col].values, color=COLORS['nat_gas'],
            linewidth=0.8, label='Natural Gas (EUR/MWh)')
    ax.set_ylabel('Natural Gas (EUR/MWh)')
    ax.legend(loc='upper left')

    # Oil on right axis
    brent_col = 'DE_commodity_brent_oil' if 'DE_commodity_brent_oil' in df_defr_daily.columns else 'DE_brent_oil'
    wti_col = 'DE_commodity_wti_oil' if 'DE_commodity_wti_oil' in df_defr_daily.columns else 'DE_wti_oil'
    ax2.plot(dates_defr, df_defr_daily[brent_col].values, color=COLORS['brent'],
             linewidth=0.8, label='Brent Oil (USD/bbl)')
    ax2.plot(dates_defr, df_defr_daily[wti_col].values, color=COLORS['wti'],
             linewidth=0.8, label='WTI Oil (USD/bbl)')
    ax2.set_ylabel('Oil Price (USD/bbl)')
    ax2.legend(loc='upper right')

    ax.set_xlabel('Date')
    ax.set_title('Commodity Prices')
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_de_weather_generation_group(df_de: pd.DataFrame, output_path: Path,
                                     ylim_renewable: tuple = None,
                                     ylim_conventional: tuple = None,
                                     ylim_precip: tuple = None):
    """Plot DE weather and generation group (3 panels).

    Args:
        df_de: German dataset
        output_path: Path to save SVG
        ylim_renewable: Optional (min, max) for renewable generation panel
        ylim_conventional: Optional (min, max) for conventional generation panel
        ylim_precip: Optional (min, max) for precipitation axis
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Resample to daily
    df_daily = df_de.resample('D').mean()
    dates = df_daily.index.to_numpy()

    # Panel 1: Weather (dual y-axis)
    ax = axes[0]
    ax2 = ax.twinx()

    ax.plot(dates, df_daily['DE_temperature_2m'].values, color=COLORS['temperature'],
            linewidth=0.8, label='Temperature (°C)')
    ax.plot(dates, df_daily['DE_wind_speed_100m'].values, color=COLORS['wind'],
            linewidth=0.8, label='Wind Speed (m/s)')
    ax.set_ylabel('Temperature (°C) / Wind (m/s)')
    ax.legend(loc='upper left')

    ax2.plot(dates, df_daily['DE_precipitation'].values, color=COLORS['precipitation'],
             linewidth=0.8, alpha=0.7, label='Precipitation (mm)')
    ax2.set_ylabel('Precipitation (mm)')
    ax2.legend(loc='upper right')
    if ylim_precip:
        ax2.set_ylim(ylim_precip)

    ax.set_title('DE Weather')
    ax.grid(True, alpha=0.3)

    # Panel 2: Renewable Generation
    ax = axes[1]
    ax.plot(dates, df_daily['Solar_Actual Aggregated'].values / 1000,
            color=COLORS['solar'], linewidth=0.8, label='Solar')
    ax.plot(dates, df_daily['Wind Onshore_Actual Aggregated'].values / 1000,
            color=COLORS['wind_on'], linewidth=0.8, label='Wind Onshore')
    if 'Wind Offshore_Actual Aggregated' in df_daily.columns:
        ax.plot(dates, df_daily['Wind Offshore_Actual Aggregated'].values / 1000,
                color=COLORS['wind_off'], linewidth=0.8, label='Wind Offshore')
    ax.set_ylabel('Generation (GW)')
    ax.set_title('DE Renewable Generation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if ylim_renewable:
        ax.set_ylim(ylim_renewable)

    # Panel 3: Conventional Generation
    ax = axes[2]
    ax.plot(dates, df_daily['Nuclear_Actual Aggregated'].values / 1000,
            color=COLORS['nuclear'], linewidth=0.8, label='Nuclear')
    ax.plot(dates, df_daily['Fossil Gas_Actual Aggregated'].values / 1000,
            color=COLORS['gas'], linewidth=0.8, label='Gas')
    ax.plot(dates, df_daily['Fossil Hard coal_Actual Aggregated'].values / 1000,
            color=COLORS['hard_coal'], linewidth=0.8, label='Hard Coal')
    if 'Fossil Brown coal/Lignite_Actual Aggregated' in df_daily.columns:
        ax.plot(dates, df_daily['Fossil Brown coal/Lignite_Actual Aggregated'].values / 1000,
                color=COLORS['lignite'], linewidth=0.8, label='Lignite')
    ax.set_ylabel('Generation (GW)')
    ax.set_xlabel('Date')
    ax.set_title('DE Conventional Generation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if ylim_conventional:
        ax.set_ylim(ylim_conventional)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fr_weather_generation_group(df_fr: pd.DataFrame, output_path: Path,
                                     ylim_renewable: tuple = None,
                                     ylim_conventional: tuple = None,
                                     ylim_precip: tuple = None):
    """Plot FR weather and generation group (3 panels).

    Args:
        df_fr: French dataset
        output_path: Path to save SVG
        ylim_renewable: Optional (min, max) for renewable generation panel
        ylim_conventional: Optional (min, max) for conventional generation panel
        ylim_precip: Optional (min, max) for precipitation axis
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Resample to daily
    df_daily = df_fr.resample('D').mean()
    dates = df_daily.index.to_numpy()

    # Panel 1: Weather (dual y-axis)
    ax = axes[0]
    ax2 = ax.twinx()

    ax.plot(dates, df_daily['FR_temperature_2m'].values, color=COLORS['temperature'],
            linewidth=0.8, label='Temperature (°C)')
    ax.plot(dates, df_daily['FR_wind_speed_100m'].values, color=COLORS['wind'],
            linewidth=0.8, label='Wind Speed (m/s)')
    ax.set_ylabel('Temperature (°C) / Wind (m/s)')
    ax.legend(loc='upper left')

    ax2.plot(dates, df_daily['FR_precipitation'].values, color=COLORS['precipitation'],
             linewidth=0.8, alpha=0.7, label='Precipitation (mm)')
    ax2.set_ylabel('Precipitation (mm)')
    ax2.legend(loc='upper right')
    if ylim_precip:
        ax2.set_ylim(ylim_precip)

    ax.set_title('FR Weather')
    ax.grid(True, alpha=0.3)

    # Panel 2: Renewable Generation
    ax = axes[1]
    ax.plot(dates, df_daily['Solar_Actual Aggregated'].values / 1000,
            color=COLORS['solar'], linewidth=0.8, label='Solar')
    ax.plot(dates, df_daily['Wind Onshore_Actual Aggregated'].values / 1000,
            color=COLORS['wind_on'], linewidth=0.8, label='Wind Onshore')
    if 'Wind Offshore_Actual Aggregated' in df_daily.columns:
        ax.plot(dates, df_daily['Wind Offshore_Actual Aggregated'].values / 1000,
                color=COLORS['wind_off'], linewidth=0.8, label='Wind Offshore')
    ax.set_ylabel('Generation (GW)')
    ax.set_title('FR Renewable Generation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if ylim_renewable:
        ax.set_ylim(ylim_renewable)

    # Panel 3: Conventional Generation
    ax = axes[2]
    ax.plot(dates, df_daily['Nuclear_Actual Aggregated'].values / 1000,
            color=COLORS['nuclear'], linewidth=0.8, label='Nuclear')
    ax.plot(dates, df_daily['Fossil Gas_Actual Aggregated'].values / 1000,
            color=COLORS['gas'], linewidth=0.8, label='Gas')
    if 'Fossil Hard coal_Actual Aggregated' in df_daily.columns:
        ax.plot(dates, df_daily['Fossil Hard coal_Actual Aggregated'].values / 1000,
                color=COLORS['hard_coal'], linewidth=0.8, label='Hard Coal')
    ax.set_ylabel('Generation (GW)')
    ax.set_xlabel('Date')
    ax.set_title('FR Conventional Generation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if ylim_conventional:
        ax.set_ylim(ylim_conventional)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_price_vs_generation_scatter(df_de: pd.DataFrame, df_fr: pd.DataFrame,
                                      output_path: Path):
    """Create scatter plots of Price vs Generation features.

    Layout: 4 rows × 2 columns
    - Columns: DE | FR
    - Rows: Wind, Solar, Nuclear, Gas generation
    """
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))

    # Resample to daily for cleaner visualization
    df_de_daily = df_de.resample('D').mean()
    df_fr_daily = df_fr.resample('D').mean()

    generation_features = [
        ('Wind Onshore', 'Wind Onshore_Actual Aggregated', COLORS['wind_on']),
        ('Solar', 'Solar_Actual Aggregated', COLORS['solar']),
        ('Nuclear', 'Nuclear_Actual Aggregated', COLORS['nuclear']),
        ('Gas', 'Fossil Gas_Actual Aggregated', COLORS['gas']),
    ]

    for row, (name, col, color) in enumerate(generation_features):
        # DE column (left)
        if col in df_de_daily.columns:
            x_de = df_de_daily[col].dropna().values / 1000  # Convert to GW
            y_de = df_de_daily.loc[df_de_daily[col].notna(), 'Day_Ahead_Price'].values
            axes[row, 0].scatter(x_de, y_de, alpha=0.3, s=5, c=color, edgecolors='none')
        axes[row, 0].set_ylabel('Price (EUR/MWh)')
        axes[row, 0].set_xlabel(f'{name} (GW)')
        axes[row, 0].grid(True, alpha=0.3)

        # FR column (right)
        if col in df_fr_daily.columns:
            x_fr = df_fr_daily[col].dropna().values / 1000  # Convert to GW
            y_fr = df_fr_daily.loc[df_fr_daily[col].notna(), 'Day_Ahead_Price'].values
            axes[row, 1].scatter(x_fr, y_fr, alpha=0.3, s=5, c=color, edgecolors='none')
        axes[row, 1].set_xlabel(f'{name} (GW)')
        axes[row, 1].grid(True, alpha=0.3)

    # Set titles
    axes[0, 0].set_title('Germany (DE)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('France (FR)', fontsize=12, fontweight='bold')

    # Add row labels on the right
    for row, (name, _, _) in enumerate(generation_features):
        axes[row, 1].yaxis.set_label_position('right')

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_gas_storage_timeseries(df: pd.DataFrame, country: str, output_path: Path,
                                ylim_fill: tuple = None, ylim_flow: tuple = None,
                                ylim_net: tuple = None):
    """Plot gas storage time series (3 panels): fill level, injection/withdrawal, net flow.

    Args:
        df: Dataset with gas_storage_* columns
        country: 'DE' or 'FR'
        output_path: Path to save SVG
        ylim_fill: Optional shared y-axis limits for fill level panel
        ylim_flow: Optional shared y-axis limits for injection/withdrawal panel
        ylim_net: Optional shared y-axis limits for net flow panel
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    df_daily = df.resample('D').mean()
    dates = df_daily.index.to_numpy()

    country_name = 'Germany' if country == 'DE' else 'France'

    # Panel 1: Fill Level (%) with working gas volume on secondary axis
    ax = axes[0]
    ax.plot(dates, df_daily['gas_storage_fill_pct'].values,
            color=COLORS['fill_level'], linewidth=0.8, label='Fill Level (%)')
    ax.axhline(y=90, color='grey', linestyle='--', linewidth=0.5, alpha=0.5, label='90% Target')
    ax.set_ylabel('Fill Level (%)')
    ax.set_title(f'{country} Gas Storage Fill Level')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    if ylim_fill:
        ax.set_ylim(ylim_fill)

    # Panel 2: Injection and Withdrawal (GWh/d)
    ax = axes[1]
    ax.plot(dates, df_daily['gas_storage_injection_gwh'].values,
            color=COLORS['injection'], linewidth=0.8, label='Injection')
    ax.plot(dates, df_daily['gas_storage_withdrawal_gwh'].values,
            color=COLORS['withdrawal'], linewidth=0.8, label='Withdrawal')
    ax.set_ylabel('Flow (GWh/d)')
    ax.set_title(f'{country} Gas Injection & Withdrawal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if ylim_flow:
        ax.set_ylim(ylim_flow)

    # Panel 3: Net Flow (injection - withdrawal) with trend on secondary axis
    ax = axes[2]
    ax2 = ax.twinx()

    ax.fill_between(dates, 0, df_daily['gas_storage_net_flow_gwh'].values,
                    where=df_daily['gas_storage_net_flow_gwh'].values >= 0,
                    color=COLORS['injection'], alpha=0.3, interpolate=True)
    ax.fill_between(dates, 0, df_daily['gas_storage_net_flow_gwh'].values,
                    where=df_daily['gas_storage_net_flow_gwh'].values < 0,
                    color=COLORS['withdrawal'], alpha=0.3, interpolate=True)
    ax.plot(dates, df_daily['gas_storage_net_flow_gwh'].values,
            color=COLORS['net_flow'], linewidth=0.8, label='Net Flow')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylabel('Net Flow (GWh/d)')
    ax.legend(loc='upper left')

    ax2.plot(dates, df_daily['gas_storage_trend_pct'].values,
             color=COLORS['trend'], linewidth=0.6, alpha=0.7, label='Trend (%/d)')
    ax2.set_ylabel('Trend (%/d)')
    ax2.legend(loc='upper right')

    ax.set_xlabel('Date')
    ax.set_title(f'{country} Net Storage Flow & Trend')
    ax.grid(True, alpha=0.3)
    if ylim_net:
        ax.set_ylim(ylim_net)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 70)
    print("Generating Data Time Series Plots")
    print("=" * 70)

    # Ensure output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load both DE and FR datasets first for shared axis computation
    print("\n--- Loading datasets ---")
    df_de = None
    df_fr = None

    try:
        df_de = load_data('DE')
        print(f"  DE: Loaded {len(df_de)} rows, {len(df_de.columns)} columns")
    except Exception as e:
        print(f"  DE Error: {e}")

    try:
        df_fr = load_data('FR')
        print(f"  FR: Loaded {len(df_fr)} rows, {len(df_fr.columns)} columns")
    except Exception as e:
        print(f"  FR Error: {e}")

    # Compute shared axis limits (using daily resampled data)
    ylim_renewable = None
    ylim_conventional = None
    ylim_precip = None

    if df_de is not None and df_fr is not None:
        df_de_daily = df_de.resample('D').mean()
        df_fr_daily = df_fr.resample('D').mean()

        # Renewable generation columns
        renewable_cols = ['Solar_Actual Aggregated', 'Wind Onshore_Actual Aggregated',
                          'Wind Offshore_Actual Aggregated']
        ylim_renewable = compute_shared_ylim(df_de_daily, df_fr_daily, renewable_cols, convert_gw=True)
        print(f"  Shared renewable y-axis: {ylim_renewable}")

        # Conventional generation columns
        conventional_cols = ['Nuclear_Actual Aggregated', 'Fossil Gas_Actual Aggregated',
                             'Fossil Hard coal_Actual Aggregated', 'Fossil Brown coal/Lignite_Actual Aggregated']
        ylim_conventional = compute_shared_ylim(df_de_daily, df_fr_daily, conventional_cols, convert_gw=True)
        print(f"  Shared conventional y-axis: {ylim_conventional}")

        # Precipitation columns
        precip_cols = ['DE_precipitation', 'FR_precipitation']
        ylim_precip = compute_shared_ylim(df_de_daily, df_fr_daily, precip_cols)
        print(f"  Shared precipitation y-axis: {ylim_precip}")

    # Plot 1: DE
    print("\n--- DE: Price, Weather & Generation ---")
    if df_de is not None:
        try:
            plot_de_timeseries(df_de, FIGURES_DIR / 'de_price_weather_generation.svg',
                               ylim_renewable=ylim_renewable,
                               ylim_conventional=ylim_conventional,
                               ylim_precip=ylim_precip)
        except Exception as e:
            print(f"  Error: {e}")

    # Plot 2: FR
    print("\n--- FR: Price, Weather & Generation ---")
    if df_fr is not None:
        try:
            plot_fr_timeseries(df_fr, FIGURES_DIR / 'fr_price_weather_generation.svg',
                               ylim_renewable=ylim_renewable,
                               ylim_conventional=ylim_conventional,
                               ylim_precip=ylim_precip)
        except Exception as e:
            print(f"  Error: {e}")

    # Plot 3: DE-FR
    print("\n--- DE-FR: Spread & Commodities ---")
    df_defr = None
    try:
        df_defr = load_data('DE_FR')
        print(f"  Loaded {len(df_defr)} rows, {len(df_defr.columns)} columns")
        plot_defr_timeseries(df_defr, FIGURES_DIR / 'defr_spread_commodities.svg')
    except Exception as e:
        print(f"  Error: {e}")

    # Plot 4: Price vs Generation Scatter
    print("\n--- Price vs Generation Scatter Plots ---")
    if df_de is not None and df_fr is not None:
        try:
            plot_price_vs_generation_scatter(df_de, df_fr,
                                              FIGURES_DIR / 'price_vs_generation_scatter.svg')
        except Exception as e:
            print(f"  Error: {e}")

    # Plot 5: Price Group (DE, FR, Spread, Commodities)
    print("\n--- Price Group (DE, FR, Spread, Commodities) ---")
    if df_de is not None and df_fr is not None and df_defr is not None:
        try:
            plot_price_group(df_de, df_fr, df_defr,
                             FIGURES_DIR / 'group_prices.svg')
        except Exception as e:
            print(f"  Error: {e}")

    # Plot 6: DE Weather + Generation Group
    print("\n--- DE Weather + Generation Group ---")
    if df_de is not None:
        try:
            plot_de_weather_generation_group(df_de,
                                             FIGURES_DIR / 'group_de_weather_generation.svg',
                                             ylim_renewable=ylim_renewable,
                                             ylim_conventional=ylim_conventional,
                                             ylim_precip=ylim_precip)
        except Exception as e:
            print(f"  Error: {e}")

    # Plot 7: FR Weather + Generation Group
    print("\n--- FR Weather + Generation Group ---")
    if df_fr is not None:
        try:
            plot_fr_weather_generation_group(df_fr,
                                             FIGURES_DIR / 'group_fr_weather_generation.svg',
                                             ylim_renewable=ylim_renewable,
                                             ylim_conventional=ylim_conventional,
                                             ylim_precip=ylim_precip)
        except Exception as e:
            print(f"  Error: {e}")

    # Plot 8 & 9: Gas Storage (DE and FR)
    print("\n--- Gas Storage Time Series ---")
    if df_de is not None and df_fr is not None:
        # Compute shared y-axis limits for gas storage
        df_de_daily = df_de.resample('D').mean()
        df_fr_daily = df_fr.resample('D').mean()

        ylim_fill = compute_shared_ylim(
            df_de_daily, df_fr_daily, ['gas_storage_fill_pct'])
        ylim_flow = compute_shared_ylim(
            df_de_daily, df_fr_daily,
            ['gas_storage_injection_gwh', 'gas_storage_withdrawal_gwh'])
        ylim_net = compute_shared_ylim(
            df_de_daily, df_fr_daily, ['gas_storage_net_flow_gwh'])

        print(f"  Shared fill level y-axis: {ylim_fill}")
        print(f"  Shared flow y-axis: {ylim_flow}")
        print(f"  Shared net flow y-axis: {ylim_net}")

        try:
            plot_gas_storage_timeseries(df_de, 'DE',
                                        FIGURES_DIR / 'de_gas_storage.svg',
                                        ylim_fill=ylim_fill,
                                        ylim_flow=ylim_flow,
                                        ylim_net=ylim_net)
        except Exception as e:
            print(f"  DE Error: {e}")

        try:
            plot_gas_storage_timeseries(df_fr, 'FR',
                                        FIGURES_DIR / 'fr_gas_storage.svg',
                                        ylim_fill=ylim_fill,
                                        ylim_flow=ylim_flow,
                                        ylim_net=ylim_net)
        except Exception as e:
            print(f"  FR Error: {e}")
    else:
        for country, df in [('DE', df_de), ('FR', df_fr)]:
            if df is not None:
                try:
                    plot_gas_storage_timeseries(
                        df, country, FIGURES_DIR / f'{country.lower()}_gas_storage.svg')
                except Exception as e:
                    print(f"  {country} Error: {e}")

    print("\n" + "=" * 70)
    print(f"Plots saved to {FIGURES_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
