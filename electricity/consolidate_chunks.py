"""
Consolidate chunked download files into single per-country files
expected by collect_all_data.py's create_merged_dataset function.

Reads:
  - data/entsoe/day_ahead_prices_{CC}_*.csv
  - data/entsoe/generation_{CC}_*.csv
  - data/entsoe/load_{CC}_*.csv
  - data/entsoe/flow_{A}_to_{B}_*.csv
  - data/weather/{CC}_hourly_*.csv
  - data/gas_storage/gas_storage_{CC}_*.csv
  - data/outages/outage_daily_{CC}_*.csv

Writes:
  - data/entsoe/entsoe_{CC}_{start}_{end}.csv
  - data/weather/weather_{CC}_{start}_{end}.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from country_config import get_neighbors, get_registered_countries, has_gas_storage
from paths import DATA_DIR

ENTSOE_DIR = DATA_DIR / "entsoe"
WEATHER_DIR = DATA_DIR / "weather"
GAS_DIR = DATA_DIR / "gas_storage"
OUTAGE_DIR = DATA_DIR / "outages"


def load_and_concat(pattern, data_dir, index_col=0, parse_dates=True):
    """Load all files matching a glob pattern and concatenate."""
    files = sorted(data_dir.glob(pattern))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, index_col=index_col, parse_dates=parse_dates)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()
    return combined


def consolidate_entsoe(country, start_year, end_year):
    """Consolidate all ENTSO-E data for a country into one file."""
    output_file = ENTSOE_DIR / f"entsoe_{country}_{start_year}_{end_year}.csv"
    if output_file.exists():
        print(f"  Already exists: {output_file.name}")
        return pd.read_csv(output_file, index_col=0, parse_dates=True)

    print(f"  Loading prices...")
    prices = load_and_concat(f"day_ahead_prices_{country}_*.csv", ENTSOE_DIR)
    if not prices.empty:
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name='Day_Ahead_Price')
        elif 'Day_Ahead_Price' not in prices.columns and len(prices.columns) == 1:
            prices.columns = ['Day_Ahead_Price']

    print(f"  Loading generation...")
    gen = load_and_concat(f"generation_{country}_*.csv", ENTSOE_DIR)

    print(f"  Loading load...")
    load = load_and_concat(f"load_{country}_*.csv", ENTSOE_DIR)

    # Start with prices
    if prices.empty:
        print(f"  No price data for {country}!")
        return pd.DataFrame()

    combined = prices.copy()

    if not load.empty:
        # Handle load column naming
        if isinstance(load, pd.Series):
            load = load.to_frame(name='Load')
        combined = combined.join(load, how='outer')

    if not gen.empty:
        combined = combined.join(gen, how='outer')

    # Load flows
    neighbors = get_neighbors(country)
    print(f"  Loading flows for neighbors: {neighbors}")
    for neighbor in neighbors:
        # Outbound flow
        flow_out = load_and_concat(f"flow_{country}_to_{neighbor}_*.csv", ENTSOE_DIR)
        if not flow_out.empty:
            if isinstance(flow_out, pd.Series):
                flow_out = flow_out.to_frame(name=f'Flow_to_{neighbor}')
            elif len(flow_out.columns) == 1:
                flow_out.columns = [f'Flow_to_{neighbor}']
            combined = combined.join(flow_out, how='outer')

        # Inbound flow
        flow_in = load_and_concat(f"flow_{neighbor}_to_{country}_*.csv", ENTSOE_DIR)
        if not flow_in.empty:
            if isinstance(flow_in, pd.Series):
                flow_in = flow_in.to_frame(name=f'Flow_from_{neighbor}')
            elif len(flow_in.columns) == 1:
                flow_in.columns = [f'Flow_from_{neighbor}']
            combined = combined.join(flow_in, how='outer')

        # Net flow
        to_col = f'Flow_to_{neighbor}'
        from_col = f'Flow_from_{neighbor}'
        if to_col in combined.columns and from_col in combined.columns:
            combined[f'Net_Flow_{neighbor}'] = combined[from_col] - combined[to_col]

    combined = combined.sort_index()
    combined.to_csv(output_file)
    print(f"  Saved: {output_file.name} — shape={combined.shape}, range={combined.index.min()} to {combined.index.max()}")
    return combined


def consolidate_weather(country, start_year, end_year):
    """Consolidate all weather files for a country into one file."""
    output_file = WEATHER_DIR / f"weather_{country}_{start_year}_{end_year}.csv"
    if output_file.exists():
        print(f"  Already exists: {output_file.name}")
        return

    weather = load_and_concat(f"{country}_hourly_*.csv", WEATHER_DIR)
    if weather.empty:
        print(f"  No weather data for {country}!")
        return

    weather.to_csv(output_file)
    print(f"  Saved: {output_file.name} — shape={weather.shape}, range={weather.index.min()} to {weather.index.max()}")


def consolidate_gas(country, start_year, end_year):
    """Consolidate gas storage files."""
    output_file = GAS_DIR / f"gas_storage_{country}_{start_year}_{end_year}.csv"
    if output_file.exists():
        print(f"  Already exists: {output_file.name}")
        return

    gas = load_and_concat(f"gas_storage_{country}_*.csv", GAS_DIR)
    if gas.empty:
        print(f"  No gas storage data for {country}!")
        return

    gas.to_csv(output_file)
    print(f"  Saved: {output_file.name} — shape={gas.shape}, range={gas.index.min()} to {gas.index.max()}")


def consolidate_outages(country, start_year, end_year):
    """Consolidate outage files."""
    output_file = OUTAGE_DIR / f"outage_daily_{country}_{start_year}_{end_year}.csv"
    if output_file.exists():
        print(f"  Already exists: {output_file.name}")
        return

    outages = load_and_concat(f"outage_daily_{country}_*.csv", OUTAGE_DIR)
    if outages.empty:
        print(f"  No outage data for {country}!")
        return

    outages.to_csv(output_file)
    print(f"  Saved: {output_file.name} — shape={outages.shape}, range={outages.index.min()} to {outages.index.max()}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Consolidate chunked data files')
    parser.add_argument('--countries', type=str, default=None)
    parser.add_argument('--start-year', type=int, default=2015)
    parser.add_argument('--end-year', type=int, default=2026)
    args = parser.parse_args()

    countries = args.countries.split(',') if args.countries else get_registered_countries()

    print("=" * 60)
    print("CONSOLIDATING CHUNKED DATA FILES")
    print("=" * 60)

    for country in countries:
        print(f"\n{'#' * 40}")
        print(f"# {country}")
        print(f"{'#' * 40}")

        print("\nENTSO-E:")
        consolidate_entsoe(country, args.start_year, args.end_year)

        print("\nWeather:")
        consolidate_weather(country, args.start_year, args.end_year)

        if has_gas_storage(country):
            print("\nGas storage:")
            consolidate_gas(country, args.start_year, args.end_year)

        print("\nOutages:")
        consolidate_outages(country, args.start_year, args.end_year)

    print("\n" + "=" * 60)
    print("CONSOLIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
