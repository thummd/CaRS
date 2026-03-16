"""
Comprehensive data collection script for electricity price forecasting.

Downloads all available data from:
1. ENTSO-E: Day-ahead prices, generation, load, cross-border flows
2. Open-Meteo: Historical weather (temperature, wind, solar, precipitation)
3. Commodity prices: Natural gas, oil from Yahoo Finance
4. epftoolbox: Pre-packaged electricity datasets
5. Gas storage: AGSI+ fill levels, injection/withdrawal for DE and FR

Run with: python3 collect_all_data.py --years 2015-2023

Requires:
- ENTSOE_API_KEY environment variable
- pip install entsoe-py yfinance openpyxl requests
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from country_config import (
    get_entsoe_zone, get_neighbors, get_registered_countries,
    has_gas_storage, COUNTRY_REGISTRY,
)
from paths import DATA_DIR

# Data directories
BASE_DATA_DIR = DATA_DIR
ENTSOE_DIR = BASE_DATA_DIR / "entsoe"
WEATHER_DIR = BASE_DATA_DIR / "weather"
COMMODITY_DIR = BASE_DATA_DIR / "commodities"
EPFTOOLBOX_DIR = BASE_DATA_DIR / "epftoolbox"
GAS_STORAGE_DIR = BASE_DATA_DIR / "gas_storage"
MACRO_DIR = BASE_DATA_DIR / "macro"
SENTIMENT_DIR = BASE_DATA_DIR / "sentiment"
OIL_FUND_DIR = BASE_DATA_DIR / "oil_fundamentals"
TRANSPORT_DIR = BASE_DATA_DIR / "transport"
TRADE_DIR = BASE_DATA_DIR / "trade"
HYDROGEN_DIR = BASE_DATA_DIR / "hydrogen"

# Ensure directories exist
for d in [ENTSOE_DIR, WEATHER_DIR, COMMODITY_DIR, EPFTOOLBOX_DIR, GAS_STORAGE_DIR,
          MACRO_DIR, SENTIMENT_DIR, OIL_FUND_DIR, TRANSPORT_DIR, TRADE_DIR, HYDROGEN_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def download_entsoe_long_term(
    country: str,
    start_year: int,
    end_year: int,
    chunk_months: int = 6
):
    """
    Download ENTSO-E data in chunks to avoid API timeouts.

    Args:
        country: Country code (e.g., 'DE', 'FR', 'NL')
        start_year: Start year
        end_year: End year (inclusive)
        chunk_months: Months per API call
    """
    from download_entsoe_data import (
        download_day_ahead_prices,
        download_generation_by_source,
        download_load,
        download_cross_border_flows,
        get_api_key
    )

    api_key = get_api_key()

    print(f"\n{'='*60}")
    print(f"Downloading ENTSO-E data for {country}: {start_year}-{end_year}")
    print(f"{'='*60}")

    all_prices = []
    all_generation = []
    all_load = []
    # Per-neighbor flow storage
    neighbors = get_neighbors(country)
    all_flows = {n: {'out': [], 'in': []} for n in neighbors}

    current_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year + 1, 1, 1)

    while current_date < end_date:
        chunk_end = current_date + timedelta(days=chunk_months * 30)
        if chunk_end > end_date:
            chunk_end = end_date

        start_str = current_date.strftime('%Y-%m-%d')
        end_str = (chunk_end - timedelta(days=1)).strftime('%Y-%m-%d')

        print(f"\nChunk: {start_str} to {end_str}")

        # Day-ahead prices
        try:
            prices = download_day_ahead_prices(
                country, start_str, end_str, api_key, ENTSOE_DIR
            )
            if isinstance(prices, pd.Series):
                all_prices.append(prices.to_frame(name='Day_Ahead_Price'))
            else:
                all_prices.append(prices)
            time.sleep(1)
        except Exception as e:
            print(f"  Prices error: {e}")

        # Load
        try:
            load = download_load(country, start_str, end_str, api_key, ENTSOE_DIR)
            all_load.append(load)
            time.sleep(1)
        except Exception as e:
            print(f"  Load error: {e}")

        # Generation
        try:
            gen = download_generation_by_source(
                country, start_str, end_str, api_key, ENTSOE_DIR
            )
            all_generation.append(gen)
            time.sleep(1)
        except Exception as e:
            print(f"  Generation error: {e}")

        # Cross-border flows with all neighbors
        for neighbor in neighbors:
            try:
                flow_out = download_cross_border_flows(
                    country, neighbor, start_str, end_str, api_key, ENTSOE_DIR
                )
                all_flows[neighbor]['out'].append(
                    flow_out.to_frame(name=f'Flow_to_{neighbor}')
                )
                time.sleep(1)

                flow_in = download_cross_border_flows(
                    neighbor, country, start_str, end_str, api_key, ENTSOE_DIR
                )
                all_flows[neighbor]['in'].append(
                    flow_in.to_frame(name=f'Flow_from_{neighbor}')
                )
                time.sleep(1)
            except Exception as e:
                print(f"  Flows {country}<->{neighbor} error: {e}")

        current_date = chunk_end

    # Combine all chunks
    print(f"\nCombining {country} data...")

    combined = pd.DataFrame()

    if all_prices:
        prices_df = pd.concat(all_prices)
        prices_df = prices_df[~prices_df.index.duplicated(keep='first')]
        combined = prices_df

    if all_load:
        load_df = pd.concat(all_load)
        load_df = load_df[~load_df.index.duplicated(keep='first')]
        combined = combined.join(load_df, how='outer')

    if all_generation:
        gen_df = pd.concat(all_generation)
        gen_df = gen_df[~gen_df.index.duplicated(keep='first')]
        combined = combined.join(gen_df, how='outer')

    # Add flows for each neighbor
    for neighbor in neighbors:
        if all_flows[neighbor]['out']:
            flows_out_df = pd.concat(all_flows[neighbor]['out'])
            flows_out_df = flows_out_df[~flows_out_df.index.duplicated(keep='first')]
            combined = combined.join(flows_out_df, how='outer')

        if all_flows[neighbor]['in']:
            flows_in_df = pd.concat(all_flows[neighbor]['in'])
            flows_in_df = flows_in_df[~flows_in_df.index.duplicated(keep='first')]
            combined = combined.join(flows_in_df, how='outer')

        # Add net flow
        to_col = f'Flow_to_{neighbor}'
        from_col = f'Flow_from_{neighbor}'
        if to_col in combined.columns and from_col in combined.columns:
            combined[f'Net_Flow_{neighbor}'] = combined[from_col] - combined[to_col]

    combined = combined.sort_index()

    # Save combined dataset
    output_file = ENTSOE_DIR / f"entsoe_{country}_{start_year}_{end_year}.csv"
    combined.to_csv(output_file)
    print(f"Saved to {output_file}")
    print(f"Shape: {combined.shape}")
    print(f"Date range: {combined.index.min()} to {combined.index.max()}")

    return combined


def download_weather_long_term(
    country: str,
    start_year: int,
    end_year: int,
    chunk_years: int = 2
):
    """
    Download weather data in chunks to avoid rate limits.

    Args:
        country: 'DE' or 'FR'
        start_year: Start year
        end_year: End year (inclusive)
        chunk_years: Years per API call
    """
    from download_weather_data import download_country_weather

    print(f"\n{'='*60}")
    print(f"Downloading Weather data for {country}: {start_year}-{end_year}")
    print(f"{'='*60}")

    all_weather = []
    current_year = start_year

    while current_year <= end_year:
        chunk_end_year = min(current_year + chunk_years - 1, end_year)
        start_str = f"{current_year}-01-01"
        end_str = f"{chunk_end_year}-12-31"

        print(f"\nChunk: {start_str} to {end_str}")

        try:
            # Clear cache to force re-download
            weather = download_country_weather(
                country=country,
                start_date=start_str,
                end_date=end_str,
                hourly=True,
                data_dir=WEATHER_DIR
            )
            all_weather.append(weather)
            time.sleep(2)  # Respect rate limits
        except Exception as e:
            print(f"  Error: {e}")
            print("  Waiting 30s before retry...")
            time.sleep(30)
            try:
                weather = download_country_weather(
                    country=country,
                    start_date=start_str,
                    end_date=end_str,
                    hourly=True,
                    data_dir=WEATHER_DIR
                )
                all_weather.append(weather)
            except Exception as e2:
                print(f"  Retry failed: {e2}")

        current_year = chunk_end_year + 1

    if all_weather:
        combined = pd.concat(all_weather)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()

        output_file = WEATHER_DIR / f"weather_{country}_{start_year}_{end_year}.csv"
        combined.to_csv(output_file)
        print(f"\nSaved to {output_file}")
        print(f"Shape: {combined.shape}")

        return combined

    return pd.DataFrame()


def download_commodities_long_term(
    start_year: int,
    end_year: int
):
    """Download commodity prices for the full period."""
    from download_commodity_data import create_commodity_dataset

    print(f"\n{'='*60}")
    print(f"Downloading Commodity data: {start_year}-{end_year}")
    print(f"{'='*60}")

    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"

    commodities = create_commodity_dataset(
        start=start_str,
        end=end_str,
        data_dir=COMMODITY_DIR,
        use_cache=False
    )

    if not commodities.empty:
        output_file = COMMODITY_DIR / f"commodities_{start_year}_{end_year}.csv"
        commodities.to_csv(output_file)
        print(f"Saved to {output_file}")
        print(f"Shape: {commodities.shape}")

    return commodities


def download_gas_storage_long_term(
    start_year: int,
    end_year: int,
    countries: list = None
):
    """Download gas storage data for the full period."""
    from download_gas_storage_data import create_gas_storage_dataset

    if countries is None:
        countries = [c for c in get_registered_countries() if has_gas_storage(c)]

    print(f"\n{'='*60}")
    print(f"Downloading Gas Storage data: {start_year}-{end_year}")
    print(f"{'='*60}")

    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"

    results = create_gas_storage_dataset(
        start=start_str,
        end=end_str,
        countries=countries,
        data_dir=GAS_STORAGE_DIR,
        use_cache=False
    )

    return results


def download_macro_long_term(start_year: int, end_year: int):
    """Download macroeconomic indicator data."""
    from download_macro_data import create_macro_dataset

    print(f"\n{'='*60}")
    print(f"Downloading Macro data: {start_year}-{end_year}")
    print(f"{'='*60}")

    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"

    return create_macro_dataset(start=start_str, end=end_str,
                                data_dir=MACRO_DIR, use_cache=False)


def download_sentiment_long_term(start_year: int, end_year: int):
    """Download market sentiment indicator data."""
    from download_sentiment_data import create_sentiment_dataset

    print(f"\n{'='*60}")
    print(f"Downloading Sentiment data: {start_year}-{end_year}")
    print(f"{'='*60}")

    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"

    return create_sentiment_dataset(start=start_str, end=end_str,
                                    data_dir=SENTIMENT_DIR, use_cache=False)


def download_oil_fundamentals_long_term(start_year: int, end_year: int):
    """Download oil fundamentals and OPEC data."""
    from download_oil_fundamentals_data import create_oil_fundamentals_dataset

    print(f"\n{'='*60}")
    print(f"Downloading Oil Fundamentals data: {start_year}-{end_year}")
    print(f"{'='*60}")

    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"

    return create_oil_fundamentals_dataset(start=start_str, end=end_str,
                                           data_dir=OIL_FUND_DIR, use_cache=False)


def download_transport_long_term(start_year: int, end_year: int):
    """Download transport/freight index data."""
    from download_transport_data import create_transport_dataset

    print(f"\n{'='*60}")
    print(f"Downloading Transport data: {start_year}-{end_year}")
    print(f"{'='*60}")

    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"

    return create_transport_dataset(start=start_str, end=end_str,
                                    data_dir=TRANSPORT_DIR, use_cache=False)


def download_trade_long_term(start_year: int, end_year: int):
    """Download EU trade data."""
    from download_trade_data import create_trade_dataset

    print(f"\n{'='*60}")
    print(f"Downloading Trade data: {start_year}-{end_year}")
    print(f"{'='*60}")

    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"

    return create_trade_dataset(start=start_str, end=end_str,
                                data_dir=TRADE_DIR, use_cache=False)


def download_hydrogen_long_term(start_year: int, end_year: int):
    """Download hydrogen cost data."""
    from download_hydrogen_data import create_hydrogen_dataset

    print(f"\n{'='*60}")
    print(f"Downloading Hydrogen data: {start_year}-{end_year}")
    print(f"{'='*60}")

    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"

    return create_hydrogen_dataset(start=start_str, end=end_str,
                                   data_dir=HYDROGEN_DIR, use_cache=False)


def download_epftoolbox_data():
    """Download epftoolbox datasets."""
    from download_epftoolbox_data import download_dataset, explore_dataset

    print(f"\n{'='*60}")
    print("Downloading epftoolbox datasets")
    print(f"{'='*60}")

    for country in ['DE', 'FR']:
        data = download_dataset(country, EPFTOOLBOX_DIR)
        explore_dataset(data, country)

    return True


def create_merged_dataset(
    country: str,
    start_year: int,
    end_year: int,
    output_dir: Path = BASE_DATA_DIR
):
    """
    Create a merged dataset with all features for a country.

    Combines:
    - ENTSO-E prices and fundamentals (hourly)
    - Weather data (hourly)
    - Commodity prices (daily, broadcast to hourly)

    Outputs:
    - Hourly dataset
    - Daily aggregated dataset
    """
    print(f"\n{'='*60}")
    print(f"Creating merged dataset for {country}")
    print(f"{'='*60}")

    # Load ENTSO-E data
    entsoe_file = ENTSOE_DIR / f"entsoe_{country}_{start_year}_{end_year}.csv"
    if entsoe_file.exists():
        entsoe = pd.read_csv(entsoe_file, index_col=0, parse_dates=True)
        print(f"ENTSO-E: {entsoe.shape}")
    else:
        print(f"ENTSO-E file not found: {entsoe_file}")
        entsoe = pd.DataFrame()

    # Load weather data
    weather_file = WEATHER_DIR / f"weather_{country}_{start_year}_{end_year}.csv"
    if weather_file.exists():
        weather = pd.read_csv(weather_file, index_col=0, parse_dates=True)
        print(f"Weather: {weather.shape}")
    else:
        print(f"Weather file not found: {weather_file}")
        weather = pd.DataFrame()

    # Load commodity data
    commodity_file = COMMODITY_DIR / f"commodities_{start_year}_{end_year}.csv"
    if commodity_file.exists():
        commodities = pd.read_csv(commodity_file, index_col=0, parse_dates=True)
        print(f"Commodities: {commodities.shape}")
    else:
        print(f"Commodity file not found: {commodity_file}")
        commodities = pd.DataFrame()

    # Merge datasets
    if entsoe.empty:
        print("Cannot create merged dataset without ENTSO-E data")
        return None

    merged = entsoe.copy()

    # Add weather (align hourly)
    if not weather.empty:
        # Ensure timezone consistency
        if merged.index.tz is not None and weather.index.tz is None:
            weather.index = weather.index.tz_localize('UTC')
        elif merged.index.tz is None and weather.index.tz is not None:
            merged.index = merged.index.tz_localize('UTC')

        merged = merged.join(weather, how='left')

    # Add commodities (broadcast daily to hourly)
    if not commodities.empty:
        # Ensure timezone consistency
        if commodities.index.tz is None:
            commodities.index = commodities.index.tz_localize('UTC')
        if merged.index.tz is None:
            merged.index = merged.index.tz_localize('UTC')

        # Forward fill commodities to hourly
        merged['date'] = merged.index.date
        commodities['date'] = commodities.index.date
        commodities_dict = commodities.set_index('date').to_dict()
        for col in [c for c in commodities.columns if c != 'date']:
            merged[col] = merged['date'].map(commodities_dict.get(col, {}))
        merged = merged.drop(columns=['date'])

    # Compute target variable (daily price variation)
    if 'Day_Ahead_Price' in merged.columns:
        merged['Price_Change'] = merged['Day_Ahead_Price'].diff()
        merged['Price_Return'] = np.log(merged['Day_Ahead_Price']).diff()

    # Fill missing values
    merged = merged.ffill().bfill()

    # Save hourly dataset
    hourly_file = output_dir / f"merged_hourly_{country}_{start_year}_{end_year}.csv"
    merged.to_csv(hourly_file)
    print(f"\nHourly dataset saved to {hourly_file}")
    print(f"Shape: {merged.shape}")

    # Create daily aggregated dataset
    daily = merged.resample('D').agg({
        col: 'mean' if 'temperature' in col.lower() or 'price' in col.lower()
        else 'sum' if 'radiation' in col.lower() or 'precipitation' in col.lower()
        else 'max' if 'wind' in col.lower()
        else 'mean'
        for col in merged.columns
    })

    # Recompute daily target variable
    if 'Day_Ahead_Price' in daily.columns:
        daily['Price_Change'] = daily['Day_Ahead_Price'].diff()
        daily['Price_Return'] = np.log(daily['Day_Ahead_Price']).diff()

    daily_file = output_dir / f"merged_daily_{country}_{start_year}_{end_year}.csv"
    daily.to_csv(daily_file)
    print(f"\nDaily dataset saved to {daily_file}")
    print(f"Shape: {daily.shape}")

    return merged, daily


def main():
    parser = argparse.ArgumentParser(description='Collect all electricity data')
    parser.add_argument('--years', type=str, default='2015-2023',
                       help='Year range (e.g., 2015-2023)')
    parser.add_argument('--countries', type=str, default=None,
                       help='Countries to download (comma-separated, default: all registered)')
    parser.add_argument('--skip-entsoe', action='store_true',
                       help='Skip ENTSO-E download')
    parser.add_argument('--skip-weather', action='store_true',
                       help='Skip weather download')
    parser.add_argument('--skip-commodities', action='store_true',
                       help='Skip commodity download')
    parser.add_argument('--skip-gas-storage', action='store_true',
                       help='Skip gas storage download')
    parser.add_argument('--skip-merge', action='store_true',
                       help='Skip creating merged dataset')
    parser.add_argument('--skip-macro', action='store_true',
                       help='Skip macro indicator download')
    parser.add_argument('--skip-sentiment', action='store_true',
                       help='Skip sentiment indicator download')
    parser.add_argument('--skip-oil-fundamentals', action='store_true',
                       help='Skip oil fundamentals / OPEC download')
    parser.add_argument('--skip-transport', action='store_true',
                       help='Skip transport index download')
    parser.add_argument('--skip-trade', action='store_true',
                       help='Skip trade data download')
    parser.add_argument('--skip-hydrogen', action='store_true',
                       help='Skip hydrogen data download')
    parser.add_argument('--frequency', type=str, default='D', choices=['D', 'H'],
                       help='Output frequency: D (daily, default) or H (hourly)')

    args = parser.parse_args()

    # Parse years
    start_year, end_year = map(int, args.years.split('-'))
    countries = args.countries.split(',') if args.countries else get_registered_countries()

    print("="*60)
    print("COMPREHENSIVE DATA COLLECTION")
    print("="*60)
    print(f"Years: {start_year}-{end_year}")
    print(f"Countries: {countries}")
    print(f"Data directory: {BASE_DATA_DIR}")

    # Download epftoolbox data first (quick)
    try:
        download_epftoolbox_data()
    except Exception as e:
        print(f"epftoolbox error: {e}")

    # Download commodity data (needed for all countries)
    if not args.skip_commodities:
        try:
            download_commodities_long_term(start_year, end_year)
        except Exception as e:
            print(f"Commodity error: {e}")

    # Download gas storage data (country-specific AGSI+ fill levels)
    if not args.skip_gas_storage:
        try:
            gas_countries = [c for c in countries if has_gas_storage(c)]
            download_gas_storage_long_term(start_year, end_year, gas_countries)
        except Exception as e:
            print(f"Gas storage error: {e}")

    # Download country-agnostic data sources (run once, shared across all countries)
    if not args.skip_macro:
        try:
            download_macro_long_term(start_year, end_year)
        except Exception as e:
            print(f"Macro error: {e}")

    if not args.skip_sentiment:
        try:
            download_sentiment_long_term(start_year, end_year)
        except Exception as e:
            print(f"Sentiment error: {e}")

    if not args.skip_oil_fundamentals:
        try:
            download_oil_fundamentals_long_term(start_year, end_year)
        except Exception as e:
            print(f"Oil fundamentals error: {e}")

    if not args.skip_transport:
        try:
            download_transport_long_term(start_year, end_year)
        except Exception as e:
            print(f"Transport error: {e}")

    if not args.skip_trade:
        try:
            download_trade_long_term(start_year, end_year)
        except Exception as e:
            print(f"Trade error: {e}")

    if not args.skip_hydrogen:
        try:
            download_hydrogen_long_term(start_year, end_year)
        except Exception as e:
            print(f"Hydrogen error: {e}")

    # Download per-country data
    for country in countries:
        print(f"\n{'#'*60}")
        print(f"# COUNTRY: {country}")
        print(f"{'#'*60}")

        # ENTSO-E
        if not args.skip_entsoe:
            try:
                download_entsoe_long_term(country, start_year, end_year)
            except Exception as e:
                print(f"ENTSO-E error for {country}: {e}")

        # Weather
        if not args.skip_weather:
            try:
                download_weather_long_term(country, start_year, end_year)
            except Exception as e:
                print(f"Weather error for {country}: {e}")

        # Create merged dataset
        if not args.skip_merge:
            try:
                create_merged_dataset(country, start_year, end_year)
            except Exception as e:
                print(f"Merge error for {country}: {e}")

    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE")
    print("="*60)
    print(f"\nData saved to: {BASE_DATA_DIR}")
    print("\nFiles created:")
    for f in sorted(BASE_DATA_DIR.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
