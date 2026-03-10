"""
Download macroeconomic indicator data relevant for electricity price forecasting.

Key indicators:
- Eurozone Manufacturing Production (monthly)
- Eurozone HICP Inflation / CPI (monthly)
- Eurozone Real GDP Growth (quarterly)
- German IFO Business Climate Index (monthly)
- EU Unemployment Rate (monthly)

All sourced from FRED (Federal Reserve Economic Data) - free API.

Usage:
    python3 download_macro_data.py
    python3 download_macro_data.py --start 2015-01-01 --end 2024-12-31
    python3 download_macro_data.py --no-cache
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CARS_ROOT, MACRO_DIR
# Data directory
DATA_DIR = MACRO_DIR

# FRED series IDs for macroeconomic indicators
FRED_MACRO_SERIES = {
    'macro_eu_industrial_prod': {
        'series_id': 'PRMNTO01EZM661S',
        'description': 'Eurozone Manufacturing Production Index (monthly)',
        'frequency': 'M',
    },
    'macro_eu_cpi': {
        'series_id': 'CP0000EZ19M086NEST',
        'description': 'Eurozone HICP - All Items (monthly)',
        'frequency': 'M',
    },
    'macro_eu_gdp_growth': {
        'series_id': 'NAEXKP01EZQ659S',
        'description': 'Eurozone Real GDP (quarterly)',
        'frequency': 'Q',
    },
    'macro_de_ifo': {
        'series_id': 'BSCICP03DEM665S',
        'description': 'German IFO Business Climate Index (monthly)',
        'frequency': 'M',
    },
    'macro_eu_unemployment': {
        'series_id': 'LRHUTTTTEZM156S',
        'description': 'EU Unemployment Rate (monthly)',
        'frequency': 'M',
    },
}


def _load_env_credentials():
    """Load API keys from .env file."""
    env_file = CARS_ROOT / ".env"
    creds = {}
    if env_file.exists():
        for line in env_file.read_text().split('\n'):
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                creds[key.strip()] = value.strip().strip('"\'')
    return creds


def get_fred_api_key():
    """Get FRED API key from environment or .env file."""
    api_key = os.environ.get('FRED_API_KEY')
    if api_key:
        return api_key

    creds = _load_env_credentials()
    api_key = creds.get('FRED_API_KEY')
    if api_key:
        return api_key

    print("Error: FRED_API_KEY not found.")
    print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
    print(f"Then add to {CARS_ROOT / '.env'}:")
    print("  FRED_API_KEY=your-key-here")
    return None


def download_fred_macro_data(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
) -> pd.DataFrame:
    """
    Download macroeconomic indicators from FRED.

    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)

    Returns:
        DataFrame with daily-resampled macro indicators (forward-filled)
    """
    try:
        from fredapi import Fred
    except ImportError:
        print("Please install fredapi: pip install fredapi")
        return pd.DataFrame()

    api_key = get_fred_api_key()
    if api_key is None:
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    all_series = []

    for col_name, config in FRED_MACRO_SERIES.items():
        series_id = config['series_id']
        desc = config['description']
        print(f"  Downloading {col_name}: {desc} ({series_id})...")

        try:
            data = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            if data is not None and len(data) > 0:
                series = pd.Series(data, name=col_name)
                all_series.append(series)
                print(f"    Got {len(data)} observations ({config['frequency']})")
            else:
                print(f"    No data returned")
        except Exception as e:
            print(f"    Error: {e}")

    if not all_series:
        print("No macro data downloaded.")
        return pd.DataFrame()

    # Combine all series
    combined = pd.concat(all_series, axis=1)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()

    print(f"\n  Raw shape: {combined.shape}")
    print(f"  Date range: {combined.index.min()} to {combined.index.max()}")

    # Resample to daily frequency via forward-fill
    # Monthly/quarterly values are assigned to their observation date,
    # then propagated forward to maintain causality (no lookahead)
    combined_daily = combined.resample('D').ffill()

    # Trim to requested date range
    combined_daily = combined_daily[
        (combined_daily.index >= start) & (combined_daily.index <= end)
    ]

    print(f"  Daily shape (after ffill): {combined_daily.shape}")
    print(f"  Date range: {combined_daily.index.min()} to {combined_daily.index.max()}")

    return combined_daily


def create_macro_dataset(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Create a macro indicator dataset, with caching.

    Args:
        start: Start date
        end: End date
        data_dir: Directory to store cached data
        use_cache: Whether to use cached data

    Returns:
        DataFrame with daily macro indicators
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / f"macro_{start}_{end}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading cached macro data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    data = download_fred_macro_data(start, end)

    if not data.empty:
        data.to_csv(cache_file)
        print(f"\nSaved macro data to {cache_file}")

    return data


def main():
    """Download macroeconomic indicator data."""
    parser = argparse.ArgumentParser(description='Download macroeconomic indicator data')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--no-cache', action='store_true', help='Force re-download')
    args = parser.parse_args()

    print("=" * 60)
    print("Macroeconomic Indicator Data Download")
    print("=" * 60)
    print(f"Date range: {args.start} to {args.end}")
    print()

    print("Series to download:")
    for name, config in FRED_MACRO_SERIES.items():
        print(f"  {name}: {config['description']}")
    print()

    data = create_macro_dataset(args.start, args.end, use_cache=not args.no_cache)

    if data.empty:
        print("\nNo macro data downloaded.")
        print("Ensure FRED_API_KEY is set (free key from fred.stlouisfed.org)")
    else:
        print(f"\nFinal shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"\nSample statistics:")
        print(data.describe())

    print("\n" + "=" * 60)
    print("Macro data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
