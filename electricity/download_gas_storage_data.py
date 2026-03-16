"""
Download gas storage data from AGSI+ (GIE) for electricity price forecasting.

European gas storage fill levels directly impact electricity prices:
- Low storage -> supply scarcity -> higher gas & electricity prices
- High injection demand -> increased gas consumption -> price effects
- Seasonal patterns in storage track heating/cooling demand

Data source: https://agsi.gie.eu/
API docs: https://www.gie.eu/transparency-platform/GIE_API_documentation_v007.pdf

Usage:
    python3 download_gas_storage_data.py
    python3 download_gas_storage_data.py --start 2015-01-01 --end 2024-12-31
    python3 download_gas_storage_data.py --no-cache
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta

from country_config import has_gas_storage, get_registered_countries, COUNTRY_REGISTRY

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CARS_ROOT, GAS_STORAGE_DIR
# Data directory
DATA_DIR = GAS_STORAGE_DIR

# AGSI+ API configuration
AGSI_BASE_URL = "https://agsi.gie.eu/api"


def get_agsi_api_key() -> str:
    """
    Get AGSI+ API key from environment variable or .env file.

    Priority: environment variable > .env file > hardcoded fallback.
    """
    # 1. Try environment variable
    api_key = os.environ.get('AGSI_API_KEY')
    if api_key:
        return api_key

    # 2. Try .env file
    env_file = CARS_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().split('\n'):
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                if key.strip() == 'AGSI_API_KEY':
                    return value.strip().strip('"\'')

    raise ValueError(
        "AGSI_API_KEY not found. Set it as an environment variable or add to .env file.\n"
        "Register at https://agsi.gie.eu/account to get a free API key."
    )


def _parse_agsi_response(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw AGSI+ API response into a clean daily DataFrame."""
    # Set date index
    df['date'] = pd.to_datetime(df['gasDayStart'])
    df = df.set_index('date').sort_index()

    # Map API fields to clean column names
    numeric_cols = {
        'gasInStorage': 'level_twh',
        'full': 'fill_pct',
        'injection': 'injection_gwh',
        'withdrawal': 'withdrawal_gwh',
        'workingGasVolume': 'working_volume_twh',
        'injectionCapacity': 'injection_capacity_gwh',
        'withdrawalCapacity': 'withdrawal_capacity_gwh',
        'trend': 'trend_pct',
    }

    result = pd.DataFrame(index=df.index)
    for api_col, new_col in numeric_cols.items():
        if api_col in df.columns:
            result[new_col] = pd.to_numeric(df[api_col], errors='coerce')

    # Derived feature: net flow (injection - withdrawal)
    if 'injection_gwh' in result.columns and 'withdrawal_gwh' in result.columns:
        result['net_flow_gwh'] = result['injection_gwh'] - result['withdrawal_gwh']

    # Remove duplicate dates
    result = result[~result.index.duplicated(keep='first')]

    return result


def download_agsi_country_storage(
    country: str,
    start: str,
    end: str,
    api_key: str = None,
    chunk_days: int = 290,
    sleep_seconds: float = 1.0
) -> pd.DataFrame:
    """
    Download country-level gas storage data from AGSI+ API.

    Downloads in chunks to respect the 300-record page size limit.

    Args:
        country: 'DE' or 'FR'
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        api_key: AGSI+ API key (auto-detected if None)
        chunk_days: Days per request (max ~290 to stay under 300 limit)
        sleep_seconds: Delay between API calls

    Returns:
        DataFrame indexed by date with storage columns
    """
    if api_key is None:
        api_key = get_agsi_api_key()

    headers = {'x-key': api_key}
    all_data = []

    current_start = datetime.strptime(start, '%Y-%m-%d')
    final_end = datetime.strptime(end, '%Y-%m-%d')

    while current_start <= final_end:
        chunk_end = min(current_start + timedelta(days=chunk_days), final_end)

        params = {
            'type': 'eu',
            'country': country,
            'from': current_start.strftime('%Y-%m-%d'),
            'to': chunk_end.strftime('%Y-%m-%d'),
            'size': 300,
        }

        print(f"  Fetching {country}: {params['from']} to {params['to']}...")

        try:
            response = requests.get(
                AGSI_BASE_URL, headers=headers, params=params, timeout=30
            )
            response.raise_for_status()

            json_data = response.json()
            records = json_data.get('data', [])

            if records:
                all_data.extend(records)
                print(f"    Got {len(records)} records")
            else:
                print(f"    No records returned")

        except requests.exceptions.RequestException as e:
            print(f"    Error: {e}")
            print(f"    Waiting 10s before continuing...")
            time.sleep(10)

        time.sleep(sleep_seconds)
        current_start = chunk_end + timedelta(days=1)

    if not all_data:
        print(f"  WARNING: No data retrieved for {country}")
        return pd.DataFrame()

    # Parse into DataFrame
    df = pd.DataFrame(all_data)
    df = _parse_agsi_response(df)

    print(f"  {country}: {len(df)} records, {df.index.min()} to {df.index.max()}")
    return df


def create_gas_storage_dataset(
    start: str,
    end: str,
    countries: list = None,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Create gas storage datasets for specified countries.

    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        countries: List of country codes (default: all with gas storage)
        data_dir: Directory to save/cache data
        use_cache: Whether to use cached data

    Returns:
        Dictionary mapping country code to DataFrame
    """
    if countries is None:
        countries = [c for c in get_registered_countries() if has_gas_storage(c)]

    data_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for country in countries:
        cache_file = data_dir / f"gas_storage_{country}_{start}_{end}.csv"

        if use_cache and cache_file.exists():
            print(f"Loading cached gas storage data for {country} from {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            print(f"\nDownloading gas storage data for {country}...")
            df = download_agsi_country_storage(country, start, end)

            if not df.empty:
                # Forward-fill small gaps (weekends/holidays)
                df = df.ffill()

                # Save cache
                df.to_csv(cache_file)
                print(f"  Saved to {cache_file}")
            else:
                print(f"  WARNING: No gas storage data for {country}")

        if not df.empty:
            results[country] = df
            print(f"  {country}: shape={df.shape}, "
                  f"range={df.index.min()} to {df.index.max()}")

    return results


def main():
    """Download gas storage data."""
    parser = argparse.ArgumentParser(description='Download AGSI+ gas storage data')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end', type=str,
                        default=datetime.now().strftime('%Y-%m-%d'), help='End date')
    parser.add_argument('--countries', type=str, default=None,
                        help='Countries (comma-separated, default: all with gas storage)')
    parser.add_argument('--no-cache', action='store_true', help='Force re-download')
    args = parser.parse_args()

    print("=" * 60)
    print("AGSI+ Gas Storage Data Download")
    print("=" * 60)
    print(f"Date range: {args.start} to {args.end}")

    countries = args.countries.split(',') if args.countries else None
    results = create_gas_storage_dataset(
        start=args.start,
        end=args.end,
        countries=countries,
        use_cache=not args.no_cache
    )

    for country, df in results.items():
        print(f"\n{country} Summary:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(df.describe())

    print("\n" + "=" * 60)
    print("Gas storage data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
