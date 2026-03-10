"""
Download EU import/export and trade balance data for electricity price forecasting.

Key indicators:
- EU Trade Balance (monthly, Eurostat)
- EU Energy Imports (monthly, Eurostat)
- German Trade Balance (monthly, FRED)

Eurostat provides free access to EU statistical data via the eurostat Python package.
FRED provides German trade data as a free alternative.

Usage:
    python3 download_trade_data.py
    python3 download_trade_data.py --start 2015-01-01 --end 2024-12-31
    python3 download_trade_data.py --no-cache
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CARS_ROOT, TRADE_DIR
# Data directory
DATA_DIR = TRADE_DIR

# FRED series for trade data (fallback)
FRED_TRADE_SERIES = {
    'trade_de_balance': {
        'series_id': 'XTEXVA01DEM664S',
        'description': 'German Trade Balance (monthly)',
        'frequency': 'M',
    },
}

# Eurostat dataset codes
EUROSTAT_DATASETS = {
    'trade_eu_balance': {
        'code': 'ext_lt_maineu',
        'description': 'EU Trade Balance with rest of world (monthly)',
    },
    'trade_eu_energy_imports': {
        'code': 'nrg_ti_m',
        'description': 'EU Energy Trade - Imports (monthly)',
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


def download_eurostat_trade(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
) -> pd.DataFrame:
    """
    Download EU trade data from Eurostat.

    Args:
        start: Start date
        end: End date

    Returns:
        DataFrame with monthly trade data (forward-filled to daily)
    """
    try:
        import eurostat
    except ImportError:
        print("  Eurostat package not installed. Install with: pip install eurostat")
        return pd.DataFrame()

    start_year = int(start[:4])
    end_year = int(end[:4])
    all_series = []

    # Try to get EU trade balance
    print("  Downloading EU trade balance from Eurostat...")
    try:
        # ext_lt_maineu: EU trade with rest of world
        df = eurostat.get_data_df('ext_lt_maineu')

        if df is not None and len(df) > 0:
            # Eurostat data comes in wide format with time periods as columns
            # Find the time columns (format: YYYY-MM or YYYYMXX)
            time_cols = [c for c in df.columns if str(c).startswith('20')]
            meta_cols = [c for c in df.columns if c not in time_cols]

            if time_cols:
                # Filter for balance/total rows
                # Look for total trade balance rows
                filter_cols = [c for c in meta_cols if 'partner' in c.lower() or 'flow' in c.lower()]

                # Melt to long format
                df_long = df.melt(id_vars=meta_cols, value_vars=time_cols,
                                  var_name='period', value_name='value')
                df_long['date'] = pd.to_datetime(df_long['period'], format='mixed', errors='coerce')
                df_long = df_long.dropna(subset=['date', 'value'])

                if len(df_long) > 0:
                    # Aggregate to get a single trade balance series
                    trade_balance = df_long.groupby('date')['value'].sum()
                    trade_balance = trade_balance.sort_index()
                    trade_balance = trade_balance[
                        (trade_balance.index.year >= start_year) &
                        (trade_balance.index.year <= end_year)
                    ]

                    if len(trade_balance) > 0:
                        series = pd.Series(trade_balance.values, index=trade_balance.index,
                                          name='trade_eu_balance')
                        all_series.append(series)
                        print(f"    Got {len(trade_balance)} monthly observations")
                    else:
                        print("    No data in requested date range")
                else:
                    print("    Could not parse Eurostat data")
            else:
                print("    No time columns found in Eurostat data")
        else:
            print("    No data returned from Eurostat")

    except Exception as e:
        print(f"    Eurostat trade balance error: {e}")

    # Try to get EU energy imports
    print("  Downloading EU energy imports from Eurostat...")
    try:
        df = eurostat.get_data_df('nrg_ti_m')

        if df is not None and len(df) > 0:
            time_cols = [c for c in df.columns if str(c).startswith('20')]
            meta_cols = [c for c in df.columns if c not in time_cols]

            if time_cols:
                df_long = df.melt(id_vars=meta_cols, value_vars=time_cols,
                                  var_name='period', value_name='value')
                df_long['date'] = pd.to_datetime(df_long['period'], format='mixed', errors='coerce')
                df_long = df_long.dropna(subset=['date', 'value'])

                if len(df_long) > 0:
                    energy_imports = df_long.groupby('date')['value'].sum()
                    energy_imports = energy_imports.sort_index()
                    energy_imports = energy_imports[
                        (energy_imports.index.year >= start_year) &
                        (energy_imports.index.year <= end_year)
                    ]

                    if len(energy_imports) > 0:
                        series = pd.Series(energy_imports.values, index=energy_imports.index,
                                          name='trade_eu_energy_imports')
                        all_series.append(series)
                        print(f"    Got {len(energy_imports)} monthly observations")

        else:
            print("    No data returned from Eurostat")

    except Exception as e:
        print(f"    Eurostat energy imports error: {e}")

    if all_series:
        combined = pd.concat(all_series, axis=1)
        combined.index = pd.to_datetime(combined.index)
        combined = combined.sort_index()

        # Resample to daily via forward-fill
        combined_daily = combined.resample('D').ffill()
        combined_daily = combined_daily[
            (combined_daily.index >= start) & (combined_daily.index <= end)
        ]

        print(f"  Eurostat daily shape (after ffill): {combined_daily.shape}")
        return combined_daily

    return pd.DataFrame()


def download_fred_trade(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
) -> pd.DataFrame:
    """
    Download German trade balance from FRED as fallback.

    Args:
        start: Start date
        end: End date

    Returns:
        DataFrame with daily trade data (forward-filled from monthly)
    """
    try:
        from fredapi import Fred
    except ImportError:
        print("  FRED API not available. Install with: pip install fredapi")
        return pd.DataFrame()

    api_key = os.environ.get('FRED_API_KEY')
    if api_key is None:
        creds = _load_env_credentials()
        api_key = creds.get('FRED_API_KEY')

    if api_key is None:
        print("  Warning: FRED_API_KEY not set. Skipping FRED trade data.")
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    all_series = []

    for col_name, config in FRED_TRADE_SERIES.items():
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
                print(f"    Got {len(data)} observations")
            else:
                print(f"    No data returned")
        except Exception as e:
            print(f"    Error: {e}")

    if not all_series:
        return pd.DataFrame()

    combined = pd.concat(all_series, axis=1)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()

    # Resample to daily via forward-fill
    combined_daily = combined.resample('D').ffill()
    combined_daily = combined_daily[
        (combined_daily.index >= start) & (combined_daily.index <= end)
    ]

    print(f"  FRED trade daily shape (after ffill): {combined_daily.shape}")
    return combined_daily


def create_trade_dataset(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Create a combined trade dataset, with caching.

    Args:
        start: Start date
        end: End date
        data_dir: Directory to store cached data
        use_cache: Whether to use cached data

    Returns:
        DataFrame with daily trade indicators
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / f"trade_{start}_{end}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading cached trade data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    all_data = []

    # 1. Eurostat (EU trade balance, energy imports)
    print("\n--- Eurostat Trade Data ---")
    eurostat_data = download_eurostat_trade(start, end)
    if not eurostat_data.empty:
        all_data.append(eurostat_data)

    # 2. FRED (German trade balance)
    print("\n--- FRED Trade Data ---")
    fred_data = download_fred_trade(start, end)
    if not fred_data.empty:
        all_data.append(fred_data)

    if not all_data:
        print("\nNo trade data downloaded.")
        return pd.DataFrame()

    # Combine all sources
    combined = pd.concat(all_data, axis=1)
    combined = combined.sort_index()
    combined = combined.ffill()

    # Trim to date range
    combined = combined[(combined.index >= start) & (combined.index <= end)]

    print(f"\nCombined shape: {combined.shape}")
    print(f"Date range: {combined.index.min()} to {combined.index.max()}")

    # Save
    combined.to_csv(cache_file)
    print(f"Saved trade data to {cache_file}")

    return combined


def main():
    """Download trade data."""
    parser = argparse.ArgumentParser(description='Download EU trade data')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--no-cache', action='store_true', help='Force re-download')
    args = parser.parse_args()

    print("=" * 60)
    print("EU Trade Data Download")
    print("=" * 60)
    print(f"Date range: {args.start} to {args.end}")
    print()

    data = create_trade_dataset(args.start, args.end, use_cache=not args.no_cache)

    if data.empty:
        print("\nNo trade data downloaded.")
        print("\nTo enable data sources:")
        print("  1. Eurostat: pip install eurostat")
        print("  2. FRED: pip install fredapi, set FRED_API_KEY in .env")
    else:
        print(f"\nFinal shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"\nSample statistics:")
        print(data.describe())

    print("\n" + "=" * 60)
    print("Trade data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
