"""
Download hydrogen cost data for electricity price forecasting.

Key indicators:
- Green hydrogen production cost (SPGCI, PEM electrolysis)
- Grey hydrogen production cost (SPGCI, SMR natural gas)
- Derived grey H2 cost from TTF gas price (fallback)

Green hydrogen is an emerging factor in European electricity markets as
electrolyzers become large electricity consumers and hydrogen storage
may become a flexibility resource.

Usage:
    python3 download_hydrogen_data.py
    python3 download_hydrogen_data.py --start 2015-01-01 --end 2024-12-31
    python3 download_hydrogen_data.py --no-cache
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (
    CARS_ROOT,
    COMMODITY_DIR,
    DATA_DIR,
    HYDROGEN_DIR,
)
# Data directory
DATA_DIR = HYDROGEN_DIR

# SPGCI hydrogen assessment symbols
SPGCI_HYDROGEN_SYMBOLS = {
    'hydrogen_green_cost': {
        'symbols': ['AAKZJ00'],
        'description': 'Green Hydrogen - PEM Electrolysis (EUR/kg)',
    },
    'hydrogen_grey_cost': {
        'symbols': ['AAKZK00'],
        'description': 'Grey Hydrogen - SMR Natural Gas (EUR/kg)',
    },
}

# Grey hydrogen derivation parameters
# SMR process: ~50 kWh natural gas per kg H2 (9 kg CH4 per kg H2 at ~33.3 kWh/kg LHV)
# In terms of gas price: grey_h2_cost ≈ ttf_gas_price_eur_mwh * 0.050
GREY_H2_GAS_FACTOR = 0.050  # kg H2 per MWh of gas input (approximate)


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


def download_spgci_hydrogen(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
) -> pd.DataFrame:
    """
    Download hydrogen price assessments from SPGCI.

    Args:
        start: Start date
        end: End date

    Returns:
        DataFrame with daily hydrogen costs
    """
    try:
        import spgci as ci
    except ImportError:
        print("  SPGCI not available. Install with: pip install spgci")
        return pd.DataFrame()

    # Setup credentials
    creds = _load_env_credentials()
    username = os.environ.get('SPGCI_USERNAME') or creds.get('SPGCI_USERNAME')
    password = os.environ.get('SPGCI_PASSWORD') or creds.get('SPGCI_PASSWORD')

    if not username or not password:
        print("  SPGCI credentials not found. Skipping hydrogen data.")
        return pd.DataFrame()

    ci.set_credentials(username, password)

    appkey = os.environ.get('SPGCI_APPKEY') or creds.get('SPGCI_APPKEY')
    if appkey:
        import spgci.config as config
        config.appkey = appkey

    md = ci.MarketData()
    all_series = []

    for col_name, sym_config in SPGCI_HYDROGEN_SYMBOLS.items():
        desc = sym_config['description']
        print(f"  Downloading {col_name}: {desc}...")

        for symbol in sym_config['symbols']:
            try:
                from datetime import date as dt_date
                start_date = dt_date.fromisoformat(start)
                end_date = dt_date.fromisoformat(end)
                data = md.get_assessments_by_symbol_historical(
                    symbol=symbol,
                    assess_date_gte=start_date,
                    assess_date_lte=end_date,
                    paginate=True,
                )

                if data is not None and len(data) > 0:
                    # Find date and price columns
                    date_col = None
                    for c in data.columns:
                        if 'date' in c.lower():
                            date_col = c
                            break

                    price_col = None
                    for c in ['close', 'value', 'price', 'settlement']:
                        if c in data.columns:
                            price_col = c
                            break

                    if date_col and price_col:
                        ts = data.groupby(date_col)[price_col].mean()
                        ts.index = pd.to_datetime(ts.index)
                        ts.name = col_name
                        all_series.append(ts)
                        print(f"    Got {len(ts)} observations for {symbol}")
                    else:
                        print(f"    Unexpected columns: {list(data.columns)}")
                else:
                    print(f"    No data for {symbol}")

            except Exception as e:
                err_str = str(e)
                if 'Invalid symbols' in err_str or '400' in err_str:
                    print(f"    Symbol {symbol} not in SPGCI subscription (requires Hydrogen add-on)")
                else:
                    print(f"    Error for {symbol}: {e}")

    if all_series:
        combined = pd.concat(all_series, axis=1)
        combined = combined.sort_index()
        combined = combined.resample('D').ffill()
        combined = combined[(combined.index >= start) & (combined.index <= end)]
        print(f"  SPGCI hydrogen shape: {combined.shape}")
        return combined

    return pd.DataFrame()


def derive_grey_hydrogen_cost(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
) -> pd.DataFrame:
    """
    Derive grey hydrogen production cost from natural gas (TTF) prices.

    Grey hydrogen via Steam Methane Reforming (SMR) cost is primarily
    determined by natural gas feedstock cost. Approximate conversion:
    ~50 kWh of gas per kg of H2 produced.

    Args:
        start: Start date
        end: End date

    Returns:
        DataFrame with derived daily grey H2 cost
    """
    print("  Deriving grey hydrogen cost from gas prices...")

    # Try to load SPGCI TTF gas price first
    spgci_dir = DATA_DIR / "spgci"
    spgci_files = sorted(spgci_dir.glob("commodity_timeseries_*.csv")) if spgci_dir.exists() else []

    gas_price = None

    if spgci_files:
        try:
            df = pd.read_csv(spgci_files[-1], index_col=0, parse_dates=True)
            if 'TTF_Gas' in df.columns:
                gas_price = df['TTF_Gas']
                print(f"    Using SPGCI TTF gas price ({len(gas_price)} observations)")
        except Exception as e:
            print(f"    Could not load SPGCI data: {e}")

    # Fallback: use FRED Henry Hub gas price
    if gas_price is None:
        try:
            from fredapi import Fred
            api_key = os.environ.get('FRED_API_KEY')
            if api_key is None:
                creds = _load_env_credentials()
                api_key = creds.get('FRED_API_KEY')
            if api_key:
                fred = Fred(api_key=api_key)
                data = fred.get_series('DHHNGSP', observation_start=start, observation_end=end)
                if data is not None and len(data) > 0:
                    # Henry Hub ($/MMBtu) -> approximate EUR/MWh: * 3.412 (MMBtu/MWh)
                    gas_price = data * 3.412
                    print(f"    Using FRED Henry Hub gas price ({len(data)} obs, converted to $/MWh)")
        except Exception as e:
            print(f"    Could not load FRED gas data: {e}")

    # Fallback: use commodity CSV files
    if gas_price is None:
        commodity_dir = COMMODITY_DIR
        commodity_files = sorted(commodity_dir.glob("commodities_*.csv"))

        if commodity_files:
            try:
                df = pd.read_csv(commodity_files[-1], index_col=0, parse_dates=True)
                gas_col = [c for c in df.columns if 'gas' in c.lower() or 'natural' in c.lower()]
                if gas_col:
                    # Henry Hub ($/MMBtu) -> approximate EUR/MWh: * 3.0 / 0.293
                    gas_price = df[gas_col[0]] * 3.0 / 0.293
                    print(f"    Using Yahoo Finance gas price (Henry Hub -> EUR/MWh proxy)")
            except Exception as e:
                print(f"    Could not load commodity data: {e}")

    if gas_price is None:
        print("    No gas price data available for hydrogen cost derivation.")
        return pd.DataFrame()

    gas_price.index = pd.to_datetime(gas_price.index)

    # Derive grey H2 cost: gas_price (EUR/MWh) * 0.050 (MWh per kg H2)
    grey_h2_cost = gas_price * GREY_H2_GAS_FACTOR

    result = pd.DataFrame({
        'hydrogen_grey_cost': grey_h2_cost
    })
    result = result.sort_index()
    result = result[(result.index >= start) & (result.index <= end)]
    result = result.resample('D').ffill()

    print(f"    Derived {len(result)} daily grey H2 cost estimates")
    return result


def create_hydrogen_dataset(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Create a hydrogen cost dataset, with caching.

    Tries SPGCI first, then falls back to derived costs from gas prices.

    Args:
        start: Start date
        end: End date
        data_dir: Directory to store cached data
        use_cache: Whether to use cached data

    Returns:
        DataFrame with daily hydrogen cost indicators
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / f"hydrogen_{start}_{end}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading cached hydrogen data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    all_data = []

    # 1. SPGCI hydrogen assessments (primary)
    print("\n--- SPGCI Hydrogen Assessments ---")
    spgci_data = download_spgci_hydrogen(start, end)
    if not spgci_data.empty:
        all_data.append(spgci_data)

    # 2. Derived grey H2 cost (fallback if SPGCI doesn't have it)
    has_grey = any('hydrogen_grey_cost' in (d.columns if hasattr(d, 'columns') else []) for d in all_data)
    if not has_grey:
        print("\n--- Derived Grey Hydrogen Cost ---")
        derived = derive_grey_hydrogen_cost(start, end)
        if not derived.empty:
            all_data.append(derived)

    if not all_data:
        print("\nNo hydrogen data available.")
        return pd.DataFrame()

    # Combine
    combined = pd.concat(all_data, axis=1)
    # Deduplicate columns (prefer SPGCI over derived)
    combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]
    combined = combined.sort_index()
    combined = combined.ffill()

    # Trim
    combined = combined[(combined.index >= start) & (combined.index <= end)]

    print(f"\nCombined shape: {combined.shape}")
    print(f"Date range: {combined.index.min()} to {combined.index.max()}")

    # Save
    combined.to_csv(cache_file)
    print(f"Saved hydrogen data to {cache_file}")

    return combined


def main():
    """Download hydrogen cost data."""
    parser = argparse.ArgumentParser(description='Download hydrogen cost data')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--no-cache', action='store_true', help='Force re-download')
    args = parser.parse_args()

    print("=" * 60)
    print("Hydrogen Cost Data Download")
    print("=" * 60)
    print(f"Date range: {args.start} to {args.end}")
    print()

    data = create_hydrogen_dataset(args.start, args.end, use_cache=not args.no_cache)

    if data.empty:
        print("\nNo hydrogen data downloaded.")
        print("\nTo enable data sources:")
        print("  1. SPGCI: pip install spgci, set credentials in .env")
        print("  2. Derived: requires gas price data (commodity or SPGCI)")
    else:
        print(f"\nFinal shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"\nSample statistics:")
        print(data.describe())

    print("\n" + "=" * 60)
    print("Hydrogen data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
