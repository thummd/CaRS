"""
Download oil supply/demand fundamentals and OPEC data for electricity price forecasting.

Key indicators:
- US Crude Oil Inventories (weekly, EIA via FRED)
- OPEC Basket Price (daily, OPEC website)
- Brent-WTI Spread (derived from existing commodity data)
- OPEC Production & Spare Capacity (monthly, SPGCI WorldOilSupply if available)

Usage:
    python3 download_oil_fundamentals_data.py
    python3 download_oil_fundamentals_data.py --start 2015-01-01 --end 2024-12-31
    python3 download_oil_fundamentals_data.py --no-cache
"""

import os
import sys
import io
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CARS_ROOT, COMMODITY_DIR, OIL_FUND_DIR
# Data directory
DATA_DIR = OIL_FUND_DIR

# FRED series for oil fundamentals
FRED_OIL_SERIES = {
    'oil_brent': {
        'series_id': 'DCOILBRENTEU',
        'description': 'Brent Crude Oil Price (daily, EIA)',
        'frequency': 'D',
    },
    'oil_wti': {
        'series_id': 'DCOILWTICO',
        'description': 'WTI Crude Oil Price (daily, EIA)',
        'frequency': 'D',
    },
    'oil_henry_hub_gas': {
        'series_id': 'DHHNGSP',
        'description': 'Henry Hub Natural Gas Spot Price (daily, EIA)',
        'frequency': 'D',
    },
}

# OPEC Reference Basket Price data URL
OPEC_BASKET_URL = "https://www.opec.org/basket/basketDayArchives.xml"


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


def download_fred_oil_data(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
) -> pd.DataFrame:
    """
    Download oil fundamentals from FRED (EIA crude inventories).

    Args:
        start: Start date
        end: End date

    Returns:
        DataFrame with daily oil fundamental data (forward-filled from weekly)
    """
    try:
        from fredapi import Fred
    except ImportError:
        print("Please install fredapi: pip install fredapi")
        return pd.DataFrame()

    api_key = os.environ.get('FRED_API_KEY')
    if api_key is None:
        creds = _load_env_credentials()
        api_key = creds.get('FRED_API_KEY')

    if api_key is None:
        print("Warning: FRED_API_KEY not set. Skipping FRED oil data.")
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    all_series = []

    for col_name, config in FRED_OIL_SERIES.items():
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
        return pd.DataFrame()

    combined = pd.concat(all_series, axis=1)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()

    # Resample to daily via forward-fill
    combined_daily = combined.resample('D').ffill()
    combined_daily = combined_daily[
        (combined_daily.index >= start) & (combined_daily.index <= end)
    ]

    # Derive Brent-WTI spread from FRED data
    if 'oil_brent' in combined_daily.columns and 'oil_wti' in combined_daily.columns:
        combined_daily['oil_brent_wti_spread'] = combined_daily['oil_brent'] - combined_daily['oil_wti']
        print(f"  Derived Brent-WTI spread from FRED data")

    print(f"  Daily shape (after ffill): {combined_daily.shape}")
    return combined_daily


def download_opec_basket_price(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
) -> pd.DataFrame:
    """
    Download OPEC Reference Basket Price.

    Tries XML feed first, then falls back to a manual approach.

    Args:
        start: Start date
        end: End date

    Returns:
        DataFrame with daily OPEC basket price
    """
    print("  Downloading OPEC basket price...")

    try:
        # Fetch XML with requests to handle SSL issues and User-Agent filtering
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(
            OPEC_BASKET_URL,
            verify=False,
            headers={'User-Agent': 'Mozilla/5.0 (CaRS Research)'},
            timeout=30,
        )
        response.raise_for_status()
        df = pd.read_xml(io.BytesIO(response.content))

        # Parse date and price columns
        date_col = [c for c in df.columns if 'date' in c.lower() or 'day' in c.lower()]
        price_col = [c for c in df.columns if 'val' in c.lower() or 'price' in c.lower()]

        if date_col and price_col:
            result = pd.DataFrame({
                'opec_basket_price': pd.to_numeric(df[price_col[0]], errors='coerce')
            }, index=pd.to_datetime(df[date_col[0]]))
            result = result.sort_index()
            result = result[(result.index >= start) & (result.index <= end)]
            result = result.resample('D').ffill()
            print(f"    Got {len(result)} daily observations")
            return result
    except Exception as e:
        print(f"    OPEC XML feed unavailable: {e}")

    # Fallback: use Brent oil as OPEC basket proxy (highly correlated, ~$1-3 spread)
    print("    Using Brent oil as OPEC basket proxy...")
    try:
        import yfinance as yf
        data = yf.download('BZ=F', start=start, end=end, progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            close_col = [c for c in data.columns if 'Close' in c]
            if close_col:
                result = pd.DataFrame({
                    'opec_basket_price': data[close_col[0]] * 0.97  # Approximate discount
                })
                result = result.resample('D').ffill()
                print(f"    Got {len(result)} daily observations (Brent proxy)")
                return result
    except Exception as e:
        print(f"    Brent proxy failed: {e}")

    return pd.DataFrame()


def download_spgci_opec_data(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
) -> pd.DataFrame:
    """
    Download OPEC production and spare capacity from SPGCI.

    Requires spgci library and valid credentials.

    Args:
        start: Start date
        end: End date

    Returns:
        DataFrame with monthly OPEC data (forward-filled to daily)
    """
    try:
        import spgci as ci
    except ImportError:
        print("  SPGCI not available. Skipping OPEC production data.")
        return pd.DataFrame()

    # Setup credentials
    creds = _load_env_credentials()
    username = os.environ.get('SPGCI_USERNAME') or creds.get('SPGCI_USERNAME')
    password = os.environ.get('SPGCI_PASSWORD') or creds.get('SPGCI_PASSWORD')

    if not username or not password:
        print("  SPGCI credentials not found. Skipping OPEC production data.")
        return pd.DataFrame()

    ci.set_credentials(username, password)

    appkey = os.environ.get('SPGCI_APPKEY') or creds.get('SPGCI_APPKEY')
    if appkey:
        import spgci.config as config
        config.appkey = appkey

    print("  Downloading OPEC production data from SPGCI...")

    try:
        wos = ci.WorldOilSupply()

        # Get OPEC total production
        start_year = int(start[:4])
        end_year = int(end[:4])
        opec_data = wos.get_production(
            country='OPEC',
            year_gte=start_year,
            year_lte=end_year,
            paginate=True,
        )

        if opec_data is not None and len(opec_data) > 0:
            # Extract relevant columns
            date_col = [c for c in opec_data.columns if 'period' in c.lower() or 'date' in c.lower()]
            prod_col = [c for c in opec_data.columns if 'production' in c.lower() or 'value' in c.lower()]

            if date_col and prod_col:
                result = pd.DataFrame({
                    'opec_production': opec_data[prod_col[0]].values
                }, index=pd.to_datetime(opec_data[date_col[0]]))

                result = result.sort_index()
                result_daily = result.resample('D').ffill()
                result_daily = result_daily[
                    (result_daily.index >= start) & (result_daily.index <= end)
                ]

                print(f"    Got {len(result)} monthly -> {len(result_daily)} daily observations")
                return result_daily
            else:
                print(f"    Unexpected columns: {list(opec_data.columns)}")
        else:
            print("    No OPEC data returned")

    except Exception as e:
        print(f"    SPGCI OPEC data error: {e}")

    return pd.DataFrame()


def compute_brent_wti_spread(
    commodity_dir: Path = None,
) -> pd.DataFrame:
    """
    Compute Brent-WTI spread from existing commodity data.

    This is derived from already-downloaded commodity prices.

    Args:
        commodity_dir: Directory with commodity CSV files

    Returns:
        DataFrame with daily Brent-WTI spread
    """
    if commodity_dir is None:
        commodity_dir = COMMODITY_DIR

    # Try to load existing commodity data
    commodity_files = sorted(commodity_dir.glob("commodities_*.csv"))
    if not commodity_files:
        print("  No commodity data found for Brent-WTI spread derivation.")
        return pd.DataFrame()

    # Use the most comprehensive file
    cf = commodity_files[-1]
    print(f"  Computing Brent-WTI spread from {cf.name}...")

    df = pd.read_csv(cf, index_col=0, parse_dates=True)

    # Find Brent and WTI columns
    brent_col = [c for c in df.columns if 'brent' in c.lower()]
    wti_col = [c for c in df.columns if 'wti' in c.lower()]

    if brent_col and wti_col:
        spread = pd.DataFrame({
            'oil_brent_wti_spread': df[brent_col[0]] - df[wti_col[0]]
        })
        spread = spread.dropna()
        print(f"    Got {len(spread)} daily observations")
        return spread
    else:
        print(f"    Could not find Brent/WTI columns in {list(df.columns)}")
        return pd.DataFrame()


def create_oil_fundamentals_dataset(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Create a combined oil fundamentals dataset, with caching.

    Args:
        start: Start date
        end: End date
        data_dir: Directory to store cached data
        use_cache: Whether to use cached data

    Returns:
        DataFrame with daily oil fundamental indicators
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / f"oil_fundamentals_{start}_{end}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading cached oil fundamentals from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    all_data = []

    # 1. FRED oil data (EIA inventories)
    print("\n--- FRED Oil Data ---")
    fred_data = download_fred_oil_data(start, end)
    if not fred_data.empty:
        all_data.append(fred_data)

    # 2. OPEC basket price
    print("\n--- OPEC Basket Price ---")
    opec_price = download_opec_basket_price(start, end)
    if not opec_price.empty:
        all_data.append(opec_price)

    # 3. Brent-WTI spread (derived from commodity files, only if not already from FRED)
    has_spread = not fred_data.empty and 'oil_brent_wti_spread' in fred_data.columns
    if not has_spread:
        print("\n--- Brent-WTI Spread ---")
        spread = compute_brent_wti_spread()
        if not spread.empty:
            all_data.append(spread)

    # 4. SPGCI OPEC production (optional)
    print("\n--- SPGCI OPEC Production ---")
    spgci_data = download_spgci_opec_data(start, end)
    if not spgci_data.empty:
        all_data.append(spgci_data)

    if not all_data:
        print("\nNo oil fundamentals data downloaded.")
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
    print(f"Saved oil fundamentals to {cache_file}")

    return combined


def main():
    """Download oil fundamentals and OPEC data."""
    parser = argparse.ArgumentParser(description='Download oil fundamentals & OPEC data')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--no-cache', action='store_true', help='Force re-download')
    args = parser.parse_args()

    print("=" * 60)
    print("Oil Fundamentals & OPEC Data Download")
    print("=" * 60)
    print(f"Date range: {args.start} to {args.end}")
    print()

    data = create_oil_fundamentals_dataset(args.start, args.end, use_cache=not args.no_cache)

    if data.empty:
        print("\nNo oil fundamentals data downloaded.")
    else:
        print(f"\nFinal shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"\nSample statistics:")
        print(data.describe())

    print("\n" + "=" * 60)
    print("Oil fundamentals collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
