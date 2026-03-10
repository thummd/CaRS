"""
Download commodity data from S&P Global Commodity Insights API.

This script downloads:
- TTF Natural Gas prices (European benchmark)
- API2 Coal prices
- EU Carbon (EUA) prices

SETUP REQUIRED:
1. You need an API key from S&P Global developer portal
2. Go to https://developer.spglobal.com/
3. Log in and navigate to your API keys
4. Copy your API key for Market Data
5. Set environment variable: SPGCI_APPKEY='your-api-key'

Or set credentials in .env file:
SPGCI_USERNAME=your-email@domain.com
SPGCI_PASSWORD=your-password
SPGCI_APPKEY=your-api-key

Note: The spgci library handles authentication automatically once credentials are set.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CARS_ROOT, DATA_DIR
# Data directory
DATA_DIR = DATA_DIR / "spgci"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Key commodity symbols for electricity price forecasting
COMMODITY_SYMBOLS = {
    # TTF Natural Gas
    'TTF_Gas': {
        'symbols': ['PCAAS00', 'AAVOB00'],  # TTF day-ahead, TTF month-ahead
        'description': 'TTF Natural Gas Hub (Netherlands) - European benchmark'
    },
    # Coal
    'API2_Coal': {
        'symbols': ['AAQZB00', 'AAQZC00'],  # API2 Coal CIF ARA
        'description': 'API2 Coal CIF ARA - European coal benchmark'
    },
    # Carbon
    'EU_Carbon': {
        'symbols': ['EECRA00', 'EECRS00'],  # EU ETS Carbon
        'description': 'EU ETS Carbon (EUA) prices'
    },
    # Power
    'DE_Power': {
        'symbols': ['EADEA00', 'EADEB00'],  # German power
        'description': 'German baseload power'
    },
    'FR_Power': {
        'symbols': ['EAFRA00', 'EAFRB00'],  # French power
        'description': 'French baseload power'
    }
}


def setup_spgci():
    """
    Set up SPGCI library with credentials.

    Returns True if setup successful, False otherwise.
    """
    try:
        import spgci as ci
    except ImportError:
        print("Please install spgci: pip install spgci")
        return False

    # Try environment variables
    username = os.environ.get('SPGCI_USERNAME')
    password = os.environ.get('SPGCI_PASSWORD')
    appkey = os.environ.get('SPGCI_APPKEY')

    # Try .env file
    env_file = CARS_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().split('\n'):
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                if key == 'SPGCI_USERNAME' and not username:
                    username = value
                elif key == 'SPGCI_PASSWORD' and not password:
                    password = value
                elif key == 'SPGCI_APPKEY' and not appkey:
                    appkey = value

    if not username or not password:
        print("Error: SPGCI_USERNAME and SPGCI_PASSWORD are required")
        print(f"Set environment variables or add to {CARS_ROOT / '.env'}")
        return False

    ci.set_credentials(username, password)

    if appkey:
        # Set appkey if available (may be needed for some endpoints)
        import spgci.config as config
        config.appkey = appkey

    return True


def get_price_assessments(
    symbols: list,
    start_date: str,
    end_date: str,
    save_path: Path = None
) -> pd.DataFrame:
    """
    Get price assessments for given symbols.

    Args:
        symbols: List of S&P Global symbol codes
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        save_path: Path to save CSV (optional)

    Returns:
        DataFrame with price assessments
    """
    import spgci as ci

    md = ci.MarketData()

    all_data = []
    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        try:
            # Get historical assessments
            data = md.get_assessments_by_symbol(
                symbol=symbol,
                assess_date_gte=start_date,
                assess_date_lte=end_date
            )

            if data is not None and len(data) > 0:
                data['symbol'] = symbol
                all_data.append(data)
                print(f"    Got {len(data)} records")
            else:
                print(f"    No data returned")

        except Exception as e:
            print(f"    Error: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)

        if save_path:
            combined.to_csv(save_path, index=False)
            print(f"  Saved to {save_path}")

        return combined

    return pd.DataFrame()


def download_all_commodities(
    start_date: str = '2015-01-01',
    end_date: str = None
) -> dict:
    """
    Download all relevant commodities for electricity price forecasting.

    Args:
        start_date: Start date
        end_date: End date (defaults to yesterday)

    Returns:
        Dictionary of DataFrames by commodity
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"Downloading SPGCI data: {start_date} to {end_date}")
    print("="*60)

    if not setup_spgci():
        print("\nFailed to set up SPGCI. Please check credentials.")
        return {}

    results = {}

    for name, config in COMMODITY_SYMBOLS.items():
        print(f"\n--- {name}: {config['description']} ---")

        save_path = DATA_DIR / f"{name}_{start_date}_{end_date}.csv"

        data = get_price_assessments(
            symbols=config['symbols'],
            start_date=start_date,
            end_date=end_date,
            save_path=save_path
        )

        if not data.empty:
            results[name] = data
            print(f"  Shape: {data.shape}")
        else:
            print(f"  No data obtained")

    return results


def create_commodity_timeseries(
    raw_data: dict,
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Create a clean timeseries from raw SPGCI data.

    Args:
        raw_data: Dictionary of DataFrames from download_all_commodities
        freq: Resampling frequency ('D' for daily, 'H' for hourly)

    Returns:
        DataFrame indexed by date with commodity prices
    """
    dfs = []

    for name, data in raw_data.items():
        if data.empty:
            continue

        # Pivot by date and symbol
        if 'assessDate' in data.columns:
            data['date'] = pd.to_datetime(data['assessDate'])
        elif 'assess_date' in data.columns:
            data['date'] = pd.to_datetime(data['assess_date'])
        else:
            print(f"Warning: No date column in {name}")
            continue

        # Get close/settlement price
        price_col = None
        for col in ['close', 'value', 'price', 'settlement']:
            if col in data.columns:
                price_col = col
                break

        if price_col is None:
            print(f"Warning: No price column in {name}")
            continue

        # Create simple timeseries
        ts = data.groupby('date')[price_col].mean()
        ts.name = name
        dfs.append(ts)

    if dfs:
        combined = pd.concat(dfs, axis=1)
        combined = combined.sort_index()
        combined = combined.resample(freq).last().ffill()
        return combined

    return pd.DataFrame()


def test_authentication():
    """Test SPGCI authentication."""
    print("Testing S&P Global CI authentication...")
    print("="*60)

    if not setup_spgci():
        return False

    try:
        import spgci as ci
        md = ci.MarketData()

        # Try to get symbols
        print("\nSearching for TTF symbols...")
        symbols = md.get_symbols(q='TTF', page_size=5)
        print(f"Found {len(symbols)} symbols:")
        print(symbols)
        return True

    except Exception as e:
        print(f"\nAuthentication test failed: {e}")
        print("\nPossible issues:")
        print("1. Invalid credentials")
        print("2. Missing API key (SPGCI_APPKEY)")
        print("3. No API access for Market Data")
        print("\nTo get API access:")
        print("1. Go to https://developer.spglobal.com/")
        print("2. Log in with your S&P Global credentials")
        print("3. Request access to Market Data API")
        print("4. Copy your API key and set SPGCI_APPKEY environment variable")
        return False


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Download SPGCI commodity data')
    parser.add_argument('--test', action='store_true', help='Test authentication')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end', type=str, default=None, help='End date')

    args = parser.parse_args()

    if args.test:
        test_authentication()
        return

    results = download_all_commodities(args.start, args.end)

    if results:
        print("\n" + "="*60)
        print("Creating commodity timeseries...")
        ts = create_commodity_timeseries(results)

        if not ts.empty:
            ts_file = DATA_DIR / f"commodity_timeseries_{args.start}_{args.end or 'latest'}.csv"
            ts.to_csv(ts_file)
            print(f"Saved to {ts_file}")
            print(f"Shape: {ts.shape}")
            print(f"Columns: {list(ts.columns)}")
            print(f"\nSample:")
            print(ts.head())


if __name__ == "__main__":
    main()
