"""
Download market sentiment and risk indicator data for electricity price forecasting.

Key indicators:
- VIX (CBOE Volatility Index) - market fear gauge (daily, Yahoo Finance)
- Gold futures - safe-haven demand proxy (daily, Yahoo Finance)
- EUR/USD exchange rate - currency risk (daily, FRED)
- European Economic Policy Uncertainty (EPU) index (monthly, policyuncertainty.com)

Usage:
    python3 download_sentiment_data.py
    python3 download_sentiment_data.py --start 2015-01-01 --end 2024-12-31
    python3 download_sentiment_data.py --no-cache
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CARS_ROOT, SENTIMENT_DIR
# Data directory
DATA_DIR = SENTIMENT_DIR

# Yahoo Finance tickers for daily sentiment indicators
YFINANCE_SENTIMENT = {
    'sentiment_vix': {
        'ticker': '^VIX',
        'description': 'CBOE Volatility Index (VIX)',
    },
    'sentiment_gold': {
        'ticker': 'GC=F',
        'description': 'Gold Futures (COMEX)',
    },
}

# FRED series for sentiment-related data
FRED_SENTIMENT = {
    'sentiment_eurusd': {
        'series_id': 'DEXUSEU',
        'description': 'EUR/USD Exchange Rate (daily)',
    },
}

# EPU data URL
EPU_URL = "https://www.policyuncertainty.com/media/Europe_Policy_Uncertainty_Data.xlsx"


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


def download_yfinance_sentiment(
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download VIX and Gold from Yahoo Finance.

    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)

    Returns:
        DataFrame with daily sentiment indicators
    """
    try:
        import yfinance as yf
    except ImportError:
        print("Please install yfinance: pip install yfinance")
        return pd.DataFrame()

    all_data = []

    for col_name, config in YFINANCE_SENTIMENT.items():
        ticker = config['ticker']
        desc = config['description']
        print(f"  Downloading {col_name}: {desc} ({ticker})...")

        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data is not None and not data.empty:
                # Flatten multi-index columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                # Keep only Close price
                close_col = [c for c in data.columns if 'Close' in c]
                if close_col:
                    series = data[close_col[0]].rename(col_name)
                    all_data.append(series)
                    print(f"    Got {len(series)} daily observations")
            else:
                print(f"    No data returned")
        except Exception as e:
            print(f"    Error: {e}")

    if all_data:
        combined = pd.concat(all_data, axis=1)
        combined.index = pd.to_datetime(combined.index)
        return combined

    return pd.DataFrame()


def download_fred_sentiment(
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download EUR/USD exchange rate from FRED.

    Args:
        start: Start date
        end: End date

    Returns:
        DataFrame with daily EUR/USD rate
    """
    try:
        from fredapi import Fred
    except ImportError:
        print("FRED API not available. Install with: pip install fredapi")
        return pd.DataFrame()

    api_key = os.environ.get('FRED_API_KEY')
    if api_key is None:
        creds = _load_env_credentials()
        api_key = creds.get('FRED_API_KEY')

    if api_key is None:
        print("Warning: FRED_API_KEY not set. Skipping EUR/USD.")
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    all_series = []

    for col_name, config in FRED_SENTIMENT.items():
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

    if all_series:
        combined = pd.concat(all_series, axis=1)
        combined.index = pd.to_datetime(combined.index)
        return combined

    return pd.DataFrame()


def download_epu_data(
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download European Economic Policy Uncertainty (EPU) index.

    Source: https://www.policyuncertainty.com/europe_monthly.html

    The EPU index measures policy-related economic uncertainty based on
    newspaper coverage frequency. Monthly data, forward-filled to daily.

    Args:
        start: Start date
        end: End date

    Returns:
        DataFrame with daily EPU index (forward-filled from monthly)
    """
    print(f"  Downloading European EPU index from policyuncertainty.com...")

    try:
        df = pd.read_excel(EPU_URL)

        # The Excel file has Year, Month, and policy uncertainty columns
        # Find year and month columns
        year_col = [c for c in df.columns if 'year' in str(c).lower()]
        month_col = [c for c in df.columns if 'month' in str(c).lower()]

        if not year_col or not month_col:
            # Try numeric column pattern: first two columns are year/month
            print(f"    Columns found: {list(df.columns)}")
            # Assume first col = year, second col = month
            year_col = df.columns[0]
            month_col = df.columns[1]
        else:
            year_col = year_col[0]
            month_col = month_col[0]

        # Find the EPU value column (usually the last or one named with 'Europe')
        value_cols = [c for c in df.columns if c not in [year_col, month_col]]
        epu_cols = [c for c in value_cols if 'europ' in str(c).lower() or 'epu' in str(c).lower()]
        if epu_cols:
            epu_col = epu_cols[0]
        elif value_cols:
            epu_col = value_cols[0]
        else:
            print("    Could not identify EPU value column")
            return pd.DataFrame()

        # Create datetime index
        df['date'] = pd.to_datetime(
            df[year_col].astype(int).astype(str) + '-' +
            df[month_col].astype(int).astype(str).str.zfill(2) + '-01'
        )
        df = df.set_index('date')

        # Extract EPU series
        epu = df[[epu_col]].rename(columns={epu_col: 'sentiment_epu_eu'})
        epu = epu.sort_index()

        # Convert to numeric, coerce errors
        epu['sentiment_epu_eu'] = pd.to_numeric(epu['sentiment_epu_eu'], errors='coerce')
        epu = epu.dropna()

        # Filter date range
        epu = epu[(epu.index >= start) & (epu.index <= end)]

        # Resample to daily via forward-fill
        epu_daily = epu.resample('D').ffill()

        # Trim to requested range
        epu_daily = epu_daily[(epu_daily.index >= start) & (epu_daily.index <= end)]

        print(f"    Got {len(epu)} monthly observations -> {len(epu_daily)} daily (ffill)")
        return epu_daily

    except Exception as e:
        print(f"    Error downloading EPU data: {e}")
        return pd.DataFrame()


def create_sentiment_dataset(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Create a combined sentiment indicator dataset, with caching.

    Args:
        start: Start date
        end: End date
        data_dir: Directory to store cached data
        use_cache: Whether to use cached data

    Returns:
        DataFrame with daily sentiment indicators
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / f"sentiment_{start}_{end}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading cached sentiment data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    all_data = []

    # 1. Yahoo Finance (VIX, Gold)
    print("\n--- Yahoo Finance Sentiment Data ---")
    yf_data = download_yfinance_sentiment(start, end)
    if not yf_data.empty:
        all_data.append(yf_data)

    # 2. FRED (EUR/USD)
    print("\n--- FRED Sentiment Data ---")
    fred_data = download_fred_sentiment(start, end)
    if not fred_data.empty:
        all_data.append(fred_data)

    # 3. EPU (European Policy Uncertainty)
    print("\n--- EPU Data ---")
    epu_data = download_epu_data(start, end)
    if not epu_data.empty:
        all_data.append(epu_data)

    if not all_data:
        print("\nNo sentiment data downloaded.")
        return pd.DataFrame()

    # Combine all sources
    combined = pd.concat(all_data, axis=1)
    combined = combined.sort_index()

    # Forward-fill gaps (weekends, holidays for market data)
    combined = combined.ffill()

    # Trim to date range
    combined = combined[(combined.index >= start) & (combined.index <= end)]

    print(f"\nCombined shape: {combined.shape}")
    print(f"Date range: {combined.index.min()} to {combined.index.max()}")

    # Save
    combined.to_csv(cache_file)
    print(f"Saved sentiment data to {cache_file}")

    return combined


def main():
    """Download market sentiment indicator data."""
    parser = argparse.ArgumentParser(description='Download market sentiment data')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--no-cache', action='store_true', help='Force re-download')
    args = parser.parse_args()

    print("=" * 60)
    print("Market Sentiment Data Download")
    print("=" * 60)
    print(f"Date range: {args.start} to {args.end}")
    print()

    print("Indicators to download:")
    for name, config in YFINANCE_SENTIMENT.items():
        print(f"  {name}: {config['description']} (Yahoo Finance)")
    for name, config in FRED_SENTIMENT.items():
        print(f"  {name}: {config['description']} (FRED)")
    print(f"  sentiment_epu_eu: European Economic Policy Uncertainty (policyuncertainty.com)")
    print()

    data = create_sentiment_dataset(args.start, args.end, use_cache=not args.no_cache)

    if data.empty:
        print("\nNo sentiment data downloaded.")
    else:
        print(f"\nFinal shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"\nSample statistics:")
        print(data.describe())

    print("\n" + "=" * 60)
    print("Sentiment data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
