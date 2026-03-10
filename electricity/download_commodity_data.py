"""
Download commodity price data relevant for electricity price forecasting.

Key commodities affecting electricity prices:
- Natural Gas (TTF for Europe) - major input for gas power plants
- Coal (API2) - input for coal power plants
- Brent Crude Oil - general energy benchmark
- Carbon/CO2 emissions (EUA) - affects thermal generation costs

PUBLIC DATA SOURCES:

1. Quandl/Nasdaq Data Link (Free tier available)
   - Historical commodity prices
   - pip install nasdaq-data-link

2. Yahoo Finance (Free)
   - Natural gas, oil futures via yfinance
   - pip install yfinance

3. FRED (Federal Reserve Economic Data - Free)
   - Some energy commodity prices
   - pip install fredapi

4. World Bank Commodity Prices (Free)
   - Monthly commodity price data
   - https://www.worldbank.org/en/research/commodity-markets

NOTE: For production use, consider:
- S&P Global Platts (commercial) - the gold standard for energy commodities
- ICE (commercial) - European energy futures
- EEX (commercial) - European carbon prices

This script uses free sources where available, with notes on commercial alternatives.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import COMMODITY_DIR
# Data directory
DATA_DIR = COMMODITY_DIR

# Commodity symbols/tickers
COMMODITY_TICKERS = {
    # Yahoo Finance tickers
    'yfinance': {
        'natural_gas': 'NG=F',      # Natural Gas Futures (NYMEX Henry Hub)
        'brent_oil': 'BZ=F',        # Brent Crude Oil Futures
        'wti_oil': 'CL=F',          # WTI Crude Oil Futures
        'coal': None,               # No direct ticker, use World Bank
        'carbon': None,             # EUA not available on Yahoo
    },
    # Nasdaq Data Link / Quandl codes
    'nasdaq': {
        'natural_gas_eu': 'CHRIS/ICE_TFM1',  # ICE TTF Natural Gas (needs subscription)
        'brent_oil': 'CHRIS/ICE_B1',          # ICE Brent Crude
        'coal_api2': 'CHRIS/ICE_ATW1',        # ICE API2 Coal
    }
}


def download_yfinance_commodity(
    ticker: str,
    start: str,
    end: str,
    name: str = None
) -> pd.DataFrame:
    """
    Download commodity data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        name: Friendly name for the commodity

    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Please install yfinance: pip install yfinance")

    print(f"Downloading {name or ticker} from Yahoo Finance...")

    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Rename columns with commodity name
    if name:
        data.columns = [f"{name}_{col}" for col in data.columns]

    return data


def download_worldbank_commodities(
    start_year: int = 2010,
    end_year: int = 2023
) -> pd.DataFrame:
    """
    Download commodity prices from World Bank Pink Sheet.

    World Bank provides monthly commodity price data including:
    - Energy (crude oil, natural gas, coal)
    - Metals
    - Agriculture

    Source: https://www.worldbank.org/en/research/commodity-markets

    Args:
        start_year: Start year
        end_year: End year

    Returns:
        DataFrame with monthly commodity prices
    """
    # World Bank commodity price URL (updated monthly)
    url = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"

    print("Downloading World Bank commodity prices...")

    try:
        # Read the Excel file - energy prices are typically in first sheet
        df = pd.read_excel(url, sheet_name=0, skiprows=4)

        # First column is typically the date
        df = df.rename(columns={df.columns[0]: 'Date'})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.set_index('Date')

        # Filter date range
        df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]

        # Select relevant energy commodities
        energy_cols = [
            col for col in df.columns
            if any(term in col.lower() for term in
                   ['crude', 'oil', 'gas', 'coal', 'lng', 'brent', 'wti'])
        ]

        if energy_cols:
            df = df[energy_cols]

        print(f"Downloaded {len(df)} months of World Bank commodity data")
        return df

    except Exception as e:
        print(f"Warning: Could not download World Bank data: {e}")
        return pd.DataFrame()


def download_fred_data(
    series_ids: List[str],
    start: str,
    end: str,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Download data from FRED (Federal Reserve Economic Data).

    Relevant series:
    - DCOILBRENTEU: Crude Oil Prices - Brent Europe
    - DCOILWTICO: Crude Oil Prices - WTI
    - DHHNGSP: Henry Hub Natural Gas Spot Price

    Args:
        series_ids: List of FRED series IDs
        start: Start date
        end: End date
        api_key: FRED API key (optional but recommended)

    Returns:
        DataFrame with requested series
    """
    try:
        from fredapi import Fred
    except ImportError:
        print("FRED API not available. Install with: pip install fredapi")
        return pd.DataFrame()

    if api_key is None:
        import os
        api_key = os.environ.get('FRED_API_KEY')

    if api_key is None:
        print("Warning: FRED_API_KEY not set. Get free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    dfs = []

    for series_id in series_ids:
        try:
            print(f"Downloading {series_id} from FRED...")
            data = fred.get_series(series_id, observation_start=start, observation_end=end)
            dfs.append(pd.DataFrame({series_id: data}))
        except Exception as e:
            print(f"Warning: Could not download {series_id}: {e}")

    if dfs:
        return pd.concat(dfs, axis=1)
    return pd.DataFrame()


def create_commodity_dataset(
    start: str,
    end: str,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Create a comprehensive commodity price dataset.

    Combines data from multiple free sources:
    1. Yahoo Finance: Natural gas, crude oil futures
    2. World Bank: Monthly commodity benchmarks
    3. FRED: US energy prices

    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        data_dir: Directory to cache data
        use_cache: Whether to use cached data

    Returns:
        DataFrame with commodity prices
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / f"commodities_{start}_{end}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading cached commodity data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    all_data = []

    # 1. Yahoo Finance data (daily)
    print("\n--- Yahoo Finance Data ---")
    for name, ticker in COMMODITY_TICKERS['yfinance'].items():
        if ticker is None:
            continue
        try:
            df = download_yfinance_commodity(ticker, start, end, name)
            # Keep only Close price
            close_col = [c for c in df.columns if 'Close' in c][0]
            all_data.append(pd.DataFrame({name: df[close_col]}))
        except Exception as e:
            print(f"Warning: Could not download {name}: {e}")

    # 2. World Bank data (monthly - will need interpolation)
    print("\n--- World Bank Data ---")
    try:
        start_year = int(start[:4])
        end_year = int(end[:4])
        wb_data = download_worldbank_commodities(start_year, end_year)
        if not wb_data.empty:
            # Resample to daily and forward-fill
            wb_daily = wb_data.resample('D').ffill()
            wb_daily.columns = [f"WB_{col}" for col in wb_daily.columns]
            all_data.append(wb_daily)
    except Exception as e:
        print(f"Warning: World Bank data error: {e}")

    # 3. FRED data
    print("\n--- FRED Data ---")
    fred_series = ['DCOILBRENTEU', 'DCOILWTICO', 'DHHNGSP']
    fred_data = download_fred_data(fred_series, start, end)
    if not fred_data.empty:
        all_data.append(fred_data)

    # Combine all data
    if all_data:
        combined = pd.concat(all_data, axis=1)
        combined = combined.sort_index()

        # Forward-fill gaps (commodities often have holidays)
        combined = combined.ffill()

        # Filter to date range
        combined = combined[(combined.index >= start) & (combined.index <= end)]

        # Cache the data
        combined.to_csv(cache_file)
        print(f"\nSaved commodity data to {cache_file}")

        return combined

    return pd.DataFrame()


def align_with_electricity_prices(
    commodity_df: pd.DataFrame,
    electricity_df: pd.DataFrame,
    resample: str = 'D'
) -> pd.DataFrame:
    """
    Align commodity prices with electricity price data.

    Args:
        commodity_df: Commodity prices (typically daily)
        electricity_df: Electricity prices (typically hourly)
        resample: Resampling frequency for electricity data

    Returns:
        DataFrame with aligned data
    """
    # Resample electricity data if needed
    if resample == 'D':
        elec_resampled = electricity_df.resample('D').mean()
    else:
        elec_resampled = electricity_df

    # Align indices
    common_dates = elec_resampled.index.intersection(commodity_df.index)

    if len(common_dates) == 0:
        print("Warning: No overlapping dates")
        print(f"Commodities: {commodity_df.index.min()} to {commodity_df.index.max()}")
        print(f"Electricity: {elec_resampled.index.min()} to {elec_resampled.index.max()}")
        return pd.DataFrame()

    merged = elec_resampled.loc[common_dates].join(
        commodity_df.loc[common_dates],
        how='inner'
    )

    return merged


def get_european_gas_proxy(
    natural_gas_df: pd.DataFrame,
    method: str = 'scaled'
) -> pd.Series:
    """
    Create a proxy for European natural gas prices from US Henry Hub.

    European TTF gas prices are highly correlated with Henry Hub but at
    different price levels. This creates an approximate proxy.

    Args:
        natural_gas_df: DataFrame with Henry Hub natural gas prices
        method: 'scaled' (multiply by factor) or 'spread' (add constant)

    Returns:
        Series with estimated European gas price proxy

    Note:
        This is an approximation. For accurate TTF prices, use:
        - ICE TTF Futures (commercial)
        - S&P Global Platts (commercial)
        - ENTSO-E gas day-ahead prices
    """
    # Historical TTF/Henry Hub ratio has varied significantly:
    # - Pre-2021: ~2-3x
    # - 2021-2022: ~5-10x (energy crisis)
    # - 2023+: ~2-3x
    # Use a moderate factor as rough proxy
    TTF_FACTOR = 3.0  # Approximate TTF/Henry Hub ratio

    if 'natural_gas' in natural_gas_df.columns:
        ng = natural_gas_df['natural_gas']
    elif 'Close' in natural_gas_df.columns:
        ng = natural_gas_df['Close']
    else:
        raise ValueError("Cannot find natural gas price column")

    if method == 'scaled':
        # Convert from $/MMBtu to €/MWh (rough)
        # 1 MMBtu ≈ 0.293 MWh
        # Multiply by factor and convert
        ttf_proxy = ng * TTF_FACTOR / 0.293
        ttf_proxy.name = 'TTF_proxy_EUR_MWh'
    else:
        ttf_proxy = ng + 5.0  # Add spread
        ttf_proxy.name = 'TTF_proxy'

    return ttf_proxy


def main():
    """Test commodity data download."""
    import argparse

    parser = argparse.ArgumentParser(description='Download commodity price data')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--no-cache', action='store_true', help='Force re-download')
    args = parser.parse_args()

    print("=" * 60)
    print("Commodity Price Data Download")
    print("=" * 60)

    start = args.start
    end = args.end

    print(f"\nDate range: {start} to {end}")

    # Download commodity data
    data = create_commodity_dataset(start, end, use_cache=not args.no_cache)

    if data.empty:
        print("\nNo commodity data downloaded.")
        print("\nTo enable more data sources:")
        print("1. Install required packages:")
        print("   pip install yfinance fredapi")
        print("2. Get FRED API key (free):")
        print("   https://fred.stlouisfed.org/docs/api/api_key.html")
        print("3. Set environment variable:")
        print("   export FRED_API_KEY='your-key-here'")
    else:
        print(f"\nShape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"\nColumns: {list(data.columns)}")
        print(f"\nSample statistics:")
        print(data.describe())

        # Check for European gas proxy
        if 'natural_gas' in data.columns:
            print("\n--- European Gas Price Proxy ---")
            ttf_proxy = get_european_gas_proxy(data[['natural_gas']])
            print(f"TTF proxy (EUR/MWh): mean={ttf_proxy.mean():.2f}, std={ttf_proxy.std():.2f}")

    print("\n" + "=" * 60)
    print("Commodity data collection complete!")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("NOTES ON DATA SOURCES")
    print("=" * 60)
    print("""
FREE SOURCES (used in this script):
- Yahoo Finance: Natural gas (Henry Hub), Brent/WTI oil futures
- World Bank: Monthly commodity price indices
- FRED: US energy prices

COMMERCIAL SOURCES (recommended for production):
- S&P Global Platts: TTF gas, API2 coal, carbon prices (gold standard)
- ICE: European energy futures (TTF, EUA, power)
- EEX: European carbon (EUA), power futures
- Refinitiv/LSEG: Comprehensive energy data

The user mentioned S&P Global Energy as a potential source -
this would provide accurate European TTF gas and API2 coal prices
which are the most relevant for DE/FR electricity price modeling.
    """)


if __name__ == "__main__":
    main()
