"""
Download transportation/freight index data for electricity price forecasting.

Key indicators:
- Baltic Dry Index (BDI) - bulk commodity shipping costs (daily)
  Primary source: WRDS (Bloomberg-sourced)
  Fallback: Nasdaq Data Link (Quandl)

The BDI correlates with coal transport costs to European power plants
and is a leading indicator of global economic activity.

Usage:
    python3 download_transport_data.py
    python3 download_transport_data.py --start 2015-01-01 --end 2024-12-31
    python3 download_transport_data.py --no-cache
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CARS_ROOT, TRANSPORT_DIR
# Data directory
DATA_DIR = TRANSPORT_DIR


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


def download_bdi_wrds(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
) -> pd.DataFrame:
    """
    Download Baltic Dry Index from WRDS (Bloomberg-sourced).

    Requires WRDS account with Bloomberg data access.

    Args:
        start: Start date
        end: End date

    Returns:
        DataFrame with daily BDI values
    """
    creds = _load_env_credentials()
    username = os.environ.get('WRDS_USERNAME') or creds.get('WRDS_USERNAME')
    password = os.environ.get('WRDS_PASSWORD') or creds.get('WRDS_PASSWORD')

    if not username:
        print("  WRDS_USERNAME not found. Set in .env or environment.")
        return pd.DataFrame()

    print(f"  Connecting to WRDS as {username}...")

    try:
        # Set credentials in environment so WRDS doesn't prompt interactively
        os.environ['WRDS_USERNAME'] = username
        if password:
            os.environ['WRDS_PASSWORD'] = password

        # Create .pgpass file for non-interactive WRDS auth
        pgpass_path = Path.home() / '.pgpass'
        pgpass_entry = f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{username}:{password}"
        needs_pgpass = True
        if pgpass_path.exists():
            existing = pgpass_path.read_text()
            if f"wrds:{username}:" in existing:
                needs_pgpass = False
        if needs_pgpass and password:
            with open(pgpass_path, 'a') as f:
                f.write(pgpass_entry + '\n')
            pgpass_path.chmod(0o600)

        # Connect via sqlalchemy to avoid interactive prompts from wrds library
        from sqlalchemy import create_engine, text
        from urllib.parse import quote_plus
        engine = create_engine(
            f"postgresql://{username}:{quote_plus(password)}@wrds-pgdata.wharton.upenn.edu:9737/wrds",
            connect_args={'sslmode': 'require'},
        )
        # Verify connection before proceeding
        with engine.connect() as test_conn:
            test_conn.execute(text("SELECT 1"))

        # Try multiple possible table locations for BDI
        queries = [
            # Bloomberg daily data
            f"""
            SELECT date, prc as transport_bdi
            FROM bdiy.bdiy
            WHERE date BETWEEN '{start}' AND '{end}'
            ORDER BY date
            """,
            # Alternative: CRSP-like structure
            f"""
            SELECT caldt as date, sprtrn as transport_bdi
            FROM crsp.dsi
            WHERE caldt BETWEEN '{start}' AND '{end}'
            ORDER BY caldt
            """,
        ]

        for i, query in enumerate(queries):
            try:
                df = pd.read_sql(query, engine)
                if df is not None and len(df) > 0:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    df = df.sort_index()
                    print(f"    Got {len(df)} daily BDI observations from WRDS")
                    engine.dispose()
                    return df
            except Exception as e:
                if i == 0:
                    print(f"    Query {i+1} failed: {e}")
                continue

        # Try listing available schemas to help debug
        print("  Could not find BDI data in WRDS. Listing available schemas...")
        try:
            schemas_df = pd.read_sql(
                "SELECT schema_name FROM information_schema.schemata ORDER BY schema_name",
                engine,
            )
            libraries = schemas_df['schema_name'].tolist()
            relevant = [lib for lib in libraries if any(
                k in lib.lower() for k in ['bloom', 'baltic', 'freight', 'ship', 'transport']
            )]
            if relevant:
                print(f"    Potentially relevant schemas: {relevant}")
            else:
                print(f"    No shipping-related schemas found. Available: {libraries[:20]}...")
        except Exception:
            pass

        engine.dispose()

    except Exception as e:
        print(f"  WRDS connection error: {e}")

    return pd.DataFrame()


def download_bdi_quandl(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
) -> pd.DataFrame:
    """
    Download Baltic Dry Index from Nasdaq Data Link (Quandl).

    Free tier may have limited history.

    Args:
        start: Start date
        end: End date

    Returns:
        DataFrame with daily BDI values
    """
    print("  Trying Nasdaq Data Link (Quandl) for BDI...")

    try:
        import nasdaqdatalink
    except ImportError:
        try:
            import quandl as nasdaqdatalink
        except ImportError:
            print("  Neither nasdaqdatalink nor quandl installed.")
            print("  Install with: pip install nasdaq-data-link")
            return pd.DataFrame()

    # Set API key from environment / .env
    creds = _load_env_credentials()
    api_key = (os.environ.get('QUANDL_API_KEY') or os.environ.get('QUANDL_API')
               or creds.get('QUANDL_API_KEY') or creds.get('QUANDL_API'))
    if api_key:
        nasdaqdatalink.ApiConfig.api_key = api_key
    else:
        print("  Warning: QUANDL_API not set — requests may be rate-limited")

    # Try various dataset codes
    dataset_codes = [
        'LLOYDS/BDI',       # Lloyd's List BDI
        'CHRIS/CME_BDI1',   # CME BDI futures
    ]

    for code in dataset_codes:
        try:
            print(f"    Trying {code}...")
            data = nasdaqdatalink.get(
                code,
                start_date=start,
                end_date=end,
            )
            if data is not None and len(data) > 0:
                # Take the first numeric column as BDI
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    result = pd.DataFrame({
                        'transport_bdi': data[numeric_cols[0]]
                    })
                    result.index = pd.to_datetime(result.index)
                    result = result.sort_index()
                    print(f"    Got {len(result)} daily observations from {code}")
                    return result
        except Exception as e:
            print(f"    {code} failed: {e}")

    return pd.DataFrame()


def download_bdi_yfinance(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
) -> pd.DataFrame:
    """
    Attempt to download BDI-related ETF as a proxy.

    BDRY (Breakwave Dry Bulk Shipping ETF) tracks dry bulk shipping futures.

    Args:
        start: Start date
        end: End date

    Returns:
        DataFrame with daily BDI proxy values
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    print("  Trying BDRY ETF (BDI proxy) from Yahoo Finance...")

    try:
        data = yf.download('BDRY', start=start, end=end, progress=False)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            close_col = [c for c in data.columns if 'Close' in c]
            if close_col:
                result = pd.DataFrame({
                    'transport_bdi': data[close_col[0]]
                })
                # Note: BDRY started trading ~2018, so may not cover full range
                print(f"    Got {len(result)} daily observations (BDRY ETF proxy)")
                return result
    except Exception as e:
        print(f"    BDRY download failed: {e}")

    return pd.DataFrame()


def create_transport_dataset(
    start: str = '2015-01-01',
    end: str = '2024-12-31',
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Create a transport index dataset, with caching.

    Tries sources in order: WRDS -> Quandl -> Yahoo Finance (BDRY proxy).

    Args:
        start: Start date
        end: End date
        data_dir: Directory to store cached data
        use_cache: Whether to use cached data

    Returns:
        DataFrame with daily transport indices
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / f"transport_{start}_{end}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading cached transport data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Try sources in priority order
    print("\n--- WRDS (Bloomberg) ---")
    data = download_bdi_wrds(start, end)

    if data.empty:
        print("\n--- Nasdaq Data Link (Quandl) ---")
        data = download_bdi_quandl(start, end)

    if data.empty:
        print("\n--- Yahoo Finance (BDRY ETF proxy) ---")
        data = download_bdi_yfinance(start, end)

    if data.empty:
        print("\nNo transport data downloaded from any source.")
        return pd.DataFrame()

    # Forward-fill gaps (weekends, holidays)
    data = data.sort_index()
    data = data.resample('D').ffill()

    # Trim to date range
    data = data[(data.index >= start) & (data.index <= end)]

    print(f"\nFinal shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    # Save
    data.to_csv(cache_file)
    print(f"Saved transport data to {cache_file}")

    return data


def main():
    """Download transport index data."""
    parser = argparse.ArgumentParser(description='Download transport/freight index data')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--no-cache', action='store_true', help='Force re-download')
    args = parser.parse_args()

    print("=" * 60)
    print("Transport / Freight Index Data Download")
    print("=" * 60)
    print(f"Date range: {args.start} to {args.end}")
    print()

    data = create_transport_dataset(args.start, args.end, use_cache=not args.no_cache)

    if data.empty:
        print("\nNo transport data downloaded.")
        print("\nTo enable data sources:")
        print("  1. WRDS: pip install wrds, set WRDS_USERNAME/WRDS_PASSWORD in .env")
        print("  2. Quandl: pip install nasdaq-data-link")
        print("  3. Yahoo: pip install yfinance (BDRY ETF, limited history)")
    else:
        print(f"\nFinal shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"\nSample statistics:")
        print(data.describe())

    print("\n" + "=" * 60)
    print("Transport data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
