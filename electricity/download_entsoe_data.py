"""
Download electricity market data from ENTSO-E Transparency Platform.

ENTSO-E provides:
- Day-ahead prices (hourly)
- Actual generation by source (nuclear, wind, solar, etc.)
- Cross-border physical flows
- Load/consumption data

Data source: https://transparency.entsoe.eu/
API access: Free after registration at transparency@entsoe.eu

IMPORTANT: You need an API key to use this script.
1. Register at https://transparency.entsoe.eu/
2. Email transparency@entsoe.eu with "Restful API access" in subject
3. Save your API key in environment variable ENTSOE_API_KEY or in a .env file

Python client: pip install entsoe-py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import os
from datetime import datetime, timedelta
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CARS_ROOT, ENTSOE_DIR

from country_config import (
    get_entsoe_zone, get_entsoe_zones, is_multi_zone,
    get_neighbors, get_flow_zone_pairs, get_registered_countries,
    COUNTRY_REGISTRY,
)

# Data directory
DATA_DIR = ENTSOE_DIR


def get_api_key() -> str:
    """
    Get ENTSO-E API key from environment or .env file.

    Returns:
        API key string

    Raises:
        ValueError if no API key found
    """
    # Try environment variable
    api_key = os.environ.get('ENTSOE_API_KEY')

    if api_key:
        return api_key

    # Try .env file
    env_file = Path.home() / '.entsoe_api_key'
    if env_file.exists():
        return env_file.read_text().strip()

    # Try local .env
    local_env = CARS_ROOT / ".env"
    if local_env.exists():
        for line in local_env.read_text().split('\n'):
            if line.startswith('ENTSOE_API_KEY='):
                return line.split('=', 1)[1].strip().strip('"\'')

    raise ValueError(
        "ENTSO-E API key not found. Set ENTSOE_API_KEY environment variable "
        "or create ~/.entsoe_api_key file with your API key.\n"
        "To get an API key:\n"
        "1. Register at https://transparency.entsoe.eu/\n"
        "2. Email transparency@entsoe.eu with 'Restful API access' in subject\n"
        "3. Save your API key"
    )


def download_day_ahead_prices(
    country: str,
    start: str,
    end: str,
    api_key: Optional[str] = None,
    data_dir: Path = DATA_DIR
) -> pd.Series:
    """
    Download day-ahead electricity prices from ENTSO-E.

    Args:
        country: Country code (e.g., 'DE', 'FR', 'NL')
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        api_key: ENTSO-E API key (optional, will try to get from env)
        data_dir: Directory to cache data

    Returns:
        Series with hourly day-ahead prices in EUR/MWh
    """
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        raise ImportError("Please install entsoe-py: pip install entsoe-py")

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / f"day_ahead_prices_{country}_{start}_{end}.csv"

    if cache_file.exists():
        print(f"Loading cached day-ahead prices from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df['Price']

    if api_key is None:
        api_key = get_api_key()

    client = EntsoePandasClient(api_key=api_key)

    country_code = get_entsoe_zone(country, start)

    start_dt = pd.Timestamp(start, tz='Europe/Brussels')
    end_dt = pd.Timestamp(end, tz='Europe/Brussels')

    print(f"Downloading day-ahead prices for {country} ({country_code})...")
    print(f"  Period: {start} to {end}")

    if is_multi_zone(country):
        # Download from all zones and average
        prices = _download_multi_zone_prices(client, country, start_dt, end_dt)
    else:
        try:
            prices = client.query_day_ahead_prices(
                country_code,
                start=start_dt,
                end=end_dt
            )
        except Exception as e:
            # Try historical zone code if available
            hist_code = get_entsoe_zone(country, start)
            if hist_code != country_code:
                print(f"  Trying historical zone code {hist_code}...")
                prices = client.query_day_ahead_prices(
                    hist_code,
                    start=start_dt,
                    end=end_dt
                )
            else:
                raise e

    # Convert to DataFrame for caching
    df = pd.DataFrame({'Price': prices})
    df.index = df.index.tz_convert('UTC')
    df.to_csv(cache_file)
    print(f"Saved to {cache_file}")

    return prices


def download_generation_by_source(
    country: str,
    start: str,
    end: str,
    api_key: Optional[str] = None,
    data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """
    Download actual generation per production type from ENTSO-E.

    Returns generation breakdown by source:
    - Nuclear
    - Wind (onshore/offshore)
    - Solar
    - Hydro (run-of-river, reservoir, pumped storage)
    - Gas
    - Coal/Lignite
    - Other (biomass, oil, etc.)

    Args:
        country: Country code (e.g., 'DE', 'FR')
        start: Start date
        end: End date
        api_key: ENTSO-E API key
        data_dir: Cache directory

    Returns:
        DataFrame with hourly generation by source (MW)
    """
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        raise ImportError("Please install entsoe-py: pip install entsoe-py")

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / f"generation_{country}_{start}_{end}.csv"

    if cache_file.exists():
        print(f"Loading cached generation data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    if api_key is None:
        api_key = get_api_key()

    client = EntsoePandasClient(api_key=api_key)
    country_code = get_entsoe_zone(country, start)

    start_dt = pd.Timestamp(start, tz='Europe/Brussels')
    end_dt = pd.Timestamp(end, tz='Europe/Brussels')

    print(f"Downloading generation data for {country}...")
    print(f"  Period: {start} to {end}")

    if is_multi_zone(country):
        generation = _download_multi_zone_data(
            client, country, start_dt, end_dt, 'generation'
        )
    else:
        try:
            generation = client.query_generation(
                country_code,
                start=start_dt,
                end=end_dt,
                psr_type=None
            )
        except Exception as e:
            hist_code = get_entsoe_zone(country, start)
            if hist_code != country_code:
                generation = client.query_generation(
                    hist_code, start=start_dt, end=end_dt, psr_type=None
                )
            else:
                raise e

    # Handle multi-index columns
    if isinstance(generation.columns, pd.MultiIndex):
        # Flatten column names
        generation.columns = ['_'.join(col).strip() for col in generation.columns.values]

    generation.index = generation.index.tz_convert('UTC')
    generation.to_csv(cache_file)
    print(f"Saved to {cache_file}")

    return generation


def download_load(
    country: str,
    start: str,
    end: str,
    api_key: Optional[str] = None,
    data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """
    Download actual total load (consumption) from ENTSO-E.

    Args:
        country: Country code (e.g., 'DE', 'FR')
        start: Start date
        end: End date
        api_key: ENTSO-E API key
        data_dir: Cache directory

    Returns:
        DataFrame with hourly load data (MW)
    """
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        raise ImportError("Please install entsoe-py: pip install entsoe-py")

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / f"load_{country}_{start}_{end}.csv"

    if cache_file.exists():
        print(f"Loading cached load data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    if api_key is None:
        api_key = get_api_key()

    client = EntsoePandasClient(api_key=api_key)
    country_code = get_entsoe_zone(country, start)

    start_dt = pd.Timestamp(start, tz='Europe/Brussels')
    end_dt = pd.Timestamp(end, tz='Europe/Brussels')

    print(f"Downloading load data for {country}...")

    if is_multi_zone(country):
        load = _download_multi_zone_data(
            client, country, start_dt, end_dt, 'load'
        )
    else:
        try:
            load = client.query_load(
                country_code,
                start=start_dt,
                end=end_dt
            )
        except Exception as e:
            hist_code = get_entsoe_zone(country, start)
            if hist_code != country_code:
                load = client.query_load(
                    hist_code, start=start_dt, end=end_dt
                )
            else:
                raise e

    if isinstance(load, pd.Series):
        load = load.to_frame(name='Actual Load')

    load.index = load.index.tz_convert('UTC')
    load.to_csv(cache_file)
    print(f"Saved to {cache_file}")

    return load


def download_cross_border_flows(
    country_from: str,
    country_to: str,
    start: str,
    end: str,
    api_key: Optional[str] = None,
    data_dir: Path = DATA_DIR
) -> pd.Series:
    """
    Download cross-border physical flows between two countries.

    Args:
        country_from: Source country
        country_to: Destination country
        start: Start date
        end: End date
        api_key: ENTSO-E API key
        data_dir: Cache directory

    Returns:
        Series with hourly cross-border flows (MW)
    """
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        raise ImportError("Please install entsoe-py: pip install entsoe-py")

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / f"flow_{country_from}_to_{country_to}_{start}_{end}.csv"

    if cache_file.exists():
        print(f"Loading cached flow data from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df.iloc[:, 0]

    if api_key is None:
        api_key = get_api_key()

    client = EntsoePandasClient(api_key=api_key)

    start_dt = pd.Timestamp(start, tz='Europe/Brussels')
    end_dt = pd.Timestamp(end, tz='Europe/Brussels')

    # Get the specific zone pairs for this country pair
    zone_pairs = get_flow_zone_pairs(country_from, country_to)

    print(f"Downloading cross-border flows {country_from} -> {country_to}...")

    all_flows = []
    for zone_from, zone_to in zone_pairs:
        print(f"  Zone pair: {zone_from} -> {zone_to}")
        flow = client.query_crossborder_flows(
            zone_from,
            zone_to,
            start=start_dt,
            end=end_dt
        )
        all_flows.append(flow)

    # Sum flows across all zone pairs
    if len(all_flows) == 1:
        flows = all_flows[0]
    else:
        flows = pd.concat(all_flows, axis=1).sum(axis=1)

    df = pd.DataFrame({f'{country_from}_to_{country_to}': flows})
    df.index = df.index.tz_convert('UTC')
    df.to_csv(cache_file)
    print(f"Saved to {cache_file}")

    return flows


def create_entsoe_dataset(
    country: str,
    start: str,
    end: str,
    api_key: Optional[str] = None,
    include_generation: bool = True,
    include_load: bool = True,
    include_flows: bool = False,
    data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """
    Create a comprehensive dataset from ENTSO-E data.

    Args:
        country: Country code (e.g., 'DE', 'FR')
        start: Start date
        end: End date
        api_key: ENTSO-E API key
        include_generation: Include generation by source
        include_load: Include load data
        include_flows: Include cross-border flows with all neighbors
        data_dir: Cache directory

    Returns:
        DataFrame with all requested data
    """
    if api_key is None:
        try:
            api_key = get_api_key()
        except ValueError as e:
            print(f"Warning: {e}")
            print("Returning empty DataFrame - configure API key to download data")
            return pd.DataFrame()

    # Start with day-ahead prices
    prices = download_day_ahead_prices(
        country, start, end, api_key, data_dir
    )
    data = pd.DataFrame({'Day_Ahead_Price': prices})

    if include_load:
        try:
            load = download_load(country, start, end, api_key, data_dir)
            data = data.join(load, how='outer')
        except Exception as e:
            print(f"Warning: Could not download load data: {e}")

    if include_generation:
        try:
            gen = download_generation_by_source(country, start, end, api_key, data_dir)
            data = data.join(gen, how='outer')
        except Exception as e:
            print(f"Warning: Could not download generation data: {e}")

    if include_flows:
        neighbors = get_neighbors(country)
        for neighbor in neighbors:
            try:
                flow_out = download_cross_border_flows(
                    country, neighbor, start, end, api_key, data_dir
                )
                flow_in = download_cross_border_flows(
                    neighbor, country, start, end, api_key, data_dir
                )
                data[f'Flow_to_{neighbor}'] = flow_out
                data[f'Flow_from_{neighbor}'] = flow_in
                data[f'Net_Flow_{neighbor}'] = flow_in - flow_out
            except Exception as e:
                print(f"Warning: Could not download flows {country}<->{neighbor}: {e}")

    # Sort by index and fill gaps
    data = data.sort_index()
    data = data.ffill().bfill()

    return data


def compute_target_variable(
    prices: pd.Series,
    method: str = 'daily_return'
) -> pd.Series:
    """
    Compute target variable from day-ahead prices.

    For electricity futures price variation prediction:
    - Daily price change captures day-to-day movements
    - Log returns are more stable for percentage changes

    Args:
        prices: Hourly day-ahead prices
        method: 'daily_return' (default), 'hourly_return', 'daily_diff'

    Returns:
        Series with target variable
    """
    if method == 'daily_return':
        # Daily average price, then log returns
        daily_price = prices.resample('D').mean()
        target = np.log(daily_price).diff()
    elif method == 'hourly_return':
        # Hourly log returns
        target = np.log(prices).diff()
    elif method == 'daily_diff':
        # Simple daily price difference
        daily_price = prices.resample('D').mean()
        target = daily_price.diff()
    else:
        raise ValueError(f"Unknown method: {method}")

    return target.dropna()


def _download_multi_zone_prices(client, country, start_dt, end_dt):
    """Download prices from all sub-zones of a multi-zone country and average."""
    zones = get_entsoe_zones(country)
    all_prices = []
    for zone in zones:
        print(f"  Downloading prices for zone {zone}...")
        p = client.query_day_ahead_prices(zone, start=start_dt, end=end_dt)
        all_prices.append(p)
    # Simple average across zones
    return pd.concat(all_prices, axis=1).mean(axis=1)


def _download_multi_zone_data(client, country, start_dt, end_dt, data_type):
    """Download generation or load from all sub-zones and sum."""
    zones = get_entsoe_zones(country)
    all_data = []
    for zone in zones:
        print(f"  Downloading {data_type} for zone {zone}...")
        if data_type == 'generation':
            d = client.query_generation(zone, start=start_dt, end=end_dt, psr_type=None)
        elif data_type == 'load':
            d = client.query_load(zone, start=start_dt, end=end_dt)
            if isinstance(d, pd.Series):
                d = d.to_frame(name='Actual Load')
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = ['_'.join(col).strip() for col in d.columns.values]
        all_data.append(d)
    # Sum across zones (total generation / total load)
    combined = pd.concat(all_data, axis=1)
    # Group duplicate column names and sum
    return combined.groupby(combined.columns, axis=1).sum()


def main():
    """Download ENTSO-E data for all registered countries."""
    import argparse
    parser = argparse.ArgumentParser(description='Download ENTSO-E data')
    parser.add_argument('--countries', type=str, default=None,
                        help='Comma-separated country codes (default: all registered)')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--include-flows', action='store_true',
                        help='Include cross-border flows')
    args = parser.parse_args()

    print("=" * 60)
    print("ENTSO-E Transparency Platform Data Download")
    print("=" * 60)

    try:
        api_key = get_api_key()
        print(f"API key found: {api_key[:8]}...")
    except ValueError as e:
        print(f"\n{e}")
        return

    countries = args.countries.split(',') if args.countries else get_registered_countries()
    print(f"Countries: {countries}")

    for country in countries:
        print(f"\n{'='*40}")
        print(f"Country: {country}")
        print(f"{'='*40}")

        try:
            data = create_entsoe_dataset(
                country=country,
                start=args.start,
                end=args.end,
                api_key=api_key,
                include_generation=True,
                include_load=True,
                include_flows=args.include_flows,
                data_dir=DATA_DIR
            )

            print(f"\nShape: {data.shape}")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            print(f"Columns: {list(data.columns)[:10]}...")

            if 'Day_Ahead_Price' in data.columns:
                print(f"\nDay-ahead price statistics:")
                print(data['Day_Ahead_Price'].describe())

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("ENTSO-E data download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
