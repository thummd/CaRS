"""
Download power plant outage data from ENTSO-E.

ENTSO-E provides data on:
- Unavailability of generation units (planned and unplanned)
- Unavailability of production units (aggregated)
- Transmission unavailability

This data is crucial for electricity price forecasting as plant outages
directly affect supply and thus prices.

Usage: python3 download_outage_data.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from country_config import get_entsoe_zone, get_registered_countries

from paths import CARS_ROOT, DATA_DIR
# Data directory
DATA_DIR = DATA_DIR
OUTAGE_DIR = DATA_DIR / "outages"
OUTAGE_DIR.mkdir(parents=True, exist_ok=True)


def get_entsoe_client():
    """Get ENTSO-E client with API key."""
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        print("Please install entsoe-py: pip install entsoe-py")
        return None

    # Get API key
    api_key = os.environ.get('ENTSOE_API_KEY')

    if not api_key:
        env_file = CARS_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().split('\n'):
                if line.startswith('ENTSOE_API_KEY='):
                    api_key = line.split('=', 1)[1].strip().strip('"\'')
                    break

    if not api_key:
        print("Error: ENTSOE_API_KEY not found")
        return None

    return EntsoePandasClient(api_key=api_key)


def download_unavailability_of_generation_units(
    client,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp
) -> pd.DataFrame:
    """
    Download unavailability of generation units.

    Returns DataFrame with outage events.
    """
    try:
        print(f"  Fetching generation unit unavailability for {country_code}...")
        data = client.query_unavailability_of_generation_units(
            country_code,
            start=start,
            end=end,
            docstatus=None  # Get all statuses
        )
        if data is not None and len(data) > 0:
            print(f"    Got {len(data)} outage events")
            return data
        else:
            print(f"    No data returned")
            return pd.DataFrame()
    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()


def download_unavailability_of_production_units(
    client,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp
) -> pd.DataFrame:
    """
    Download aggregated unavailability of production units.

    This is a more aggregated view than individual generation units.
    """
    try:
        print(f"  Fetching production unit unavailability for {country_code}...")
        data = client.query_unavailability_of_production_units(
            country_code,
            start=start,
            end=end,
            docstatus=None
        )
        if data is not None and len(data) > 0:
            print(f"    Got {len(data)} records")
            return data
        else:
            print(f"    No data returned")
            return pd.DataFrame()
    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()


def aggregate_outages_to_daily(
    outage_events: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Aggregate outage events to daily unavailable capacity.

    Takes raw outage events and computes daily aggregates:
    - Total unavailable capacity (MW)
    - Number of active outages
    - Planned vs unplanned breakdown
    """
    if outage_events.empty:
        return pd.DataFrame()

    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_data = pd.DataFrame(index=dates)
    daily_data.index.name = 'date'

    # Initialize columns
    daily_data['total_unavailable_mw'] = 0.0
    daily_data['planned_unavailable_mw'] = 0.0
    daily_data['unplanned_unavailable_mw'] = 0.0
    daily_data['num_outages'] = 0
    daily_data['num_planned'] = 0
    daily_data['num_unplanned'] = 0

    # Process each outage event
    for idx, event in outage_events.iterrows():
        try:
            # Get event start and end
            event_start = pd.to_datetime(event.get('start', event.get('Start')))
            event_end = pd.to_datetime(event.get('end', event.get('End')))

            if pd.isna(event_start) or pd.isna(event_end):
                continue

            # Get capacity
            capacity = event.get('unavailable_capacity',
                       event.get('Unavailable Capacity',
                       event.get('unavailableCapacity', 0)))

            if pd.isna(capacity):
                capacity = 0

            # Determine if planned or unplanned
            event_type = str(event.get('type', event.get('Type',
                           event.get('businessType', '')))).lower()
            is_planned = 'planned' in event_type or 'maintenance' in event_type

            # Add to all days the outage is active
            for date in dates:
                if event_start.date() <= date.date() <= event_end.date():
                    daily_data.loc[date, 'total_unavailable_mw'] += capacity
                    daily_data.loc[date, 'num_outages'] += 1

                    if is_planned:
                        daily_data.loc[date, 'planned_unavailable_mw'] += capacity
                        daily_data.loc[date, 'num_planned'] += 1
                    else:
                        daily_data.loc[date, 'unplanned_unavailable_mw'] += capacity
                        daily_data.loc[date, 'num_unplanned'] += 1

        except Exception as e:
            continue

    return daily_data


def aggregate_outages_to_hourly(
    outage_events: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Aggregate outage events to hourly unavailable capacity.

    Uses vectorized boolean masking for performance (~87K hourly rows).
    """
    if outage_events.empty:
        return pd.DataFrame()

    # Create hourly date range
    hours = pd.date_range(start=start_date, end=end_date, freq='h')
    hourly_data = pd.DataFrame(index=hours)
    hourly_data.index.name = 'datetime'

    # Initialize columns
    hourly_data['total_unavailable_mw'] = 0.0
    hourly_data['planned_unavailable_mw'] = 0.0
    hourly_data['unplanned_unavailable_mw'] = 0.0
    hourly_data['num_outages'] = 0
    hourly_data['num_planned'] = 0
    hourly_data['num_unplanned'] = 0

    # Process each outage event using vectorized masking
    for idx, event in outage_events.iterrows():
        try:
            event_start = pd.to_datetime(event.get('start', event.get('Start')))
            event_end = pd.to_datetime(event.get('end', event.get('End')))

            if pd.isna(event_start) or pd.isna(event_end):
                continue

            # Strip timezone for comparison
            if hasattr(event_start, 'tz') and event_start.tz is not None:
                event_start = event_start.tz_localize(None)
            if hasattr(event_end, 'tz') and event_end.tz is not None:
                event_end = event_end.tz_localize(None)

            capacity = event.get('unavailable_capacity',
                       event.get('Unavailable Capacity',
                       event.get('unavailableCapacity', 0)))
            if pd.isna(capacity):
                capacity = 0

            event_type = str(event.get('type', event.get('Type',
                           event.get('businessType', '')))).lower()
            is_planned = 'planned' in event_type or 'maintenance' in event_type

            # Vectorized boolean mask
            mask = (hourly_data.index >= event_start) & (hourly_data.index <= event_end)
            hourly_data.loc[mask, 'total_unavailable_mw'] += capacity
            hourly_data.loc[mask, 'num_outages'] += 1

            if is_planned:
                hourly_data.loc[mask, 'planned_unavailable_mw'] += capacity
                hourly_data.loc[mask, 'num_planned'] += 1
            else:
                hourly_data.loc[mask, 'unplanned_unavailable_mw'] += capacity
                hourly_data.loc[mask, 'num_unplanned'] += 1

        except Exception:
            continue

    return hourly_data


def download_outages_in_chunks(
    country: str,
    start_year: int,
    end_year: int,
    chunk_months: int = 3,
    frequency: str = 'D'
) -> pd.DataFrame:
    """
    Download outage data in chunks to avoid API timeouts.
    """
    client = get_entsoe_client()
    if client is None:
        return pd.DataFrame()

    country_code = get_entsoe_zone(country)

    print(f"\nDownloading outage data for {country} ({start_year}-{end_year})...")
    print("=" * 60)

    all_events = []

    # Download in chunks
    current_start = pd.Timestamp(f'{start_year}-01-01', tz='Europe/Brussels')
    final_end = pd.Timestamp(f'{end_year + 1}-01-01', tz='Europe/Brussels')

    while current_start < final_end:
        chunk_end = current_start + pd.DateOffset(months=chunk_months)
        if chunk_end > final_end:
            chunk_end = final_end

        print(f"\n--- Chunk: {current_start.date()} to {chunk_end.date()} ---")

        # Try to get outage data
        try:
            events = download_unavailability_of_generation_units(
                client, country_code, current_start, chunk_end
            )
            if not events.empty:
                all_events.append(events)
        except Exception as e:
            print(f"  Error getting generation unit data: {e}")

        # Small delay to avoid rate limiting
        import time
        time.sleep(1)

        current_start = chunk_end

    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
        print(f"\nTotal outage events: {len(combined)}")

        # Save raw events
        raw_file = OUTAGE_DIR / f"outage_events_{country}_{start_year}_{end_year}.csv"
        combined.to_csv(raw_file, index=False)
        print(f"Saved raw events to {raw_file}")

        # Aggregate to daily
        start_ts = pd.Timestamp(f'{start_year}-01-01', tz='Europe/Brussels')
        end_ts = pd.Timestamp(f'{end_year}-12-31', tz='Europe/Brussels')
        daily = aggregate_outages_to_daily(combined, start_ts, end_ts)

        if not daily.empty:
            daily_file = OUTAGE_DIR / f"outage_daily_{country}_{start_year}_{end_year}.csv"
            daily.to_csv(daily_file)
            print(f"Saved daily aggregates to {daily_file}")
            print(f"Daily shape: {daily.shape}")

        # Also aggregate to hourly if requested
        if frequency == 'H':
            print("\nAggregating outages to hourly resolution...")
            hourly = aggregate_outages_to_hourly(combined, start_ts, end_ts)
            if not hourly.empty:
                hourly_file = OUTAGE_DIR / f"outage_hourly_{country}_{start_year}_{end_year}.csv"
                hourly.to_csv(hourly_file)
                print(f"Saved hourly aggregates to {hourly_file}")
                print(f"Hourly shape: {hourly.shape}")
                if frequency == 'H':
                    return hourly

        return daily

    return pd.DataFrame()


def create_simple_outage_proxy(
    generation_df: pd.DataFrame,
    country: str
) -> pd.DataFrame:
    """
    Create a simple proxy for outages based on generation capacity factors.

    When actual outage data is not available, we can estimate potential
    outages from unusually low capacity factors.
    """
    print(f"\nCreating outage proxy from generation data for {country}...")

    df = generation_df.copy()

    # Get columns related to generation by type
    gen_cols = [c for c in df.columns if any(
        x in c.lower() for x in ['generation', 'nuclear', 'coal', 'gas', 'wind', 'solar']
    ) and 'actual' in c.lower()]

    if not gen_cols:
        gen_cols = [c for c in df.columns if any(
            x in c.lower() for x in ['nuclear', 'coal', 'gas', 'wind', 'solar', 'hydro']
        )]

    if not gen_cols:
        print("  No generation columns found")
        return pd.DataFrame()

    print(f"  Using generation columns: {gen_cols[:5]}...")

    # Calculate rolling statistics for anomaly detection
    result = pd.DataFrame(index=df.index)

    for col in gen_cols:
        if col in df.columns:
            series = df[col].fillna(method='ffill')

            # Calculate rolling mean and std (30-day window)
            rolling_mean = series.rolling(window=30, min_periods=7).mean()
            rolling_std = series.rolling(window=30, min_periods=7).std()

            # Z-score (how many std below mean)
            z_score = (series - rolling_mean) / (rolling_std + 1e-6)

            # Flag unusually low generation (potential outage)
            # Z-score < -2 means more than 2 std below rolling mean
            outage_flag = (z_score < -2).astype(int)

            col_name = col.replace(' ', '_').lower()
            result[f'{col_name}_low'] = outage_flag

    # Aggregate: total number of generation types with unusual low output
    result['potential_outage_count'] = result.sum(axis=1)

    print(f"  Created proxy with shape: {result.shape}")
    return result


def main():
    """Download outage data for all registered countries."""
    import argparse
    parser = argparse.ArgumentParser(description='Download outage data')
    parser.add_argument('--countries', type=str, default=None,
                        help='Comma-separated country codes (default: all registered)')
    parser.add_argument('--start-year', type=int, default=2015)
    parser.add_argument('--end-year', type=int, default=datetime.now().year)
    args = parser.parse_args()

    print("=" * 60)
    print("Downloading Power Plant Outage Data")
    print("=" * 60)

    countries = args.countries.split(',') if args.countries else get_registered_countries()

    for country in countries:
        try:
            daily_outages = download_outages_in_chunks(
                country=country,
                start_year=args.start_year,
                end_year=args.end_year,
                chunk_months=6
            )

            if daily_outages.empty:
                print(f"\nNo outage data obtained for {country}.")
                print("Trying to create proxy from generation data...")

                # Load existing generation data
                gen_file = DATA_DIR / f"entsoe/entsoe_{country}_2015_2024.csv"
                if gen_file.exists():
                    gen_df = pd.read_csv(gen_file, index_col=0, parse_dates=True)
                    proxy = create_simple_outage_proxy(gen_df, country)
                    if not proxy.empty:
                        proxy_file = OUTAGE_DIR / f"outage_proxy_{country}_2015_2024.csv"
                        proxy.to_csv(proxy_file)
                        print(f"Saved proxy to {proxy_file}")

        except Exception as e:
            print(f"\nError processing {country}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Outage data download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
