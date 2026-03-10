"""
Download historical weather data from Open-Meteo API.

Weather variables important for electricity price forecasting:
- Temperature (affects heating/cooling demand)
- Wind speed (affects wind generation)
- Solar radiation (affects solar generation)
- Precipitation (affects hydro generation)

Data source: https://open-meteo.com/en/docs/historical-weather-api
Free for non-commercial use, no API key required.

Geographic focus: Major population/industrial centers in Germany and France
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import time
import requests
from datetime import datetime, timedelta

from country_config import COUNTRY_REGISTRY, get_registered_countries

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WEATHER_DIR
# Data directory
DATA_DIR = WEATHER_DIR

# Open-Meteo Historical Weather API
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _get_locations(country: str) -> List[Dict]:
    """Get weather station cities for a country from the registry."""
    if country not in COUNTRY_REGISTRY:
        raise ValueError(f"Country '{country}' not in registry. Available: {list(COUNTRY_REGISTRY.keys())}")
    return COUNTRY_REGISTRY[country]['weather']['cities']


def _get_grid_config(country: str) -> Dict:
    """Get grid bbox and resolution for a country from the registry."""
    if country not in COUNTRY_REGISTRY:
        raise ValueError(f"Country '{country}' not in registry. Available: {list(COUNTRY_REGISTRY.keys())}")
    weather = COUNTRY_REGISTRY[country]['weather']
    return {
        'bbox': weather['grid_bbox'],
        'resolution': weather['grid_resolution'],
        'name': COUNTRY_REGISTRY[country]['name'],
    }

# API efficiency settings for gridded downloads
MAX_LOCATIONS_PER_CALL = 50  # Conservative to stay under rate limits
BATCH_DELAY_SECONDS = 1.0    # Delay between batch API calls
YEARS_PER_CHUNK = 2          # Split long date ranges for API efficiency

# Weather variables to fetch
HOURLY_VARIABLES = [
    'temperature_2m',           # Temperature at 2m (°C)
    'relative_humidity_2m',     # Relative humidity (%)
    'apparent_temperature',     # Feels-like temperature (°C)
    'precipitation',            # Precipitation (mm)
    'rain',                     # Rain (mm)
    'snowfall',                 # Snowfall (cm)
    'cloud_cover',              # Total cloud cover (%)
    'wind_speed_10m',           # Wind speed at 10m (km/h)
    'wind_speed_100m',          # Wind speed at 100m (km/h) - relevant for wind turbines
    'wind_direction_10m',       # Wind direction (degrees)
    'shortwave_radiation',      # Solar radiation (W/m²)
    'direct_radiation',         # Direct solar radiation (W/m²)
    'diffuse_radiation',        # Diffuse solar radiation (W/m²)
]

DAILY_VARIABLES = [
    'temperature_2m_max',
    'temperature_2m_min',
    'temperature_2m_mean',
    'apparent_temperature_max',
    'apparent_temperature_min',
    'precipitation_sum',
    'rain_sum',
    'snowfall_sum',
    'wind_speed_10m_max',
    'wind_gusts_10m_max',
    'shortwave_radiation_sum',
]


def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly: bool = True,
    variables: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo API.

    Args:
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        hourly: If True, fetch hourly data; if False, fetch daily
        variables: List of weather variables to fetch

    Returns:
        DataFrame with weather data indexed by datetime
    """
    if variables is None:
        variables = HOURLY_VARIABLES if hourly else DAILY_VARIABLES

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'timezone': 'UTC',
    }

    if hourly:
        params['hourly'] = ','.join(variables)
    else:
        params['daily'] = ','.join(variables)

    response = requests.get(BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if hourly:
        time_key = 'hourly'
        time_col = 'time'
    else:
        time_key = 'daily'
        time_col = 'time'

    if time_key not in data:
        raise ValueError(f"No {time_key} data in response: {data.keys()}")

    df = pd.DataFrame(data[time_key])
    df['datetime'] = pd.to_datetime(df[time_col])
    df = df.set_index('datetime')
    df = df.drop(columns=[time_col])

    return df


def generate_country_grid(country: str) -> List[Dict]:
    """
    Generate grid points for a country with area weights.

    Args:
        country: Country code (e.g., 'DE', 'FR')

    Returns:
        List of dicts with 'lat', 'lon', 'weight' keys
    """
    config = _get_grid_config(country)
    min_lon, min_lat, max_lon, max_lat = config['bbox']
    resolution = config['resolution']

    points = []
    for lat in np.arange(min_lat + resolution/2, max_lat, resolution):
        for lon in np.arange(min_lon + resolution/2, max_lon, resolution):
            points.append({'lat': round(lat, 4), 'lon': round(lon, 4)})

    # Calculate area weights (cosine of latitude correction)
    lats = np.array([p['lat'] for p in points])
    weights = np.cos(np.radians(lats))
    weights = weights / weights.sum()

    for i, p in enumerate(points):
        p['weight'] = weights[i]

    print(f"Generated {len(points)} grid points for {config['name']}")
    return points


def fetch_weather_batch(
    points: List[Dict],
    start_date: str,
    end_date: str,
    hourly: bool = True,
    variables: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fetch weather data for multiple points and return weighted average.

    Args:
        points: List of {'lat', 'lon', 'weight'} dicts
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        hourly: If True, fetch hourly data
        variables: Weather variables to fetch

    Returns:
        DataFrame with weighted average weather data
    """
    if variables is None:
        variables = HOURLY_VARIABLES if hourly else DAILY_VARIABLES

    weights = [p['weight'] for p in points]
    all_dfs = []

    # Fetch data for each point individually (Open-Meteo batch endpoint is unreliable)
    for i, point in enumerate(points):
        params = {
            'latitude': point['lat'],
            'longitude': point['lon'],
            'start_date': start_date,
            'end_date': end_date,
            'timezone': 'UTC',
        }

        if hourly:
            params['hourly'] = ','.join(variables)
        else:
            params['daily'] = ','.join(variables)

        response = requests.get(BASE_URL, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        time_key = 'hourly' if hourly else 'daily'

        if time_key not in data:
            print(f"Warning: No {time_key} data for point {point}")
            continue

        df = pd.DataFrame(data[time_key])
        df['datetime'] = pd.to_datetime(df['time'])
        df = df.set_index('datetime').drop(columns=['time'])

        # Apply weight
        all_dfs.append(df * weights[i])

        # Small delay to be nice to API
        time.sleep(0.1)

    if not all_dfs:
        raise ValueError("No data fetched for any point")

    # Sum weighted values (weights sum to 1, so this gives weighted average)
    weighted_avg = pd.concat(all_dfs).groupby(level=0).sum()
    return weighted_avg


def download_gridded_country_weather(
    country: str,
    start_date: str,
    end_date: str,
    hourly: bool = True,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Download gridded national average weather data.

    Args:
        country: 'DE' or 'FR'
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        hourly: Fetch hourly (True) or daily (False) data
        data_dir: Directory to cache data
        use_cache: If True, use cached data if available

    Returns:
        DataFrame with area-weighted national average weather
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    freq = 'hourly' if hourly else 'daily'
    cache_file = data_dir / f"{country}_gridded_{freq}_{start_date}_{end_date}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading cached gridded weather from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    config = _get_grid_config(country)
    print(f"Downloading gridded {freq} weather for {config['name']}...")

    # Generate grid points
    points = generate_country_grid(country)

    # Split time range into chunks for API efficiency
    all_data = []
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    current_start = start
    chunk_num = 0
    while current_start <= end:
        current_end = min(current_start + pd.DateOffset(years=YEARS_PER_CHUNK) - pd.DateOffset(days=1), end)
        chunk_num += 1

        start_str = current_start.strftime('%Y-%m-%d')
        end_str = current_end.strftime('%Y-%m-%d')

        print(f"  Time chunk {chunk_num}: {start_str} to {end_str}")

        # Process location batches
        for batch_start in range(0, len(points), MAX_LOCATIONS_PER_CALL):
            batch_end = min(batch_start + MAX_LOCATIONS_PER_CALL, len(points))
            batch_points = points[batch_start:batch_end]

            # Renormalize weights for this batch
            batch_weight_sum = sum(p['weight'] for p in batch_points)
            batch_points_normalized = [
                {**p, 'weight': p['weight'] / batch_weight_sum}
                for p in batch_points
            ]

            print(f"    Locations {batch_start+1}-{batch_end} of {len(points)}...")

            try:
                batch_df = fetch_weather_batch(
                    batch_points_normalized, start_str, end_str, hourly
                )
                # Re-scale by original batch weight proportion
                batch_df = batch_df * batch_weight_sum
                all_data.append(batch_df)
                time.sleep(BATCH_DELAY_SECONDS)
            except Exception as e:
                print(f"    Error: {e}")
                time.sleep(5)  # Longer delay on error
                continue

        current_start = current_end + pd.DateOffset(days=1)

    if not all_data:
        raise ValueError("No data fetched")

    # Combine all batches
    combined = pd.concat(all_data).groupby(level=0).sum()

    # Add country prefix
    combined.columns = [f"{country}_{col}" for col in combined.columns]

    # Cache
    combined.to_csv(cache_file)
    print(f"Saved gridded weather to {cache_file}")

    return combined


def download_country_weather(
    country: str,
    start_date: str,
    end_date: str,
    hourly: bool = True,
    aggregate: str = 'mean',
    data_dir: Path = DATA_DIR,
    method: str = 'cities'
) -> pd.DataFrame:
    """
    Download weather data for a country.

    Args:
        country: Country code (e.g., 'DE', 'FR', 'NL')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        hourly: Fetch hourly (True) or daily (False) data
        aggregate: How to aggregate across cities ('mean', 'max', 'min') - only for cities method
        data_dir: Directory to cache data
        method: 'gridded' for ERA5 grid average (recommended for precipitation),
                'cities' for legacy 5-city average

    Returns:
        DataFrame with aggregated weather data
    """
    # Dispatch to gridded method if requested
    if method == 'gridded':
        return download_gridded_country_weather(
            country=country,
            start_date=start_date,
            end_date=end_date,
            hourly=hourly,
            data_dir=data_dir
        )

    # Cities method
    locations = _get_locations(country)

    data_dir.mkdir(parents=True, exist_ok=True)
    freq = 'hourly' if hourly else 'daily'
    cache_file = data_dir / f"{country}_{freq}_{start_date}_{end_date}.csv"

    if cache_file.exists():
        print(f"Loading cached weather data from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    print(f"Downloading {freq} weather data for {country}...")
    all_data = []

    for loc in locations:
        print(f"  Fetching {loc['name']}...")
        try:
            df = fetch_weather_data(
                latitude=loc['lat'],
                longitude=loc['lon'],
                start_date=start_date,
                end_date=end_date,
                hourly=hourly
            )
            df['location'] = loc['name']
            all_data.append(df)
            time.sleep(0.5)  # Be nice to the API
        except Exception as e:
            print(f"    Error fetching {loc['name']}: {e}")
            continue

    if not all_data:
        raise ValueError("No data fetched for any location")

    # Combine all locations
    combined = pd.concat(all_data)

    # Aggregate across locations
    if aggregate == 'mean':
        agg_df = combined.groupby(combined.index).mean(numeric_only=True)
    elif aggregate == 'max':
        agg_df = combined.groupby(combined.index).max(numeric_only=True)
    elif aggregate == 'min':
        agg_df = combined.groupby(combined.index).min(numeric_only=True)
    else:
        raise ValueError(f"Unknown aggregate method: {aggregate}")

    # Add country prefix to columns
    agg_df.columns = [f"{country}_{col}" for col in agg_df.columns]

    # Cache the data
    agg_df.to_csv(cache_file)
    print(f"Saved to {cache_file}")

    return agg_df


def align_weather_with_prices(
    weather_df: pd.DataFrame,
    price_df: pd.DataFrame,
    resample: str = 'D'
) -> pd.DataFrame:
    """
    Align weather data with electricity price data.

    Args:
        weather_df: Weather DataFrame (hourly or daily)
        price_df: Price DataFrame from epftoolbox (hourly)
        resample: Resampling frequency ('H' for hourly, 'D' for daily)

    Returns:
        DataFrame with weather data aligned to price timestamps
    """
    # Ensure both have UTC timezone
    if weather_df.index.tz is None:
        weather_df.index = weather_df.index.tz_localize('UTC')
    if price_df.index.tz is None:
        price_df.index = price_df.index.tz_localize('UTC')

    # Resample if needed
    if resample == 'D':
        # Aggregate hourly weather to daily
        weather_daily = weather_df.resample('D').agg({
            col: 'mean' if 'temperature' in col or 'humidity' in col or 'cloud' in col
            else 'sum' if 'precipitation' in col or 'rain' in col or 'snow' in col or 'radiation' in col
            else 'max' if 'wind_speed' in col or 'gusts' in col
            else 'mean'
            for col in weather_df.columns
        })
        weather_aligned = weather_daily
    else:
        weather_aligned = weather_df

    # Reindex to match price data timestamps
    common_dates = price_df.index.intersection(weather_aligned.index)

    if len(common_dates) == 0:
        print("Warning: No overlapping dates between weather and price data")
        print(f"Weather: {weather_aligned.index.min()} to {weather_aligned.index.max()}")
        print(f"Prices:  {price_df.index.min()} to {price_df.index.max()}")
        return weather_aligned

    return weather_aligned.loc[common_dates]


def get_epftoolbox_date_range(dataset: str = 'DE') -> tuple:
    """
    Get the date range of epftoolbox data to match weather download.

    Args:
        dataset: 'DE' or 'FR'

    Returns:
        (start_date, end_date) strings
    """
    # epftoolbox data ranges (approximate, will be refined when loading)
    # DE: 2012-01-01 to 2019-12-31
    # FR: 2012-01-01 to 2019-12-31
    return '2012-01-01', '2019-12-31'


def create_merged_dataset(
    country: str,
    hourly_weather: bool = True,
    data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """
    Create a merged dataset with electricity prices and weather features.

    Args:
        country: 'DE' or 'FR'
        hourly_weather: Whether to use hourly weather data
        data_dir: Weather data directory

    Returns:
        DataFrame with prices and weather features
    """
    from download_epftoolbox_data import download_dataset

    # Load price data
    print(f"Loading {country} price data...")
    price_df = download_dataset(country)

    # Get date range from price data
    start_date = price_df.index.min().strftime('%Y-%m-%d')
    end_date = price_df.index.max().strftime('%Y-%m-%d')
    print(f"Price data range: {start_date} to {end_date}")

    # Download weather data
    print(f"Downloading weather data...")
    weather_df = download_country_weather(
        country=country,
        start_date=start_date,
        end_date=end_date,
        hourly=hourly_weather,
        data_dir=data_dir
    )

    # Align timestamps
    print("Aligning weather with price data...")
    if hourly_weather:
        # Both hourly - direct join
        merged = price_df.join(weather_df, how='left')
    else:
        # Daily weather - need to broadcast to hourly prices
        merged = price_df.copy()
        weather_df.index = weather_df.index.date
        merged['date'] = merged.index.date
        for col in weather_df.columns:
            merged[col] = merged['date'].map(weather_df[col])
        merged = merged.drop(columns=['date'])

    # Fill any missing weather values
    n_missing_before = merged.isnull().sum().sum()
    merged = merged.ffill().bfill()
    n_missing_after = merged.isnull().sum().sum()
    if n_missing_before > 0:
        print(f"Filled {n_missing_before - n_missing_after} missing values")

    return merged


def main():
    """Download weather data for all registered countries."""
    import argparse
    parser = argparse.ArgumentParser(description='Download weather data from Open-Meteo')
    parser.add_argument('--countries', type=str, default=None,
                        help='Comma-separated country codes (default: all registered)')
    parser.add_argument('--start', type=str, default='2012-01-01')
    parser.add_argument('--end', type=str, default='2024-12-31')
    parser.add_argument('--method', type=str, default='cities',
                        choices=['cities', 'gridded'])
    args = parser.parse_args()

    print("=" * 60)
    print("Downloading Historical Weather Data from Open-Meteo")
    print("=" * 60)

    countries = args.countries.split(',') if args.countries else get_registered_countries()
    print(f"Countries: {countries}")
    print(f"Date range: {args.start} to {args.end}")

    for country in countries:
        print(f"\n{'='*40}")
        print(f"Country: {country}")
        print(f"{'='*40}")

        try:
            df = download_country_weather(
                country=country,
                start_date=args.start,
                end_date=args.end,
                hourly=True,
                method=args.method
            )

            print(f"\nShape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Columns: {list(df.columns)[:5]}...")
            print(f"\nSample statistics:")
            print(df.describe().T[['mean', 'std', 'min', 'max']].head(10))

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Weather data download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
