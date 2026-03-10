"""
Add calendar features to electricity price datasets.

Features added:
- Day of week (0=Monday, 6=Sunday)
- Is weekend
- Month
- Season (1-4)
- Day of year
- Week of year
- Public holidays (Germany and France)

Usage: python3 add_calendar_features.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date

import holidays as holidays_lib

from country_config import COUNTRY_REGISTRY, get_registered_countries

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR
# Data directory
DATA_DIR = DATA_DIR


def get_holidays_for_year(year: int, country: str = 'DE') -> set:
    """
    Get all public holidays for a given year and country.

    Uses the `holidays` library which supports all EU countries.

    Returns set of date objects.
    """
    iso_code = COUNTRY_REGISTRY.get(country, {}).get('holidays_iso', country)
    h = holidays_lib.country_holidays(iso_code, years=year)
    return set(h.keys())


def get_all_holidays(start_year: int, end_year: int, country: str) -> set:
    """Get all holidays for a range of years."""
    iso_code = COUNTRY_REGISTRY.get(country, {}).get('holidays_iso', country)
    h = holidays_lib.country_holidays(iso_code, years=range(start_year, end_year + 1))
    return set(h.keys())


def add_calendar_features(df: pd.DataFrame, country: str = 'DE') -> pd.DataFrame:
    """
    Add calendar features to a DataFrame with datetime index.

    Args:
        df: DataFrame with datetime index
        country: Country code for country-specific holidays (e.g., 'DE', 'FR', 'NL')

    Returns:
        DataFrame with additional calendar features
    """
    df = df.copy()

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Basic calendar features
    df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week.values

    # Season (1=Winter, 2=Spring, 3=Summer, 4=Autumn)
    # Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Autumn: Sep-Nov
    df['season'] = df.index.month.map({
        12: 1, 1: 1, 2: 1,  # Winter
        3: 2, 4: 2, 5: 2,   # Spring
        6: 3, 7: 3, 8: 3,   # Summer
        9: 4, 10: 4, 11: 4  # Autumn
    })

    # Year (useful for trend analysis)
    df['year'] = df.index.year

    # Get holidays
    start_year = df.index.min().year
    end_year = df.index.max().year
    holidays = get_all_holidays(start_year, end_year, country)

    # Is holiday
    df['is_holiday'] = pd.Series(df.index.date, index=df.index).isin(holidays).astype(int)

    # Is bridge day (day between holiday and weekend)
    # A bridge day is a working day between a holiday and a weekend
    # Compute on unique dates first, then broadcast (efficient for hourly data)
    unique_dates = pd.Series(df.index.date).unique()
    bridge_dates = set()
    for dt in unique_dates:
        dow = dt.weekday()
        # Check if it's a regular working day
        if dow < 5 and dt not in holidays:
            # Friday before a Monday holiday
            if dow == 4:
                monday = date.fromordinal(dt.toordinal() + 3)
                if monday in holidays:
                    bridge_dates.add(dt)
            # Monday after a Friday holiday
            elif dow == 0:
                friday = date.fromordinal(dt.toordinal() - 3)
                if friday in holidays:
                    bridge_dates.add(dt)
    df['is_bridge_day'] = pd.Series(df.index.date, index=df.index).isin(bridge_dates).astype(int)

    # One-hot encode day of week
    for i in range(7):
        df[f'dow_{i}'] = (df['day_of_week'] == i).astype(int)

    # One-hot encode seasons
    for i in range(1, 5):
        df[f'season_{i}'] = (df['season'] == i).astype(int)

    # Cyclical encoding for day of year (captures annual seasonality)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # Cyclical encoding for day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Hourly features (only if data has sub-daily resolution)
    if hasattr(df.index, 'hour') and df.index.hour.nunique() > 1:
        df['hour_of_day'] = df.index.hour
        df['is_peak_hour'] = df['hour_of_day'].isin(range(8, 20)).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

    return df


def process_dataset(input_file: Path, output_file: Path, country: str):
    """Process a single dataset file."""
    print(f"\nProcessing {input_file.name}...")

    # Read data
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    print(f"  Original shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Add calendar features
    df_enhanced = add_calendar_features(df, country)
    print(f"  Enhanced shape: {df_enhanced.shape}")

    # Show new columns
    original_cols = set(pd.read_csv(input_file, index_col=0, nrows=1).columns)
    new_cols = [c for c in df_enhanced.columns if c not in original_cols]
    print(f"  New calendar features: {len(new_cols)}")

    # Save
    df_enhanced.to_csv(output_file)
    print(f"  Saved to {output_file}")

    return df_enhanced


def main():
    """Add calendar features to all merged datasets."""
    import argparse
    parser = argparse.ArgumentParser(description='Add calendar features')
    parser.add_argument('--countries', type=str, default=None,
                        help='Comma-separated country codes (default: all registered)')
    args = parser.parse_args()

    print("=" * 60)
    print("Adding Calendar Features to Electricity Datasets")
    print("=" * 60)

    countries = args.countries.split(',') if args.countries else get_registered_countries()

    for country in countries:
        # Process daily dataset
        daily_input = DATA_DIR / f"merged_daily_{country}_2015_2024.csv"
        daily_output = DATA_DIR / f"merged_daily_{country}_2015_2024_calendar.csv"

        if daily_input.exists():
            df = process_dataset(daily_input, daily_output, country)

            calendar_cols = ['day_of_week', 'is_weekend', 'month', 'season',
                            'is_holiday', 'is_bridge_day']
            avail = [c for c in calendar_cols if c in df.columns]
            print(f"\n  Sample calendar features ({country}):")
            print(df[avail].head(5))

            hol = df[df['is_holiday'] == 1]
            print(f"  Total holidays in dataset: {len(hol)}")
        else:
            print(f"\nNOTE: {daily_input} not found")

        # Process hourly dataset (if it exists)
        hourly_input = DATA_DIR / f"merged_hourly_{country}_2015_2024.csv"
        hourly_output = DATA_DIR / f"merged_hourly_{country}_2015_2024_calendar.csv"
        if hourly_input.exists():
            process_dataset(hourly_input, hourly_output, country)
        else:
            print(f"  NOTE: {hourly_input} not found (hourly data not available)")

    print("\n" + "=" * 60)
    print("Calendar feature addition complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
