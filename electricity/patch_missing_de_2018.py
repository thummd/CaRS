#!/usr/bin/env python3
"""
Patch missing DE day-ahead prices for 2018 into the base data file.

This script:
1. Loads the newly downloaded ENTSO-E prices for 2018-06-13 to 2018-09-30
2. Updates the base calendar file with the missing prices
3. Saves back the updated file
"""

import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR
DATA_DIR = DATA_DIR


def main():
    print("=" * 60)
    print("Patching Missing DE Day-Ahead Prices for 2018")
    print("=" * 60)

    # Load the newly downloaded ENTSO-E prices
    entsoe_file = DATA_DIR / "entsoe/day_ahead_prices_DE_2018-06-13_2018-09-30.csv"
    if not entsoe_file.exists():
        print(f"Error: ENTSO-E file not found: {entsoe_file}")
        print("Run download_missing_de_2018.py first")
        return 1

    print(f"\nLoading ENTSO-E prices from {entsoe_file}")
    entsoe_prices = pd.read_csv(entsoe_file, index_col=0, parse_dates=True)
    print(f"  Shape: {entsoe_prices.shape}")
    print(f"  Date range: {entsoe_prices.index.min()} to {entsoe_prices.index.max()}")

    # Convert to UTC and resample to daily average
    if entsoe_prices.index.tz is None:
        entsoe_prices.index = entsoe_prices.index.tz_localize('UTC')
    else:
        entsoe_prices.index = entsoe_prices.index.tz_convert('UTC')

    daily_prices = entsoe_prices['Price'].resample('D').mean()
    print(f"  Daily prices: {len(daily_prices)} days")
    print(f"  Date range: {daily_prices.index.min()} to {daily_prices.index.max()}")

    # Load the base calendar file
    calendar_file = DATA_DIR / "merged_daily_DE_2015_2024_calendar.csv"
    print(f"\nLoading base calendar file: {calendar_file}")
    base_df = pd.read_csv(calendar_file, index_col=0, parse_dates=True)
    print(f"  Shape: {base_df.shape}")

    # Check how many NaN values before patching
    nan_before = base_df['Day_Ahead_Price'].isna().sum()
    print(f"  NaN in Day_Ahead_Price before: {nan_before}")

    # Skip updating ENTSO-E combined file as it has complex hourly data
    # Focus on the daily calendar file which is what we need

    # Update the base calendar file
    print("\nPatching base calendar file...")

    # Ensure base_df index has same timezone as daily_prices
    if base_df.index.tz is None:
        base_df.index = base_df.index.tz_localize('UTC')
    elif str(base_df.index.tz) != 'UTC':
        base_df.index = base_df.index.tz_convert('UTC')

    print(f"  Base index tz: {base_df.index.tz}")
    print(f"  Daily prices tz: {daily_prices.index.tz}")

    patched_count = 0
    for date, price in daily_prices.items():
        if date in base_df.index:
            if pd.isna(base_df.loc[date, 'Day_Ahead_Price']):
                base_df.loc[date, 'Day_Ahead_Price'] = price
                patched_count += 1
        else:
            # Try without timezone
            date_naive = date.tz_localize(None) if date.tz is not None else date
            # Check with UTC
            try:
                date_utc = date_naive.tz_localize('UTC')
                if date_utc in base_df.index and pd.isna(base_df.loc[date_utc, 'Day_Ahead_Price']):
                    base_df.loc[date_utc, 'Day_Ahead_Price'] = price
                    patched_count += 1
            except:
                pass

    print(f"  Patched {patched_count} missing prices")

    nan_after = base_df['Day_Ahead_Price'].isna().sum()
    print(f"  NaN in Day_Ahead_Price after: {nan_after}")

    if nan_after > 0:
        # Show which dates are still missing
        missing_dates = base_df[base_df['Day_Ahead_Price'].isna()].index
        print(f"  Still missing dates: {len(missing_dates)}")
        print(f"    First 5: {list(missing_dates[:5])}")
        print(f"    Last 5: {list(missing_dates[-5:])}")

    # Remove timezone for saving (keep dates timezone-naive in CSV)
    base_df.index = base_df.index.tz_localize(None)

    # Save updated base file
    base_df.to_csv(calendar_file)
    print(f"\nSaved updated calendar file: {calendar_file}")

    # Verify
    print("\nVerifying 2018 data completeness...")
    df_2018 = base_df[base_df.index.year == 2018]
    print(f"  2018 rows: {len(df_2018)}")
    print(f"  2018 NaN in Day_Ahead_Price: {df_2018['Day_Ahead_Price'].isna().sum()}")

    if df_2018['Day_Ahead_Price'].isna().sum() == 0:
        print("\n✓ All 2018 prices are now complete!")
    else:
        print("\n✗ Some 2018 prices are still missing")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
