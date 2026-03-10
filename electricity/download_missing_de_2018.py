#!/usr/bin/env python3
"""
Download missing DE day-ahead prices for 2018-06-13 to 2018-09-30.

This period falls before the German bidding zone change (Oct 1, 2018),
so it requires the old DE_AT_LU bidding zone code.
"""

import os
import sys
from pathlib import Path

# Ensure CASTOR electricity module is importable
sys.path.insert(0, str(Path(__file__).parent))

from download_entsoe_data import download_day_ahead_prices, get_api_key

from paths import CARS_ROOT, ENTSOE_DIR
# Load API key from .env if not set
env_file = CARS_ROOT / ".env"
if env_file.exists() and 'ENTSOE_API_KEY' not in os.environ:
    for line in env_file.read_text().split('\n'):
        if line.startswith('ENTSOE_API_KEY='):
            os.environ['ENTSOE_API_KEY'] = line.split('=', 1)[1].strip().strip('"\'')
            break

def main():
    print("=" * 60)
    print("Downloading Missing DE Day-Ahead Prices")
    print("Period: 2018-06-13 to 2018-09-30")
    print("Bidding Zone: DE_AT_LU (pre-Oct 2018)")
    print("=" * 60)

    data_dir = Path(str(ENTSOE_DIR))

    try:
        api_key = get_api_key()
        print(f"API key found: {api_key[:8]}...")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Download the missing period
    try:
        prices = download_day_ahead_prices(
            country='DE',
            start='2018-06-13',
            end='2018-09-30',
            api_key=api_key,
            data_dir=data_dir
        )

        print(f"\nDownload successful!")
        print(f"  Hourly records: {len(prices)}")
        print(f"  Date range: {prices.index.min()} to {prices.index.max()}")
        print(f"  Price range: {prices.min():.2f} to {prices.max():.2f} EUR/MWh")

        # Verify daily coverage
        daily_dates = prices.resample('D').count()
        print(f"  Daily coverage: {len(daily_dates)} days")

        return 0

    except Exception as e:
        print(f"Error downloading data: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
