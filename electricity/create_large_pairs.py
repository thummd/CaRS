#!/usr/bin/env python3
"""
Create merged country-pair datasets for large pairs that OOM with the standard approach.
Uses chunked CSV processing to stay within memory limits.
"""
import pandas as pd
import numpy as np
import gc
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))
from paths import UNIFIED_DIR as OUTPUT_DIR

CHUNK_SIZE = 50_000  # rows per chunk


def create_large_pair(country_a: str, country_b: str):
    """Create merged hourly pair dataset using chunked processing."""
    file_a = OUTPUT_DIR / f"unified_{country_a}_2015_2026_hourly_clean.csv"
    file_b = OUTPUT_DIR / f"unified_{country_b}_2015_2026_hourly_clean.csv"

    print(f"\n{'='*60}")
    print(f"Creating {country_a}-{country_b} Merged Hourly Dataset (chunked)")
    print(f"{'='*60}")

    # Step 1: Read headers to get column names
    cols_a = pd.read_csv(file_a, nrows=0).columns.tolist()
    cols_b = pd.read_csv(file_b, nrows=0).columns.tolist()
    idx_col = cols_a[0]  # datetime index column name

    calendar_cols = ['day_of_week', 'day_of_year', 'week_of_year', 'month', 'quarter',
                     'is_weekend', 'is_holiday', 'year', 'season']
    common_cols = [c for c in cols_a[1:] if c in cols_b[1:] and c in calendar_cols]

    # Build rename maps (excluding index col)
    a_rename = {c: f"{country_a}_{c}" for c in cols_a[1:] if c not in common_cols}
    b_rename = {c: f"{country_b}_{c}" for c in cols_b[1:] if c not in common_cols}

    print(f"  {country_a}: {len(cols_a)-1} features, {country_b}: {len(cols_b)-1} features")
    print(f"  Common calendar cols: {len(common_cols)}")

    # Step 2: Load dataset B into memory (index for joining) with float32
    print(f"  Loading {country_b} dataset...")
    df_b = pd.read_csv(file_b, index_col=0, parse_dates=True)
    for col in df_b.select_dtypes(include=['float64']).columns:
        df_b[col] = df_b[col].astype('float32')
    # Drop common cols from B and rename
    df_b = df_b.drop(columns=common_cols, errors='ignore').rename(columns=b_rename)
    gc.collect()
    print(f"    Shape: {df_b.shape}")

    # Step 3: Process A in chunks, join with B, write output
    output_file = OUTPUT_DIR / f"unified_{country_a}_{country_b}_2015_2026_hourly.csv"
    first_chunk = True

    print(f"  Processing {country_a} in chunks of {CHUNK_SIZE}...")
    reader = pd.read_csv(file_a, index_col=0, parse_dates=True, chunksize=CHUNK_SIZE)

    for i, chunk_a in enumerate(reader):
        # Downcast to float32
        for col in chunk_a.select_dtypes(include=['float64']).columns:
            chunk_a[col] = chunk_a[col].astype('float32')

        # Save common data from this chunk
        common_data = chunk_a[common_cols].copy() if common_cols else None

        # Rename A columns
        chunk_a = chunk_a.drop(columns=common_cols, errors='ignore').rename(columns=a_rename)

        # Join with B
        merged = chunk_a.join(df_b, how='inner')
        del chunk_a

        # Add common cols back
        if common_data is not None:
            for col in common_cols:
                if col in common_data.columns:
                    merged[col] = common_data.loc[merged.index, col]
            del common_data

        # Write chunk
        merged.to_csv(output_file, mode='a' if not first_chunk else 'w',
                      header=first_chunk)
        rows_written = len(merged)
        del merged
        gc.collect()

        if first_chunk:
            first_chunk = False
        print(f"    Chunk {i}: {rows_written} rows written", flush=True)

    del df_b
    gc.collect()

    print(f"\n  Saved raw to {output_file}")
    print(f"  File size: {output_file.stat().st_size / (1024*1024):.1f} MB")

    # Step 4: Add spread features by reading back the merged file in chunks
    print("\n  Computing spread features...")
    a_price = f"{country_a}_Day_Ahead_Price"
    b_price = f"{country_b}_Day_Ahead_Price"

    # Read just the price columns + index to compute spreads
    df_prices = pd.read_csv(output_file, index_col=0, parse_dates=True,
                            usecols=[idx_col, a_price, b_price])

    spread = df_prices[a_price] - df_prices[b_price]
    spread_df = pd.DataFrame(index=df_prices.index)
    spread_df['price_spread'] = spread
    spread_df['price_spread_change'] = spread.diff()

    spread_lag = spread.shift(1)
    safe_mask = spread_lag.abs() > 0.01
    spread_df['price_spread_change_pct'] = np.where(
        safe_mask, spread.diff() / spread_lag.abs() * 100, 0)

    # Hourly lags
    for lag in [1, 3, 6, 12, 24, 48, 168]:
        spread_df[f'price_spread_lag{lag}h'] = spread.shift(lag)

    # Rolling stats
    for window in [24, 48, 168]:
        spread_df[f'price_spread_rolling_mean_{window}h'] = spread.rolling(window).mean()
        spread_df[f'price_spread_rolling_std_{window}h'] = spread.rolling(window).std()

    del df_prices, spread, spread_lag
    gc.collect()

    print(f"    Spread features: {list(spread_df.columns)}")

    # Step 5: Append spread columns to the output file by re-reading and writing chunks
    print("  Appending spread features to merged file...")

    # Read merged file in chunks and append spread columns
    final_file = OUTPUT_DIR / f"unified_{country_a}_{country_b}_2015_2026_hourly_final.csv"
    reader = pd.read_csv(output_file, index_col=0, parse_dates=True, chunksize=CHUNK_SIZE)
    first_chunk = True

    for i, chunk in enumerate(reader):
        # Join spread features
        chunk = chunk.join(spread_df, how='left')
        for col in chunk.select_dtypes(include=['float64']).columns:
            chunk[col] = chunk[col].astype('float32')
        chunk.to_csv(final_file, mode='a' if not first_chunk else 'w', header=first_chunk)
        first_chunk = False
        del chunk
        gc.collect()

    del spread_df
    gc.collect()

    # Replace original with final
    import shutil
    output_file.unlink()
    shutil.move(str(final_file), str(output_file))

    # Create clean version (drop rows with NaN in spread_change)
    clean_file = OUTPUT_DIR / f"unified_{country_a}_{country_b}_2015_2026_hourly_clean.csv"
    reader = pd.read_csv(output_file, index_col=0, parse_dates=True, chunksize=CHUNK_SIZE)
    first_chunk = True

    for chunk in reader:
        chunk_clean = chunk.dropna(subset=['price_spread_change'])
        if not chunk_clean.empty:
            chunk_clean.to_csv(clean_file, mode='a' if not first_chunk else 'w',
                               header=first_chunk)
            first_chunk = False
        del chunk, chunk_clean
        gc.collect()

    print(f"  Saved clean to {clean_file}")
    print(f"  Clean file size: {clean_file.stat().st_size / (1024*1024):.1f} MB")
    print(f"  Done: {country_a}-{country_b}")


if __name__ == '__main__':
    pairs = sys.argv[1:] if len(sys.argv) > 1 else ['BE-DE', 'DE-NL']
    for pair in pairs:
        a, b = pair.split('-')
        create_large_pair(a, b)
        gc.collect()
    print("\nAll done!")
