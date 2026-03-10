"""
Create a unified dataset combining all features for electricity price forecasting.

This script merges:
1. ENTSO-E data (prices, generation, load, cross-border flows)
2. Weather data (temperature, wind, solar radiation)
3. Calendar features (day of week, holidays, seasons)
4. Outage data (unavailable capacity)
5. Commodity prices (when available)
6. Gas storage data (AGSI+ fill levels, injection, withdrawal)

Output: A single CSV file per country with all features aligned by date.

Usage:
    python3 create_unified_dataset.py              # Create DE and FR datasets
    python3 create_unified_dataset.py --create_merged  # Also create DE-FR merged dataset
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from country_config import (
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR
    get_registered_countries, get_all_pairs, get_neighbors,
    has_gas_storage, COUNTRY_REGISTRY,
)

# Data directories
DATA_DIR = DATA_DIR
OUTPUT_DIR = DATA_DIR / "unified"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def upsample_daily_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill daily data to hourly resolution.

    Each daily value is assigned to midnight (00:00) and then
    forward-filled across all 24 hours of that day.
    """
    if df.empty:
        return df
    # Ensure timezone-naive for consistency
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    start = df.index.min()
    end = df.index.max() + pd.Timedelta(hours=23)
    hourly_index = pd.date_range(start=start, end=end, freq='h')
    df_hourly = df.reindex(hourly_index, method='ffill')
    df_hourly.index.name = df.index.name
    return df_hourly


def _is_hourly_data(df: pd.DataFrame) -> bool:
    """Detect if a DataFrame contains hourly (sub-daily) data."""
    if len(df) < 10:
        return False
    inferred = pd.infer_freq(df.index[:min(100, len(df))])
    if inferred and ('H' in inferred.upper() or 'T' in inferred.upper()):
        return True
    # Fallback heuristic: check if there are multiple entries per day
    if hasattr(df.index, 'date'):
        sample_dates = pd.Series(df.index.date).value_counts()
        if sample_dates.median() > 1:
            return True
    return False


def load_and_prepare_data(
    file_path: Path,
    name: str,
    frequency: str = 'D',
    source_frequency: str = 'auto'
) -> pd.DataFrame:
    """Load a CSV file and prepare it for merging.

    Args:
        file_path: Path to the CSV file
        name: Display name for logging
        frequency: Target output frequency ('D' for daily, 'H' for hourly)
        source_frequency: Source data frequency ('auto', 'hourly', 'daily')
    """
    if not file_path.exists():
        print(f"  WARNING: {name} file not found: {file_path}")
        return pd.DataFrame()

    print(f"  Loading {name}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Ensure index is a proper DatetimeIndex (handles mixed/offset timezone strings)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    # Ensure timezone-naive for merging
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Detect source frequency if auto
    if source_frequency == 'auto':
        is_hourly = _is_hourly_data(df)
    else:
        is_hourly = source_frequency == 'hourly'

    if frequency == 'D':
        # Daily mode: resample hourly sources to daily (original behavior)
        if is_hourly:
            print(f"    Resampling to daily (original shape: {df.shape})")
            df = df.resample('D').mean()
    elif frequency == 'H':
        if is_hourly:
            # Already hourly, keep as-is
            print(f"    Keeping hourly resolution (shape: {df.shape})")
        else:
            # Daily source: upsample to hourly via forward-fill
            print(f"    Upsampling daily to hourly (original shape: {df.shape})")
            df = upsample_daily_to_hourly(df)

    print(f"    Shape: {df.shape}, Date range: {df.index.min()} to {df.index.max()}")
    return df


def create_target_variable(df: pd.DataFrame, price_col: str = None) -> pd.DataFrame:
    """
    Create the target variable: daily price variation.

    The QRT challenge targets the daily price variation of electricity futures
    contracts with 24h maturity. We use day-ahead prices as a proxy.

    Target = (Price_t - Price_{t-1}) / Price_{t-1} * 100  (percentage change)
    """
    df = df.copy()

    # Find price column
    if price_col is None:
        price_cols = [c for c in df.columns if 'price' in c.lower() or 'da_price' in c.lower()]
        if price_cols:
            price_col = price_cols[0]
        else:
            print("  WARNING: No price column found for target variable")
            return df

    print(f"  Creating target variable from: {price_col}")

    # Daily price change (absolute)
    df['price_change'] = df[price_col].diff()

    # Daily price change (percentage)
    df['price_change_pct'] = df[price_col].pct_change() * 100

    # Lagged price for reference
    df['price_lag1'] = df[price_col].shift(1)

    # Direction of change
    df['price_direction'] = (df['price_change'] > 0).astype(int)

    return df


def add_lag_features(
    df: pd.DataFrame,
    cols: list,
    lags: list = None,
    frequency: str = 'D'
) -> pd.DataFrame:
    """Add lagged features for time series modeling."""
    df = df.copy()

    if lags is None:
        lags = [1, 2, 3, 7] if frequency == 'D' else [1, 3, 6, 12, 24, 48, 168]

    suffix = 'h' if frequency == 'H' else ''
    for col in cols:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag{lag}{suffix}'] = df[col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    cols: list,
    windows: list = None,
    frequency: str = 'D'
) -> pd.DataFrame:
    """Add rolling mean/std features."""
    df = df.copy()

    if windows is None:
        windows = [7, 14, 30] if frequency == 'D' else [24, 48, 168, 336, 720]

    unit = 'h' if frequency == 'H' else 'd'
    for col in cols:
        if col in df.columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}{unit}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}{unit}'] = df[col].rolling(window=window, min_periods=1).std()

    return df


def create_unified_dataset(country: str, frequency: str = 'D') -> pd.DataFrame:
    """
    Create unified dataset for a country.

    Args:
        country: Country code (e.g., 'DE', 'FR', 'NL')
        frequency: 'D' for daily (default), 'H' for hourly

    Returns:
        DataFrame with all features merged
    """
    freq_label = "hourly" if frequency == 'H' else "daily"
    print(f"\n{'='*60}")
    print(f"Creating unified {freq_label} dataset for {country}")
    print(f"{'='*60}")

    # Load base data with calendar features
    if frequency == 'H':
        # Hourly mode: use existing merged hourly files
        calendar_file = DATA_DIR / f"merged_hourly_{country}_2015_2024_calendar.csv"
        base_df = load_and_prepare_data(calendar_file, "Hourly calendar-enhanced base data",
                                         frequency=frequency, source_frequency='hourly')
        if base_df.empty:
            base_file = DATA_DIR / f"merged_hourly_{country}_2015_2024.csv"
            base_df = load_and_prepare_data(base_file, "Hourly base data",
                                             frequency=frequency, source_frequency='hourly')
    else:
        calendar_file = DATA_DIR / f"merged_daily_{country}_2015_2024_calendar.csv"
        base_df = load_and_prepare_data(calendar_file, "Calendar-enhanced base data")

    if base_df.empty:
        # Fall back to non-calendar version
        base_file = DATA_DIR / f"merged_daily_{country}_2015_2024.csv"
        base_df = load_and_prepare_data(base_file, "Base data")

    if base_df.empty:
        print(f"  ERROR: No base data found for {country}")
        return pd.DataFrame()

    # Add calendar features if not already present
    if 'day_of_week' not in base_df.columns:
        try:
            from add_calendar_features import add_calendar_features
            print("  Adding calendar features...")
            base_df = add_calendar_features(base_df, country)
        except ImportError:
            print("  WARNING: Could not import add_calendar_features")

    # Load outage data
    if frequency == 'H':
        outage_file = DATA_DIR / f"outages/outage_hourly_{country}_2015_2024.csv"
        outage_df = load_and_prepare_data(outage_file, "Hourly outage data",
                                           frequency=frequency, source_frequency='hourly')
        if outage_df.empty:
            # Fallback: forward-fill daily outages to hourly
            outage_file = DATA_DIR / f"outages/outage_daily_{country}_2015_2024.csv"
            outage_df = load_and_prepare_data(outage_file, "Outage data (daily, upsampled)",
                                               frequency=frequency, source_frequency='daily')
    else:
        outage_file = DATA_DIR / f"outages/outage_daily_{country}_2015_2024.csv"
        outage_df = load_and_prepare_data(outage_file, "Outage data")

    # Load commodity data (try multiple possible filenames)
    commodity_files = [
        DATA_DIR / "commodities/commodity_prices.csv",
        DATA_DIR / "commodities/commodities_2015_2024.csv",
        DATA_DIR / "commodities/commodities_2015-01-01_2024-12-31.csv",
    ]
    commodity_df = pd.DataFrame()
    for cf in commodity_files:
        if cf.exists():
            commodity_df = load_and_prepare_data(cf, f"Commodity prices ({cf.name})",
                                                  frequency=frequency, source_frequency='daily')
            break
    if commodity_df.empty:
        print("  WARNING: No commodity price data found")

    # Load S&P Global Commodity Insights (SPGCI) data if available
    spgci_files = list((DATA_DIR / "spgci").glob("commodity_timeseries_*.csv")) if (DATA_DIR / "spgci").exists() else []
    spgci_df = pd.DataFrame()
    if spgci_files:
        # Use the most recent file
        spgci_file = sorted(spgci_files)[-1]
        spgci_df = load_and_prepare_data(spgci_file, f"SPGCI data ({spgci_file.name})",
                                            frequency=frequency, source_frequency='daily')

        # Rename SPGCI columns to be more descriptive
        spgci_rename = {
            'TTF_Gas': 'spgci_ttf_gas',
            'API2_Coal': 'spgci_coal',
            'EU_Carbon': 'spgci_carbon',
            'DE_Power': 'spgci_de_power',
            'FR_Power': 'spgci_fr_power'
        }
        spgci_df = spgci_df.rename(columns={c: spgci_rename.get(c, f'spgci_{c.lower()}') for c in spgci_df.columns})
    else:
        print("  NOTE: SPGCI data not found - run download_spgci_data.py to add TTF gas, coal, carbon prices")

    # Load gas storage data (AGSI+ fill levels, injection/withdrawal)
    gas_storage_files = [
        DATA_DIR / f"gas_storage/gas_storage_{country}_2015-01-01_2024-12-31.csv",
        DATA_DIR / f"gas_storage/gas_storage_{country}_2015_2024.csv",
    ]
    gas_storage_df = pd.DataFrame()
    for gsf in gas_storage_files:
        if gsf.exists():
            gas_storage_df = load_and_prepare_data(gsf, f"Gas storage ({gsf.name})",
                                                    frequency=frequency, source_frequency='daily')
            break
    if gas_storage_df.empty:
        print("  NOTE: Gas storage data not found - run download_gas_storage_data.py to add AGSI+ fill levels")

    # Merge datasets
    print("\n  Merging datasets...")

    # Start with base
    unified = base_df.copy()
    print(f"    Base shape: {unified.shape}")

    # Add outage data
    if not outage_df.empty:
        # Prefix outage columns
        outage_df = outage_df.add_prefix('outage_')
        unified = unified.join(outage_df, how='left')
        print(f"    After outage merge: {unified.shape}")

    # Add commodity data
    if not commodity_df.empty:
        # Select relevant columns
        commodity_cols = [c for c in commodity_df.columns if any(
            x in c.lower() for x in ['gas', 'oil', 'coal', 'carbon', 'ttf', 'brent']
        )]
        if commodity_cols:
            commodity_subset = commodity_df[commodity_cols].add_prefix('commodity_')
            unified = unified.join(commodity_subset, how='left')
            print(f"    After commodity merge: {unified.shape}")

    # Add SPGCI data (TTF gas, coal, carbon - European benchmarks)
    if not spgci_df.empty:
        unified = unified.join(spgci_df, how='left')
        print(f"    After SPGCI merge: {unified.shape}")

    # Add gas storage data (AGSI+ fill levels, injection, withdrawal)
    if not gas_storage_df.empty:
        gas_storage_df = gas_storage_df.add_prefix('gas_storage_')
        unified = unified.join(gas_storage_df, how='left')
        print(f"    After gas storage merge: {unified.shape}")

    # Add macro data (EU-wide indicators)
    if not macro_df.empty:
        unified = unified.join(macro_df, how='left')
        print(f"    After macro merge: {unified.shape}")

    # Add sentiment data (VIX, gold, EPU, EUR/USD)
    if not sentiment_df.empty:
        unified = unified.join(sentiment_df, how='left')
        print(f"    After sentiment merge: {unified.shape}")

    # Add oil fundamentals / OPEC data
    if not oil_df.empty:
        unified = unified.join(oil_df, how='left')
        print(f"    After oil fundamentals merge: {unified.shape}")

    # Add transport index data
    if not transport_df.empty:
        unified = unified.join(transport_df, how='left')
        print(f"    After transport merge: {unified.shape}")

    # Add trade data
    if not trade_df.empty:
        unified = unified.join(trade_df, how='left')
        print(f"    After trade merge: {unified.shape}")

    # Add hydrogen data
    if not hydrogen_df.empty:
        unified = unified.join(hydrogen_df, how='left')
        print(f"    After hydrogen merge: {unified.shape}")

    # Load macroeconomic indicators (EU-wide, country-agnostic)
    macro_files = [
        DATA_DIR / "macro/macro_2015-01-01_2024-12-31.csv",
        DATA_DIR / "macro/macro_2015_2024.csv",
    ]
    macro_df = pd.DataFrame()
    for mf in macro_files:
        if mf.exists():
            macro_df = load_and_prepare_data(mf, f"Macro data ({mf.name})",
                                              frequency=frequency, source_frequency='daily')
            break
    if macro_df.empty:
        print("  NOTE: Macro data not found - run download_macro_data.py")

    # Load sentiment indicators (global, country-agnostic)
    sentiment_files = [
        DATA_DIR / "sentiment/sentiment_2015-01-01_2024-12-31.csv",
        DATA_DIR / "sentiment/sentiment_2015_2024.csv",
    ]
    sentiment_df = pd.DataFrame()
    for sf in sentiment_files:
        if sf.exists():
            sentiment_df = load_and_prepare_data(sf, f"Sentiment data ({sf.name})",
                                                  frequency=frequency, source_frequency='daily')
            break
    if sentiment_df.empty:
        print("  NOTE: Sentiment data not found - run download_sentiment_data.py")

    # Load oil fundamentals / OPEC data (global, country-agnostic)
    oil_files = [
        DATA_DIR / "oil_fundamentals/oil_fundamentals_2015-01-01_2024-12-31.csv",
        DATA_DIR / "oil_fundamentals/oil_fundamentals_2015_2024.csv",
    ]
    oil_df = pd.DataFrame()
    for of_ in oil_files:
        if of_.exists():
            oil_df = load_and_prepare_data(of_, f"Oil fundamentals ({of_.name})",
                                            frequency=frequency, source_frequency='daily')
            break
    if oil_df.empty:
        print("  NOTE: Oil fundamentals data not found - run download_oil_fundamentals_data.py")

    # Load transport index data (global, country-agnostic)
    transport_files = [
        DATA_DIR / "transport/transport_2015-01-01_2024-12-31.csv",
        DATA_DIR / "transport/transport_2015_2024.csv",
    ]
    transport_df = pd.DataFrame()
    for tf in transport_files:
        if tf.exists():
            transport_df = load_and_prepare_data(tf, f"Transport data ({tf.name})",
                                                  frequency=frequency, source_frequency='daily')
            break
    if transport_df.empty:
        print("  NOTE: Transport data not found - run download_transport_data.py")

    # Load trade data (EU-wide, country-agnostic)
    trade_files = [
        DATA_DIR / "trade/trade_2015-01-01_2024-12-31.csv",
        DATA_DIR / "trade/trade_2015_2024.csv",
    ]
    trade_df = pd.DataFrame()
    for tdf in trade_files:
        if tdf.exists():
            trade_df = load_and_prepare_data(tdf, f"Trade data ({tdf.name})",
                                              frequency=frequency, source_frequency='daily')
            break
    if trade_df.empty:
        print("  NOTE: Trade data not found - run download_trade_data.py")

    # Load hydrogen data (global, country-agnostic)
    hydrogen_files = [
        DATA_DIR / "hydrogen/hydrogen_2015-01-01_2024-12-31.csv",
        DATA_DIR / "hydrogen/hydrogen_2015_2024.csv",
    ]
    hydrogen_df = pd.DataFrame()
    for hf in hydrogen_files:
        if hf.exists():
            hydrogen_df = load_and_prepare_data(hf, f"Hydrogen data ({hf.name})",
                                                 frequency=frequency, source_frequency='daily')
            break
    if hydrogen_df.empty:
        print("  NOTE: Hydrogen data not found - run download_hydrogen_data.py")

    # Fix FR Wind Offshore placeholder: France had no offshore wind before June 2023
    # ENTSO-E reports a constant ~362 MW placeholder for 2015-2023
    if country.upper() == 'FR':
        offshore_cols = [c for c in unified.columns if 'wind' in c.lower() and 'offshore' in c.lower()]
        for col in offshore_cols:
            mask = unified.index < pd.Timestamp('2023-06-01')
            if mask.any():
                unified.loc[mask, col] = 0.0
                print(f"    Zeroed {col} before 2023-06-01 ({mask.sum()} rows)")

    # Create target variable
    unified = create_target_variable(unified)

    # Find price column for lag/rolling features
    price_cols = [c for c in unified.columns if 'price' in c.lower() and 'lag' not in c.lower()]
    if price_cols:
        # Add lag features for key columns
        key_cols_for_lags = [price_cols[0]]

        # Add load if available
        load_cols = [c for c in unified.columns if 'load' in c.lower() and 'lag' not in c.lower()]
        if load_cols:
            key_cols_for_lags.append(load_cols[0])

        # Add gas storage fill percentage if available
        fill_cols = [c for c in unified.columns if 'gas_storage_fill_pct' in c.lower() and 'lag' not in c.lower()]
        if fill_cols:
            key_cols_for_lags.append(fill_cols[0])

        print(f"\n  Adding lag features for: {key_cols_for_lags}")
        unified = add_lag_features(unified, key_cols_for_lags, frequency=frequency)

        # Add rolling features
        print(f"  Adding rolling features...")
        unified = add_rolling_features(unified, [price_cols[0]], frequency=frequency)

    # Clean up
    print("\n  Final cleanup...")

    # Fill NaN in outage columns with 0
    outage_cols = [c for c in unified.columns if 'outage' in c.lower()]
    for col in outage_cols:
        unified[col] = unified[col].fillna(0)

    # Forward fill commodity, SPGCI, and all new slowly-changing data sources
    ffill_prefixes = ['commodity', 'spgci', 'gas_storage', 'macro_', 'sentiment_',
                      'oil_', 'opec_', 'transport_', 'trade_', 'hydrogen_']
    ffill_cols = [c for c in unified.columns if any(p in c.lower() for p in ffill_prefixes)]
    for col in ffill_cols:
        unified[col] = unified[col].fillna(method='ffill')

    # Report missing values
    missing = unified.isnull().sum()
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) > 0:
        print(f"\n  Columns with missing values:")
        print(cols_with_missing.head(20))

    # Summary statistics
    print(f"\n  Final shape: {unified.shape}")
    print(f"  Date range: {unified.index.min()} to {unified.index.max()}")
    print(f"  Total features: {len(unified.columns)}")

    # Categorize features
    feature_categories = {
        'Price': [c for c in unified.columns if 'price' in c.lower()],
        'Generation': [c for c in unified.columns if 'generation' in c.lower() or 'nuclear' in c.lower() or 'solar' in c.lower() or 'wind' in c.lower()],
        'Load': [c for c in unified.columns if 'load' in c.lower()],
        'Weather': [c for c in unified.columns if 'temp' in c.lower() or 'wind' in c.lower() or 'solar_rad' in c.lower()],
        'Calendar': [c for c in unified.columns if any(x in c for x in ['day_of', 'is_', 'month', 'season', 'week_of', 'dow_', 'year'])],
        'Outage': [c for c in unified.columns if 'outage' in c.lower()],
        'Commodity': [c for c in unified.columns if 'commodity' in c.lower()],
        'SPGCI': [c for c in unified.columns if 'spgci' in c.lower()],  # TTF gas, coal, carbon
        'Gas_Storage': [c for c in unified.columns if 'gas_storage' in c.lower()],
        'Flow': [c for c in unified.columns if 'flow' in c.lower()],
        'Macro': [c for c in unified.columns if c.lower().startswith('macro_')],
        'Sentiment': [c for c in unified.columns if c.lower().startswith('sentiment_')],
        'Oil_Fundamentals': [c for c in unified.columns if c.lower().startswith('oil_') or c.lower().startswith('opec_')],
        'Transport': [c for c in unified.columns if c.lower().startswith('transport_')],
        'Trade': [c for c in unified.columns if c.lower().startswith('trade_')],
        'Hydrogen': [c for c in unified.columns if c.lower().startswith('hydrogen_')],
    }

    print("\n  Feature categories:")
    for cat, cols in feature_categories.items():
        if cols:
            print(f"    {cat}: {len(cols)} features")

    return unified


def create_pair_dataset(country_a: str, country_b: str, frequency: str = 'D') -> pd.DataFrame:
    """
    Create a merged dataset for two countries for spread prediction.

    Args:
        country_a: First country code (e.g., 'DE')
        country_b: Second country code (e.g., 'FR')
        frequency: 'D' for daily (default), 'H' for hourly

    Returns:
        DataFrame with features from both countries and spread target variables.
    """
    freq_label = "hourly" if frequency == 'H' else "daily"
    freq_suffix = '_hourly' if frequency == 'H' else ''
    print(f"\n{'='*60}")
    print(f"Creating {country_a}-{country_b} Merged {freq_label.title()} Dataset for Spread Prediction")
    print(f"{'='*60}")

    # Load clean unified datasets for both countries
    file_a = OUTPUT_DIR / f"unified_{country_a}_2015_2024{freq_suffix}_clean.csv"
    file_b = OUTPUT_DIR / f"unified_{country_b}_2015_2024{freq_suffix}_clean.csv"

    if not file_a.exists() or not file_b.exists():
        print(f"  ERROR: {country_a} and {country_b} unified datasets must exist first.")
        missing = file_a if not file_a.exists() else file_b
        print(f"  Missing: {missing}")
        print("  Run this script without --create_merged first to create them.")
        return pd.DataFrame()

    print(f"  Loading {country_a} dataset...")
    df_a = pd.read_csv(file_a, index_col=0, parse_dates=True)
    print(f"    Shape: {df_a.shape}, Date range: {df_a.index.min()} to {df_a.index.max()}")

    print(f"  Loading {country_b} dataset...")
    df_b = pd.read_csv(file_b, index_col=0, parse_dates=True)
    print(f"    Shape: {df_b.shape}, Date range: {df_b.index.min()} to {df_b.index.max()}")

    # Add country prefixes to all columns (except common calendar features)
    calendar_cols = ['day_of_week', 'day_of_year', 'week_of_year', 'month', 'quarter',
                     'is_weekend', 'is_holiday', 'year', 'season']

    # Identify common columns to keep only once
    common_cols = [c for c in df_a.columns if c in df_b.columns and c in calendar_cols]
    common_data = df_a[common_cols].copy() if common_cols else pd.DataFrame(index=df_a.index)

    # Rename columns with prefixes
    a_cols = {c: f"{country_a}_{c}" for c in df_a.columns if c not in common_cols}
    b_cols = {c: f"{country_b}_{c}" for c in df_b.columns if c not in common_cols}

    df_a_renamed = df_a.drop(columns=common_cols, errors='ignore').rename(columns=a_cols)
    df_b_renamed = df_b.drop(columns=common_cols, errors='ignore').rename(columns=b_cols)

    # Merge on date index (inner join to get only overlapping dates)
    print("\n  Merging datasets...")
    df_merged = df_a_renamed.join(df_b_renamed, how='inner')

    # Add common columns back
    if not common_data.empty:
        for col in common_cols:
            if col in common_data.columns:
                df_merged[col] = common_data.loc[df_merged.index, col]

    print(f"    Merged shape: {df_merged.shape}")
    print(f"    Date range: {df_merged.index.min()} to {df_merged.index.max()}")

    # Create spread target variables
    print("\n  Creating spread target variables...")

    # Find price columns
    a_price_col = f"{country_a}_Day_Ahead_Price" if f"{country_a}_Day_Ahead_Price" in df_merged.columns else None
    b_price_col = f"{country_b}_Day_Ahead_Price" if f"{country_b}_Day_Ahead_Price" in df_merged.columns else None

    if a_price_col is None:
        for col in df_merged.columns:
            if f'{country_a}_Day_Ahead_Price' in col and 'lag' not in col.lower() and 'rolling' not in col.lower():
                a_price_col = col
                break
    if b_price_col is None:
        for col in df_merged.columns:
            if f'{country_b}_Day_Ahead_Price' in col and 'lag' not in col.lower() and 'rolling' not in col.lower():
                b_price_col = col
                break

    if a_price_col and b_price_col:
        print(f"    Using price columns: {a_price_col}, {b_price_col}")

        # Spread = country_a price - country_b price
        df_merged['price_spread'] = df_merged[a_price_col] - df_merged[b_price_col]

        # Change in spread (absolute)
        df_merged['price_spread_change'] = df_merged['price_spread'].diff()

        # Change in spread (percentage) - handle zero spread carefully
        spread_lag = df_merged['price_spread'].shift(1)
        df_merged['price_spread_change_pct'] = np.where(
            spread_lag.abs() > 0.01,
            (df_merged['price_spread'] - spread_lag) / spread_lag.abs() * 100,
            0.0
        )

        # Add lagged spread features
        if frequency == 'H':
            spread_lags = [1, 3, 6, 12, 24, 48, 168]
            suffix = 'h'
        else:
            spread_lags = [1, 2, 3, 7]
            suffix = ''
        for lag in spread_lags:
            df_merged[f'price_spread_lag{lag}{suffix}'] = df_merged['price_spread'].shift(lag)

        # Rolling features for spread
        if frequency == 'H':
            spread_windows = [24, 48, 168]
            unit = 'h'
        else:
            spread_windows = [7, 14]
            unit = 'd'
        for window in spread_windows:
            df_merged[f'price_spread_rolling_mean_{window}{unit}'] = df_merged['price_spread'].rolling(window=window, min_periods=1).mean()
            df_merged[f'price_spread_rolling_std_{window}{unit}'] = df_merged['price_spread'].rolling(window=window, min_periods=1).std()

        print(f"    Created spread features (lags: {spread_lags}, rolling: {spread_windows})")
    else:
        print("    WARNING: Could not find price columns for spread calculation")

    # Summary
    a_features = [c for c in df_merged.columns if c.startswith(f'{country_a}_')]
    b_features = [c for c in df_merged.columns if c.startswith(f'{country_b}_')]
    spread_features = [c for c in df_merged.columns if 'spread' in c.lower()]

    print(f"\n  Final: shape={df_merged.shape}, {country_a}={len(a_features)}, "
          f"{country_b}={len(b_features)}, spread={len(spread_features)}")

    return df_merged


def create_de_fr_merged_dataset(frequency: str = 'D') -> pd.DataFrame:
    """Backward-compatible wrapper for create_pair_dataset('DE', 'FR')."""
    return create_pair_dataset('DE', 'FR', frequency=frequency)


def main():
    """Create unified datasets for all countries and pairs."""
    parser = argparse.ArgumentParser(description='Create unified electricity datasets')
    parser.add_argument('--countries', type=str, default=None,
                        help='Comma-separated country codes (default: all registered)')
    parser.add_argument('--create_merged', action='store_true',
                        help='Also create pair datasets for spread prediction')
    parser.add_argument('--pairs', type=str, default=None,
                        help='Country pairs for spread datasets, e.g. "DE-FR,DE-NL" or "all"')
    parser.add_argument('--only_merged', action='store_true',
                        help='Only create pair datasets (assumes individual datasets exist)')
    parser.add_argument('--frequency', type=str, default='D', choices=['D', 'H'],
                        help='Output frequency: D (daily, default) or H (hourly)')
    args = parser.parse_args()

    frequency = args.frequency
    freq_label = "Hourly" if frequency == 'H' else "Daily"
    freq_suffix = '_hourly' if frequency == 'H' else ''

    countries = args.countries.split(',') if args.countries else get_registered_countries()

    print("=" * 70)
    print(f"Creating {freq_label} Unified Electricity Price Forecasting Datasets")
    print("=" * 70)
    print(f"Countries: {countries}")

    # Create individual country datasets (unless --only_merged)
    if not args.only_merged:
        for country in countries:
            unified = create_unified_dataset(country, frequency=frequency)

            if not unified.empty:
                output_file = OUTPUT_DIR / f"unified_{country}_2015_2024{freq_suffix}.csv"
                unified.to_csv(output_file)
                print(f"\n  Saved to {output_file}")

                unified_clean = unified.dropna(subset=['price_change'])
                clean_file = OUTPUT_DIR / f"unified_{country}_2015_2024{freq_suffix}_clean.csv"
                unified_clean.to_csv(clean_file)
                print(f"  Saved clean version to {clean_file}")
                print(f"  Clean shape: {unified_clean.shape}")

    # Create pair datasets if requested
    if args.create_merged or args.only_merged or args.pairs:
        # Determine which pairs to create
        if args.pairs == 'all':
            pairs = get_all_pairs()
        elif args.pairs:
            pairs = [tuple(p.split('-')) for p in args.pairs.split(',')]
        else:
            # Default: DE-FR for backward compatibility
            pairs = [('DE', 'FR')]

        for country_a, country_b in pairs:
            merged = create_pair_dataset(country_a, country_b, frequency=frequency)

            if not merged.empty:
                pair_label = f"{country_a}_{country_b}"
                output_file = OUTPUT_DIR / f"unified_{pair_label}_2015_2024{freq_suffix}.csv"
                merged.to_csv(output_file)
                print(f"\n  Saved to {output_file}")

                if 'price_spread_change' in merged.columns:
                    merged_clean = merged.dropna(subset=['price_spread_change'])
                    clean_file = OUTPUT_DIR / f"unified_{pair_label}_2015_2024{freq_suffix}_clean.csv"
                    merged_clean.to_csv(clean_file)
                    print(f"  Clean shape: {merged_clean.shape}")

    print("\n" + "=" * 70)
    print("Unified dataset creation complete!")
    print("=" * 70)

    print("\nDataset files:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
