"""
Unified Data Loader for Shared Backbone Experiments.

Loads the unified datasets (with ENTSO-E, weather, calendar, outages, commodities)
and prepares them for DS3M-Causal models with shared backbone.

Supports:
- 'DE': Germany only
- 'FR': France only
- 'DE_FR': Merged DE-FR dataset for spread prediction

Usage:
import sys
    from data_loader import prepare_unified_ds3m_data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import UNIFIED_DIR
    # For shared backbone experiments
    data = prepare_unified_ds3m_data('DE', timestep=14)
    data = prepare_unified_ds3m_data('DE_FR', timestep=14)  # Spread prediction
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on path for paths import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import UNIFIED_DIR

# Feature groups for easy selection (single-country datasets)
FEATURE_GROUPS = {
    'price': ['Day_Ahead_Price', 'price_lag1', 'price_change', 'price_change_pct'],
    'generation': [
        'Biomass_Actual Aggregated', 'Fossil Brown coal/Lignite_Actual Aggregated',
        'Fossil Gas_Actual Aggregated', 'Fossil Hard coal_Actual Aggregated',
        'Hydro Pumped Storage_Actual Aggregated', 'Nuclear_Actual Aggregated',
        'Solar_Actual Aggregated', 'Wind Onshore_Actual Aggregated',
        'Wind Offshore_Actual Aggregated'
    ],
    'load': ['Actual Load'],
    'weather': [
        'DE_temperature_2m', 'DE_wind_speed_10m', 'DE_wind_speed_100m',
        'DE_shortwave_radiation', 'DE_cloud_cover', 'DE_precipitation'
    ],
    'calendar': [
        'day_of_week', 'is_weekend', 'month', 'season', 'is_holiday',
        'day_of_year_sin', 'day_of_year_cos', 'dow_sin', 'dow_cos'
    ],
    'outage': [
        'outage_total_unavailable_mw', 'outage_planned_unavailable_mw',
        'outage_unplanned_unavailable_mw', 'outage_num_outages'
    ],
    'commodity': ['commodity_natural_gas', 'commodity_brent_oil', 'commodity_wti_oil'],
    'gas_storage': [
        'gas_storage_level_twh', 'gas_storage_fill_pct',
        'gas_storage_injection_gwh', 'gas_storage_withdrawal_gwh',
        'gas_storage_net_flow_gwh', 'gas_storage_trend_pct',
    ],
    'spgci': ['spgci_ttf_gas', 'spgci_coal', 'spgci_carbon'],
    'macro': [
        'macro_eu_cpi', 'macro_eu_gdp_growth', 'macro_de_ifo',
        'macro_eu_consumer_confidence', 'macro_eu_energy_hicp',
        'macro_eu_bond_yield', 'macro_eu_m3_money',
    ],
    'sentiment': ['sentiment_vix', 'sentiment_gold', 'sentiment_eurusd', 'sentiment_epu_eu'],
    'gen_forecast': ['forecast_Solar', 'forecast_Wind Onshore', 'forecast_Wind Offshore', 'gen_forecast_total'],
    'demand_forecast': ['load_forecast', 'load_forecast_Forecasted Load'],
    'imbalance': ['imbalance_Long', 'imbalance_Short', 'imbalance_price'],
    'ntc': [],  # Dynamically matched (ntc_* columns)
    'pipeline': [
        'pipeline_gas_import_gwh', 'pipeline_gas_export_gwh',
        'pipeline_gas_net_import_gwh', 'pipeline_gas_total_flow_gwh',
    ],
    'hydro_reservoir': ['hydro_reservoir_total_mwh'],
    'flow': ['Flow_to_FR', 'Flow_from_FR', 'Net_Flow_FR'],
    'spillover': [],  # Dynamically matched: *_price_lag*h, *_flow_lag*h
}

# Feature groups for DE-FR merged dataset (prefixed with DE_ or FR_)
FEATURE_GROUPS_DE_FR = {
    'price_de': ['DE_Day_Ahead_Price', 'DE_price_lag1', 'DE_price_change', 'DE_price_change_pct'],
    'price_fr': ['FR_Day_Ahead_Price', 'FR_price_lag1', 'FR_price_change', 'FR_price_change_pct'],
    'spread': ['price_spread', 'price_spread_change', 'price_spread_change_pct',
               'price_spread_lag1', 'price_spread_lag2', 'price_spread_lag3', 'price_spread_lag7',
               'price_spread_rolling_mean_7d', 'price_spread_rolling_std_7d',
               'price_spread_rolling_mean_14d', 'price_spread_rolling_std_14d'],
    'load_de': ['DE_Actual Load'],
    'load_fr': ['FR_Actual Load'],
    'generation_de': [f'DE_{g}' for g in FEATURE_GROUPS['generation']],
    'generation_fr': [f'FR_{g}' for g in FEATURE_GROUPS['generation']],
    'weather_de': ['DE_DE_temperature_2m', 'DE_DE_wind_speed_10m', 'DE_DE_wind_speed_100m',
                   'DE_DE_shortwave_radiation', 'DE_DE_cloud_cover', 'DE_DE_precipitation'],
    'weather_fr': ['FR_FR_temperature_2m', 'FR_FR_wind_speed_10m', 'FR_FR_wind_speed_100m',
                   'FR_FR_shortwave_radiation', 'FR_FR_cloud_cover', 'FR_FR_precipitation'],
    'calendar': [
        'day_of_week', 'is_weekend', 'month', 'season', 'is_holiday',
        'day_of_year_sin', 'day_of_year_cos', 'dow_sin', 'dow_cos'
    ],
    'flow': ['DE_Flow_to_FR', 'DE_Flow_from_FR', 'DE_Net_Flow_FR'],
    'commodity': ['DE_commodity_natural_gas', 'DE_commodity_brent_oil', 'DE_commodity_wti_oil'],
}


def load_unified_dataset(
    country: str,
    clean: bool = True,
    date_range: str = '2015_2026',
    frequency: str = 'H'
) -> pd.DataFrame:
    """
    Load unified dataset for a country or country pair.

    Args:
        country: Country code ('DE', 'FR', etc.) or pair ('DE_FR')
        clean: If True, load the clean version (no NaN in target)
        date_range: Dataset date range, e.g. '2015_2026' or '2015_2024'
        frequency: 'H' for hourly (default for extended datasets),
                   'D' for daily, or '' for no frequency suffix
    """
    freq_suffix = '_hourly' if frequency == 'H' else ''
    suffix = '_clean' if clean else ''
    file_path = UNIFIED_DIR / f"unified_{country}_{date_range}{freq_suffix}{suffix}.csv"

    if not file_path.exists():
        # Fallback: try without frequency suffix (old format)
        fallback = UNIFIED_DIR / f"unified_{country}_{date_range}{suffix}.csv"
        if fallback.exists():
            file_path = fallback
        else:
            raise FileNotFoundError(
                f"Dataset not found: {file_path}\n"
                f"Also tried: {fallback}"
            )

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df


ALL_COUNTRIES = ['DE', 'FR', 'NL', 'BE', 'AT', 'IT', 'ES', 'PL', 'DK', 'SE', 'HU', 'CZ']

# Physical interconnections (bidirectional) — used for flow spillover features
INTERCONNECTIONS = {
    'DE': ['FR', 'NL', 'BE', 'AT', 'CZ', 'PL', 'DK', 'SE'],
    'FR': ['DE', 'BE', 'ES', 'IT'],
    'NL': ['DE', 'BE', 'DK'],
    'BE': ['DE', 'FR', 'NL'],
    'AT': ['DE', 'CZ', 'HU', 'IT'],
    'IT': ['FR', 'AT'],
    'ES': ['FR'],
    'PL': ['DE', 'CZ', 'SE'],
    'DK': ['DE', 'NL', 'SE'],
    'SE': ['DE', 'DK', 'PL'],
    'HU': ['AT'],
    'CZ': ['DE', 'AT', 'PL'],
}


def load_resampled_dataset(country: str) -> pd.DataFrame:
    """Load hourly resampled dataset (common period, identical row counts)."""
    path = UNIFIED_DIR / f"unified_{country}_hourly_common.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Resampled dataset not found: {path}\n"
            f"Run: python -m electricity.resample_to_hourly"
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def prepare_spillover_features(
    target_country: str,
    all_countries: Optional[List[str]] = None,
    lag_horizons: Optional[List[int]] = None,
    include_flows: bool = True,
    frequency: str = 'H',
) -> pd.DataFrame:
    """
    Load target country's full features + LAGGED prices and flows from other countries.

    Only uses historical information — no contemporaneous cross-country data.
    This is a forecasting task: at time t, other countries' prices at t are unknown.

    Loads from the Luminus-extended unified hourly datasets and aligns all
    markets to a common date range (intersection of available periods).

    Args:
        target_country: Country to forecast (e.g., 'DE')
        all_countries: All countries to include spillover from
        lag_horizons: Lag horizons in hours for price spillover features
        include_flows: Whether to include lagged cross-border flow features

    Returns:
        DataFrame with domestic features + lagged spillover features
    """
    if all_countries is None:
        all_countries = ALL_COUNTRIES

    # Frequency-aware lag suffix and default horizons.
    # Hourly: lag1h captures next-hour momentum, lag24h captures the
    # same-hour-yesterday diurnal cycle. Daily: lag1d captures
    # day-to-day momentum, lag7d captures the same-day-last-week
    # weekly cycle. Either way the (horizon, suffix) units in the
    # column name match what `shift(lag)` actually applies on the
    # frequency-specific index.
    if frequency == 'D':
        lag_suffix = 'd'
        default_lags = [1, 7]
    else:
        lag_suffix = 'h'
        default_lags = [1, 24]
    if lag_horizons is None:
        lag_horizons = default_lags

    # Load target country's full dataset at the requested frequency
    try:
        df = load_unified_dataset(target_country, clean=True, frequency=frequency)
    except FileNotFoundError:
        # Fallback to old resampled common-period data
        df = load_resampled_dataset(target_country)

    other_countries = [c for c in all_countries if c != target_country]

    # Collect other countries' prices for lagging
    other_prices = {}
    for country in other_countries:
        try:
            other_df = load_unified_dataset(country, clean=True, frequency=frequency)
        except FileNotFoundError:
            try:
                other_df = load_resampled_dataset(country)
            except FileNotFoundError:
                continue

        if 'Day_Ahead_Price' in other_df.columns:
            other_prices[country] = other_df['Day_Ahead_Price']

    # Find common date range across all markets
    common_start = df.index.min()
    common_end = df.index.max()
    for country, price_series in other_prices.items():
        common_start = max(common_start, price_series.index.min())
        common_end = min(common_end, price_series.index.max())

    # Align to common period
    df = df.loc[common_start:common_end]

    # Add lagged price features from other countries (index-aligned).
    # The suffix matches the data frequency so e.g. daily features are
    # *_price_lag1d / *_price_lag7d, hourly features are
    # *_price_lag1h / *_price_lag24h.
    for country, price_series in other_prices.items():
        price_aligned = price_series.reindex(df.index)
        for lag in lag_horizons:
            col_name = f'{country}_price_lag{lag}{lag_suffix}'
            df[col_name] = price_aligned.shift(lag)

    # Add lagged flow features for physically connected countries
    # (uses the same per-frequency lag pair as the price lags).
    if include_flows:
        neighbors = INTERCONNECTIONS.get(target_country, [])
        for neighbor in neighbors:
            flow_to = f'Flow_to_{neighbor}'
            flow_from = f'Flow_from_{neighbor}'

            for flow_col in [flow_to, flow_from]:
                if flow_col in df.columns:
                    for lag in lag_horizons:
                        lag_col = f'{flow_col}_lag{lag}{lag_suffix}'
                        df[lag_col] = df[flow_col].shift(lag)

    # Drop rows with NaN from lag computation
    max_lag = max(lag_horizons)
    df = df.iloc[max_lag:]

    return df


def get_feature_columns(
    df: pd.DataFrame,
    groups: Optional[List[str]] = None,
    exclude_target: bool = True,
    country: str = None
) -> List[str]:
    """Get feature column names based on group selection."""
    is_de_fr = country == 'DE_FR' or any(c.startswith('DE_') or c.startswith('FR_') for c in df.columns[:20])

    target_cols = ['price_change', 'price_change_pct', 'price_direction',
                   'Price_Change', 'Price_Return', 'price_return',
                   'price_spread_change', 'price_spread_change_pct']

    if groups is None:
        cols = [c for c in df.columns if c not in target_cols]
    else:
        cols = []
        feature_groups = FEATURE_GROUPS_DE_FR if is_de_fr else FEATURE_GROUPS

        # Dynamic group matching: maps group name to column-matching patterns
        # Used when static lists don't match (e.g., different country prefixes)
        dynamic_patterns = {
            'weather': ['temperature', 'wind_speed', 'shortwave_radiation',
                        'cloud_cover', 'precipitation', 'humidity'],
            'ntc': ['ntc_'],
            'imbalance': ['imbalance_'],
            'pipeline': ['pipeline_'],
            'hydro_reservoir': ['hydro_'],
            'gen_forecast': ['forecast_', 'gen_forecast'],
            'demand_forecast': ['load_forecast'],
            'flow': ['Flow_to_', 'Flow_from_', 'Net_Flow_'],
            'spillover': ['_price_lag', '_flow_lag'],
        }

        for group in groups:
            n_before = len(cols)
            # Try static list first
            if group in feature_groups:
                for col in feature_groups[group]:
                    if col in df.columns:
                        cols.append(col)

            # Also try dynamic pattern matching (catches country-prefixed columns
            # and columns not in the static list)
            if group in dynamic_patterns:
                patterns = dynamic_patterns[group]
                for col in df.columns:
                    if any(p in col for p in patterns) and col not in target_cols:
                        cols.append(col)

            # Original fallback for DE_FR renaming
            if len(cols) == n_before and group not in dynamic_patterns and group in FEATURE_GROUPS:
                for col in FEATURE_GROUPS[group]:
                    if col in df.columns:
                        cols.append(col)
                    elif col.replace('DE_', 'FR_') in df.columns:
                        cols.append(col.replace('DE_', 'FR_'))

    if exclude_target:
        cols = [c for c in cols if c not in target_cols]

    # Remove duplicates while preserving order
    seen = set()
    unique_cols = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)

    return unique_cols


def normalize_features(
    train_data: np.ndarray,
    val_data: np.ndarray = None,
    test_data: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features using training data statistics only."""
    n_features = train_data.shape[1]
    moments = np.zeros((n_features, 2))

    for i in range(n_features):
        moments[i, 0] = np.nanmean(train_data[:, i])
        moments[i, 1] = np.nanstd(train_data[:, i])
        if moments[i, 1] == 0 or np.isnan(moments[i, 1]):
            moments[i, 1] = 1.0
        if np.isnan(moments[i, 0]):
            moments[i, 0] = 0.0

    train_norm = (train_data - moments[:, 0]) / moments[:, 1]
    val_norm = (val_data - moments[:, 0]) / moments[:, 1] if val_data is not None else None
    test_norm = (test_data - moments[:, 0]) / moments[:, 1] if test_data is not None else None

    return train_norm, val_norm, test_norm, moments


def create_temporal_windows(
    X: np.ndarray,
    Y: np.ndarray,
    timestep: int,
    task_type: str = 'prediction',
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series tasks.

    task_type='prediction': X[t:t+T], Y[t+h:t+T+h] (forecast h steps ahead)
    task_type='estimation': X[t:t+T], Y[t:t+T] (concurrent estimation)

    Args:
        horizon: Forecast horizon in steps (default 1 = next step).
                 h=6 means predict 6 hours ahead at each position.
    """
    assert horizon >= 1, "horizon must be >= 1"
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    # Reduce sample count for h>1 to prevent out-of-bounds access
    n_samples = len(X) - timestep - (horizon - 1)
    n_features = X.shape[1]

    X_win = np.zeros((n_samples, timestep, n_features))
    Y_win = np.zeros((n_samples, timestep, 1))

    for i in range(n_samples):
        X_win[i] = X[i:i + timestep]
        if task_type == 'prediction':
            Y_win[i] = Y[i + horizon:i + timestep + horizon]
        else:
            Y_win[i] = Y[i:i + timestep]

    # Transpose to DS3M format: (timestep, batch, features)
    X_out = np.transpose(X_win, (1, 0, 2))
    Y_out = np.transpose(Y_win, (1, 0, 2))

    return X_out, Y_out


def prepare_unified_ds3m_data(
    country: str = 'DE',
    timestep: int = 14,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    feature_groups: Optional[List[str]] = None,
    target_col: str = None,
    handle_outliers: bool = True,
    outlier_threshold: float = 5.0,
    task_type: str = 'prediction',
    train_end_date: Optional[str] = None,
    test_end_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    resampled: bool = False,
    spillover: bool = False,
    horizon: int = 1,
    frequency: str = 'H',
) -> Dict:
    """
    Prepare unified data for DS3M training.

    Args:
        country: 'DE', 'FR', or 'DE_FR' (for spread prediction)
        timestep: Lookback window size
        test_ratio: Fraction for test set
        val_ratio: Fraction for validation set
        feature_groups: List of feature groups to include
        target_col: Target column name (auto-selected if None)
        handle_outliers: If True, clip extreme target values
        outlier_threshold: Z-score threshold for outlier clipping
        task_type: 'prediction' or 'estimation'
        train_end_date: If set, use date-based split instead of ratio.
            Training ends at this date (exclusive). Format: 'YYYY-MM-DD'.
        test_end_date: If set, test period ends at this date (exclusive).
            If None with train_end_date, uses all remaining data as test.
        resampled: If True, load hourly resampled datasets (common period)
        spillover: If True, include lagged prices + flows from all other countries

    Returns:
        Dictionary with trainX, trainY, testX, testY, etc.
    """
    if target_col is None:
        if '_' in country and len(country) <= 5:
            # Country pair: predict spread
            target_col = 'price_spread_change_pct'
        else:
            target_col = 'price_return'

    if spillover:
        # Load domestic + lagged cross-country features (no contemporaneous leakage)
        df = prepare_spillover_features(country, frequency=frequency)
        df = df.ffill().bfill()
    elif resampled:
        df = load_resampled_dataset(country)
        df = df.ffill().bfill()
    else:
        df = load_unified_dataset(country, clean=True, frequency=frequency)
        df = df.ffill().bfill()

    # Restrict to a structural-break era (e.g. pre/post a regulatory change) so
    # the causal graph is fit on a single mechanism regime. Applied before the
    # return computation so the first in-era return is the only edge effect.
    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index < end_date]

    # Compute hourly price return on the fly if requested
    if target_col == 'price_return' and 'price_return' not in df.columns:
        price_col = 'Day_Ahead_Price'
        if price_col in df.columns:
            # Percentage return with denominator offset to avoid division by zero
            df['price_return'] = df[price_col].diff() / (df[price_col].shift(1).abs() + 1.0)
            df['price_return'] = df['price_return'].fillna(0.0)
            # Replace inf values from near-zero denominator
            df['price_return'] = df['price_return'].replace([np.inf, -np.inf], 0.0)

    feature_cols = get_feature_columns(df, feature_groups, exclude_target=True, country=country)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    X_all = df[feature_cols].values.astype(np.float32)
    Y_all = df[target_col].values.astype(np.float32)
    timestamps = df.index

    if handle_outliers:
        Y_mean = np.nanmean(Y_all)
        Y_std = np.nanstd(Y_all)
        lower_bound = Y_mean - outlier_threshold * Y_std
        upper_bound = Y_mean + outlier_threshold * Y_std
        Y_all = np.clip(Y_all, lower_bound, upper_bound)

    X_all = np.nan_to_num(X_all, nan=0.0)
    Y_all = np.nan_to_num(Y_all, nan=0.0)

    # Temporal split (no shuffling!)
    n_total = len(X_all)

    if train_end_date is not None:
        # Date-based split for rolling-window evaluation
        train_mask = timestamps < train_end_date
        n_train_val = int(train_mask.sum())
        n_val = int(n_train_val * val_ratio)
        n_train = n_train_val - n_val

        if test_end_date is not None:
            test_mask = (timestamps >= train_end_date) & (timestamps < test_end_date)
            n_test = int(test_mask.sum())
        else:
            n_test = n_total - n_train_val
    else:
        # Ratio-based split (default)
        n_test = int(n_total * test_ratio)
        n_val = int((n_total - n_test) * val_ratio)
        n_train = n_total - n_test - n_val

    X_train, X_val, X_test = X_all[:n_train], X_all[n_train:n_train + n_val], X_all[n_train + n_val:n_train + n_val + n_test]
    Y_train, Y_val, Y_test = Y_all[:n_train], Y_all[n_train:n_train + n_val], Y_all[n_train + n_val:n_train + n_val + n_test]
    ts_train = timestamps[:n_train]
    ts_val = timestamps[n_train:n_train + n_val]
    ts_test = timestamps[n_train + n_val:n_train + n_val + n_test]

    # Normalize using training data ONLY
    X_train_norm, X_val_norm, X_test_norm, X_moments = normalize_features(X_train, X_val, X_test)

    Y_moments = np.array([Y_train.mean(), Y_train.std()])
    if Y_moments[1] == 0:
        Y_moments[1] = 1.0
    Y_train_norm = (Y_train - Y_moments[0]) / Y_moments[1]
    Y_val_norm = (Y_val - Y_moments[0]) / Y_moments[1]
    Y_test_norm = (Y_test - Y_moments[0]) / Y_moments[1]

    # Create windows
    trainX, trainY = create_temporal_windows(X_train_norm, Y_train_norm, timestep, task_type, horizon)
    valX, valY = create_temporal_windows(X_val_norm, Y_val_norm, timestep, task_type, horizon)
    testX, testY = create_temporal_windows(X_test_norm, Y_test_norm, timestep, task_type, horizon)

    return {
        'trainX': torch.from_numpy(trainX).float(),
        'trainY': torch.from_numpy(trainY).float(),
        'valX': torch.from_numpy(valX).float(),
        'valY': torch.from_numpy(valY).float(),
        'testX': torch.from_numpy(testX).float(),
        'testY': torch.from_numpy(testY).float(),
        'X_moments': X_moments,
        'Y_moments': Y_moments,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'task_type': task_type,
        'horizon': horizon,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'timestamps': {'train': ts_train, 'val': ts_val, 'test': ts_test}
    }


if __name__ == "__main__":
    print("Testing Shared Backbone Data Loader")
    print("=" * 60)

    for country in ['DE', 'FR', 'DE_FR']:
        try:
            print(f"\n--- {country} ---")
            data = prepare_unified_ds3m_data(
                country=country,
                timestep=14,
                feature_groups=['price', 'load', 'weather', 'calendar'],
            )
            print(f"Features: {len(data['feature_cols'])}")
            print(f"Train: {data['trainX'].shape}, Test: {data['testX'].shape}")
            print(f"Target: {data['target_col']}")
        except FileNotFoundError as e:
            print(f"  Dataset not found: {e}")

    print("\nData loader test passed!")
