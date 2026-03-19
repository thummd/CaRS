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
    'flow': ['Flow_to_FR', 'Flow_from_FR', 'Net_Flow_FR']
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


def get_feature_columns(
    df: pd.DataFrame,
    groups: Optional[List[str]] = None,
    exclude_target: bool = True,
    country: str = None
) -> List[str]:
    """Get feature column names based on group selection."""
    is_de_fr = country == 'DE_FR' or any(c.startswith('DE_') or c.startswith('FR_') for c in df.columns[:20])

    target_cols = ['price_change', 'price_change_pct', 'price_direction',
                   'Price_Change', 'Price_Return',
                   'price_spread_change', 'price_spread_change_pct']

    if groups is None:
        cols = [c for c in df.columns if c not in target_cols]
    else:
        cols = []
        feature_groups = FEATURE_GROUPS_DE_FR if is_de_fr else FEATURE_GROUPS

        for group in groups:
            if group in feature_groups:
                for col in feature_groups[group]:
                    if col in df.columns:
                        cols.append(col)
            elif group in FEATURE_GROUPS:
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
    task_type: str = 'prediction'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series tasks.

    task_type='prediction': X[t:t+T], Y[t+1:t+T+1] (forecast)
    task_type='estimation': X[t:t+T], Y[t:t+T] (concurrent estimation)
    """
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    n_samples = len(X) - timestep
    n_features = X.shape[1]

    X_win = np.zeros((n_samples, timestep, n_features))
    Y_win = np.zeros((n_samples, timestep, 1))

    for i in range(n_samples):
        X_win[i] = X[i:i + timestep]
        if task_type == 'prediction':
            Y_win[i] = Y[i + 1:i + timestep + 1]
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
    test_end_date: Optional[str] = None
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

    Returns:
        Dictionary with trainX, trainY, testX, testY, etc.
    """
    if target_col is None:
        if '_' in country and len(country) <= 5:
            # Country pair: predict spread
            target_col = 'price_spread_change_pct'
        else:
            target_col = 'Day_Ahead_Price'

    df = load_unified_dataset(country, clean=True)
    df = df.ffill().bfill()

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
    trainX, trainY = create_temporal_windows(X_train_norm, Y_train_norm, timestep, task_type)
    valX, valY = create_temporal_windows(X_val_norm, Y_val_norm, timestep, task_type)
    testX, testY = create_temporal_windows(X_test_norm, Y_test_norm, timestep, task_type)

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
