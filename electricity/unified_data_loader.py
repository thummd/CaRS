"""
Unified Data Loader for Electricity Price Forecasting.

Loads the unified datasets (with ENTSO-E, weather, calendar, outages, commodities)
and prepares them for DS3M and FANTOM models.

Supports:
- 'DE': Germany only
- 'FR': France only
- 'DE_FR': Merged DE-FR dataset for spread prediction (matches QRT challenge)

Usage:
import sys
    from unified_data_loader import prepare_unified_ds3m_data, prepare_unified_fantom_data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import UNIFIED_DIR
    # For DS3M
    data = prepare_unified_ds3m_data('DE', timestep=14)
    data = prepare_unified_ds3m_data('DE_FR', timestep=14)  # Spread prediction

    # For FANTOM
    data = prepare_unified_fantom_data('DE', lag=1)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from country_config import get_neighbors, get_registered_countries

# Data directory
UNIFIED_DIR = UNIFIED_DIR

# Feature groups for easy selection (single-country datasets)
FEATURE_GROUPS = {
    'price': ['Day_Ahead_Price', 'price_lag1', 'price_change', 'price_change_pct'],
    'generation': [
        'Biomass', 'Fossil Brown coal/Lignite_Actual Aggregated',
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
    'spgci': ['spgci_ttf_gas', 'spgci_coal', 'spgci_carbon'],  # S&P Global European benchmarks
    'gas_storage': [
        'gas_storage_level_twh', 'gas_storage_fill_pct',
        'gas_storage_injection_gwh', 'gas_storage_withdrawal_gwh',
        'gas_storage_net_flow_gwh', 'gas_storage_trend_pct',
    ],
    'macro': [
        'macro_eu_cpi', 'macro_eu_gdp_growth', 'macro_de_ifo',
        'macro_eu_consumer_confidence', 'macro_eu_energy_hicp',
        'macro_eu_bond_yield', 'macro_eu_m3_money',
    ],
    'sentiment': [
        'sentiment_vix', 'sentiment_gold', 'sentiment_eurusd', 'sentiment_epu_eu',
    ],
    'oil_fundamentals': [
        'oil_us_inventories', 'oil_brent_wti_spread',
        'opec_production', 'opec_spare_capacity', 'opec_basket_price',
    ],
    'transport': ['transport_bdi', 'transport_container_freight'],
    'trade': ['trade_eu_balance', 'trade_eu_energy_imports', 'trade_de_balance'],
    'hydrogen': ['hydrogen_green_cost', 'hydrogen_grey_cost'],
    'flow': ['Flow_to_FR', 'Flow_from_FR', 'Net_Flow_FR'],
    'calendar_hourly': [
        'hour_of_day', 'is_peak_hour', 'hour_sin', 'hour_cos',
        'day_of_week', 'is_weekend', 'month', 'season', 'is_holiday',
        'day_of_year_sin', 'day_of_year_cos', 'dow_sin', 'dow_cos'
    ],
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
    'calendar': [  # Shared calendar features (not prefixed)
        'day_of_week', 'is_weekend', 'month', 'season', 'is_holiday',
        'day_of_year_sin', 'day_of_year_cos', 'dow_sin', 'dow_cos'
    ],
    'spgci': ['spgci_ttf_gas', 'spgci_coal', 'spgci_carbon'],  # Shared European commodities
    'macro': FEATURE_GROUPS['macro'],  # Shared EU-wide macro indicators
    'sentiment': FEATURE_GROUPS['sentiment'],  # Shared global sentiment
    'oil_fundamentals': FEATURE_GROUPS['oil_fundamentals'],  # Shared global oil/OPEC
    'transport': FEATURE_GROUPS['transport'],  # Shared global transport indices
    'trade': FEATURE_GROUPS['trade'],  # Shared EU trade data
    'hydrogen': FEATURE_GROUPS['hydrogen'],  # Shared hydrogen costs
    'gas_storage_de': [f'DE_gas_storage_{c}' for c in [
        'level_twh', 'fill_pct', 'injection_gwh', 'withdrawal_gwh',
        'net_flow_gwh', 'trend_pct',
    ]],
    'gas_storage_fr': [f'FR_gas_storage_{c}' for c in [
        'level_twh', 'fill_pct', 'injection_gwh', 'withdrawal_gwh',
        'net_flow_gwh', 'trend_pct',
    ]],
}


def load_unified_dataset(
    country: str,
    clean: bool = True,
    frequency: str = 'D'
) -> pd.DataFrame:
    """
    Load unified dataset for a country or the merged DE-FR dataset.

    Args:
        country: Country code ('DE', 'FR', 'NL', etc.) or pair ('DE_FR', 'DE_NL', etc.)
        clean: If True, load the clean version (no NaN in target)
        frequency: 'D' for daily (default), 'H' for hourly

    Returns:
        DataFrame with all features
    """
    freq_suffix = '_hourly' if frequency == 'H' else ''
    suffix = '_clean' if clean else ''
    file_path = UNIFIED_DIR / f"unified_{country}_2015_2024{freq_suffix}{suffix}.csv"

    if not file_path.exists():
        is_pair = '_' in country and len(country) <= 5
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            f"Run: python create_unified_dataset.py --countries {country.replace('_', ',')} "
            f"--frequency {'H' if frequency == 'H' else 'D'}"
            + (f" --pairs {country.replace('_', '-')}" if is_pair else "")
        )

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df


def build_feature_groups(df: pd.DataFrame, country: str = None) -> Dict[str, List[str]]:
    """
    Dynamically build feature groups by inspecting DataFrame column names.

    Works for any country or pair dataset without hardcoded column names.

    Args:
        df: DataFrame to inspect
        country: Country code or pair string (e.g., 'DE', 'NL', 'DE_FR')

    Returns:
        Dictionary mapping group names to lists of column names
    """
    cols = list(df.columns)
    groups = {}

    # Target columns to exclude from feature groups
    target_cols = {'price_change', 'price_change_pct', 'price_direction',
                   'Price_Change', 'Price_Return',
                   'price_spread_change', 'price_spread_change_pct'}

    # Detect if this is a pair dataset (columns prefixed with country codes)
    is_pair = country and '_' in country and len(country) <= 5
    if is_pair:
        ca, cb = country.split('_')
    else:
        ca = country

    # Price features
    groups['price'] = [c for c in cols if
                       ('price' in c.lower() or 'da_price' in c.lower())
                       and c not in target_cols]

    # Generation features
    gen_keywords = ['biomass', 'fossil', 'nuclear', 'solar', 'wind', 'hydro', 'geothermal']
    groups['generation'] = [c for c in cols if
                            any(k in c.lower() for k in gen_keywords)
                            and 'actual' in c.lower()]
    # Fallback: any column with generation keywords
    if not groups['generation']:
        groups['generation'] = [c for c in cols if any(k in c.lower() for k in gen_keywords)]

    # Load features
    groups['load'] = [c for c in cols if 'load' in c.lower() and 'lag' not in c.lower()]

    # Weather features
    weather_keywords = ['temperature', 'wind_speed', 'shortwave_radiation', 'direct_radiation',
                        'diffuse_radiation', 'cloud_cover', 'precipitation', 'humidity',
                        'snowfall', 'rain', 'apparent_temperature']
    groups['weather'] = [c for c in cols if any(k in c.lower() for k in weather_keywords)]

    # Calendar features
    calendar_names = {'day_of_week', 'is_weekend', 'month', 'season', 'is_holiday',
                      'is_bridge_day', 'day_of_year', 'week_of_year', 'year',
                      'day_of_year_sin', 'day_of_year_cos', 'dow_sin', 'dow_cos',
                      'month_sin', 'month_cos', 'hour_of_day', 'is_peak_hour',
                      'hour_sin', 'hour_cos'}
    groups['calendar'] = [c for c in cols if c in calendar_names]
    # Also include one-hot encoded day/season columns
    groups['calendar'] += [c for c in cols if c.startswith('dow_') or c.startswith('season_')]
    groups['calendar'] = list(dict.fromkeys(groups['calendar']))  # Deduplicate

    # Outage features
    groups['outage'] = [c for c in cols if 'outage' in c.lower()]

    # Commodity features
    groups['commodity'] = [c for c in cols if 'commodity' in c.lower()]

    # SPGCI features
    groups['spgci'] = [c for c in cols if 'spgci' in c.lower()]

    # Gas storage features
    groups['gas_storage'] = [c for c in cols if 'gas_storage' in c.lower()]

    # Macro features (EU-wide, country-agnostic)
    groups['macro'] = [c for c in cols if c.lower().startswith('macro_')]

    # Sentiment features (global, country-agnostic)
    groups['sentiment'] = [c for c in cols if c.lower().startswith('sentiment_')]

    # Oil fundamentals / OPEC features (global, country-agnostic)
    groups['oil_fundamentals'] = [c for c in cols if
                                   c.lower().startswith('oil_') or c.lower().startswith('opec_')]

    # Transport features (global, country-agnostic)
    groups['transport'] = [c for c in cols if c.lower().startswith('transport_')]

    # Trade features (EU-wide, country-agnostic)
    groups['trade'] = [c for c in cols if c.lower().startswith('trade_')]

    # Hydrogen features (global, country-agnostic)
    groups['hydrogen'] = [c for c in cols if c.lower().startswith('hydrogen_')]

    # Flow features
    groups['flow'] = [c for c in cols if 'flow' in c.lower() and 'gas_storage' not in c.lower()]

    # Spread features (pair datasets only)
    groups['spread'] = [c for c in cols if 'spread' in c.lower()]

    # For pair datasets, also create per-country sub-groups
    if is_pair:
        for prefix, label in [(ca, ca.lower()), (cb, cb.lower())]:
            groups[f'price_{label}'] = [c for c in groups['price'] if c.startswith(f'{prefix}_')]
            groups[f'generation_{label}'] = [c for c in groups['generation'] if c.startswith(f'{prefix}_')]
            groups[f'load_{label}'] = [c for c in groups['load'] if c.startswith(f'{prefix}_')]
            groups[f'weather_{label}'] = [c for c in groups['weather'] if c.startswith(f'{prefix}_')]
            groups[f'gas_storage_{label}'] = [c for c in groups['gas_storage'] if c.startswith(f'{prefix}_')]

    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}

    return groups


def get_feature_columns(
    df: pd.DataFrame,
    groups: Optional[List[str]] = None,
    exclude_target: bool = True,
    country: str = None
) -> List[str]:
    """
    Get feature column names based on group selection.

    Args:
        df: DataFrame with all columns
        groups: List of feature groups to include (None = all available)
        exclude_target: If True, exclude target-related columns
        country: 'DE', 'FR', or 'DE_FR' - helps select correct feature groups

    Returns:
        List of column names
    """
    # Determine which feature group mapping to use
    is_de_fr = country == 'DE_FR' or any(c.startswith('DE_') or c.startswith('FR_') for c in df.columns[:20])

    # Target columns to exclude
    target_cols = ['price_change', 'price_change_pct', 'price_direction',
                   'Price_Change', 'Price_Return',
                   'price_spread_change', 'price_spread_change_pct']

    if groups is None:
        # Use all columns except target
        cols = [c for c in df.columns if c not in target_cols]
    else:
        cols = []
        # Try dynamic feature groups first, fall back to static
        dynamic_groups = build_feature_groups(df, country=country)

        # Also keep static groups as fallback
        static_groups = FEATURE_GROUPS_DE_FR if is_de_fr else FEATURE_GROUPS

        for group in groups:
            matched = False
            # Try dynamic groups first
            if group in dynamic_groups:
                for col in dynamic_groups[group]:
                    if col in df.columns:
                        cols.append(col)
                matched = True
            # Fall back to static groups
            if not matched and group in static_groups:
                for col in static_groups[group]:
                    if col in df.columns:
                        cols.append(col)
            elif not matched and group in FEATURE_GROUPS:
                for col in FEATURE_GROUPS[group]:
                    if col in df.columns:
                        cols.append(col)

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
    """
    Standardize features using training data statistics only.

    Args:
        train_data: Training features (n_train, n_features)
        val_data: Validation features (optional)
        test_data: Test features (optional)

    Returns:
        train_norm, val_norm, test_norm, moments
    """
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

    val_norm = None
    if val_data is not None:
        val_norm = (val_data - moments[:, 0]) / moments[:, 1]

    test_norm = None
    if test_data is not None:
        test_norm = (test_data - moments[:, 0]) / moments[:, 1]

    return train_norm, val_norm, test_norm, moments


def create_temporal_windows(
    X: np.ndarray,
    Y: np.ndarray,
    timestep: int,
    task_type: str = 'prediction'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series tasks.

    Supports two task types:

    task_type='prediction' (forecasting):
        - X[t:t+T], Y[t+1:t+T+1]
        - At position k: X[k] is at time t+k, Y[k] is at time t+k+1
        - Model predicts Y[T] (one step ahead of last X)
        - Use case: "Given today's features and prices, forecast tomorrow's price"

    task_type='estimation' (autoregressive, QRT format):
        - X[t:t+T], Y[t:t+T]
        - At position k: X[k] is at time t+k, Y[k] is at time t+k
        - Model estimates Y[T-1] (same time as last X)
        - Use case: "Given today's features and yesterday's prices, estimate today's price"
        - Note: Model should only use Y[0:T-1] as input to estimate Y[T-1]

    Args:
        X: Feature array (n_samples, n_features)
        Y: Target array (n_samples,) or (n_samples, 1)
        timestep: Window length
        task_type: 'prediction' or 'estimation'

    Returns:
        X_windows: Shape (timestep, n_windows, n_features)
        Y_windows: Shape (timestep, n_windows, 1)
    """
    if task_type not in ['prediction', 'estimation']:
        raise ValueError(f"task_type must be 'prediction' or 'estimation', got '{task_type}'")

    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    n_samples = len(X) - timestep
    n_features = X.shape[1]

    X_win = np.zeros((n_samples, timestep, n_features))
    Y_win = np.zeros((n_samples, timestep, 1))

    for i in range(n_samples):
        X_win[i] = X[i:i + timestep]
        if task_type == 'prediction':
            # Y is one step ahead of X
            Y_win[i] = Y[i + 1:i + timestep + 1]
        else:  # estimation
            # Y is aligned with X (same time)
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
    frequency: str = 'D'
) -> Dict:
    """
    Prepare unified data for DS3M training.

    Performs temporal train/val/test split (no shuffling!).

    Args:
        country: Country code ('DE', 'FR', 'NL', etc.) or pair ('DE_FR', 'DE_NL', etc.)
        timestep: Lookback window size (units match frequency: hours if 'H', days if 'D')
        test_ratio: Fraction for test set
        val_ratio: Fraction for validation set (from remaining after test)
        feature_groups: List of feature groups to include (None = all)
        target_col: Target column name (auto-selected if None)
            - 'price_change_pct' for DE/FR
            - 'price_spread_change_pct' for DE_FR
        handle_outliers: If True, clip extreme target values
        outlier_threshold: Z-score threshold for outlier clipping
        task_type: 'prediction' or 'estimation'
            - 'prediction': Y shifted forward by 1 (forecast next timestep)
            - 'estimation': Y aligned with X (estimate current timestep)
        frequency: 'D' for daily (default), 'H' for hourly

    Returns:
        Dictionary with:
        - trainX, trainY, valX, valY, testX, testY: PyTorch tensors
        - X_moments, Y_moments: Normalization parameters
        - feature_cols: Feature column names
        - timestamps: Original timestamps for each split
        - task_type: The task type used
    """
    # Detect pair dataset
    is_pair = '_' in country and len(country) <= 5

    # Set default target column based on country
    if target_col is None:
        target_col = 'price_spread_change_pct' if is_pair else 'price_change_pct'

    # Load data
    df = load_unified_dataset(country, clean=True, frequency=frequency)

    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Get feature columns
    feature_cols = get_feature_columns(df, feature_groups, exclude_target=True, country=country)

    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    # Extract arrays
    X_all = df[feature_cols].values.astype(np.float32)
    Y_all = df[target_col].values.astype(np.float32)
    timestamps = df.index

    # Handle outliers in target (clip extreme values)
    if handle_outliers:
        Y_mean = np.nanmean(Y_all)
        Y_std = np.nanstd(Y_all)
        lower_bound = Y_mean - outlier_threshold * Y_std
        upper_bound = Y_mean + outlier_threshold * Y_std
        Y_all = np.clip(Y_all, lower_bound, upper_bound)

    # Replace any remaining NaN with 0
    X_all = np.nan_to_num(X_all, nan=0.0)
    Y_all = np.nan_to_num(Y_all, nan=0.0)

    # Temporal split (IMPORTANT: no shuffling!)
    n_total = len(X_all)
    n_test = int(n_total * test_ratio)
    n_val = int((n_total - n_test) * val_ratio)
    n_train = n_total - n_test - n_val

    X_train = X_all[:n_train]
    X_val = X_all[n_train:n_train + n_val]
    X_test = X_all[n_train + n_val:]

    Y_train = Y_all[:n_train]
    Y_val = Y_all[n_train:n_train + n_val]
    Y_test = Y_all[n_train + n_val:]

    ts_train = timestamps[:n_train]
    ts_val = timestamps[n_train:n_train + n_val]
    ts_test = timestamps[n_train + n_val:]

    # Normalize using training data ONLY
    X_train_norm, X_val_norm, X_test_norm, X_moments = normalize_features(
        X_train, X_val, X_test
    )

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

    # Convert to tensors
    trainX = torch.from_numpy(trainX).float()
    trainY = torch.from_numpy(trainY).float()
    valX = torch.from_numpy(valX).float()
    valY = torch.from_numpy(valY).float()
    testX = torch.from_numpy(testX).float()
    testY = torch.from_numpy(testY).float()

    return {
        'trainX': trainX,
        'trainY': trainY,
        'valX': valX,
        'valY': valY,
        'testX': testX,
        'testY': testY,
        'X_moments': X_moments,
        'Y_moments': Y_moments,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'task_type': task_type,
        'frequency': frequency,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'timestamps': {
            'train': ts_train,
            'val': ts_val,
            'test': ts_test
        }
    }


def prepare_unified_fantom_data(
    country: str = 'DE',
    lag: int = 1,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    feature_groups: Optional[List[str]] = None,
    target_col: str = None,
    max_features: int = 20,
    task_type: str = 'estimation',
    frequency: str = 'D'
) -> Dict:
    """
    Prepare unified data for FANTOM training.

    FANTOM expects data in shape [N, lag+1, num_nodes] where:
    - N = number of samples
    - lag+1 = temporal dimensions (lag features + current)
    - num_nodes = number of variables

    Args:
        country: Country code ('DE', 'FR', 'NL', etc.) or pair ('DE_FR', 'DE_NL', etc.)
        lag: Number of lagged timesteps (units match frequency: hours if 'H', days if 'D')
        test_ratio: Fraction for test set
        val_ratio: Fraction for validation
        feature_groups: Feature groups to include
        target_col: Target column name (auto-selected if None)
        max_features: Maximum number of features (for memory)
        task_type: 'estimation' or 'prediction'
            - 'estimation': Estimate Y[t] using X[0:t] (concurrent)
            - 'prediction': Forecast Y[t+1] using X[0:t] (one-step ahead)
        frequency: 'D' for daily (default), 'H' for hourly

    Returns:
        Dictionary with FANTOM-compatible data
    """
    # Detect pair dataset
    is_pair = '_' in country and len(country) <= 5

    # Set default target column based on country
    if target_col is None:
        target_col = 'price_spread_change_pct' if is_pair else 'price_change_pct'

    # Load data
    df = load_unified_dataset(country, clean=True, frequency=frequency)

    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Get feature columns (limited)
    feature_cols = get_feature_columns(df, feature_groups, exclude_target=True, country=country)

    # Limit features if too many
    if len(feature_cols) > max_features - 1:  # -1 for target
        # Prioritize certain features
        priority = ['Day_Ahead_Price', 'Actual Load', 'price_lag1']
        priority_cols = [c for c in priority if c in feature_cols]
        other_cols = [c for c in feature_cols if c not in priority_cols]
        feature_cols = priority_cols + other_cols[:max_features - 1 - len(priority_cols)]

    # Add target column
    all_cols = feature_cols + [target_col]
    target_idx = len(feature_cols)  # Target is last column

    # Extract data
    data = df[all_cols].values.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)
    timestamps = df.index

    # Create lagged data structure
    # For prediction task, we need one extra sample to shift the target
    extra = 1 if task_type == 'prediction' else 0
    n_samples = len(data) - lag - extra
    n_nodes = len(all_cols)

    # Shape: [N, lag+1, num_nodes]
    X = np.zeros((n_samples, lag + 1, n_nodes), dtype=np.float32)
    for i in range(n_samples):
        for l in range(lag + 1):
            X[i, l] = data[i + lag - l]  # lag=0 is current, lag=1 is t-1

    # For prediction task: shift target to be one step ahead
    # X[i, 0, target_idx] should be the target at time t+1 instead of t
    if task_type == 'prediction':
        for i in range(n_samples):
            X[i, 0, target_idx] = data[i + lag + 1, target_idx]  # Target at t+1

    # Temporal split
    n_total = n_samples
    n_test = int(n_total * test_ratio)
    n_val = int((n_total - n_test) * val_ratio)
    n_train = n_total - n_test - n_val

    X_train = X[:n_train]
    X_val = X[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]

    ts_train = timestamps[lag:lag + n_train]
    ts_val = timestamps[lag + n_train:lag + n_train + n_val]
    ts_test = timestamps[lag + n_train + n_val:]

    # Normalize per-node using training data
    moments = np.zeros((n_nodes, 2))
    for j in range(n_nodes):
        train_vals = X_train[:, :, j].flatten()
        moments[j, 0] = np.mean(train_vals)
        moments[j, 1] = np.std(train_vals)
        if moments[j, 1] == 0:
            moments[j, 1] = 1.0

    X_train_norm = (X_train - moments[:, 0]) / moments[:, 1]
    X_val_norm = (X_val - moments[:, 0]) / moments[:, 1]
    X_test_norm = (X_test - moments[:, 0]) / moments[:, 1]

    return {
        'train': torch.from_numpy(X_train_norm).float(),
        'val': torch.from_numpy(X_val_norm).float(),
        'test': torch.from_numpy(X_test_norm).float(),
        'moments': moments,
        'feature_cols': all_cols,
        'target_idx': target_idx,
        'target_col': target_col,
        'n_nodes': n_nodes,
        'lag': lag,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'task_type': task_type,
        'frequency': frequency,
        'timestamps': {
            'train': ts_train,
            'val': ts_val,
            'test': ts_test
        }
    }


def main():
    """Test the unified data loader."""
    print("Testing Unified Data Loader")
    print("=" * 60)

    # Test DS3M data preparation
    print("\n--- DS3M Data (Germany) ---")
    ds3m_data = prepare_unified_ds3m_data(
        country='DE',
        timestep=14,
        feature_groups=['price', 'load', 'weather', 'calendar'],
        target_col='price_change_pct'
    )

    print(f"Features used: {len(ds3m_data['feature_cols'])}")
    print(f"Feature columns: {ds3m_data['feature_cols'][:10]}...")
    print(f"Train X: {ds3m_data['trainX'].shape}")  # (timestep, n_windows, n_features)
    print(f"Train Y: {ds3m_data['trainY'].shape}")  # (timestep, n_windows, 1)
    print(f"Val X:   {ds3m_data['valX'].shape}")
    print(f"Test X:  {ds3m_data['testX'].shape}")
    print(f"\nSplit: train={ds3m_data['n_train']}, val={ds3m_data['n_val']}, test={ds3m_data['n_test']}")
    print(f"Y moments: mean={ds3m_data['Y_moments'][0]:.4f}, std={ds3m_data['Y_moments'][1]:.4f}")
    print(f"\nTrain date range: {ds3m_data['timestamps']['train'].min()} to {ds3m_data['timestamps']['train'].max()}")
    print(f"Test date range:  {ds3m_data['timestamps']['test'].min()} to {ds3m_data['timestamps']['test'].max()}")

    # Test FANTOM data preparation
    print("\n--- FANTOM Data (Germany) ---")
    fantom_data = prepare_unified_fantom_data(
        country='DE',
        lag=1,
        feature_groups=['price', 'load', 'weather'],
        max_features=15
    )

    print(f"Nodes: {fantom_data['n_nodes']}")
    print(f"Features: {fantom_data['feature_cols']}")
    print(f"Target index: {fantom_data['target_idx']}")
    print(f"Train shape: {fantom_data['train'].shape}")  # (N, lag+1, num_nodes)
    print(f"Val shape:   {fantom_data['val'].shape}")
    print(f"Test shape:  {fantom_data['test'].shape}")

    # Test France
    print("\n--- DS3M Data (France) ---")
    ds3m_fr = prepare_unified_ds3m_data(
        country='FR',
        timestep=14,
        feature_groups=None,  # All features
        target_col='price_change_pct'
    )

    print(f"Features used: {len(ds3m_fr['feature_cols'])}")
    print(f"Train X: {ds3m_fr['trainX'].shape}")
    print(f"Test X:  {ds3m_fr['testX'].shape}")

    # Test DE-FR merged (spread prediction)
    print("\n--- DS3M Data (DE-FR Spread) ---")
    try:
        ds3m_de_fr = prepare_unified_ds3m_data(
            country='DE_FR',
            timestep=14,
            feature_groups=['spread', 'calendar'],  # Use DE_FR specific groups
        )
        print(f"Target: {ds3m_de_fr['target_col']}")
        print(f"Features used: {len(ds3m_de_fr['feature_cols'])}")
        print(f"Feature columns: {ds3m_de_fr['feature_cols'][:10]}...")
        print(f"Train X: {ds3m_de_fr['trainX'].shape}")
        print(f"Test X:  {ds3m_de_fr['testX'].shape}")
        print(f"Y moments: mean={ds3m_de_fr['Y_moments'][0]:.4f}, std={ds3m_de_fr['Y_moments'][1]:.4f}")
    except FileNotFoundError as e:
        print(f"DE_FR dataset not found (run create_unified_dataset.py --create_merged first)")
        print(f"  {e}")

    print("\n" + "=" * 60)
    print("Unified Data Loader tests passed!")


if __name__ == "__main__":
    main()
