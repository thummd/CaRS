"""
Feature Engineering for Spread Prediction

This module provides enhanced feature engineering specifically for DE-FR spread prediction.
The spread (DE price - FR price) has different dynamics than individual country prices
and requires features that capture cross-market relationships.

Key insight: Univariate models fail on spread because spread prediction inherently
requires understanding relationships between DE and FR markets.

Feature categories:
1. Spread-specific: Direct spread history and derivatives
2. Cross-market: Relationships between DE and FR (ratios, differences)
3. Flow-based: Cross-border electricity flows
4. Fundamental drivers: Generation mix differences, load differences
5. Convergence indicators: Features indicating price convergence/divergence
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import UNIFIED_DIR

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


def add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive spread-specific features.

    Args:
        df: DataFrame with DE_ and FR_ prefixed columns

    Returns:
        DataFrame with additional spread features
    """
    df = df.copy()

    # === 1. Basic Spread Features ===
    if 'price_spread' not in df.columns:
        if 'DE_Day_Ahead_Price' in df.columns and 'FR_Day_Ahead_Price' in df.columns:
            df['price_spread'] = df['DE_Day_Ahead_Price'] - df['FR_Day_Ahead_Price']

    # Spread changes
    if 'price_spread' in df.columns:
        df['price_spread_change'] = df['price_spread'].diff()
        df['price_spread_change_pct'] = df['price_spread'].pct_change() * 100

        # Lagged spread
        for lag in [1, 2, 3, 5, 7, 14]:
            df[f'price_spread_lag{lag}'] = df['price_spread'].shift(lag)

        # Rolling statistics
        for window in [3, 7, 14, 30]:
            df[f'price_spread_ma{window}'] = df['price_spread'].rolling(window).mean()
            df[f'price_spread_std{window}'] = df['price_spread'].rolling(window).std()
            df[f'price_spread_min{window}'] = df['price_spread'].rolling(window).min()
            df[f'price_spread_max{window}'] = df['price_spread'].rolling(window).max()

        # Mean reversion indicator
        df['spread_z_score'] = (
            (df['price_spread'] - df['price_spread_ma14']) /
            (df['price_spread_std14'] + 1e-6)
        )

        # Spread momentum
        df['spread_momentum_3d'] = df['price_spread'] - df['price_spread_lag3']
        df['spread_momentum_7d'] = df['price_spread'] - df['price_spread_lag7']

    return df


def add_cross_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features capturing relationships between DE and FR markets.

    Args:
        df: DataFrame with DE_ and FR_ prefixed columns

    Returns:
        DataFrame with cross-market features
    """
    df = df.copy()

    # === 2. Price Relationship Features ===
    if 'DE_Day_Ahead_Price' in df.columns and 'FR_Day_Ahead_Price' in df.columns:
        # Price ratio (captures relative pricing)
        df['price_ratio_de_fr'] = df['DE_Day_Ahead_Price'] / (df['FR_Day_Ahead_Price'] + 1e-6)

        # Log price difference
        df['log_price_diff'] = np.log(df['DE_Day_Ahead_Price'] + 100) - np.log(df['FR_Day_Ahead_Price'] + 100)

        # Price correlation (rolling)
        df['price_corr_7d'] = df['DE_Day_Ahead_Price'].rolling(7).corr(df['FR_Day_Ahead_Price'])
        df['price_corr_30d'] = df['DE_Day_Ahead_Price'].rolling(30).corr(df['FR_Day_Ahead_Price'])

        # Price change correlation
        de_change = df['DE_Day_Ahead_Price'].diff()
        fr_change = df['FR_Day_Ahead_Price'].diff()
        df['price_change_corr_7d'] = de_change.rolling(7).corr(fr_change)

        # Lead-lag indicators
        df['de_leads_fr'] = df['DE_Day_Ahead_Price'].shift(1) - df['FR_Day_Ahead_Price']
        df['fr_leads_de'] = df['FR_Day_Ahead_Price'].shift(1) - df['DE_Day_Ahead_Price']

    # === 3. Load Relationship Features ===
    if 'DE_Actual Load' in df.columns and 'FR_Actual Load' in df.columns:
        df['load_ratio_de_fr'] = df['DE_Actual Load'] / (df['FR_Actual Load'] + 1e-6)
        df['load_diff_de_fr'] = df['DE_Actual Load'] - df['FR_Actual Load']

        # Relative load change
        de_load_change = df['DE_Actual Load'].pct_change()
        fr_load_change = df['FR_Actual Load'].pct_change()
        df['load_change_diff'] = de_load_change - fr_load_change

    return df


def add_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-border flow features.

    Cross-border flows are critical for spread prediction as they:
    - Indicate price convergence pressure
    - Reflect market coupling efficiency
    - Signal supply/demand imbalances

    Args:
        df: DataFrame with flow columns

    Returns:
        DataFrame with flow features
    """
    df = df.copy()

    # Find flow columns
    flow_cols = [c for c in df.columns if 'Flow' in c or 'flow' in c or 'EXCHANGE' in c.upper()]

    for col in flow_cols:
        # Lagged flows
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag7'] = df[col].shift(7)

        # Flow momentum
        df[f'{col}_momentum'] = df[col] - df[col].shift(1)

        # Rolling average
        df[f'{col}_ma7'] = df[col].rolling(7).mean()

    # Net flow from DE to FR perspective
    if 'DE_Flow_to_FR' in df.columns and 'DE_Flow_from_FR' in df.columns:
        df['net_flow_de_to_fr'] = df['DE_Flow_to_FR'] - df['DE_Flow_from_FR']
        df['net_flow_de_to_fr_lag1'] = df['net_flow_de_to_fr'].shift(1)

        # Flow direction indicator
        df['flow_direction'] = np.sign(df['net_flow_de_to_fr'])

        # Flow as percentage of load
        if 'DE_Actual Load' in df.columns:
            df['flow_pct_de_load'] = df['net_flow_de_to_fr'] / (df['DE_Actual Load'] + 1e-6) * 100

    return df


def add_generation_mix_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add generation mix difference features.

    Different generation mixes (e.g., DE has more renewables, FR has more nuclear)
    create different marginal cost structures that affect spread dynamics.

    Args:
        df: DataFrame with generation columns

    Returns:
        DataFrame with generation mix features
    """
    df = df.copy()

    # Generation type differences
    gen_types = ['Nuclear', 'Solar', 'Wind Onshore', 'Fossil Gas', 'Hydro']

    for gen_type in gen_types:
        de_col = [c for c in df.columns if c.startswith('DE_') and gen_type in c and 'Actual Aggregated' in c]
        fr_col = [c for c in df.columns if c.startswith('FR_') and gen_type in c and 'Actual Aggregated' in c]

        if de_col and fr_col:
            de_col = de_col[0]
            fr_col = fr_col[0]
            gen_name = gen_type.replace(' ', '_').lower()

            # Absolute difference
            df[f'{gen_name}_diff'] = df[de_col] - df[fr_col]

            # Ratio
            df[f'{gen_name}_ratio'] = df[de_col] / (df[fr_col] + 1e-6)

    # Renewable share difference
    de_renewable_cols = [c for c in df.columns if c.startswith('DE_') and
                         any(r in c for r in ['Solar', 'Wind', 'Hydro']) and 'Actual Aggregated' in c]
    fr_renewable_cols = [c for c in df.columns if c.startswith('FR_') and
                         any(r in c for r in ['Solar', 'Wind', 'Hydro']) and 'Actual Aggregated' in c]

    if de_renewable_cols and fr_renewable_cols and 'DE_Actual Load' in df.columns and 'FR_Actual Load' in df.columns:
        de_renewable = df[de_renewable_cols].sum(axis=1)
        fr_renewable = df[fr_renewable_cols].sum(axis=1)

        df['renewable_share_de'] = de_renewable / (df['DE_Actual Load'] + 1e-6)
        df['renewable_share_fr'] = fr_renewable / (df['FR_Actual Load'] + 1e-6)
        df['renewable_share_diff'] = df['renewable_share_de'] - df['renewable_share_fr']

    return df


def add_convergence_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add indicators for price convergence/divergence.

    These features help predict whether prices will converge (spread decreases)
    or diverge (spread increases).

    Args:
        df: DataFrame with spread and flow features

    Returns:
        DataFrame with convergence indicators
    """
    df = df.copy()

    if 'price_spread' in df.columns:
        # Spread regime indicators
        spread_median = df['price_spread'].rolling(30, min_periods=7).median()
        df['spread_above_median'] = (df['price_spread'] > spread_median).astype(int)

        # Spread percentile (historical)
        df['spread_percentile'] = df['price_spread'].rolling(90, min_periods=30).apply(
            lambda x: (x.iloc[-1] > x).mean() if len(x) > 1 else 0.5
        )

        # Mean reversion pressure
        if 'price_spread_ma14' in df.columns and 'price_spread_std14' in df.columns:
            df['mean_reversion_signal'] = -(df['price_spread'] - df['price_spread_ma14']) / (df['price_spread_std14'] + 1e-6)

        # Extreme spread indicator
        if 'price_spread_std30' in df.columns:
            df['extreme_spread'] = (abs(df['spread_z_score']) > 2).astype(int)

    # Flow-based convergence signal
    if 'net_flow_de_to_fr' in df.columns and 'price_spread' in df.columns:
        # High flow + positive spread suggests convergence pressure
        df['convergence_pressure'] = df['net_flow_de_to_fr'] * np.sign(df['price_spread'])

    return df


def add_weather_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather differential features between DE and FR.

    Weather differences affect renewable generation and thus spread dynamics.

    Args:
        df: DataFrame with weather columns

    Returns:
        DataFrame with weather differential features
    """
    df = df.copy()

    weather_vars = ['temperature_2m', 'wind_speed_10m', 'wind_speed_100m',
                    'shortwave_radiation', 'cloud_cover']

    for var in weather_vars:
        de_col = [c for c in df.columns if 'DE_DE_' in c and var in c]
        fr_col = [c for c in df.columns if 'FR_FR_' in c and var in c]

        if de_col and fr_col:
            de_col = de_col[0]
            fr_col = fr_col[0]
            var_name = var.replace('_', '')

            df[f'{var_name}_diff'] = df[de_col] - df[fr_col]
            df[f'{var_name}_ratio'] = df[de_col] / (df[fr_col] + 1e-6)

    return df


def engineer_spread_features(
    df: pd.DataFrame,
    include_groups: List[str] = None
) -> pd.DataFrame:
    """
    Apply comprehensive feature engineering for spread prediction.

    Args:
        df: Raw unified DE-FR DataFrame
        include_groups: List of feature groups to include
            Options: 'spread', 'cross_market', 'flow', 'generation', 'convergence', 'weather'
            Default: all groups

    Returns:
        DataFrame with engineered features
    """
    if include_groups is None:
        include_groups = ['spread', 'cross_market', 'flow', 'generation', 'convergence', 'weather']

    if 'spread' in include_groups:
        df = add_spread_features(df)

    if 'cross_market' in include_groups:
        df = add_cross_market_features(df)

    if 'flow' in include_groups:
        df = add_flow_features(df)

    if 'generation' in include_groups:
        df = add_generation_mix_features(df)

    if 'convergence' in include_groups:
        df = add_convergence_indicators(df)

    if 'weather' in include_groups:
        df = add_weather_differential_features(df)

    return df


def get_spread_feature_groups() -> dict:
    """
    Get organized feature groups for spread prediction.

    Returns:
        Dictionary mapping group names to feature patterns
    """
    return {
        'spread_basic': [
            'price_spread', 'price_spread_change', 'price_spread_change_pct',
            'price_spread_lag1', 'price_spread_lag2', 'price_spread_lag3',
            'price_spread_lag5', 'price_spread_lag7'
        ],
        'spread_stats': [
            'price_spread_ma3', 'price_spread_ma7', 'price_spread_ma14',
            'price_spread_std3', 'price_spread_std7', 'price_spread_std14',
            'spread_z_score', 'spread_momentum_3d', 'spread_momentum_7d'
        ],
        'cross_market': [
            'price_ratio_de_fr', 'log_price_diff',
            'price_corr_7d', 'price_corr_30d', 'price_change_corr_7d',
            'de_leads_fr', 'fr_leads_de',
            'load_ratio_de_fr', 'load_diff_de_fr', 'load_change_diff'
        ],
        'flow': [
            'net_flow_de_to_fr', 'net_flow_de_to_fr_lag1',
            'flow_direction', 'flow_pct_de_load'
        ],
        'generation_diff': [
            'nuclear_diff', 'solar_diff', 'wind_onshore_diff', 'fossil_gas_diff',
            'renewable_share_diff'
        ],
        'convergence': [
            'spread_above_median', 'spread_percentile',
            'mean_reversion_signal', 'extreme_spread', 'convergence_pressure'
        ],
        'weather_diff': [
            'temperature2m_diff', 'windspeed10m_diff', 'windspeed100m_diff',
            'shortwaveradiation_diff', 'cloudcover_diff'
        ],
        'calendar': [
            'day_of_week', 'is_weekend', 'month', 'season', 'is_holiday'
        ],
        'commodity': [
            'spgci_ttf_gas', 'spgci_coal', 'spgci_carbon'
        ]
    }


def select_features_for_model(
    df: pd.DataFrame,
    groups: List[str],
    max_features: int = 50
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features based on specified groups with optional limit.

    Args:
        df: DataFrame with all features
        groups: List of feature groups to include
        max_features: Maximum number of features to select

    Returns:
        Tuple of (selected DataFrame, list of column names)
    """
    feature_groups = get_spread_feature_groups()

    selected_cols = []
    for group in groups:
        if group in feature_groups:
            for pattern in feature_groups[group]:
                matching = [c for c in df.columns if pattern in c or c == pattern]
                selected_cols.extend(matching)

    # Remove duplicates while preserving order
    seen = set()
    selected_cols = [c for c in selected_cols if not (c in seen or seen.add(c))]

    # Filter to existing columns
    selected_cols = [c for c in selected_cols if c in df.columns]

    # Limit features
    if len(selected_cols) > max_features:
        selected_cols = selected_cols[:max_features]

    return df[selected_cols], selected_cols


if __name__ == "__main__":
    # Test feature engineering
    from pathlib import Path

    data_path = UNIFIED_DIR / "unified_DE_FR_2015_2024_clean.csv"
    if data_path.exists():
        print("Loading DE-FR data...")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"Original shape: {df.shape}")

        print("\nEngineering features...")
        df_eng = engineer_spread_features(df)
        print(f"After engineering: {df_eng.shape}")

        print("\nFeature groups:")
        for group, patterns in get_spread_feature_groups().items():
            matching = [c for c in df_eng.columns if any(p in c for p in patterns)]
            print(f"  {group}: {len(matching)} features")

        print("\nSample of new features:")
        new_cols = [c for c in df_eng.columns if c not in df.columns]
        print(f"  Added {len(new_cols)} new features")
        for col in new_cols[:10]:
            print(f"    {col}: {df_eng[col].dropna().head(3).tolist()}")
    else:
        print(f"Data file not found: {data_path}")
