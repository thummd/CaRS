"""
Data loader for electricity price prediction challenge.
Handles loading, preprocessing, and transforming data for FANTOM.
"""

import sys
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR
DATA_DIR = DATA_DIR / "qrt"

# Feature groups
COMMODITY_FEATURES = ['GAS_RET', 'COAL_RET', 'CARBON_RET']

DE_FEATURES = [
    'DE_CONSUMPTION', 'DE_GAS', 'DE_COAL', 'DE_HYDRO', 'DE_NUCLEAR',
    'DE_SOLAR', 'DE_WINDPOW', 'DE_LIGNITE', 'DE_RESIDUAL_LOAD',
    'DE_RAIN', 'DE_WIND', 'DE_TEMP', 'DE_NET_EXPORT', 'DE_NET_IMPORT'
]

FR_FEATURES = [
    'FR_CONSUMPTION', 'FR_GAS', 'FR_COAL', 'FR_HYDRO', 'FR_NUCLEAR',
    'FR_SOLAR', 'FR_WINDPOW', 'FR_RESIDUAL_LOAD',
    'FR_RAIN', 'FR_WIND', 'FR_TEMP', 'FR_NET_EXPORT', 'FR_NET_IMPORT'
]

EXCHANGE_FEATURES = ['DE_FR_EXCHANGE', 'FR_DE_EXCHANGE']


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw training data."""
    X_train = pd.read_csv(DATA_DIR / "X_train_NHkHMNU.csv")
    Y_train = pd.read_csv(DATA_DIR / "y_train_ZAN5mwg.csv")
    return X_train, Y_train


def merge_data(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """Merge features and target on ID."""
    return X.merge(Y, on='ID')


def impute_missing(df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values.

    Args:
        df: DataFrame with potential NaN values
        method: 'mean', 'median', 'ffill', or 'drop'

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if method == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif method == 'ffill':
        # Sort by DAY_ID first for forward fill to make sense
        df = df.sort_values('DAY_ID')
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        # Fill any remaining NaN at the start with mean
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == 'drop':
        df = df.dropna()
    elif method == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    return df


def get_feature_columns(country: Optional[str] = None) -> List[str]:
    """
    Get feature columns for a specific country or all.

    Args:
        country: 'DE', 'FR', or None for all features

    Returns:
        List of feature column names
    """
    if country == 'DE':
        return DE_FEATURES + COMMODITY_FEATURES + EXCHANGE_FEATURES
    elif country == 'FR':
        return FR_FEATURES + COMMODITY_FEATURES + EXCHANGE_FEATURES
    else:
        # All features for joint model
        return DE_FEATURES + FR_FEATURES + COMMODITY_FEATURES + EXCHANGE_FEATURES


def prepare_country_data(
    df: pd.DataFrame,
    country: str,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter and prepare data for a specific country.

    Args:
        df: Full merged DataFrame
        country: 'DE' or 'FR'
        feature_cols: Specific features to use (default: country-specific)

    Returns:
        Filtered DataFrame for that country
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(country)

    country_df = df[df['COUNTRY'] == country].copy()
    country_df = country_df.sort_values('DAY_ID').reset_index(drop=True)

    return country_df


def create_temporal_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    lag: int = 1,
    include_target: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create temporal windows for FANTOM input.

    FANTOM expects data of shape [N, lag+1, num_nodes].
    We create windows where:
    - X[i, 0, :] = features at time t-lag
    - X[i, 1, :] = features at time t-lag+1
    - ...
    - X[i, lag, :] = features at time t (current)

    Args:
        df: DataFrame sorted by time (DAY_ID)
        feature_cols: Columns to include as features
        lag: Number of lagged time steps
        include_target: Whether to include TARGET as the last node

    Returns:
        X: Temporal windows of shape [N-lag, lag+1, num_features]
        target: Target values of shape [N-lag]
        ids: Row IDs of shape [N-lag]
    """
    # Select features
    cols = feature_cols.copy()
    if include_target and 'TARGET' not in cols:
        cols.append('TARGET')

    data = df[cols].values
    ids_all = df['ID'].values

    n_samples = len(df) - lag
    n_features = len(cols)

    X = np.zeros((n_samples, lag + 1, n_features))

    for i in range(n_samples):
        for l in range(lag + 1):
            X[i, l, :] = data[i + l]

    # Target is the last feature at the current time step
    if include_target:
        target_idx = cols.index('TARGET')
        target = X[:, -1, target_idx]
    else:
        target = df['TARGET'].values[lag:]

    ids = ids_all[lag:]

    return X, target, ids


def create_instantaneous_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    include_target: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create data without temporal structure (single time point).
    This treats each observation independently.

    For FANTOM, we still need shape [N, lag+1, num_nodes] so we use lag=0
    effectively giving [N, 1, num_nodes].

    Args:
        df: DataFrame
        feature_cols: Columns to include as features
        include_target: Whether to include TARGET as the last node

    Returns:
        X: Data of shape [N, 1, num_features]
        target: Target values of shape [N]
        ids: Row IDs of shape [N]
    """
    cols = feature_cols.copy()
    if include_target and 'TARGET' not in cols:
        cols.append('TARGET')

    data = df[cols].values
    X = data.reshape(len(df), 1, len(cols))

    target = df['TARGET'].values
    ids = df['ID'].values

    return X, target, ids


def standardize_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize data to zero mean and unit variance.

    Args:
        X: Data of shape [N, lag+1, num_features]

    Returns:
        X_std: Standardized data
        mean: Feature means
        std: Feature standard deviations
    """
    # Compute mean and std across samples and time
    mean = X.mean(axis=(0, 1))
    std = X.std(axis=(0, 1))
    std[std == 0] = 1  # Avoid division by zero

    X_std = (X - mean) / std

    return X_std, mean, std


def get_target_index(feature_cols: List[str], include_target: bool = True) -> int:
    """Get the index of TARGET in the feature list."""
    if include_target:
        cols = feature_cols.copy()
        if 'TARGET' not in cols:
            cols.append('TARGET')
        return cols.index('TARGET')
    return -1


class ElectricityDataset:
    """
    Dataset class for electricity price data.
    Handles preprocessing and provides data in FANTOM-compatible format.
    """

    def __init__(
        self,
        country: Optional[str] = None,
        lag: int = 1,
        imputation: str = 'mean',
        use_temporal: bool = True,
        standardize: bool = True
    ):
        """
        Initialize dataset.

        Args:
            country: 'DE', 'FR', or None for joint model
            lag: Temporal lag (0 for instantaneous)
            imputation: Method for handling missing values
            use_temporal: Whether to use temporal windows
            standardize: Whether to standardize features
        """
        self.country = country
        self.lag = lag
        self.imputation = imputation
        self.use_temporal = use_temporal
        self.standardize = standardize

        self._load_and_prepare()

    def _load_and_prepare(self):
        """Load and preprocess data."""
        # Load raw data
        X_raw, Y_raw = load_raw_data()
        df = merge_data(X_raw, Y_raw)

        # Impute missing values
        df = impute_missing(df, method=self.imputation)

        # Get feature columns
        self.feature_cols = get_feature_columns(self.country)

        # Filter by country if specified
        if self.country is not None:
            df = prepare_country_data(df, self.country, self.feature_cols)
        else:
            df = df.sort_values('DAY_ID').reset_index(drop=True)

        self.df = df

        # Create temporal or instantaneous data
        if self.use_temporal and self.lag > 0:
            self.X, self.target, self.ids = create_temporal_windows(
                df, self.feature_cols, self.lag, include_target=True
            )
        else:
            self.X, self.target, self.ids = create_instantaneous_data(
                df, self.feature_cols, include_target=True
            )

        # Standardize
        if self.standardize:
            self.X, self.mean, self.std = standardize_data(self.X)
        else:
            self.mean = None
            self.std = None

        # Get target index
        cols = self.feature_cols.copy()
        if 'TARGET' not in cols:
            cols.append('TARGET')
        self.target_idx = cols.index('TARGET')
        self.num_features = len(cols)

    def get_fantom_data(self) -> np.ndarray:
        """Get data in FANTOM format [N, lag+1, num_nodes]."""
        return self.X

    def get_target(self) -> np.ndarray:
        """Get target values."""
        return self.target

    def get_ids(self) -> np.ndarray:
        """Get sample IDs."""
        return self.ids

    def get_num_nodes(self) -> int:
        """Get number of nodes (features) for FANTOM."""
        return self.num_features

    def get_target_idx(self) -> int:
        """Get index of TARGET variable in node list."""
        return self.target_idx

    def get_feature_names(self) -> List[str]:
        """Get list of feature names including TARGET."""
        cols = self.feature_cols.copy()
        if 'TARGET' not in cols:
            cols.append('TARGET')
        return cols

    def train_test_split(
        self,
        test_ratio: float = 0.2,
        random: bool = False
    ) -> Tuple['ElectricityDataset', 'ElectricityDataset']:
        """
        Split data into train and test sets.

        For temporal data, we typically use the last portion as test.
        For random split, we shuffle first.
        """
        n = len(self.X)
        n_test = int(n * test_ratio)
        n_train = n - n_test

        if random:
            idx = np.random.permutation(n)
        else:
            idx = np.arange(n)

        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        # Create new dataset objects
        train_ds = ElectricityDatasetSubset(self, train_idx)
        test_ds = ElectricityDatasetSubset(self, test_idx)

        return train_ds, test_ds


class ElectricityDatasetSubset:
    """Subset of ElectricityDataset for train/test splits."""

    def __init__(self, parent: ElectricityDataset, indices: np.ndarray):
        self.parent = parent
        self.indices = indices

        self.X = parent.X[indices]
        self.target = parent.target[indices]
        self.ids = parent.ids[indices]

        self.feature_cols = parent.feature_cols
        self.target_idx = parent.target_idx
        self.num_features = parent.num_features
        self.mean = parent.mean
        self.std = parent.std

    def get_fantom_data(self) -> np.ndarray:
        return self.X

    def get_target(self) -> np.ndarray:
        return self.target

    def get_ids(self) -> np.ndarray:
        return self.ids

    def get_num_nodes(self) -> int:
        return self.num_features

    def get_target_idx(self) -> int:
        return self.target_idx

    def get_feature_names(self) -> List[str]:
        return self.parent.get_feature_names()


def load_test_data(
    train_dataset: 'ElectricityDataset',
    test_path: str = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load and transform test data using training statistics.

    Args:
        train_dataset: Training dataset (for mean/std normalization and feature columns)
        test_path: Path to X_test file (default: auto-detect)

    Returns:
        X_test: Transformed test data [N, lag+1, num_features] (without TARGET)
        test_ids: Sample IDs for submission
        test_df: Raw test DataFrame
    """
    # Find X_test file
    if test_path is None:
        test_files = list(DATA_DIR.glob("*test*.csv"))
        x_test_candidates = [f for f in test_files
                            if 'X' in f.name.upper() and 'y_' not in f.name.lower()]
        if not x_test_candidates:
            raise FileNotFoundError(
                f"X_test file not found in {DATA_DIR}. "
                f"Available files: {[f.name for f in test_files]}"
            )
        test_path = x_test_candidates[0]

    # Load test data
    test_df = pd.read_csv(test_path)

    # Get feature columns from training dataset (without TARGET)
    feature_cols = train_dataset.feature_cols.copy()

    # Impute missing values using training column means
    # First, we need to compute training means for imputation
    X_train_raw, _ = load_raw_data()
    train_means = X_train_raw[feature_cols].mean()

    for col in feature_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(train_means.get(col, 0))

    return test_df, feature_cols


class TestDataset:
    """
    Test dataset class for generating predictions.
    Uses training dataset's normalization parameters.
    """

    def __init__(
        self,
        train_dataset: 'ElectricityDataset',
        test_path: str = None
    ):
        """
        Initialize test dataset.

        Args:
            train_dataset: Training dataset for normalization parameters
            test_path: Path to X_test file
        """
        self.train_dataset = train_dataset
        self.country = train_dataset.country
        self.lag = train_dataset.lag
        self.feature_cols = train_dataset.feature_cols.copy()
        self.mean = train_dataset.mean
        self.std = train_dataset.std

        self._load_and_prepare(test_path)

    def _load_and_prepare(self, test_path: str = None):
        """Load and preprocess test data."""
        # Find X_test file
        if test_path is None:
            test_files = list(DATA_DIR.glob("*test*.csv"))
            x_test_candidates = [f for f in test_files
                                if 'X' in f.name.upper() and 'y_' not in f.name.lower()]
            if not x_test_candidates:
                raise FileNotFoundError(f"X_test file not found in {DATA_DIR}")
            test_path = x_test_candidates[0]

        # Load test data
        test_df = pd.read_csv(test_path)

        # Filter by country if specified
        if self.country is not None:
            test_df = test_df[test_df['COUNTRY'] == self.country].copy()

        test_df = test_df.sort_values('DAY_ID').reset_index(drop=True)

        # Compute training means for imputation
        X_train_raw, _ = load_raw_data()

        # If country-specific, filter training data too
        if self.country is not None:
            X_train_raw = X_train_raw[X_train_raw['COUNTRY'] == self.country]

        train_means = X_train_raw[self.feature_cols].mean()

        # Impute missing values in test data using training means
        for col in self.feature_cols:
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna(train_means.get(col, 0))

        self.test_df = test_df
        self.ids = test_df['ID'].values

        # Create temporal windows (without TARGET)
        # For test data, we need to handle the fact that we don't have TARGET
        # We'll create windows and fill TARGET column with 0 (placeholder)

        if self.lag > 0:
            self._create_temporal_windows()
        else:
            self._create_instantaneous_data()

        # Standardize using training statistics
        if self.mean is not None and self.std is not None:
            self.X = (self.X - self.mean) / self.std

    def _create_temporal_windows(self):
        """Create temporal windows for test data."""
        # For test data, we need sequential days
        # Since we don't have TARGET, we use placeholder

        df = self.test_df
        n_samples = len(df) - self.lag

        # Include all feature columns + placeholder for TARGET
        cols = self.feature_cols.copy()
        n_features = len(cols) + 1  # +1 for TARGET placeholder

        data = df[cols].values

        X = np.zeros((n_samples, self.lag + 1, n_features))

        for i in range(n_samples):
            for l in range(self.lag + 1):
                # Fill features
                X[i, l, :-1] = data[i + l]
                # TARGET placeholder is 0

        self.X = X
        self.ids = self.ids[self.lag:]  # Adjust IDs for lagged samples

    def _create_instantaneous_data(self):
        """Create instantaneous data for test (lag=0)."""
        df = self.test_df
        cols = self.feature_cols.copy()

        data = df[cols].values
        n_features = len(cols) + 1  # +1 for TARGET placeholder

        X = np.zeros((len(df), 1, n_features))
        X[:, 0, :-1] = data

        self.X = X

    def get_fantom_data(self) -> np.ndarray:
        """Get data in FANTOM format [N, lag+1, num_nodes]."""
        return self.X

    def get_ids(self) -> np.ndarray:
        """Get sample IDs for submission."""
        return self.ids

    def get_num_nodes(self) -> int:
        """Get number of nodes."""
        return self.X.shape[2]


def load_all_test_data() -> pd.DataFrame:
    """Load raw test data without any processing."""
    test_files = list(DATA_DIR.glob("*test*.csv"))
    x_test_candidates = [f for f in test_files
                        if 'X' in f.name.upper() and 'y_' not in f.name.lower()]
    if not x_test_candidates:
        raise FileNotFoundError(f"X_test file not found in {DATA_DIR}")
    return pd.read_csv(x_test_candidates[0])


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")

    # Test loading raw data
    X, Y = load_raw_data()
    print(f"X_train shape: {X.shape}")
    print(f"Y_train shape: {Y.shape}")

    # Test merged data
    df = merge_data(X, Y)
    print(f"Merged shape: {df.shape}")

    # Test imputation
    df_imputed = impute_missing(df, method='mean')
    print(f"Missing values after imputation: {df_imputed.isnull().sum().sum()}")

    # Test dataset class
    print("\n--- Testing ElectricityDataset ---")

    # Joint model
    ds_joint = ElectricityDataset(country=None, lag=1)
    print(f"Joint model - X shape: {ds_joint.X.shape}, "
          f"num_nodes: {ds_joint.get_num_nodes()}, "
          f"target_idx: {ds_joint.get_target_idx()}")

    # DE model
    ds_de = ElectricityDataset(country='DE', lag=1)
    print(f"DE model - X shape: {ds_de.X.shape}, "
          f"num_nodes: {ds_de.get_num_nodes()}")

    # FR model
    ds_fr = ElectricityDataset(country='FR', lag=1)
    print(f"FR model - X shape: {ds_fr.X.shape}, "
          f"num_nodes: {ds_fr.get_num_nodes()}")

    # Instantaneous (no temporal)
    ds_inst = ElectricityDataset(country='DE', lag=0, use_temporal=False)
    print(f"Instantaneous DE - X shape: {ds_inst.X.shape}")

    print("\nData loader tests passed!")
