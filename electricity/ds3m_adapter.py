"""
Data adapter for DS3M on QRT electricity data.

Transforms QRT tabular data into the sequential time series format
required by DS3M (Deep Switching State Space Model).
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, Optional, List, Dict


sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR
# Data paths
DATA_DIR = DATA_DIR / "qrt"


def normalize_moments(dataset: np.ndarray) -> np.ndarray:
    """
    Compute normalization moments (mean, std) from dataset.
    Compatible with DS3M's normalization.

    Args:
        dataset: Array to compute moments from

    Returns:
        moments: Array [mean, std]
    """
    moments = np.zeros(2)
    moments[0] = dataset.mean()
    moments[1] = dataset.std()
    if moments[1] == 0:
        moments[1] = 1.0  # Avoid division by zero
    return moments


def normalize_fit(dataset: np.ndarray, moments: np.ndarray) -> np.ndarray:
    """
    Normalize dataset using precomputed moments.

    Args:
        dataset: Data to normalize
        moments: Array [mean, std]

    Returns:
        Normalized data
    """
    return (dataset - moments[0]) / moments[1]


def normalize_invert(dataset: np.ndarray, moments: np.ndarray) -> np.ndarray:
    """
    Invert normalization.

    Args:
        dataset: Normalized data
        moments: Array [mean, std]

    Returns:
        Original scale data
    """
    return dataset * moments[1] + moments[0]


def load_qrt_data(
    x_train_path: Optional[str] = None,
    y_train_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load and merge QRT training data.

    Args:
        x_train_path: Path to X_train CSV (default: auto-detect)
        y_train_path: Path to Y_train CSV (default: auto-detect)

    Returns:
        Merged DataFrame with features and TARGET
    """
    if x_train_path is None:
        x_train_path = DATA_DIR / "X_train_NHkHMNU.csv"
    if y_train_path is None:
        y_train_path = DATA_DIR / "y_train_ZAN5mwg.csv"

    X_train = pd.read_csv(x_train_path)
    Y_train = pd.read_csv(y_train_path)

    # Merge on ID
    df = X_train.merge(Y_train, on='ID')

    return df


def load_qrt_test_data(
    x_test_path: Optional[str] = None,
    y_test_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load and merge QRT test data.

    Args:
        x_test_path: Path to X_test CSV
        y_test_path: Path to Y_test CSV (if available)

    Returns:
        DataFrame with test features (and TARGET if y_test provided)
    """
    if x_test_path is None:
        x_test_path = DATA_DIR / "X_test_final.csv"
    if y_test_path is None:
        y_test_path = DATA_DIR / "y_test_random_final.csv"

    X_test = pd.read_csv(x_test_path)

    if y_test_path and Path(y_test_path).exists():
        Y_test = pd.read_csv(y_test_path)
        df = X_test.merge(Y_test, on='ID')
    else:
        df = X_test

    return df


def create_dataset_windows(
    data: np.ndarray,
    timestep: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create X, Y pairs using sliding windows.

    DS3M expects:
    - X: input sequence of length timestep
    - Y: target sequence (shifted by 1)

    Args:
        data: Time series of shape (T, D) where T is time, D is features
        timestep: Length of lookback window

    Returns:
        X: Shape (n_windows, timestep, D)
        Y: Shape (n_windows, timestep, D)
    """
    n_samples = len(data) - timestep
    n_features = data.shape[1] if len(data.shape) > 1 else 1

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    X = np.zeros((n_samples, timestep, n_features))
    Y = np.zeros((n_samples, timestep, n_features))

    for i in range(n_samples):
        X[i] = data[i:i + timestep]
        Y[i] = data[i + 1:i + timestep + 1]

    return X, Y


def qrt_to_ds3m_format(
    df: pd.DataFrame,
    country: Optional[str] = None,
    target_col: str = 'TARGET',
    timestep: int = 14,
    return_df: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, Optional[pd.DataFrame]]:
    """
    Convert QRT data to DS3M format.

    DS3M expects data in shape (timestep, batch, features).
    This function:
    1. Filters by country if specified
    2. Sorts by DAY_ID for temporal ordering
    3. Extracts target column
    4. Creates sliding windows
    5. Normalizes data
    6. Transposes to DS3M format

    Args:
        df: Merged DataFrame with features and TARGET
        country: 'FR', 'DE', or None for combined
        target_col: Column to use as target (e.g., 'TARGET', 'FR_CONSUMPTION')
        timestep: Lookback window size
        return_df: Whether to return filtered DataFrame

    Returns:
        X: Tensor of shape (timestep, n_windows, 1)
        Y: Tensor of shape (timestep, n_windows, 1)
        moments: Normalization parameters [mean, std]
        filtered_df: (optional) Filtered and sorted DataFrame
    """
    # Filter by country
    if country is not None:
        df_filtered = df[df['COUNTRY'] == country].copy()
    else:
        df_filtered = df.copy()

    # Sort by DAY_ID for temporal ordering
    df_filtered = df_filtered.sort_values('DAY_ID').reset_index(drop=True)

    # Extract target column
    if target_col not in df_filtered.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {df_filtered.columns.tolist()}")

    values = df_filtered[target_col].values.astype(np.float32)

    # Compute normalization from all data (train portion)
    moments = normalize_moments(values)
    values_norm = normalize_fit(values, moments)

    # Create windows
    X, Y = create_dataset_windows(values_norm, timestep)

    # Transpose to DS3M format: (timestep, batch, features)
    X = np.transpose(X, (1, 0, 2))  # (timestep, n_windows, features)
    Y = np.transpose(Y, (1, 0, 2))

    # Convert to tensors
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    if return_df:
        return X, Y, moments, df_filtered
    return X, Y, moments, None


def prepare_train_test_split(
    df: pd.DataFrame,
    country: Optional[str] = None,
    target_col: str = 'TARGET',
    timestep: int = 14,
    test_ratio: float = 0.2
) -> Dict[str, torch.Tensor]:
    """
    Prepare train/test split for DS3M.

    Uses temporal split (last portion for test) to respect time series nature.

    Args:
        df: Merged DataFrame
        country: Country filter
        target_col: Target column
        timestep: Lookback window
        test_ratio: Fraction for test set

    Returns:
        Dictionary with trainX, trainY, testX, testY, moments
    """
    # Filter and sort
    if country is not None:
        df_filtered = df[df['COUNTRY'] == country].copy()
    else:
        df_filtered = df.copy()

    df_filtered = df_filtered.sort_values('DAY_ID').reset_index(drop=True)

    # Extract values
    values = df_filtered[target_col].values.astype(np.float32)

    # Temporal split
    n_total = len(values)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test

    train_values = values[:n_train]
    test_values = values[n_train - timestep:]  # Include timestep for context

    # Normalize using training data
    moments = normalize_moments(train_values)
    train_norm = normalize_fit(train_values, moments)
    test_norm = normalize_fit(test_values, moments)

    # Create windows
    trainX, trainY = create_dataset_windows(train_norm, timestep)
    testX, testY = create_dataset_windows(test_norm, timestep)

    # Transpose to DS3M format
    trainX = torch.from_numpy(np.transpose(trainX, (1, 0, 2))).float()
    trainY = torch.from_numpy(np.transpose(trainY, (1, 0, 2))).float()
    testX = torch.from_numpy(np.transpose(testX, (1, 0, 2))).float()
    testY = torch.from_numpy(np.transpose(testY, (1, 0, 2))).float()

    return {
        'trainX': trainX,
        'trainY': trainY,
        'testX': testX,
        'testY': testY,
        'moments': moments,
        'n_train': n_train,
        'n_test': n_test
    }


def prepare_official_test_data(
    train_moments: np.ndarray,
    country: Optional[str] = None,
    target_col: str = 'TARGET',
    timestep: int = 14
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Prepare official test data using training normalization.

    Args:
        train_moments: Normalization moments from training data
        country: Country filter
        target_col: Target column
        timestep: Lookback window

    Returns:
        testX: Tensor (timestep, n_windows, features)
        testY: Tensor (timestep, n_windows, features)
        test_ids: Original IDs for submission
        raw_target: Raw target values for evaluation
    """
    df = load_qrt_test_data()

    # Filter by country
    if country is not None:
        df_filtered = df[df['COUNTRY'] == country].copy()
    else:
        df_filtered = df.copy()

    df_filtered = df_filtered.sort_values('DAY_ID').reset_index(drop=True)

    # Get IDs for submission
    test_ids = df_filtered['ID'].values

    # Extract and normalize target
    if target_col in df_filtered.columns:
        values = df_filtered[target_col].values.astype(np.float32)
        raw_target = values.copy()
    else:
        # If target not available in test, use placeholder
        values = np.zeros(len(df_filtered), dtype=np.float32)
        raw_target = None

    values_norm = normalize_fit(values, train_moments)

    # Create windows
    X, Y = create_dataset_windows(values_norm, timestep)

    # Transpose
    X = torch.from_numpy(np.transpose(X, (1, 0, 2))).float()
    Y = torch.from_numpy(np.transpose(Y, (1, 0, 2))).float()

    # Adjust IDs to match windows (first timestep samples are dropped)
    test_ids = test_ids[timestep:]
    if raw_target is not None:
        raw_target = raw_target[timestep:]

    return X, Y, test_ids, raw_target


def get_country_sample_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Get sample counts per country."""
    return df['COUNTRY'].value_counts().to_dict()


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (excluding metadata and target).

    Returns:
        List of feature column names
    """
    exclude = ['ID', 'DAY_ID', 'COUNTRY', 'TARGET']
    return [c for c in df.columns if c not in exclude]


def qrt_to_ds3m_multivariate(
    df: pd.DataFrame,
    country: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'TARGET',
    timestep: int = 14,
    return_df: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, List[str], Optional[pd.DataFrame]]:
    """
    Convert QRT data to multivariate DS3M format.

    Uses ALL features as input (X) and predicts TARGET only (Y).
    This allows DS3M to learn cross-variable relationships.

    Args:
        df: Merged DataFrame with features and TARGET
        country: 'FR', 'DE', or None for combined
        feature_cols: List of feature columns to use (default: all except metadata)
        target_col: Column to predict (typically 'TARGET')
        timestep: Lookback window size
        return_df: Whether to return filtered DataFrame

    Returns:
        X: Tensor of shape (timestep, n_windows, n_features) - all features
        Y: Tensor of shape (timestep, n_windows, 1) - TARGET only
        X_moments: Normalization parameters for features (n_features, 2)
        Y_moments: Normalization parameters for target [mean, std]
        feature_names: List of feature column names
        filtered_df: (optional) Filtered and sorted DataFrame
    """
    # Filter by country
    if country is not None:
        df_filtered = df[df['COUNTRY'] == country].copy()
    else:
        df_filtered = df.copy()

    # Sort by DAY_ID for temporal ordering
    df_filtered = df_filtered.sort_values('DAY_ID').reset_index(drop=True)

    # Get feature columns
    if feature_cols is None:
        feature_cols = get_feature_columns(df_filtered)

    # Validate columns exist
    missing = [c for c in feature_cols if c not in df_filtered.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")
    if target_col not in df_filtered.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    # Extract features and target
    X_values = df_filtered[feature_cols].values.astype(np.float32)
    Y_values = df_filtered[target_col].values.astype(np.float32).reshape(-1, 1)

    n_features = len(feature_cols)

    # Handle NaN values in features using forward fill then backward fill
    # This preserves temporal structure better than mean imputation
    for i in range(n_features):
        col_data = X_values[:, i]
        nan_mask = np.isnan(col_data)
        if nan_mask.any():
            # Forward fill
            for j in range(1, len(col_data)):
                if nan_mask[j] and not nan_mask[j-1]:
                    col_data[j] = col_data[j-1]
                    nan_mask[j] = False
            # Backward fill for any remaining NaN at the start
            for j in range(len(col_data)-2, -1, -1):
                if nan_mask[j] and not nan_mask[j+1]:
                    col_data[j] = col_data[j+1]
                    nan_mask[j] = False
            # If still NaN (entire column was NaN), fill with 0
            col_data[np.isnan(col_data)] = 0.0
            X_values[:, i] = col_data

    # Normalize features (per-feature standardization)
    X_moments = np.zeros((n_features, 2))
    X_norm = X_values.copy()
    for i in range(n_features):
        X_moments[i, 0] = np.nanmean(X_values[:, i])
        X_moments[i, 1] = np.nanstd(X_values[:, i])
        if X_moments[i, 1] == 0 or np.isnan(X_moments[i, 1]):
            X_moments[i, 1] = 1.0
        if np.isnan(X_moments[i, 0]):
            X_moments[i, 0] = 0.0
        X_norm[:, i] = (X_values[:, i] - X_moments[i, 0]) / X_moments[i, 1]

    # Normalize target
    Y_moments = normalize_moments(Y_values)
    Y_norm = normalize_fit(Y_values, Y_moments)

    # Create windows
    n_samples = len(X_norm) - timestep
    X_windows = np.zeros((n_samples, timestep, n_features))
    Y_windows = np.zeros((n_samples, timestep, 1))

    for i in range(n_samples):
        X_windows[i] = X_norm[i:i + timestep]
        Y_windows[i] = Y_norm[i + 1:i + timestep + 1]

    # Transpose to DS3M format: (timestep, batch, features)
    X = torch.from_numpy(np.transpose(X_windows, (1, 0, 2))).float()
    Y = torch.from_numpy(np.transpose(Y_windows, (1, 0, 2))).float()

    if return_df:
        return X, Y, X_moments, Y_moments, feature_cols, df_filtered
    return X, Y, X_moments, Y_moments, feature_cols, None


def prepare_multivariate_train_test_split(
    df: pd.DataFrame,
    country: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'TARGET',
    timestep: int = 14,
    test_ratio: float = 0.2
) -> Dict:
    """
    Prepare multivariate train/test split for DS3M.

    Uses temporal split (last portion for test) to respect time series nature.

    Args:
        df: Merged DataFrame
        country: Country filter
        feature_cols: Feature columns to use
        target_col: Target column
        timestep: Lookback window
        test_ratio: Fraction for test set

    Returns:
        Dictionary with trainX, trainY, testX, testY, X_moments, Y_moments, etc.
    """
    # Filter and sort
    if country is not None:
        df_filtered = df[df['COUNTRY'] == country].copy()
    else:
        df_filtered = df.copy()

    df_filtered = df_filtered.sort_values('DAY_ID').reset_index(drop=True)

    # Get feature columns
    if feature_cols is None:
        feature_cols = get_feature_columns(df_filtered)

    n_features = len(feature_cols)

    # Extract features and target
    X_values = df_filtered[feature_cols].values.astype(np.float32)
    Y_values = df_filtered[target_col].values.astype(np.float32).reshape(-1, 1)

    # Handle NaN values in features using forward fill then backward fill
    for i in range(n_features):
        col_data = X_values[:, i]
        nan_mask = np.isnan(col_data)
        if nan_mask.any():
            # Forward fill
            for j in range(1, len(col_data)):
                if nan_mask[j] and not nan_mask[j-1]:
                    col_data[j] = col_data[j-1]
                    nan_mask[j] = False
            # Backward fill for any remaining NaN at the start
            for j in range(len(col_data)-2, -1, -1):
                if nan_mask[j] and not nan_mask[j+1]:
                    col_data[j] = col_data[j+1]
                    nan_mask[j] = False
            # If still NaN (entire column was NaN), fill with 0
            col_data[np.isnan(col_data)] = 0.0
            X_values[:, i] = col_data

    # Temporal split
    n_total = len(X_values)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test

    X_train_raw = X_values[:n_train]
    Y_train_raw = Y_values[:n_train]
    X_test_raw = X_values[n_train - timestep:]  # Include timestep for context
    Y_test_raw = Y_values[n_train - timestep:]

    # Normalize using training data only
    X_moments = np.zeros((n_features, 2))
    for i in range(n_features):
        X_moments[i, 0] = np.nanmean(X_train_raw[:, i])
        X_moments[i, 1] = np.nanstd(X_train_raw[:, i])
        if X_moments[i, 1] == 0 or np.isnan(X_moments[i, 1]):
            X_moments[i, 1] = 1.0
        if np.isnan(X_moments[i, 0]):
            X_moments[i, 0] = 0.0

    Y_moments = normalize_moments(Y_train_raw)

    # Apply normalization
    X_train_norm = (X_train_raw - X_moments[:, 0]) / X_moments[:, 1]
    X_test_norm = (X_test_raw - X_moments[:, 0]) / X_moments[:, 1]
    Y_train_norm = normalize_fit(Y_train_raw, Y_moments)
    Y_test_norm = normalize_fit(Y_test_raw, Y_moments)

    # Create windows
    def create_multivariate_windows(X, Y, timestep):
        n_samples = len(X) - timestep
        X_win = np.zeros((n_samples, timestep, X.shape[1]))
        Y_win = np.zeros((n_samples, timestep, 1))
        for i in range(n_samples):
            X_win[i] = X[i:i + timestep]
            Y_win[i] = Y[i + 1:i + timestep + 1]
        return X_win, Y_win

    trainX_win, trainY_win = create_multivariate_windows(X_train_norm, Y_train_norm, timestep)
    testX_win, testY_win = create_multivariate_windows(X_test_norm, Y_test_norm, timestep)

    # Transpose to DS3M format
    trainX = torch.from_numpy(np.transpose(trainX_win, (1, 0, 2))).float()
    trainY = torch.from_numpy(np.transpose(trainY_win, (1, 0, 2))).float()
    testX = torch.from_numpy(np.transpose(testX_win, (1, 0, 2))).float()
    testY = torch.from_numpy(np.transpose(testY_win, (1, 0, 2))).float()

    return {
        'trainX': trainX,
        'trainY': trainY,
        'testX': testX,
        'testY': testY,
        'X_moments': X_moments,
        'Y_moments': Y_moments,
        'feature_cols': feature_cols,
        'n_features': n_features,
        'n_train': n_train,
        'n_test': n_test
    }


def normalize_features_invert(data: np.ndarray, X_moments: np.ndarray) -> np.ndarray:
    """
    Invert feature normalization.

    Args:
        data: Normalized features (n_samples, n_features)
        X_moments: Per-feature moments (n_features, 2)

    Returns:
        Original scale data
    """
    return data * X_moments[:, 1] + X_moments[:, 0]


def evaluation(
    predict: np.ndarray,
    original: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics (compatible with DS3M utils).

    Args:
        predict: Predictions (n_samples, features)
        original: Ground truth (n_samples, features)

    Returns:
        Dictionary with rmse, mape
    """
    ae = np.abs(predict - original)
    se = ae ** 2
    ape = np.abs(ae / original)
    ape[ape == np.inf] = np.nan
    ape[ape == -np.inf] = np.nan

    mae = np.nanmean(ae)
    rmse = np.sqrt(np.nanmean(se))
    mape = np.nanmean(ape)

    return {
        'rmse': rmse,
        'mape': mape,
        'mae': mae
    }


if __name__ == "__main__":
    print("Testing DS3M data adapter...")

    # Load data
    df = load_qrt_data()
    print(f"Loaded {len(df)} training samples")
    print(f"Countries: {get_country_sample_counts(df)}")

    # Test France conversion (univariate)
    print("\n--- Testing France (TARGET, univariate) ---")
    X, Y, moments, _ = qrt_to_ds3m_format(df, country='FR', target_col='TARGET', timestep=14)
    print(f"X shape: {X.shape}")  # Should be (14, n_windows, 1)
    print(f"Y shape: {Y.shape}")
    print(f"Moments: mean={moments[0]:.4f}, std={moments[1]:.4f}")

    # Test Germany conversion
    print("\n--- Testing Germany (TARGET, univariate) ---")
    X, Y, moments, _ = qrt_to_ds3m_format(df, country='DE', target_col='TARGET', timestep=14)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    # Test Combined
    print("\n--- Testing Combined (TARGET, univariate) ---")
    X, Y, moments, _ = qrt_to_ds3m_format(df, country=None, target_col='TARGET', timestep=14)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    # Test univariate train/test split
    print("\n--- Testing univariate train/test split ---")
    split = prepare_train_test_split(df, country='FR', target_col='TARGET', timestep=14, test_ratio=0.2)
    print(f"Train X: {split['trainX'].shape}")
    print(f"Test X: {split['testX'].shape}")
    print(f"n_train: {split['n_train']}, n_test: {split['n_test']}")

    # Test official test data
    print("\n--- Testing official test data ---")
    testX, testY, test_ids, raw_target = prepare_official_test_data(
        split['moments'], country='FR', target_col='TARGET', timestep=14
    )
    print(f"Test X: {testX.shape}")
    print(f"Test IDs: {len(test_ids)}")
    if raw_target is not None:
        print(f"Raw target available: {len(raw_target)} samples")

    # ======== MULTIVARIATE TESTS ========
    print("\n" + "="*60)
    print("MULTIVARIATE TESTS")
    print("="*60)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols[:5]}...")

    # Test France multivariate
    print("\n--- Testing France (MULTIVARIATE) ---")
    X_mv, Y_mv, X_mom, Y_mom, feat_names, _ = qrt_to_ds3m_multivariate(
        df, country='FR', target_col='TARGET', timestep=14
    )
    print(f"X shape: {X_mv.shape}")  # Should be (14, n_windows, n_features)
    print(f"Y shape: {Y_mv.shape}")  # Should be (14, n_windows, 1)
    print(f"Number of features: {len(feat_names)}")

    # Test Germany multivariate
    print("\n--- Testing Germany (MULTIVARIATE) ---")
    X_mv, Y_mv, X_mom, Y_mom, feat_names, _ = qrt_to_ds3m_multivariate(
        df, country='DE', target_col='TARGET', timestep=14
    )
    print(f"X shape: {X_mv.shape}")
    print(f"Y shape: {Y_mv.shape}")

    # Test Combined multivariate
    print("\n--- Testing Combined (MULTIVARIATE) ---")
    X_mv, Y_mv, X_mom, Y_mom, feat_names, _ = qrt_to_ds3m_multivariate(
        df, country=None, target_col='TARGET', timestep=14
    )
    print(f"X shape: {X_mv.shape}")
    print(f"Y shape: {Y_mv.shape}")

    # Test multivariate train/test split
    print("\n--- Testing MULTIVARIATE train/test split ---")
    split_mv = prepare_multivariate_train_test_split(
        df, country='FR', target_col='TARGET', timestep=14, test_ratio=0.2
    )
    print(f"Train X: {split_mv['trainX'].shape}")  # (timestep, n_train_windows, n_features)
    print(f"Train Y: {split_mv['trainY'].shape}")  # (timestep, n_train_windows, 1)
    print(f"Test X: {split_mv['testX'].shape}")
    print(f"Test Y: {split_mv['testY'].shape}")
    print(f"n_features: {split_mv['n_features']}")
    print(f"n_train: {split_mv['n_train']}, n_test: {split_mv['n_test']}")

    print("\nDS3M adapter tests passed (including multivariate)!")
