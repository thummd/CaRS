"""
Temporal Data Loader for Electricity Price Forecasting.

Transforms epftoolbox data into DS3M/FANTOM compatible format with proper
temporal ordering and windowing.

Key differences from QRT adapter:
- Data IS temporally ordered (train before test)
- Proper sliding window creation
- No shuffling
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, Optional, List, Dict


sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import EPFTOOLBOX_DIR
# Data paths
DATA_DIR = EPFTOOLBOX_DIR


def load_epftoolbox_data(
    dataset: str = 'DE',
    data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """
    Load epftoolbox dataset from local cache.

    Args:
        dataset: 'DE' (Germany) or 'FR' (France)
        data_dir: Directory with cached data

    Returns:
        DataFrame with Price and Exogenous columns
    """
    file_path = data_dir / f"{dataset}.csv"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset {dataset} not found. Run download_epftoolbox_data.py first."
        )

    data = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Standardize column names
    columns = ['Price']
    n_exogenous = len(data.columns) - 1
    for i in range(1, n_exogenous + 1):
        columns.append(f'Exogenous_{i}')
    data.columns = columns

    return data


def compute_price_change(data: pd.DataFrame, method: str = 'diff') -> pd.DataFrame:
    """
    Compute price change as target variable (matching QRT's TARGET).

    Args:
        data: DataFrame with Price column
        method: 'diff' (price_t - price_{t-1}), 'returns' (log returns),
                'pct' (percent change)

    Returns:
        DataFrame with added TARGET column
    """
    df = data.copy()

    if method == 'diff':
        df['TARGET'] = df['Price'].diff()
    elif method == 'returns':
        df['TARGET'] = np.log(df['Price']).diff()
    elif method == 'pct':
        df['TARGET'] = df['Price'].pct_change()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Drop first row (NaN from diff)
    df = df.iloc[1:].copy()

    return df


def normalize_features(
    train_data: np.ndarray,
    test_data: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using training data statistics.

    Args:
        train_data: Training features (n_train, n_features)
        test_data: Test features (n_test, n_features), optional

    Returns:
        train_norm: Normalized training data
        test_norm: Normalized test data (or None)
        moments: Array of shape (n_features, 2) with [mean, std]
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

    if test_data is not None:
        test_norm = (test_data - moments[:, 0]) / moments[:, 1]
    else:
        test_norm = None

    return train_norm, test_norm, moments


def create_temporal_windows(
    X: np.ndarray,
    Y: np.ndarray,
    timestep: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series prediction.

    DS3M expects:
    - X: input features for window [t, t+timestep)
    - Y: target values for window [t+1, t+timestep+1)

    Args:
        X: Feature array (n_samples, n_features)
        Y: Target array (n_samples,) or (n_samples, 1)
        timestep: Window length

    Returns:
        X_windows: Shape (timestep, n_windows, n_features)
        Y_windows: Shape (timestep, n_windows, 1)
    """
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    n_samples = len(X) - timestep
    n_features = X.shape[1]

    X_win = np.zeros((n_samples, timestep, n_features))
    Y_win = np.zeros((n_samples, timestep, 1))

    for i in range(n_samples):
        X_win[i] = X[i:i + timestep]
        Y_win[i] = Y[i + 1:i + timestep + 1]

    # Transpose to DS3M format: (timestep, batch, features)
    X_out = np.transpose(X_win, (1, 0, 2))
    Y_out = np.transpose(Y_win, (1, 0, 2))

    return X_out, Y_out


def prepare_ds3m_data(
    dataset: str = 'DE',
    timestep: int = 14,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    target_method: str = 'diff',
    include_exogenous: bool = True
) -> Dict:
    """
    Prepare data for DS3M training.

    Performs temporal train/val/test split (no shuffling!).

    Args:
        dataset: 'DE' or 'FR'
        timestep: Lookback window size
        test_ratio: Fraction for test set
        val_ratio: Fraction for validation set (from remaining after test)
        target_method: How to compute TARGET
        include_exogenous: Whether to include exogenous features

    Returns:
        Dictionary with:
        - trainX, trainY: Training data in DS3M format
        - valX, valY: Validation data
        - testX, testY: Test data
        - X_moments, Y_moments: Normalization parameters
        - feature_cols: Feature column names
        - timestamps: Original timestamps for each split
    """
    # Load data
    data = load_epftoolbox_data(dataset)

    # Compute TARGET
    data = compute_price_change(data, method=target_method)

    # Select features
    if include_exogenous:
        feature_cols = ['Price'] + [c for c in data.columns if c.startswith('Exogenous')]
    else:
        feature_cols = ['Price']

    # Extract arrays
    X_all = data[feature_cols].values.astype(np.float32)
    Y_all = data['TARGET'].values.astype(np.float32)
    timestamps = data.index

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
    X_train_norm, _, X_moments = normalize_features(X_train)
    X_val_norm = (X_val - X_moments[:, 0]) / X_moments[:, 1]
    X_test_norm = (X_test - X_moments[:, 0]) / X_moments[:, 1]

    Y_moments = np.array([Y_train.mean(), Y_train.std()])
    if Y_moments[1] == 0:
        Y_moments[1] = 1.0
    Y_train_norm = (Y_train - Y_moments[0]) / Y_moments[1]
    Y_val_norm = (Y_val - Y_moments[0]) / Y_moments[1]
    Y_test_norm = (Y_test - Y_moments[0]) / Y_moments[1]

    # Create windows
    trainX, trainY = create_temporal_windows(X_train_norm, Y_train_norm, timestep)
    valX, valY = create_temporal_windows(X_val_norm, Y_val_norm, timestep)
    testX, testY = create_temporal_windows(X_test_norm, Y_test_norm, timestep)

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
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'timestamps': {
            'train': ts_train,
            'val': ts_val,
            'test': ts_test
        }
    }


def prepare_univariate_ds3m_data(
    dataset: str = 'DE',
    timestep: int = 14,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    target_method: str = 'diff'
) -> Dict:
    """
    Prepare univariate data for DS3M (TARGET only as input and output).

    This matches the original DS3M paper's approach for electricity forecasting.

    Args:
        dataset: 'DE' or 'FR'
        timestep: Lookback window
        test_ratio: Test fraction
        val_ratio: Validation fraction
        target_method: How to compute TARGET

    Returns:
        Dictionary with DS3M-compatible data
    """
    # Load data
    data = load_epftoolbox_data(dataset)

    # Compute TARGET
    data = compute_price_change(data, method=target_method)

    # Extract TARGET only
    Y_all = data['TARGET'].values.astype(np.float32)
    timestamps = data.index

    # Temporal split
    n_total = len(Y_all)
    n_test = int(n_total * test_ratio)
    n_val = int((n_total - n_test) * val_ratio)
    n_train = n_total - n_test - n_val

    Y_train = Y_all[:n_train]
    Y_val = Y_all[n_train:n_train + n_val]
    Y_test = Y_all[n_train + n_val:]

    ts_train = timestamps[:n_train]
    ts_val = timestamps[n_train:n_train + n_val]
    ts_test = timestamps[n_train + n_val:]

    # Normalize
    Y_moments = np.array([Y_train.mean(), Y_train.std()])
    if Y_moments[1] == 0:
        Y_moments[1] = 1.0

    Y_train_norm = (Y_train - Y_moments[0]) / Y_moments[1]
    Y_val_norm = (Y_val - Y_moments[0]) / Y_moments[1]
    Y_test_norm = (Y_test - Y_moments[0]) / Y_moments[1]

    # Create windows (X=Y for univariate)
    def create_univariate_windows(Y, timestep):
        Y = Y.reshape(-1, 1)
        n_samples = len(Y) - timestep
        X_win = np.zeros((n_samples, timestep, 1))
        Y_win = np.zeros((n_samples, timestep, 1))

        for i in range(n_samples):
            X_win[i] = Y[i:i + timestep]
            Y_win[i] = Y[i + 1:i + timestep + 1]

        X_out = np.transpose(X_win, (1, 0, 2))
        Y_out = np.transpose(Y_win, (1, 0, 2))
        return X_out, Y_out

    trainX, trainY = create_univariate_windows(Y_train_norm, timestep)
    valX, valY = create_univariate_windows(Y_val_norm, timestep)
    testX, testY = create_univariate_windows(Y_test_norm, timestep)

    return {
        'trainX': torch.from_numpy(trainX).float(),
        'trainY': torch.from_numpy(trainY).float(),
        'valX': torch.from_numpy(valX).float(),
        'valY': torch.from_numpy(valY).float(),
        'testX': torch.from_numpy(testX).float(),
        'testY': torch.from_numpy(testY).float(),
        'Y_moments': Y_moments,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'timestamps': {
            'train': ts_train,
            'val': ts_val,
            'test': ts_test
        }
    }


def main():
    """Test the temporal data loader."""
    print("Testing Temporal Data Loader")
    print("="*60)

    # Test multivariate
    print("\n--- Multivariate (DE) ---")
    data = prepare_ds3m_data(
        dataset='DE',
        timestep=14,
        test_ratio=0.2,
        val_ratio=0.1,
        include_exogenous=True
    )

    print(f"Features: {data['feature_cols']}")
    print(f"Train X: {data['trainX'].shape}")  # (timestep, n_windows, n_features)
    print(f"Train Y: {data['trainY'].shape}")  # (timestep, n_windows, 1)
    print(f"Val X:   {data['valX'].shape}")
    print(f"Val Y:   {data['valY'].shape}")
    print(f"Test X:  {data['testX'].shape}")
    print(f"Test Y:  {data['testY'].shape}")
    print(f"\nSplit sizes: train={data['n_train']}, val={data['n_val']}, test={data['n_test']}")
    print(f"Y moments: mean={data['Y_moments'][0]:.4f}, std={data['Y_moments'][1]:.4f}")

    print(f"\nTrain date range: {data['timestamps']['train'].min()} to {data['timestamps']['train'].max()}")
    print(f"Val date range:   {data['timestamps']['val'].min()} to {data['timestamps']['val'].max()}")
    print(f"Test date range:  {data['timestamps']['test'].min()} to {data['timestamps']['test'].max()}")

    # Test univariate
    print("\n--- Univariate (DE) ---")
    data_uni = prepare_univariate_ds3m_data(
        dataset='DE',
        timestep=14,
        test_ratio=0.2,
        val_ratio=0.1
    )

    print(f"Train X: {data_uni['trainX'].shape}")  # (timestep, n_windows, 1)
    print(f"Train Y: {data_uni['trainY'].shape}")

    # Test France
    print("\n--- Multivariate (FR) ---")
    data_fr = prepare_ds3m_data(
        dataset='FR',
        timestep=14,
        test_ratio=0.2,
        val_ratio=0.1
    )

    print(f"Features: {data_fr['feature_cols']}")
    print(f"Train X: {data_fr['trainX'].shape}")
    print(f"Test X:  {data_fr['testX'].shape}")

    print("\n" + "="*60)
    print("Temporal Data Loader tests passed!")


if __name__ == "__main__":
    main()
