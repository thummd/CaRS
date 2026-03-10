"""
Download electricity price data from epftoolbox Zenodo repository.

This script downloads temporally-ordered electricity market data for:
- DE (Germany EPEX)
- FR (France EPEX)

Data source: https://zenodo.org/records/4624805
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import EPFTOOLBOX_DIR
# Data directory
DATA_DIR = EPFTOOLBOX_DIR

# Zenodo URL for epftoolbox datasets
ZENODO_URL = "https://zenodo.org/records/4624805/files/"

# Available datasets
AVAILABLE_DATASETS = ['PJM', 'NP', 'FR', 'BE', 'DE']


def download_dataset(dataset: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Download electricity market dataset from Zenodo.

    Args:
        dataset: One of 'PJM', 'NP', 'FR', 'BE', 'DE'
        data_dir: Directory to save downloaded data

    Returns:
        DataFrame with columns: Price, Exogenous 1, Exogenous 2, ...
    """
    if dataset not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset must be one of {AVAILABLE_DATASETS}")

    # Create directory if needed
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / f"{dataset}.csv"

    # Download if not cached
    if file_path.exists():
        print(f"Loading cached {dataset} data from {file_path}")
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        url = ZENODO_URL + dataset + '.csv'
        print(f"Downloading {dataset} data from {url}")
        data = pd.read_csv(url, index_col=0, parse_dates=True)
        data.to_csv(file_path)
        print(f"Saved to {file_path}")

    # Rename columns to standard format
    columns = ['Price']
    n_exogenous = len(data.columns) - 1
    for i in range(1, n_exogenous + 1):
        columns.append(f'Exogenous {i}')
    data.columns = columns

    return data


def create_train_test_split(
    data: pd.DataFrame,
    years_test: int = 1,
    begin_test_date: str = None,
    end_test_date: str = None
) -> tuple:
    """
    Create temporal train/test split.

    Args:
        data: Full dataset
        years_test: Number of years for testing (364 days each)
        begin_test_date: Optional start date for test set
        end_test_date: Optional end date for test set

    Returns:
        df_train, df_test
    """
    if begin_test_date is None and end_test_date is None:
        n_total = len(data)
        n_train = n_total - 24 * 364 * years_test

        df_train = data.iloc[:n_train]
        df_test = data.iloc[n_train:]
    else:
        begin_test_date = pd.to_datetime(begin_test_date, dayfirst=True)
        end_test_date = pd.to_datetime(end_test_date, dayfirst=True)

        if end_test_date.hour == 0:
            end_test_date = end_test_date + pd.Timedelta(hours=23)

        df_train = data.loc[:begin_test_date - pd.Timedelta(hours=1)]
        df_test = data.loc[begin_test_date:end_test_date]

    return df_train, df_test


def explore_dataset(data: pd.DataFrame, name: str):
    """Print dataset statistics."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    print(f"Shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Duration: {(data.index.max() - data.index.min()).days} days")
    print(f"\nColumns: {list(data.columns)}")
    print(f"\nPrice statistics:")
    print(data['Price'].describe())

    # Check for missing values
    missing = data.isnull().sum()
    if missing.any():
        print(f"\nMissing values:\n{missing[missing > 0]}")
    else:
        print(f"\nNo missing values")

    # Check temporal ordering
    is_sorted = data.index.is_monotonic_increasing
    print(f"\nTemporally ordered: {is_sorted}")

    # Hourly frequency check
    freq = pd.infer_freq(data.index[:100])
    print(f"Inferred frequency: {freq}")


def main():
    print("Downloading epftoolbox electricity datasets...")

    # Download Germany and France data
    de_data = download_dataset('DE')
    fr_data = download_dataset('FR')

    # Explore datasets
    explore_dataset(de_data, 'Germany (DE)')
    explore_dataset(fr_data, 'France (FR)')

    # Create train/test splits (1 year test)
    print("\n" + "="*60)
    print("Creating train/test splits (1 year test)")
    print("="*60)

    de_train, de_test = create_train_test_split(de_data, years_test=1)
    fr_train, fr_test = create_train_test_split(fr_data, years_test=1)

    print(f"\nGermany:")
    print(f"  Train: {len(de_train)} samples ({de_train.index.min()} to {de_train.index.max()})")
    print(f"  Test:  {len(de_test)} samples ({de_test.index.min()} to {de_test.index.max()})")

    print(f"\nFrance:")
    print(f"  Train: {len(fr_train)} samples ({fr_train.index.min()} to {fr_train.index.max()})")
    print(f"  Test:  {len(fr_test)} samples ({fr_test.index.min()} to {fr_test.index.max()})")

    # Show sample data
    print("\n" + "="*60)
    print("Sample data (Germany train)")
    print("="*60)
    print(de_train.head(10))

    print("\n" + "="*60)
    print("Sample data (Germany test)")
    print("="*60)
    print(de_test.head(10))

    return de_data, fr_data


if __name__ == "__main__":
    main()
