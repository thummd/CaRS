"""
Generate submission file for the electricity price challenge.

Uses trained FANTOM models to predict TARGET on test data.

Usage:
    python generate_submission.py --model_de outputs/germany_XXX --model_fr outputs/france_XXX
    python generate_submission.py --use_latest  # Use most recent models
"""

import argparse
import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    ElectricityDataset, TestDataset, load_all_test_data,
    DATA_DIR, get_feature_columns
)
from fantom_electricity import FANTOMElectricity, create_model


def load_trained_model(
    model_dir: Path,
    device: torch.device
) -> Tuple[FANTOMElectricity, Dict]:
    """
    Load a trained model from directory.

    Args:
        model_dir: Directory containing model.pt and config.yaml
        device: PyTorch device

    Returns:
        model: Loaded FANTOMElectricity model
        config: Model configuration
    """
    # Load config
    config_path = model_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Recreate dataset to get dimensions
    dataset = ElectricityDataset(**config['dataset_kwargs'])

    # Create model
    model = create_model(
        num_nodes=dataset.get_num_nodes(),
        target_idx=dataset.get_target_idx(),
        lag=dataset.X.shape[1] - 1,
        device=str(device),
        model_config=config['model_config']
    )

    # Load weights
    model_path = model_dir / "model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, config, dataset


def find_latest_model(country: str, base_dir: Path = None) -> Optional[Path]:
    """Find the most recent model for a country."""
    if base_dir is None:
        base_dir = Path(__file__).parent / "outputs"

    # Find all matching directories
    pattern = f"{country.lower()}_*" if country else "*"
    candidates = list(base_dir.glob(pattern))

    # Filter to valid model directories (must have model.pt)
    valid = [d for d in candidates if (d / "model.pt").exists()]

    if not valid:
        return None

    # Sort by modification time
    valid.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return valid[0]


def generate_predictions_for_country(
    model: FANTOMElectricity,
    train_dataset: ElectricityDataset,
    test_df: pd.DataFrame,
    country: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions for a specific country's test data.

    Args:
        model: Trained FANTOM model
        train_dataset: Training dataset (for normalization parameters)
        test_df: Raw test DataFrame
        country: Country code ('DE' or 'FR')

    Returns:
        predictions: Predicted TARGET values
        ids: Corresponding sample IDs
    """
    # Filter test data by country
    country_df = test_df[test_df['COUNTRY'] == country].copy()
    country_df = country_df.sort_values('DAY_ID').reset_index(drop=True)

    if len(country_df) == 0:
        return np.array([]), np.array([])

    # Get feature columns
    feature_cols = train_dataset.feature_cols

    # Compute training means for imputation
    train_means = train_dataset.df[feature_cols].mean()

    # Impute missing values in test data
    for col in feature_cols:
        if col in country_df.columns:
            country_df[col] = country_df[col].fillna(train_means.get(col, 0))

    # Get IDs
    ids = country_df['ID'].values

    # Create temporal windows
    lag = train_dataset.lag
    n_samples = len(country_df) - lag if lag > 0 else len(country_df)

    if n_samples <= 0:
        # Not enough data for temporal windows - use instantaneous
        n_samples = len(country_df)
        lag = 0

    # Build feature array
    data = country_df[feature_cols].values
    n_features = len(feature_cols) + 1  # +1 for TARGET placeholder

    if lag > 0:
        X = np.zeros((n_samples, lag + 1, n_features))
        for i in range(n_samples):
            for l in range(lag + 1):
                X[i, l, :-1] = data[i + l]
        ids = ids[lag:]  # Adjust for lag
    else:
        X = np.zeros((n_samples, 1, n_features))
        X[:, 0, :-1] = data

    # Standardize using training statistics
    if train_dataset.mean is not None and train_dataset.std is not None:
        X = (X - train_dataset.mean) / train_dataset.std

    # Get predictions
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        predictions = model.predict_target(X_tensor).cpu().numpy()

    return predictions, ids


def generate_submission(
    model_de_dir: Optional[Path] = None,
    model_fr_dir: Optional[Path] = None,
    output_path: str = None,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Generate submission file using separate DE/FR models.

    Args:
        model_de_dir: Directory with trained DE model
        model_fr_dir: Directory with trained FR model
        output_path: Path to save submission CSV
        device: PyTorch device

    Returns:
        submission: DataFrame with ID and TARGET columns
    """
    device = torch.device(device)

    # Load test data
    test_df = load_all_test_data()
    print(f"Test data: {len(test_df)} samples")
    print(f"  DE: {len(test_df[test_df['COUNTRY'] == 'DE'])} samples")
    print(f"  FR: {len(test_df[test_df['COUNTRY'] == 'FR'])} samples")

    all_predictions = []
    all_ids = []

    # Process DE
    if model_de_dir:
        print(f"\nLoading DE model from: {model_de_dir}")
        model_de, config_de, train_ds_de = load_trained_model(model_de_dir, device)

        pred_de, ids_de = generate_predictions_for_country(
            model_de, train_ds_de, test_df, 'DE'
        )
        print(f"  DE predictions: {len(pred_de)}")

        all_predictions.extend(pred_de)
        all_ids.extend(ids_de)

    # Process FR
    if model_fr_dir:
        print(f"\nLoading FR model from: {model_fr_dir}")
        model_fr, config_fr, train_ds_fr = load_trained_model(model_fr_dir, device)

        pred_fr, ids_fr = generate_predictions_for_country(
            model_fr, train_ds_fr, test_df, 'FR'
        )
        print(f"  FR predictions: {len(pred_fr)}")

        all_predictions.extend(pred_fr)
        all_ids.extend(ids_fr)

    # Create submission DataFrame
    submission = pd.DataFrame({
        'ID': all_ids,
        'TARGET': all_predictions
    })

    # Sort by ID for consistency
    submission = submission.sort_values('ID').reset_index(drop=True)

    # Verify we have predictions for all test IDs
    test_ids = set(test_df['ID'].values)
    pred_ids = set(submission['ID'].values)
    missing_ids = test_ids - pred_ids

    if missing_ids:
        print(f"\nWarning: Missing predictions for {len(missing_ids)} IDs")
        # Fill missing with 0 (neutral prediction)
        for mid in missing_ids:
            submission = pd.concat([
                submission,
                pd.DataFrame({'ID': [mid], 'TARGET': [0.0]})
            ])
        submission = submission.sort_values('ID').reset_index(drop=True)

    print(f"\nSubmission: {len(submission)} samples")

    # Save
    if output_path:
        submission.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")

    return submission


def main():
    parser = argparse.ArgumentParser(description="Generate submission for electricity challenge")
    parser.add_argument('--model_de', type=str, default=None,
                        help='Directory with trained DE model')
    parser.add_argument('--model_fr', type=str, default=None,
                        help='Directory with trained FR model')
    parser.add_argument('--use_latest', action='store_true',
                        help='Use most recent models for each country')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for submission CSV')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')

    args = parser.parse_args()

    # Find models
    model_de_dir = None
    model_fr_dir = None

    if args.use_latest:
        model_de_dir = find_latest_model('germany')
        model_fr_dir = find_latest_model('france')

        if model_de_dir:
            print(f"Using latest DE model: {model_de_dir}")
        else:
            print("No DE model found!")

        if model_fr_dir:
            print(f"Using latest FR model: {model_fr_dir}")
        else:
            print("No FR model found!")

    if args.model_de:
        model_de_dir = Path(args.model_de)
    if args.model_fr:
        model_fr_dir = Path(args.model_fr)

    if not model_de_dir and not model_fr_dir:
        print("Error: No models specified. Use --model_de, --model_fr, or --use_latest")
        return

    # Set output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / f"submission_{timestamp}.csv"

    # Generate submission
    submission = generate_submission(
        model_de_dir=model_de_dir,
        model_fr_dir=model_fr_dir,
        output_path=str(output_path),
        device=args.device
    )

    # Print statistics
    print("\nSubmission statistics:")
    print(f"  Mean TARGET: {submission['TARGET'].mean():.4f}")
    print(f"  Std TARGET: {submission['TARGET'].std():.4f}")
    print(f"  Min TARGET: {submission['TARGET'].min():.4f}")
    print(f"  Max TARGET: {submission['TARGET'].max():.4f}")


if __name__ == "__main__":
    main()
