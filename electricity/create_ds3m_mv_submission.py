"""
Generate submission file using multivariate DS3M model.

Uses trained multivariate DS3M (features -> TARGET) to predict on X_test.

Usage:
    python create_ds3m_mv_submission.py
    python create_ds3m_mv_submission.py --model_dir outputs/ds3m/ALL_TARGET_mv_d2_seed44_20260120_140929
"""

import sys
import os
from pathlib import Path

from paths import (
    DATA_DIR,
    DS3M_DIR,
    OUTPUT_DIR,
    SUBMISSIONS_DIR,
)
# Add DS3M code to path
DS3M_PATH = str(DS3M_DIR)
sys.path.insert(0, DS3M_PATH)
sys.path.insert(0, os.path.join(DS3M_PATH, "src"))

import argparse
import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional

from DSSSMCode import DSSSM


# Data paths
DATA_DIR = DATA_DIR / "qrt"


def load_multivariate_model(checkpoint_dir: Path, device: str = 'cpu') -> Tuple[DSSSM, Dict]:
    """Load a trained multivariate DS3M model."""
    checkpoint_path = checkpoint_dir / 'checkpoints' / 'best.tar'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Get dimensions from config
    x_dim = config.get('x_dim', config.get('n_features', 32))
    y_dim = config.get('y_dim', 1)

    model = DSSSM(
        x_dim=x_dim,
        y_dim=y_dim,
        h_dim=config['h_dim'],
        z_dim=config['z_dim'],
        d_dim=config['d_dim'],
        n_layers=config['n_layers'],
        device=device,
        bidirection=config.get('bidirection', False)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def load_training_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load training data to get normalization parameters."""
    X_train = pd.read_csv(DATA_DIR / "X_train_NHkHMNU.csv")
    Y_train = pd.read_csv(DATA_DIR / "y_train_ZAN5mwg.csv")

    # Merge
    df = X_train.merge(Y_train, on='ID')

    # Get feature columns (exclude metadata)
    exclude = ['ID', 'DAY_ID', 'COUNTRY', 'TARGET']
    feature_cols = [c for c in df.columns if c not in exclude]

    # Compute normalization moments from training data
    X_values = df[feature_cols].values.astype(np.float32)
    Y_values = df['TARGET'].values.astype(np.float32)

    # Handle NaN in training data
    for i in range(X_values.shape[1]):
        col = X_values[:, i]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col[nan_mask] = np.nanmean(col)
            X_values[:, i] = col

    # Compute moments
    X_moments = np.zeros((len(feature_cols), 2))
    for i in range(len(feature_cols)):
        X_moments[i, 0] = np.nanmean(X_values[:, i])
        X_moments[i, 1] = np.nanstd(X_values[:, i])
        if X_moments[i, 1] == 0 or np.isnan(X_moments[i, 1]):
            X_moments[i, 1] = 1.0

    Y_moments = np.array([np.mean(Y_values), np.std(Y_values)])
    if Y_moments[1] == 0:
        Y_moments[1] = 1.0

    return df, X_moments, Y_moments, feature_cols


def load_test_data() -> pd.DataFrame:
    """Load test data."""
    X_test = pd.read_csv(DATA_DIR / "X_test_final.csv")
    return X_test


def prepare_test_features(
    test_df: pd.DataFrame,
    feature_cols: list,
    X_moments: np.ndarray,
    timestep: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare test features for model input.

    Handles per-country processing with temporal ordering.

    Returns:
        X_windows: Feature windows (n_samples, timestep, n_features)
        sample_ids: Original IDs for each window
        valid_mask: Which samples have valid predictions
    """
    all_windows = []
    all_ids = []

    for country in ['FR', 'DE']:
        country_df = test_df[test_df['COUNTRY'] == country].copy()
        if len(country_df) == 0:
            continue

        # Sort by DAY_ID for temporal ordering
        country_df = country_df.sort_values('DAY_ID').reset_index(drop=True)

        # Get IDs and features
        ids = country_df['ID'].values
        X_raw = country_df[feature_cols].values.astype(np.float32)

        # Handle NaN using forward/backward fill
        for i in range(X_raw.shape[1]):
            col = X_raw[:, i]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                # Forward fill
                for j in range(1, len(col)):
                    if nan_mask[j] and not nan_mask[j-1]:
                        col[j] = col[j-1]
                        nan_mask[j] = False
                # Backward fill
                for j in range(len(col)-2, -1, -1):
                    if nan_mask[j] and not nan_mask[j+1]:
                        col[j] = col[j+1]
                        nan_mask[j] = False
                # Fill remaining with mean from X_moments
                col[np.isnan(col)] = X_moments[i, 0]
                X_raw[:, i] = col

        # Normalize
        X_norm = (X_raw - X_moments[:, 0]) / X_moments[:, 1]

        # Create windows
        # For samples at the start, we need to pad
        n_samples = len(country_df)
        n_features = len(feature_cols)

        for i in range(n_samples):
            # Create window ending at sample i
            if i >= timestep - 1:
                # Full window available
                window = X_norm[i - timestep + 1:i + 1]
            else:
                # Pad with first sample values
                window = np.zeros((timestep, n_features))
                available = i + 1
                window[-available:] = X_norm[:available]
                # Pad with first values
                window[:-available] = X_norm[0]

            all_windows.append(window)
            all_ids.append(ids[i])

    X_windows = np.array(all_windows)  # (n_samples, timestep, n_features)
    sample_ids = np.array(all_ids)

    return X_windows, sample_ids


def generate_predictions(
    model: DSSSM,
    X_windows: np.ndarray,
    Y_moments: np.ndarray,
    device: str = 'cpu',
    batch_size: int = 64
) -> np.ndarray:
    """
    Generate predictions using the model.

    Args:
        model: Trained DSSSM model
        X_windows: Input features (n_samples, timestep, n_features)
        Y_moments: [mean, std] for denormalization
        device: PyTorch device
        batch_size: Batch size for inference

    Returns:
        predictions: Denormalized predictions (n_samples,)
    """
    model.eval()
    n_samples = len(X_windows)
    predictions = []

    # DS3M expects input shape: (timestep, batch, features)
    # Our X_windows is: (batch, timestep, features)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_X = X_windows[i:i+batch_size]

            # Transpose to DS3M format: (timestep, batch, features)
            X_tensor = torch.from_numpy(batch_X).float().to(device)
            X_tensor = X_tensor.permute(1, 0, 2)  # (timestep, batch, features)

            # Create dummy Y input (we'll use X as both input for encoder)
            # For prediction, we need Y_dim=1 output
            # DS3M uses X for encoder input, Y for decoder target
            # At inference, we pass zeros for Y and let model predict
            Y_dummy = torch.zeros(X_tensor.shape[0], X_tensor.shape[1], 1).to(device)

            # Forward pass
            # DS3M returns:
            # outputs[0-2]: losses (kld_g, kld_c, nll)
            # outputs[3]: z_posterior tuple
            # outputs[4]: y_emission tuple (mean_list, std_list)
            # outputs[5-8]: regime info
            outputs = model(X_tensor, Y_dummy)

            # Get predictions from y_emission
            y_emission = outputs[4]  # tuple: (mean_list, std_list)
            all_y_mean, all_y_std = y_emission
            # all_y_mean is a list of tensors, each (batch, y_dim)
            # Stack and take last timestep
            y_pred_stack = torch.stack(all_y_mean, dim=0)  # (timestep+1, batch, y_dim)
            pred = y_pred_stack[-1, :, 0].cpu().numpy()  # (batch,)

            predictions.extend(pred)

    predictions = np.array(predictions)

    # Denormalize
    predictions = predictions * Y_moments[1] + Y_moments[0]

    return predictions


def create_submission(
    model_dir: Path,
    output_path: Path,
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    Create submission file.

    Args:
        model_dir: Directory with trained model
        output_path: Path to save submission CSV
        device: PyTorch device

    Returns:
        submission: DataFrame with ID, TARGET
    """
    print("=" * 60)
    print("MULTIVARIATE DS3M SUBMISSION")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {model_dir}")
    model, config = load_multivariate_model(model_dir, device)
    timestep = config.get('timestep', 14)
    print(f"  Config: d_dim={config['d_dim']}, timestep={timestep}")

    # Load training data for normalization
    print("\nLoading training data for normalization...")
    train_df, X_moments, Y_moments, feature_cols = load_training_data()
    print(f"  Features: {len(feature_cols)}")
    print(f"  Y_moments: mean={Y_moments[0]:.4f}, std={Y_moments[1]:.4f}")

    # Load test data
    print("\nLoading test data...")
    test_df = load_test_data()
    print(f"  Test samples: {len(test_df)}")
    print(f"    FR: {len(test_df[test_df['COUNTRY'] == 'FR'])}")
    print(f"    DE: {len(test_df[test_df['COUNTRY'] == 'DE'])}")

    # Prepare test features
    print("\nPreparing test features...")
    X_windows, sample_ids = prepare_test_features(
        test_df, feature_cols, X_moments, timestep
    )
    print(f"  Windows shape: {X_windows.shape}")
    print(f"  Sample IDs: {len(sample_ids)}")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = generate_predictions(model, X_windows, Y_moments, device)
    print(f"  Predictions: {len(predictions)}")
    print(f"  Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")

    # Create submission DataFrame
    submission = pd.DataFrame({
        'ID': sample_ids,
        'TARGET': predictions
    })

    # Verify all test IDs are present
    test_ids = set(test_df['ID'].values)
    pred_ids = set(submission['ID'].values)
    missing = test_ids - pred_ids
    extra = pred_ids - test_ids

    print(f"\nVerification:")
    print(f"  Expected IDs: {len(test_ids)}")
    print(f"  Generated IDs: {len(pred_ids)}")
    print(f"  Missing: {len(missing)}")
    print(f"  Extra: {len(extra)}")

    if missing:
        print(f"  WARNING: Missing IDs: {list(missing)[:10]}...")

    # Sort by ID for consistency with expected format
    submission = submission.sort_values('ID').reset_index(drop=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"  Rows: {len(submission)}")

    # Show first few rows
    print(f"\nFirst 5 rows:")
    print(submission.head().to_string(index=False))

    return submission


def main():
    parser = argparse.ArgumentParser(description="Generate DS3M multivariate submission")
    parser.add_argument('--model_dir', type=str,
                        default=str(OUTPUT_DIR) + '/ds3m/ALL_TARGET_mv_d2_seed44_20260120_140929',
                        help='Directory with trained model')
    parser.add_argument('--output', type=str,
                        default=str(SUBMISSIONS_DIR) + '/DS3M-MV_Y_test.csv',
                        help='Output path for submission')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')

    args = parser.parse_args()

    submission = create_submission(
        model_dir=Path(args.model_dir),
        output_path=Path(args.output),
        device=args.device
    )

    print("\n" + "=" * 60)
    print("SUBMISSION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
