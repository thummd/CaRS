"""
FANTOM Cross-Sectional Submission for QRT Electricity Challenge.

Uses FANTOM with lag=0 to learn instantaneous causal relationships
between features and TARGET for cross-sectional prediction.

Key insight: No temporal structure! Each sample is independent.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA_DIR, FANTOM_CODE_DIR, SUBMISSIONS_DIR
FANTOM_PATH = str(FANTOM_CODE_DIR)
sys.path.insert(0, FANTOM_PATH)

from fantom_electricity import FANTOMElectricity, create_model

# Paths
DATA_DIR = DATA_DIR / "qrt"
OUTPUT_DIR = SUBMISSIONS_DIR


def prepare_cross_sectional_data(X_df: pd.DataFrame, Y_df: pd.DataFrame = None):
    """
    Prepare data for cross-sectional FANTOM (lag=0).

    Each sample is independent - no temporal windows.
    Input shape: [N, 1, num_features]  (lag+1=1 when lag=0)

    Args:
        X_df: Features DataFrame (with ID, DAY_ID, COUNTRY)
        Y_df: Optional targets DataFrame (with ID, TARGET)

    Returns:
        X_tensor: Tensor of shape [N, 1, num_nodes]
        feature_cols: List of feature column names
        ids: Sample IDs
        targets: TARGET values (if Y_df provided)
    """
    # Drop COUNTRY (categorical) - keep ID and DAY_ID as features!
    X_clean = X_df.drop(['COUNTRY'], axis=1).fillna(0)
    feature_cols = list(X_clean.columns)

    ids = X_df['ID'].values

    # Get data values
    X_values = X_clean.values

    # If we have targets, append them as an extra column
    if Y_df is not None:
        # Merge to ensure alignment
        merged = X_clean.merge(Y_df[['ID', 'TARGET']], on='ID', how='left')
        X_values = merged.drop('ID', axis=1).values  # Features without ID
        targets = merged['TARGET'].values
        feature_cols = list(merged.drop(['ID', 'TARGET'], axis=1).columns) + ['TARGET']

        # Rebuild X_values with TARGET as last column
        X_values = merged[feature_cols].values
    else:
        targets = None
        # For test data, append placeholder for TARGET
        X_values = np.hstack([X_values, np.zeros((len(X_values), 1))])
        feature_cols = feature_cols + ['TARGET']

    # Reshape for FANTOM: [N, lag+1, num_nodes] = [N, 1, num_features]
    X_tensor = torch.tensor(X_values, dtype=torch.float32).unsqueeze(1)

    return X_tensor, feature_cols, ids, targets


def train_fantom_cross_sectional(
    X_train: torch.Tensor,
    y_train: np.ndarray,
    feature_cols: list,
    device: str = "cpu",
    n_epochs: int = 500,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    verbose: bool = True
):
    """
    Train FANTOM with lag=0 for cross-sectional prediction.

    Args:
        X_train: Training data [N, 1, num_nodes]
        y_train: TARGET values [N]
        feature_cols: Feature column names (TARGET should be last)
        device: PyTorch device
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        verbose: Print progress

    Returns:
        model: Trained FANTOMElectricity model
    """
    num_nodes = X_train.shape[2]
    target_idx = num_nodes - 1  # TARGET is last column

    device_obj = torch.device(device)

    # Create model with lag=0
    model = create_model(
        num_nodes=num_nodes,
        target_idx=target_idx,
        lag=0,  # Cross-sectional: no temporal lag!
        device=device,
        model_config={
            'allow_instantaneous': True,  # Same-sample relationships
            'constrain_target': True,  # TARGET has no outgoing edges
            'lambda_dag': 50.0,  # DAG penalty
            'lambda_sparse': 0.5,  # Sparsity penalty
            'base_distribution_type': 'spline',
            'spline_bins': 8,
            'encoder_layer_sizes': [64, 64],
            'decoder_layer_sizes': [64, 64],
            'heteroscedastic': True,
        }
    )

    # Put TARGET values into the data tensor
    X_train_with_target = X_train.clone()
    X_train_with_target[:, 0, target_idx] = torch.tensor(y_train, dtype=torch.float32)

    # Move to device
    X_train_with_target = X_train_with_target.to(device_obj)

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(X_train_with_target)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_data, in dataloader:
            optimizer.zero_grad()

            # Compute ELBO loss
            loss = model.compute_loss(batch_data)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / n_batches

            # Evaluate on training data
            model.eval()
            with torch.no_grad():
                pred = model.predict_target(X_train_with_target).cpu().numpy()
            spearman, _ = spearmanr(y_train, pred)
            model.train()

            print(f"Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Train Spearman={spearman:.4f}")

    return model


def main():
    print("=" * 60)
    print("FANTOM Cross-Sectional Submission")
    print("lag=0, allow_instantaneous=True")
    print("=" * 60)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load data
    X_train_df = pd.read_csv(DATA_DIR / "X_train_NHkHMNU.csv")
    Y_train_df = pd.read_csv(DATA_DIR / "y_train_ZAN5mwg.csv")
    X_test_df = pd.read_csv(DATA_DIR / "X_test_final.csv")

    print(f"\nData shapes:")
    print(f"  X_train: {X_train_df.shape}")
    print(f"  Y_train: {Y_train_df.shape}")
    print(f"  X_test:  {X_test_df.shape}")

    # Prepare training data
    X_train, feature_cols, train_ids, y_train = prepare_cross_sectional_data(
        X_train_df, Y_train_df
    )
    print(f"\nPrepared training data: {X_train.shape}")
    print(f"Feature columns: {len(feature_cols)} (including TARGET)")
    print(f"TARGET index: {len(feature_cols) - 1}")

    # 5-fold CV to estimate performance
    print("\n" + "=" * 60)
    print("5-Fold Cross-Validation")
    print("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold+1}/5:")

        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        # Train model
        model = train_fantom_cross_sectional(
            X_tr, y_tr, feature_cols,
            device=device,
            n_epochs=200,  # Reduced for CV
            batch_size=64,
            learning_rate=0.001,
            verbose=False
        )

        # Prepare validation data with TARGET placeholder
        X_val_with_target = X_val.clone()
        X_val_with_target[:, 0, -1] = torch.tensor(y_val, dtype=torch.float32)

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_pred = model.predict_target(X_val_with_target.to(device)).cpu().numpy()

        spearman, _ = spearmanr(y_val, val_pred)
        cv_scores.append(spearman)
        print(f"  Validation Spearman: {spearman:.4f}")

    print(f"\nCV Mean: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

    # Train final model on all data
    print("\n" + "=" * 60)
    print("Training Final Model")
    print("=" * 60)

    final_model = train_fantom_cross_sectional(
        X_train, y_train, feature_cols,
        device=device,
        n_epochs=500,
        batch_size=64,
        learning_rate=0.001,
        verbose=True
    )

    # Evaluate on training data
    final_model.eval()
    X_train_with_target = X_train.clone()
    X_train_with_target[:, 0, -1] = torch.tensor(y_train, dtype=torch.float32)

    with torch.no_grad():
        train_pred = final_model.predict_target(X_train_with_target.to(device)).cpu().numpy()

    train_spearman, _ = spearmanr(y_train, train_pred)
    print(f"\nFinal Training Spearman: {train_spearman:.4f}")

    # Get causal parents of TARGET
    print("\n" + "=" * 60)
    print("Causal Parents of TARGET")
    print("=" * 60)
    parents = final_model.get_causal_parents(feature_cols, threshold=0.3)
    print("\nInstantaneous parents:")
    for name, lag, weight in parents['instantaneous'][:10]:
        print(f"  {name}: weight={weight:.4f}")

    # Prepare test data
    X_test, _, test_ids, _ = prepare_cross_sectional_data(X_test_df)
    print(f"\nPrepared test data: {X_test.shape}")

    # Generate predictions
    # Note: For test data, we don't have TARGET values, so we use placeholder
    # The model predicts TARGET based on other features
    with torch.no_grad():
        test_pred = final_model.predict_target(X_test.to(device)).cpu().numpy()

    print(f"\nTest prediction statistics:")
    print(f"  Mean: {test_pred.mean():.4f}")
    print(f"  Std:  {test_pred.std():.4f}")
    print(f"  Min:  {test_pred.min():.4f}")
    print(f"  Max:  {test_pred.max():.4f}")

    # Check for bimodal distribution
    print(f"\nPrediction distribution check:")
    print(f"  25%: {np.percentile(test_pred, 25):.4f}")
    print(f"  50%: {np.percentile(test_pred, 50):.4f}")
    print(f"  75%: {np.percentile(test_pred, 75):.4f}")

    # Create submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'TARGET': test_pred
    })
    submission = submission.sort_values('ID').reset_index(drop=True)

    # Save
    output_path = OUTPUT_DIR / "FANTOM_CrossSectional_Y_test.csv"
    submission.to_csv(output_path, index=False)

    print(f"\nSubmission saved to: {output_path}")
    print(f"Submission shape: {submission.shape}")
    print(f"\nFirst 5 rows:")
    print(submission.head())

    return submission


if __name__ == "__main__":
    main()
