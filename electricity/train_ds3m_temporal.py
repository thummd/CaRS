"""
Train DS3M on temporally-ordered electricity data.

Uses epftoolbox data (properly temporal) instead of QRT (cross-sectional).
This is the correct use case for DS3M's regime-switching SSM.
"""

import sys
import os
import argparse
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

from paths import DS3M_DIR
# Add DS3M to path
DS3M_PATH = str(DS3M_DIR)
sys.path.insert(0, DS3M_PATH)
sys.path.insert(0, os.path.join(DS3M_PATH, "src"))

from temporal_data_loader import (
    prepare_ds3m_data,
    prepare_univariate_ds3m_data
)

# Import DS3M model
from DSSSMCode import DSSSM


def train_ds3m(
    trainX: torch.Tensor,
    trainY: torch.Tensor,
    valX: torch.Tensor,
    valY: torch.Tensor,
    Y_moments: np.ndarray,
    config: dict,
    device: str = 'cpu',
    verbose: bool = True
) -> tuple:
    """
    Train DS3M model on temporal electricity data.

    Args:
        trainX: Training input (timestep, batch, x_dim)
        trainY: Training target (timestep, batch, y_dim)
        valX: Validation input
        valY: Validation target
        Y_moments: [mean, std] for denormalization
        config: Model configuration
        device: PyTorch device
        verbose: Print progress

    Returns:
        model: Trained DS3M model
        history: Training history
    """
    device = torch.device(device)

    # Model dimensions
    x_dim = trainX.shape[2]
    y_dim = trainY.shape[2]

    # Create model
    model = DSSSM(
        x_dim=x_dim,
        y_dim=y_dim,
        h_dim=config.get('h_dim', 30),
        z_dim=config.get('z_dim', 8),
        d_dim=config.get('d_dim', 2),
        n_layers=config.get('n_layers', 1),
        device=device,
        bidirection=False
    ).to(device)

    # Move data to device
    trainX = trainX.to(device)
    trainY = trainY.to(device)
    valX = valX.to(device)
    valY = valY.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.001)
    )

    # Training loop
    n_epochs = config.get('n_epochs', 200)
    batch_size = config.get('batch_size', 64)
    n_samples = trainX.shape[1]

    history = {
        'train_loss': [],
        'val_spearman': [],
        'val_rmse': []
    }

    best_val_spearman = -float('inf')
    best_state = None
    patience = config.get('patience', 30)
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Random batch indices
        indices = torch.randperm(n_samples)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            X_batch = trainX[:, batch_idx, :]
            Y_batch = trainY[:, batch_idx, :]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch, Y_batch)
            kld_g, kld_c, nll = outputs[0], outputs[1], outputs[2]
            loss = kld_g + kld_c + nll

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history['train_loss'].append(avg_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            # Get predictions
            val_outputs = model(valX, valY)
            y_emission = val_outputs[4]  # (mean_list, std_list)
            all_y_mean, _ = y_emission

            # Stack predictions
            y_pred_stack = torch.stack(all_y_mean, dim=0)  # (timestep+1, batch, y_dim)
            pred = y_pred_stack[-1, :, 0].cpu().numpy()  # Last timestep

            # True values (last timestep)
            true = valY[-1, :, 0].cpu().numpy()

            # Denormalize
            pred_denorm = pred * Y_moments[1] + Y_moments[0]
            true_denorm = true * Y_moments[1] + Y_moments[0]

            # Metrics
            spearman, _ = spearmanr(true_denorm, pred_denorm)
            rmse = np.sqrt(np.mean((true_denorm - pred_denorm) ** 2))

            history['val_spearman'].append(spearman)
            history['val_rmse'].append(rmse)

            # Early stopping
            if spearman > best_val_spearman:
                best_val_spearman = spearman
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, "
                  f"Val Spearman={spearman:.4f}, RMSE={rmse:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(
    model: DSSSM,
    testX: torch.Tensor,
    testY: torch.Tensor,
    Y_moments: np.ndarray,
    device: str = 'cpu'
) -> dict:
    """
    Evaluate DS3M model on test data.

    Args:
        model: Trained model
        testX: Test input
        testY: Test target
        Y_moments: Normalization parameters
        device: PyTorch device

    Returns:
        Dictionary with metrics
    """
    device = torch.device(device)
    model = model.to(device)
    testX = testX.to(device)
    testY = testY.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(testX, testY)
        y_emission = outputs[4]
        all_y_mean, _ = y_emission

        y_pred_stack = torch.stack(all_y_mean, dim=0)
        pred = y_pred_stack[-1, :, 0].cpu().numpy()
        true = testY[-1, :, 0].cpu().numpy()

        # Denormalize
        pred_denorm = pred * Y_moments[1] + Y_moments[0]
        true_denorm = true * Y_moments[1] + Y_moments[0]

        # Metrics
        spearman, pval = spearmanr(true_denorm, pred_denorm)
        rmse = np.sqrt(np.mean((true_denorm - pred_denorm) ** 2))
        mae = np.mean(np.abs(true_denorm - pred_denorm))

        # Prediction distribution
        pred_mean = pred_denorm.mean()
        pred_std = pred_denorm.std()
        pred_min = pred_denorm.min()
        pred_max = pred_denorm.max()

    return {
        'spearman': spearman,
        'spearman_pval': pval,
        'rmse': rmse,
        'mae': mae,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'pred_min': pred_min,
        'pred_max': pred_max,
        'n_samples': len(pred)
    }


def main():
    parser = argparse.ArgumentParser(description='Train DS3M on temporal electricity data')
    parser.add_argument('--dataset', type=str, default='DE', choices=['DE', 'FR'],
                        help='Dataset to use')
    parser.add_argument('--mode', type=str, default='univariate',
                        choices=['univariate', 'multivariate'],
                        help='Univariate (TARGET only) or multivariate (all features)')
    parser.add_argument('--timestep', type=int, default=14,
                        help='Lookback window size')
    parser.add_argument('--d_dim', type=int, default=2,
                        help='Number of discrete states (regimes)')
    parser.add_argument('--h_dim', type=int, default=30,
                        help='Hidden dimension')
    parser.add_argument('--z_dim', type=int, default=8,
                        help='Latent dimension')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("="*60)
    print(f"DS3M Training on Temporal Electricity Data")
    print(f"Dataset: {args.dataset}, Mode: {args.mode}")
    print("="*60)

    # Load data
    print("\nLoading data...")
    if args.mode == 'univariate':
        data = prepare_univariate_ds3m_data(
            dataset=args.dataset,
            timestep=args.timestep,
            test_ratio=0.2,
            val_ratio=0.1
        )
        Y_moments = data['Y_moments']
    else:
        data = prepare_ds3m_data(
            dataset=args.dataset,
            timestep=args.timestep,
            test_ratio=0.2,
            val_ratio=0.1,
            include_exogenous=True
        )
        Y_moments = data['Y_moments']

    print(f"Train: {data['trainX'].shape}")
    print(f"Val:   {data['valX'].shape}")
    print(f"Test:  {data['testX'].shape}")

    # Config
    config = {
        'h_dim': args.h_dim,
        'z_dim': args.z_dim,
        'd_dim': args.d_dim,
        'n_layers': 1,
        'learning_rate': args.lr,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'patience': 30
    }

    print(f"\nModel config: {config}")

    # Train
    print("\nTraining...")
    model, history = train_ds3m(
        data['trainX'], data['trainY'],
        data['valX'], data['valY'],
        Y_moments,
        config,
        device=args.device,
        verbose=True
    )

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate_model(
        model,
        data['testX'], data['testY'],
        Y_moments,
        device=args.device
    )

    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Spearman:    {metrics['spearman']:.4f} (p={metrics['spearman_pval']:.4e})")
    print(f"RMSE:        {metrics['rmse']:.4f}")
    print(f"MAE:         {metrics['mae']:.4f}")
    print(f"\nPrediction distribution:")
    print(f"  Mean: {metrics['pred_mean']:.4f}")
    print(f"  Std:  {metrics['pred_std']:.4f}")
    print(f"  Range: [{metrics['pred_min']:.4f}, {metrics['pred_max']:.4f}]")

    # Save results
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(__file__).parent / "outputs" / f"temporal_{args.dataset}_{args.mode}_d{args.d_dim}_{timestamp}"

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), save_dir / "model.pt")

    # Save config and results
    results = {
        'dataset': args.dataset,
        'mode': args.mode,
        'timestep': args.timestep,
        'seed': args.seed,
        'config': config,
        'metrics': {k: float(v) for k, v in metrics.items()},
        'best_val_spearman': max(history['val_spearman']),
        'final_train_loss': history['train_loss'][-1]
    }

    with open(save_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {save_dir}")

    return model, metrics


if __name__ == "__main__":
    main()
