"""
Test DS3M on DE-FR Spread Prediction.

Predicts the daily change in the electricity price spread between Germany and France.
The spread is defined as DE_price - FR_price.

Usage:
    python3 test_ds3m_spread.py
    python3 test_ds3m_spread.py --d_dim 3 --seed 123
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

from unified_data_loader import prepare_unified_ds3m_data, FEATURE_GROUPS_DE_FR

# Try to import DS3M
try:
    from DSSSMCode import DSSSM
    DS3M_AVAILABLE = True
except ImportError:
    DS3M_AVAILABLE = False
    print("WARNING: DS3M not available. Will run data verification only.")


def compute_baseline_predictions(Y_train: np.ndarray, Y_test: np.ndarray) -> dict:
    """Compute baseline predictions for comparison."""
    n_test = len(Y_test)

    # Persistence baseline (yesterday's value)
    persistence_pred = np.roll(Y_test, 1)
    persistence_pred[0] = Y_train[-1] if len(Y_train) > 0 else 0

    # Mean baseline
    mean_pred = np.full(n_test, Y_train.mean())

    # Zero baseline
    zero_pred = np.zeros(n_test)

    baselines = {}
    for name, pred in [('persistence', persistence_pred),
                       ('mean', mean_pred),
                       ('zero', zero_pred)]:
        corr, pval = spearmanr(Y_test, pred)
        rmse = np.sqrt(np.mean((Y_test - pred) ** 2))
        baselines[name] = {
            'spearman': corr,
            'spearman_pval': pval,
            'rmse': rmse
        }

    return baselines


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
    """Train DS3M model on spread data."""
    device = torch.device(device)

    x_dim = trainX.shape[2]
    y_dim = trainY.shape[2]

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

    trainX = trainX.to(device)
    trainY = trainY.to(device)
    valX = valX.to(device)
    valY = valY.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.001)
    )

    n_epochs = config.get('n_epochs', 100)
    batch_size = config.get('batch_size', 64)
    n_samples = trainX.shape[1]

    history = {
        'train_loss': [],
        'val_spearman': [],
        'val_rmse': []
    }

    best_val_spearman = -float('inf')
    best_state = None
    patience = config.get('patience', 20)
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        indices = torch.randperm(n_samples)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            X_batch = trainX[:, batch_idx, :]
            Y_batch = trainY[:, batch_idx, :]

            optimizer.zero_grad()

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
            val_outputs = model(valX, valY)
            y_emission = val_outputs[4]
            all_y_mean, _ = y_emission

            y_pred_stack = torch.stack(all_y_mean, dim=0)
            pred = y_pred_stack[-1, :, 0].cpu().numpy()
            true = valY[-1, :, 0].cpu().numpy()

            # Denormalize
            pred_denorm = pred * Y_moments[1] + Y_moments[0]
            true_denorm = true * Y_moments[1] + Y_moments[0]

            spearman, _ = spearmanr(true_denorm, pred_denorm)
            rmse = np.sqrt(np.mean((true_denorm - pred_denorm) ** 2))

            history['val_spearman'].append(spearman)
            history['val_rmse'].append(rmse)

            if spearman > best_val_spearman:
                best_val_spearman = spearman
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, "
                  f"Val Spearman={spearman:.4f}, RMSE={rmse:.2f}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(
    model,
    testX: torch.Tensor,
    testY: torch.Tensor,
    Y_moments: np.ndarray,
    device: str = 'cpu'
) -> dict:
    """Evaluate DS3M model on test data."""
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

    return {
        'spearman': spearman,
        'spearman_pval': pval,
        'rmse': rmse,
        'mae': mae,
        'pred_mean': pred_denorm.mean(),
        'pred_std': pred_denorm.std(),
        'true_mean': true_denorm.mean(),
        'true_std': true_denorm.std(),
        'predictions': pred_denorm,
        'true_values': true_denorm
    }


def main():
    parser = argparse.ArgumentParser(description='Test DS3M on DE-FR spread prediction')
    parser.add_argument('--timestep', type=int, default=14,
                        help='Lookback window size')
    parser.add_argument('--d_dim', type=int, default=2,
                        help='Number of discrete states (regimes)')
    parser.add_argument('--h_dim', type=int, default=30,
                        help='Hidden dimension')
    parser.add_argument('--z_dim', type=int, default=8,
                        help='Latent dimension')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')
    parser.add_argument('--feature_groups', type=str, nargs='+',
                        default=['spread', 'price_de', 'price_fr', 'load_de', 'load_fr', 'calendar'],
                        help='Feature groups to use')
    parser.add_argument('--data_only', action='store_true',
                        help='Only verify data loading, skip training')

    args = parser.parse_args()

    # Handle comma-separated feature groups (for HTCondor)
    if len(args.feature_groups) == 1 and ',' in args.feature_groups[0]:
        args.feature_groups = [g.strip() for g in args.feature_groups[0].split(',')]

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("DS3M Test on DE-FR Spread Prediction")
    print("=" * 70)
    print(f"Feature groups: {args.feature_groups}")
    print(f"Timestep: {args.timestep}")

    # Load data
    print("\n--- Loading Data ---")
    data = prepare_unified_ds3m_data(
        country='DE_FR',
        timestep=args.timestep,
        feature_groups=args.feature_groups,
        target_col='price_spread_change_pct'
    )

    print(f"Target: {data['target_col']}")
    print(f"Features: {len(data['feature_cols'])}")
    print(f"Train X: {data['trainX'].shape}")
    print(f"Val X:   {data['valX'].shape}")
    print(f"Test X:  {data['testX'].shape}")
    print(f"Y moments: mean={data['Y_moments'][0]:.2f}, std={data['Y_moments'][1]:.2f}")
    print(f"\nTrain period: {data['timestamps']['train'].min().date()} to {data['timestamps']['train'].max().date()}")
    print(f"Val period:   {data['timestamps']['val'].min().date()} to {data['timestamps']['val'].max().date()}")
    print(f"Test period:  {data['timestamps']['test'].min().date()} to {data['timestamps']['test'].max().date()}")

    # Compute baselines
    print("\n--- Computing Baselines ---")
    test_Y_denorm = data['testY'][-1, :, 0].numpy() * data['Y_moments'][1] + data['Y_moments'][0]
    train_Y_denorm = data['trainY'][-1, :, 0].numpy() * data['Y_moments'][1] + data['Y_moments'][0]

    baselines = compute_baseline_predictions(train_Y_denorm, test_Y_denorm)
    for name, metrics in baselines.items():
        print(f"  {name}: Spearman={metrics['spearman']:.4f}, RMSE={metrics['rmse']:.2f}")

    if args.data_only:
        print("\n--- Data Verification Complete (--data_only) ---")
        return

    if not DS3M_AVAILABLE:
        print("\n--- DS3M not available, skipping training ---")
        return

    # Config
    config = {
        'h_dim': args.h_dim,
        'z_dim': args.z_dim,
        'd_dim': args.d_dim,
        'n_layers': 1,
        'learning_rate': args.lr,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'patience': 20
    }

    print(f"\n--- Training DS3M on Spread ---")
    print(f"Config: d_dim={args.d_dim}, h_dim={args.h_dim}, z_dim={args.z_dim}")

    model, history = train_ds3m(
        data['trainX'], data['trainY'],
        data['valX'], data['valY'],
        data['Y_moments'],
        config,
        device=args.device,
        verbose=True
    )

    # Evaluate
    print("\n--- Evaluating on Test Set ---")
    metrics = evaluate_model(
        model,
        data['testX'], data['testY'],
        data['Y_moments'],
        device=args.device
    )

    print(f"\nTest Results:")
    print(f"  Spearman: {metrics['spearman']:.4f} (p={metrics['spearman_pval']:.4e})")
    print(f"  RMSE:     {metrics['rmse']:.2f}")
    print(f"  MAE:      {metrics['mae']:.2f}")
    print(f"\n  Prediction stats: mean={metrics['pred_mean']:.2f}, std={metrics['pred_std']:.2f}")
    print(f"  True stats:       mean={metrics['true_mean']:.2f}, std={metrics['true_std']:.2f}")

    # Compare with baselines
    print("\n--- Comparison with Baselines ---")
    print(f"  DS3M:        Spearman={metrics['spearman']:.4f}, RMSE={metrics['rmse']:.2f}")
    for name, baseline in baselines.items():
        improvement = metrics['spearman'] - baseline['spearman']
        print(f"  {name:12s}: Spearman={baseline['spearman']:.4f}, RMSE={baseline['rmse']:.2f} "
              f"(DS3M {'+' if improvement > 0 else ''}{improvement:.4f})")

    # Save results
    save_dir = Path(__file__).parent / "outputs" / f"test_ds3m_spread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'task': 'spread_prediction',
        'target': 'price_spread_change_pct',
        'feature_groups': args.feature_groups,
        'n_features': len(data['feature_cols']),
        'timestep': args.timestep,
        'config': config,
        'metrics': {
            'spearman': float(metrics['spearman']),
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae'])
        },
        'baselines': {k: {kk: float(vv) if not np.isnan(vv) else None for kk, vv in v.items()} for k, v in baselines.items()},
        'best_val_spearman': float(max(history['val_spearman']))
    }

    with open(save_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), save_dir / "model.pt")
    np.save(save_dir / "predictions.npy", metrics['predictions'])
    np.save(save_dir / "true_values.npy", metrics['true_values'])

    print(f"\nResults saved to: {save_dir}")

    print("\n" + "=" * 70)
    if metrics['spearman'] > baselines['persistence']['spearman']:
        print("SUCCESS: DS3M beats persistence baseline on spread prediction!")
    else:
        print("NOTE: DS3M did not beat persistence baseline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
