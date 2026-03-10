"""
Test FANTOM on unified electricity data.

This script verifies that FANTOM can train and produce meaningful predictions
using the unified dataset (ENTSO-E + weather + calendar + outages + commodities).

Usage: python3 test_fantom_unified.py --dataset DE
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

from paths import FANTOM_CODE_DIR
# Add paths
FANTOM_PATH = str(FANTOM_CODE_DIR)
sys.path.insert(0, FANTOM_PATH)
sys.path.insert(0, str(Path(__file__).parent))

from unified_data_loader import prepare_unified_fantom_data

# Try to import FANTOMElectricity
try:
    from fantom_electricity import FANTOMElectricity
    FANTOM_AVAILABLE = True
except ImportError as e:
    FANTOM_AVAILABLE = False
    print(f"WARNING: FANTOMElectricity not available: {e}")


def compute_baseline_predictions(Y_train: np.ndarray, Y_test: np.ndarray) -> dict:
    """Compute baseline predictions for comparison."""
    n_test = len(Y_test)

    # Persistence baseline
    persistence_pred = np.roll(Y_test, 1)
    persistence_pred[0] = Y_train[-1] if len(Y_train) > 0 else 0

    # Mean baseline
    mean_pred = np.full(n_test, Y_train.mean())

    baselines = {}
    for name, pred in [('persistence', persistence_pred), ('mean', mean_pred)]:
        corr, pval = spearmanr(Y_test, pred)
        rmse = np.sqrt(np.mean((Y_test - pred) ** 2))
        baselines[name] = {'spearman': corr, 'rmse': rmse}

    return baselines


def compute_loss_from_elbo_terms(elbo_terms: dict, batch_size: int, lambda_dag: float = 1.0) -> tuple:
    """
    Compute loss from ELBO terms returned by model._ELBO_terms().

    Uses a simpler, more stable loss formulation.
    """
    # Log likelihood term (main objective)
    log_p_term = elbo_terms["log_p_base"].mean()

    # Prior and entropy terms for graph
    log_p_A_term = elbo_terms["log_p_A"] / batch_size
    log_q_A_term = elbo_terms["log_q_A"] / batch_size

    # DAG penalty (should be 0 for acyclic graphs)
    penalty_dag = elbo_terms["penalty_dag"]

    # Simple ELBO: maximize log p(x) + log p(A) - log q(A) - lambda * DAG_penalty
    elbo = log_p_term + log_p_A_term - log_q_A_term - lambda_dag * penalty_dag
    loss = -elbo

    return loss, penalty_dag


def train_fantom(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: dict,
    target_idx: int,
    n_nodes: int,
    lag: int,
    device: str = 'cpu',
    verbose: bool = True
) -> tuple:
    """
    Train FANTOM model on unified electricity data.
    """
    device_obj = torch.device(device)

    # Create model using FANTOMElectricity wrapper
    model = FANTOMElectricity(
        num_nodes=n_nodes,
        device=device_obj,
        target_idx=target_idx,
        lag=lag,
        allow_instantaneous=True,
        constrain_target=True,
        lambda_dag=config.get('lambda_dag', 100.0),
        lambda_sparse=config.get('lambda_sparse', 1.0),
        tau_gumbel=config.get('tau_gumbel', 1.0),
        base_distribution_type='spline',
        spline_bins=8,
        var_dist_A_mode='temporal_three',
        norm_layers=True,
        res_connection=True,
        encoder_layer_sizes=[32, 32],
        decoder_layer_sizes=[32, 32],
        heteroscedastic=True
    ).to(device_obj)

    # Move data to device
    train_data = train_data.to(device_obj)
    val_data = val_data.to(device_obj)

    # Optimizer with lower learning rate for stability
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.0005)
    )

    # Learning rate scheduler (verbose removed in newer PyTorch)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    n_epochs = config.get('n_epochs', 100)
    batch_size = config.get('batch_size', 64)
    n_samples = train_data.shape[0]

    history = {'train_loss': [], 'val_loss': [], 'penalty_dag': [], 'log_p': []}
    best_val_loss = float('inf')
    best_state = None
    patience = config.get('patience', 30)
    patience_counter = 0

    # Fixed DAG penalty weight (smaller for stability)
    lambda_dag = config.get('lambda_dag', 100.0) / 100.0  # Scale down

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dag_penalty = 0.0
        epoch_log_p = 0.0
        n_batches = 0

        indices = torch.randperm(n_samples, device=device_obj)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            batch = train_data[batch_idx]
            current_batch_size = batch.shape[0]

            optimizer.zero_grad()

            # Compute ELBO terms and loss
            elbo_terms = model._ELBO_terms(batch)
            loss, dag_penalty = compute_loss_from_elbo_terms(elbo_terms, current_batch_size, lambda_dag)

            # Check for NaN
            if torch.isnan(loss):
                if verbose:
                    print(f"Warning: NaN loss at epoch {epoch+1}, batch {n_batches}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dag_penalty += dag_penalty.item()
            epoch_log_p += elbo_terms["log_p_base"].mean().item()
            n_batches += 1

        if n_batches == 0:
            if verbose:
                print(f"Epoch {epoch+1}: All batches had NaN loss, stopping")
            break

        avg_loss = epoch_loss / n_batches
        avg_dag_penalty = epoch_dag_penalty / n_batches
        avg_log_p = epoch_log_p / n_batches
        history['train_loss'].append(avg_loss)
        history['penalty_dag'].append(avg_dag_penalty)
        history['log_p'].append(avg_log_p)

        # Validation
        model.eval()
        with torch.no_grad():
            val_elbo_terms = model._ELBO_terms(val_data)
            val_loss, _ = compute_loss_from_elbo_terms(val_elbo_terms, val_data.shape[0], lambda_dag)
            val_loss = val_loss.item()
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

        # Update learning rate
        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.2f}, Val={val_loss:.2f}, DAG={avg_dag_penalty:.4f}, log_p={avg_log_p:.2f}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_fantom(
    model,
    test_data: torch.Tensor,
    moments: np.ndarray,
    target_idx: int,
    device: str = 'cpu'
) -> dict:
    """Evaluate FANTOM on test data."""
    device_obj = torch.device(device)
    model = model.to(device_obj)
    test_data = test_data.to(device_obj)

    model.eval()
    with torch.no_grad():
        # Get predictions using the model's predict_target method
        pred = model.predict_target(test_data)
        pred = pred.cpu().numpy()

        # True values (last timestep = current day, target variable)
        true = test_data[:, -1, target_idx].cpu().numpy()

        # Denormalize
        pred_denorm = pred * moments[target_idx, 1] + moments[target_idx, 0]
        true_denorm = true * moments[target_idx, 1] + moments[target_idx, 0]

        # Metrics
        spearman, pval = spearmanr(true_denorm, pred_denorm)
        rmse = np.sqrt(np.mean((true_denorm - pred_denorm) ** 2))
        mae = np.mean(np.abs(true_denorm - pred_denorm))

    return {
        'spearman': spearman,
        'spearman_pval': pval,
        'rmse': rmse,
        'mae': mae,
        'predictions': pred_denorm,
        'true_values': true_denorm
    }


def extract_causal_graph(model, feature_cols: list, threshold: float = 0.3) -> dict:
    """Extract learned causal graph from FANTOM."""
    model.eval()

    # Get causal parents of target
    parents = model.get_causal_parents(feature_cols, threshold=threshold)

    # Get full adjacency matrix
    A = model.get_adj_matrix(samples=1, most_likely_graph=True, squeeze=True)
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()

    return {
        'instantaneous_parents': parents['instantaneous'],
        'lagged_parents': parents['lagged'],
        'n_instantaneous': len(parents['instantaneous']),
        'n_lagged': len(parents['lagged']),
        'threshold': threshold
    }


def main():
    parser = argparse.ArgumentParser(description='Test FANTOM on unified electricity data')
    parser.add_argument('--dataset', type=str, default='DE', choices=['DE', 'FR', 'DE_FR'])
    parser.add_argument('--lag', type=int, default=1)
    parser.add_argument('--max_features', type=int, default=15)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lambda_sparse', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--feature_groups', type=str, nargs='+',
                        default=['price', 'load', 'weather'])
    parser.add_argument('--data_only', action='store_true')

    args = parser.parse_args()

    # Handle underscore-separated feature groups (for HTCondor)
    if len(args.feature_groups) == 1 and '_' in args.feature_groups[0]:
        # Check if it looks like underscore-separated groups
        parts = args.feature_groups[0].split('_')
        valid_groups = ['price', 'load', 'weather', 'calendar', 'outage', 'commodity', 'generation', 'flow']
        if all(p in valid_groups for p in parts):
            args.feature_groups = parts

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("FANTOM Test on Unified Electricity Data")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Feature groups: {args.feature_groups}")
    print(f"Lag: {args.lag}, Max features: {args.max_features}")

    # Load data
    print("\n--- Loading Data ---")
    data = prepare_unified_fantom_data(
        country=args.dataset,
        lag=args.lag,
        feature_groups=args.feature_groups,
        max_features=args.max_features
    )

    print(f"Nodes: {data['n_nodes']}")
    print(f"Features: {data['feature_cols']}")
    print(f"Target: {data['target_col']} (index {data['target_idx']})")
    print(f"Train shape: {data['train'].shape}")
    print(f"Val shape:   {data['val'].shape}")
    print(f"Test shape:  {data['test'].shape}")

    # Compute baselines
    print("\n--- Computing Baselines ---")
    train_target = data['train'][:, 0, data['target_idx']].numpy()
    train_target = train_target * data['moments'][data['target_idx'], 1] + data['moments'][data['target_idx'], 0]
    test_target = data['test'][:, 0, data['target_idx']].numpy()
    test_target = test_target * data['moments'][data['target_idx'], 1] + data['moments'][data['target_idx'], 0]

    baselines = compute_baseline_predictions(train_target, test_target)
    for name, metrics in baselines.items():
        print(f"  {name}: Spearman={metrics['spearman']:.4f}, RMSE={metrics['rmse']:.2f}")

    if args.data_only:
        print("\n--- Data Verification Complete (--data_only) ---")
        return

    if not FANTOM_AVAILABLE:
        print("\n--- FANTOM not available, skipping training ---")
        return

    # Config
    config = {
        'lambda_dag': 100.0,
        'lambda_sparse': args.lambda_sparse,
        'tau_gumbel': 1.0,
        'learning_rate': args.lr,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'patience': 20
    }

    print(f"\n--- Training FANTOM ---")
    print(f"Config: lambda_sparse={args.lambda_sparse}")

    model, history = train_fantom(
        data['train'], data['val'],
        config,
        target_idx=data['target_idx'],
        n_nodes=data['n_nodes'],
        lag=data['lag'],
        device=args.device,
        verbose=True
    )

    # Evaluate
    print("\n--- Evaluating on Test Set ---")
    metrics = evaluate_fantom(
        model,
        data['test'],
        data['moments'],
        target_idx=data['target_idx'],
        device=args.device
    )

    print(f"\nTest Results:")
    print(f"  Spearman: {metrics['spearman']:.4f}")
    print(f"  RMSE:     {metrics['rmse']:.2f}")
    print(f"  MAE:      {metrics['mae']:.2f}")

    # Extract causal graph
    print("\n--- Extracting Causal Graph ---")
    graph = extract_causal_graph(model, data['feature_cols'], threshold=0.3)
    print(f"  Instantaneous parents: {graph['n_instantaneous']}")
    print(f"  Lagged parents: {graph['n_lagged']}")

    if graph['instantaneous_parents']:
        print("  Top instantaneous:")
        for feat, lag, weight in graph['instantaneous_parents'][:5]:
            print(f"    {feat} -> TARGET (w={weight:.3f})")

    if graph['lagged_parents']:
        print("  Top lagged:")
        for feat, lag, weight in graph['lagged_parents'][:5]:
            print(f"    {feat}(t-{lag}) -> TARGET (w={weight:.3f})")

    # Compare with baselines
    print("\n--- Comparison with Baselines ---")
    print(f"  FANTOM:      Spearman={metrics['spearman']:.4f}, RMSE={metrics['rmse']:.2f}")
    for name, baseline in baselines.items():
        improvement = metrics['spearman'] - baseline['spearman']
        print(f"  {name:12s}: Spearman={baseline['spearman']:.4f} "
              f"(FANTOM {'+' if improvement > 0 else ''}{improvement:.4f})")

    # Save results
    save_dir = Path(__file__).parent / "outputs" / f"test_fantom_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': args.dataset,
        'feature_groups': args.feature_groups,
        'n_nodes': data['n_nodes'],
        'lag': args.lag,
        'config': config,
        'metrics': {k: float(v) for k, v in metrics.items() if not isinstance(v, np.ndarray)},
        'baselines': baselines,
        'causal_graph': graph
    }

    with open(save_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    torch.save(model.state_dict(), save_dir / "model.pt")

    print(f"\nResults saved to: {save_dir}")

    print("\n" + "=" * 70)
    if metrics['spearman'] > baselines['persistence']['spearman']:
        print("SUCCESS: FANTOM beats persistence baseline!")
    else:
        print("NOTE: FANTOM did not beat persistence baseline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
