"""
FANTOM Forecasting Test - Fixed Version

Key fixes from the original test_fantom_unified.py:
1. Use lag>0 for actual forecasting (not just same-day relationships)
2. Target alignment: predict T+1 from T-lag:T features
3. Simpler loss function focusing on prediction quality
4. Use FANTOM for feature importance discovery, not direct prediction

Usage:
    python3 test_fantom_forecast.py --dataset DE --lag 7
    python3 test_fantom_forecast.py --dataset FR --lag 7
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
from sklearn.linear_model import Ridge

from paths import FANTOM_CODE_DIR
# Add paths
FANTOM_PATH = str(FANTOM_CODE_DIR)
sys.path.insert(0, FANTOM_PATH)
sys.path.insert(0, str(Path(__file__).parent))

from unified_data_loader import load_unified_dataset, get_feature_columns

# Try to import FANTOM components
try:
    from fantom import FANTOM_stationary
    FANTOM_AVAILABLE = True
except ImportError as e:
    FANTOM_AVAILABLE = False
    print(f"WARNING: FANTOM not available: {e}")


def prepare_forecast_data(
    country: str = 'DE',
    lag: int = 7,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    feature_groups: list = None,
    max_features: int = 15
):
    """
    Prepare data for forecasting: predict Y[t+1] from X[t-lag:t].

    Key difference from reconstruction: we shift the target forward by 1 day.
    """
    if feature_groups is None:
        feature_groups = ['price', 'load', 'weather', 'calendar']

    # Load data
    df = load_unified_dataset(country, clean=True)
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Target: next day's price change percentage
    target_col = 'price_change_pct'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    # Get feature columns
    feature_cols = get_feature_columns(df, feature_groups, exclude_target=True, country=country)

    # Limit features
    if len(feature_cols) > max_features - 1:  # -1 for target
        # Prioritize: price, load, then others
        priority = ['price_lag1', 'Day_Ahead_Price', 'Actual Load']
        selected = [c for c in priority if c in feature_cols]
        remaining = [c for c in feature_cols if c not in selected]
        feature_cols = selected + remaining[:max_features - 1 - len(selected)]

    # Add target to feature list (FANTOM needs all nodes)
    all_cols = feature_cols + [target_col]
    target_idx = len(feature_cols)  # Target is last

    # Extract data
    data = df[all_cols].values.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)

    # Normalize
    moments = np.zeros((len(all_cols), 2))
    moments[:, 0] = data.mean(axis=0)
    moments[:, 1] = data.std(axis=0)
    moments[moments[:, 1] == 0, 1] = 1.0
    data_norm = (data - moments[:, 0]) / moments[:, 1]

    # Create windows: [N, lag+1, n_nodes]
    # IMPORTANT: For forecasting, we want to predict Y[t+lag+1] from X[t:t+lag+1]
    n_samples = len(data_norm) - lag - 1  # -1 for the forecast target
    n_nodes = len(all_cols)

    windows = np.zeros((n_samples, lag + 1, n_nodes), dtype=np.float32)
    targets = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        # Features: days i to i+lag (inclusive)
        windows[i] = data_norm[i:i + lag + 1]
        # Target: day i+lag+1 (one day after the window)
        targets[i] = data_norm[i + lag + 1, target_idx]

    # Temporal split
    n_test = int(n_samples * test_ratio)
    n_val = int((n_samples - n_test) * val_ratio)
    n_train = n_samples - n_test - n_val

    train_windows = torch.from_numpy(windows[:n_train])
    val_windows = torch.from_numpy(windows[n_train:n_train + n_val])
    test_windows = torch.from_numpy(windows[n_train + n_val:])

    train_targets = targets[:n_train]
    val_targets = targets[n_train:n_train + n_val]
    test_targets = targets[n_train + n_val:]

    return {
        'train': train_windows,
        'val': val_windows,
        'test': test_windows,
        'train_targets': train_targets,
        'val_targets': val_targets,
        'test_targets': test_targets,
        'moments': moments,
        'feature_cols': all_cols,
        'target_idx': target_idx,
        'target_col': target_col,
        'lag': lag,
        'n_nodes': n_nodes
    }


def train_fantom_discovery(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    n_nodes: int,
    lag: int,
    target_idx: int,
    config: dict,
    device: str = 'cpu'
) -> tuple:
    """
    Train FANTOM for causal discovery (not prediction).

    Returns the model and learned causal graph.
    """
    device_obj = torch.device(device)

    # Graph constraint: target has no outgoing edges
    graph_constraint = np.full((lag + 1, n_nodes, n_nodes), np.nan)
    graph_constraint[:, target_idx, :] = 0.0  # No edges FROM target
    graph_constraint[:, :, target_idx] = np.nan  # Allow edges TO target
    for l in range(lag + 1):
        np.fill_diagonal(graph_constraint[l], 0.0)  # No self-loops

    model = FANTOM_stationary(
        num_nodes=n_nodes,
        device=device_obj,
        lag=lag,
        allow_instantaneous=True,
        lambda_dag=config.get('lambda_dag', 100.0),
        lambda_sparse=config.get('lambda_sparse', 5.0),
        tau_gumbel=config.get('tau_gumbel', 1.0),
        base_distribution_type='spline',
        spline_bins=8,
        var_dist_A_mode='temporal_three',
        norm_layers=True,
        res_connection=True,
        encoder_layer_sizes=[32, 32],
        decoder_layer_sizes=[32, 32],
        heteroscedastic=True,
        graph_constraint_matrix=graph_constraint
    ).to(device_obj)

    train_data = train_data.to(device_obj)
    val_data = val_data.to(device_obj)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))

    n_epochs = config.get('n_epochs', 50)
    batch_size = config.get('batch_size', 64)
    n_samples = train_data.shape[0]

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = config.get('patience', 15)

    print("Training FANTOM for causal discovery...")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        indices = torch.randperm(n_samples, device=device_obj)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            batch = train_data[batch_idx]

            optimizer.zero_grad()

            elbo_terms = model._ELBO_terms(batch)
            log_p = elbo_terms["log_p_base"].mean()
            log_p_A = elbo_terms["log_p_A"] / len(batch)
            log_q_A = elbo_terms["log_q_A"] / len(batch)
            dag_penalty = elbo_terms["penalty_dag"]

            loss = -log_p - log_p_A + log_q_A + config.get('lambda_dag', 100.0) * dag_penalty

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            print(f"Epoch {epoch+1}: All NaN, stopping")
            break

        avg_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_elbo = model._ELBO_terms(val_data)
            val_loss = -val_elbo["log_p_base"].mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.3f}, Val={val_loss:.3f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def extract_important_features(model, feature_cols, target_idx, threshold=0.2):
    """Extract features with strong causal links to target."""
    model.eval()

    with torch.no_grad():
        A = model.get_adj_matrix(samples=1, most_likely_graph=True, squeeze=True)
        if isinstance(A, torch.Tensor):
            A = A.cpu().numpy()

    # A shape: [lag+1, n_nodes, n_nodes]
    # Edges TO target: A[:, :, target_idx]

    important_features = []
    for lag_idx in range(A.shape[0]):
        for feat_idx in range(A.shape[1]):
            if feat_idx == target_idx:
                continue
            weight = A[lag_idx, feat_idx, target_idx]
            if weight > threshold:
                lag_name = 't' if lag_idx == 0 else f't-{lag_idx}'
                important_features.append({
                    'feature': feature_cols[feat_idx],
                    'lag': lag_idx,
                    'weight': weight,
                    'name': f"{feature_cols[feat_idx]}({lag_name})"
                })

    # Sort by weight
    important_features.sort(key=lambda x: -x['weight'])

    return important_features


def build_forecast_model(
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    important_features: list,
    feature_cols: list,
    target_idx: int
):
    """
    Build a simple forecasting model using FANTOM-discovered features.

    Uses Ridge regression on the discovered causal parents.
    """
    n_samples = train_windows.shape[0]

    # Build feature matrix from important features
    feature_matrix = []
    feature_names = []

    for feat_info in important_features[:10]:  # Top 10 features
        feat_idx = feature_cols.index(feat_info['feature'])
        lag = feat_info['lag']
        # Extract feature at the specified lag from window
        # Window is [lag+1, n_nodes], lag=0 is most recent
        feature_matrix.append(train_windows[:, -(lag+1), feat_idx])
        feature_names.append(feat_info['name'])

    if len(feature_matrix) == 0:
        # Fallback: use all features at lag=0
        feature_matrix = [train_windows[:, -1, i] for i in range(len(feature_cols)) if i != target_idx]
        feature_names = [f"{c}(t)" for c in feature_cols if c != feature_cols[target_idx]]

    X_train = np.column_stack(feature_matrix)

    # Train Ridge regression
    model = Ridge(alpha=1.0)
    model.fit(X_train, train_targets)

    return model, feature_names


def evaluate_forecast(
    ridge_model,
    test_windows: np.ndarray,
    test_targets: np.ndarray,
    important_features: list,
    feature_cols: list,
    target_idx: int,
    moments: np.ndarray
):
    """Evaluate forecast model on test set."""
    # Build feature matrix
    feature_matrix = []

    for feat_info in important_features[:10]:
        feat_idx = feature_cols.index(feat_info['feature'])
        lag = feat_info['lag']
        feature_matrix.append(test_windows[:, -(lag+1), feat_idx])

    if len(feature_matrix) == 0:
        feature_matrix = [test_windows[:, -1, i] for i in range(len(feature_cols)) if i != target_idx]

    X_test = np.column_stack(feature_matrix)

    # Predict
    pred_norm = ridge_model.predict(X_test)

    # Denormalize
    target_mean, target_std = moments[target_idx]
    pred = pred_norm * target_std + target_mean
    true = test_targets * target_std + target_mean

    # Metrics
    spearman, pval = spearmanr(true, pred)
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    mae = np.mean(np.abs(true - pred))

    return {
        'spearman': spearman,
        'spearman_pval': pval,
        'rmse': rmse,
        'mae': mae,
        'predictions': pred,
        'true_values': true
    }


def main():
    parser = argparse.ArgumentParser(description='FANTOM Forecasting Test')
    parser.add_argument('--dataset', type=str, default='DE', choices=['DE', 'FR'])
    parser.add_argument('--lag', type=int, default=7)
    parser.add_argument('--max_features', type=int, default=15)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lambda_sparse', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("FANTOM Forecasting Test (Fixed Version)")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Lag: {args.lag}")
    print(f"Max features: {args.max_features}")

    # Load data
    print("\n--- Loading Data ---")
    data = prepare_forecast_data(
        country=args.dataset,
        lag=args.lag,
        max_features=args.max_features
    )

    print(f"Nodes: {data['n_nodes']}")
    print(f"Features: {data['feature_cols']}")
    print(f"Target: {data['target_col']} (index {data['target_idx']})")
    print(f"Train samples: {len(data['train'])}")
    print(f"Test samples: {len(data['test'])}")

    # Baselines
    print("\n--- Computing Baselines ---")
    target_mean, target_std = data['moments'][data['target_idx']]
    test_true = data['test_targets'] * target_std + target_mean
    train_true = data['train_targets'] * target_std + target_mean

    # Persistence: predict yesterday's change
    persistence_pred = np.roll(test_true, 1)
    persistence_pred[0] = train_true[-1]
    pers_spearman, _ = spearmanr(test_true, persistence_pred)
    print(f"  Persistence: Spearman={pers_spearman:.4f}")

    # Mean baseline
    mean_pred = np.full_like(test_true, train_true.mean())
    mean_spearman, _ = spearmanr(test_true, mean_pred)
    print(f"  Mean: Spearman={mean_spearman:.4f}")

    if not FANTOM_AVAILABLE:
        print("\n--- FANTOM not available ---")
        return

    # Train FANTOM for causal discovery
    print("\n--- Training FANTOM ---")
    config = {
        'lambda_dag': 100.0,
        'lambda_sparse': args.lambda_sparse,
        'lr': 0.001,
        'n_epochs': args.n_epochs,
        'batch_size': 64,
        'patience': 15
    }

    model = train_fantom_discovery(
        data['train'], data['val'],
        n_nodes=data['n_nodes'],
        lag=data['lag'],
        target_idx=data['target_idx'],
        config=config,
        device=args.device
    )

    # Extract important features
    print("\n--- Extracting Causal Parents ---")
    important_features = extract_important_features(
        model, data['feature_cols'], data['target_idx'], threshold=0.2
    )

    print(f"Found {len(important_features)} important features:")
    for feat in important_features[:10]:
        print(f"  {feat['name']}: weight={feat['weight']:.3f}")

    # Build forecast model using discovered features
    print("\n--- Building Forecast Model ---")
    ridge_model, used_features = build_forecast_model(
        data['train'].numpy(),
        data['train_targets'],
        important_features,
        data['feature_cols'],
        data['target_idx']
    )
    print(f"Using features: {used_features}")

    # Evaluate
    print("\n--- Evaluating ---")
    metrics = evaluate_forecast(
        ridge_model,
        data['test'].numpy(),
        data['test_targets'],
        important_features,
        data['feature_cols'],
        data['target_idx'],
        data['moments']
    )

    print(f"\nResults:")
    print(f"  FANTOM+Ridge: Spearman={metrics['spearman']:.4f}, RMSE={metrics['rmse']:.2f}")
    print(f"  Persistence:  Spearman={pers_spearman:.4f}")
    print(f"  Improvement:  {metrics['spearman'] - pers_spearman:+.4f}")

    # Save results
    save_dir = Path(__file__).parent / "outputs" / f"fantom_forecast_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': args.dataset,
        'lag': args.lag,
        'n_nodes': data['n_nodes'],
        'config': config,
        'metrics': {
            'spearman': float(metrics['spearman']),
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae'])
        },
        'baselines': {
            'persistence': float(pers_spearman),
            'mean': float(mean_spearman)
        },
        'important_features': [
            {'feature': f['feature'], 'lag': f['lag'], 'weight': float(f['weight'])}
            for f in important_features[:10]
        ]
    }

    with open(save_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {save_dir}")

    if metrics['spearman'] > pers_spearman:
        print("\n✓ SUCCESS: FANTOM+Ridge beats persistence!")
    else:
        print("\n✗ FANTOM+Ridge did not beat persistence")


if __name__ == "__main__":
    main()
