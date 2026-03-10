"""
DS3M Multivariate with Per-Regime Causal DAG Discovery

Two-stage approach:
1. Load trained DS3M multivariate → get regime assignments
2. For each regime with sufficient samples: fit FANTOM → get DAG

Reports regime distributions on both:
- Test set (for Spearman evaluation alignment)
- Full dataset (for visualization)

Usage:
    python ds3m_with_dag.py --country ALL --d_dim 2
    python ds3m_with_dag.py --country ALL --d_dim 2 \
        --ds3m_checkpoint outputs/ds3m/ALL_TARGET_mv_d2_seed42_20260118_163330/checkpoints/best.tar
"""

import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from scipy.stats import spearmanr
import random

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
from paths import DS3M_DIR, FANTOM_CODE_DIR
DS3M_PATH = str(DS3M_DIR)
sys.path.insert(0, DS3M_PATH)
sys.path.insert(0, os.path.join(DS3M_PATH, "src"))
FANTOM_PATH = str(FANTOM_CODE_DIR)
sys.path.insert(0, FANTOM_PATH)

from ds3m_electricity import QRT_MULTIVARIATE_CONFIG
from ds3m_adapter import load_qrt_data, prepare_multivariate_train_test_split
from DSSSMCode import DSSSM
from fantom_electricity import FANTOMElectricity, create_model as create_fantom, get_default_training_params


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_ds3m_checkpoint(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained DS3M model from checkpoint."""
    model = DSSSM(
        x_dim=config['x_dim'],
        y_dim=config['y_dim'],
        h_dim=config['h_dim'],
        z_dim=config['z_dim'],
        d_dim=config['d_dim'],
        n_layers=config['n_layers'],
        device=device,
        bidirection=config.get('bidirection', False)
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def get_regime_assignments(model, X, Y, d_dim):
    """
    Get regime assignments using the DS3M forward pass.

    Args:
        model: Trained DSSSM model
        X: Input features [timestep, batch, x_dim]
        Y: Target values [timestep, batch, y_dim]
        d_dim: Number of regimes

    Returns:
        regime_assignments: numpy array of shape [batch] with regime indices
    """
    model.eval()
    with torch.no_grad():
        # DS3M forward returns multiple outputs:
        # kld_g, kld_c, nll, (z_posterior), (y_emission), d_sampled_plot, z_sampled, d_posterior, d_sampled
        outputs = model(X, Y)

        # all_d_posterior is outputs[7] - list of (batch, d_dim) tensors for each timestep
        all_d_posterior = outputs[7]

        # Stack to get [timestep, batch, d_dim]
        d_posteriors = torch.stack(all_d_posterior, dim=0)

        # Use last timestep posterior, take argmax to get hard assignment
        regime_assignments = d_posteriors[-1].argmax(dim=1).cpu().numpy()

    return regime_assignments


def get_full_regime_sequence(model, X, Y, d_dim):
    """
    Get regime assignments for all timesteps (not just last).

    Returns:
        regime_sequence: numpy array of shape [timestep, batch] with regime indices
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X, Y)
        all_d_posterior = outputs[7]
        d_posteriors = torch.stack(all_d_posterior, dim=0)  # [timestep, batch, d_dim]
        regime_sequence = d_posteriors.argmax(dim=-1).cpu().numpy()  # [timestep, batch]
    return regime_sequence


def prepare_fantom_data(df, feature_cols, lag=1):
    """
    Prepare data in FANTOM format: [N, lag+1, num_nodes]

    Args:
        df: DataFrame with features and TARGET
        feature_cols: List of feature column names (TARGET should be last)
        lag: Temporal lag

    Returns:
        X: numpy array of shape [N, lag+1, num_nodes]
        valid_mask: boolean mask of rows without NaN
    """
    # Ensure TARGET is in feature_cols and is last
    all_cols = list(feature_cols)
    if 'TARGET' not in all_cols:
        all_cols.append('TARGET')
    elif all_cols[-1] != 'TARGET':
        all_cols.remove('TARGET')
        all_cols.append('TARGET')

    data = df[all_cols].values

    # Handle NaN values - fill with 0 for now (will be filtered later)
    nan_rows = np.any(np.isnan(data), axis=1)
    if np.any(nan_rows):
        print(f"  Warning: Found {nan_rows.sum()} rows with NaN values")
        data = np.nan_to_num(data, nan=0.0)

    N = len(data) - lag
    num_nodes = len(all_cols)

    X = np.zeros((N, lag + 1, num_nodes))
    for i in range(N):
        X[i] = data[i:i+lag+1]

    return X, all_cols


def fit_fantom_dag(X_regime, device, training_params=None, verbose=True):
    """
    Fit FANTOM on regime-specific data to discover causal DAG.

    Args:
        X_regime: Data in FANTOM format [N, lag+1, num_nodes]
        device: torch device
        training_params: Dict of training parameters
        verbose: Print progress

    Returns:
        Trained FANTOMElectricity model, normalization stats (mean, std)
    """
    # Filter out samples with NaN or Inf values
    valid_mask = ~np.any(np.isnan(X_regime) | np.isinf(X_regime), axis=(1, 2))
    X_regime = X_regime[valid_mask]

    if len(X_regime) < 10:
        raise ValueError(f"Too few valid samples after filtering: {len(X_regime)}")

    N, lag_plus_1, num_nodes = X_regime.shape
    target_idx = num_nodes - 1  # TARGET is last
    lag = lag_plus_1 - 1

    if verbose:
        print(f"  Fitting FANTOM: {N} samples, {num_nodes} nodes, lag={lag}")

    # Normalize data for numerical stability
    # Compute mean and std per variable (across samples and time)
    X_flat = X_regime.reshape(-1, num_nodes)
    data_mean = np.mean(X_flat, axis=0, keepdims=True)
    data_std = np.std(X_flat, axis=0, keepdims=True)
    data_std[data_std < 1e-6] = 1.0  # Avoid division by zero

    X_normalized = (X_regime - data_mean.reshape(1, 1, -1)) / data_std.reshape(1, 1, -1)

    # Clip extreme values for numerical stability
    X_normalized = np.clip(X_normalized, -10, 10)

    if verbose:
        print(f"  Data normalized: mean range [{data_mean.min():.3f}, {data_mean.max():.3f}]")
        print(f"  Data normalized: std range [{data_std.min():.3f}, {data_std.max():.3f}]")

    # Create model with more conservative settings
    model = create_fantom(
        num_nodes=num_nodes,
        target_idx=target_idx,
        lag=lag,
        device=str(device),
        model_config={
            'lambda_dag': 100.0,
            'lambda_sparse': 1.0,
            'tau_gumbel': 1.0,
            'encoder_layer_sizes': [32, 32],
            'decoder_layer_sizes': [32, 32],
            'base_distribution_type': 'gaussian',  # More stable than spline
        }
    )

    # Training parameters with numerical stability settings
    if training_params is None:
        training_params = get_default_training_params()
        # Reduce steps for per-regime fitting
        training_params['max_steps_auglag'] = 10
        training_params['max_auglag_inner_epochs'] = 500
        # Add gradient clipping for numerical stability
        training_params['gradient_clip_norm'] = 1.0
        # Lower learning rate for stability
        training_params['learning_rate'] = 0.001

    # Prepare dataloader with smaller batch size for stability
    batch_size = min(32, N)  # Smaller batches can be more stable
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    dataloader = DataLoader(X_tensor, batch_size=batch_size, shuffle=True)

    # Train
    model.train()
    model.run_train(
        dataloader=dataloader,
        num_samples=N,
        train_config_dict=training_params
    )

    # Store normalization stats on model for later use
    model.norm_mean = data_mean
    model.norm_std = data_std

    return model


def correlation_analysis(X_regime, feature_names):
    """
    Simple correlation analysis for regimes with too few samples for DAG learning.

    Args:
        X_regime: Data of shape [N, lag+1, num_nodes] or [N, num_nodes]
        feature_names: List of feature names (TARGET should be last)

    Returns:
        Dict with top correlations
    """
    # Use instantaneous data (lag=0)
    if X_regime.ndim == 3:
        data = X_regime[:, -1, :]  # [N, num_nodes] - instantaneous
    else:
        data = X_regime

    target_col = data[:, -1]  # TARGET is last

    correlations = []
    for j in range(data.shape[1] - 1):
        if np.std(data[:, j]) > 1e-6 and np.std(target_col) > 1e-6:
            corr = np.corrcoef(data[:, j], target_col)[0, 1]
            if not np.isnan(corr):
                correlations.append((feature_names[j], float(corr)))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    return {'top_correlations': correlations[:10]}


def run_ds3m_with_dag(
    country: str = "ALL",
    d_dim: int = 2,
    ds3m_checkpoint: str = None,
    output_dir: Path = None,
    seed: int = 42,
    min_regime_samples: int = 30,
    skip_fantom: bool = False
):
    """
    Main pipeline: DS3M multivariate + per-regime DAG.

    Args:
        country: Country code ('ALL', 'FR', 'DE')
        d_dim: Number of regimes
        ds3m_checkpoint: Path to DS3M checkpoint (auto-find if None)
        output_dir: Output directory (auto-generate if None)
        seed: Random seed
        min_regime_samples: Minimum samples for FANTOM DAG learning
        skip_fantom: Skip FANTOM fitting (for quick testing)

    Returns:
        Results dictionary
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs" / "ds3m_dag" / f"{country}_d{d_dim}_{timestamp}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DS3M Multivariate + Per-Regime Causal Discovery")
    print("=" * 60)
    print(f"Country: {country}")
    print(f"d_dim: {d_dim}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Load configuration
    config = QRT_MULTIVARIATE_CONFIG.get(country, QRT_MULTIVARIATE_CONFIG['ALL']).copy()
    config['d_dim'] = d_dim

    # Load data
    print("\nLoading data...")
    df = load_qrt_data()
    country_filter = None if country == 'ALL' else country
    data = prepare_multivariate_train_test_split(
        df, country=country_filter, timestep=config['timestep'], test_ratio=0.2
    )

    feature_cols = data['feature_cols']
    print(f"Features: {len(feature_cols)}")
    print(f"Train samples: {data['trainX'].shape[1]}")
    print(f"Test samples: {data['testX'].shape[1]}")

    # ===== STAGE 1: Load DS3M and get regime assignments =====
    print("\n" + "-" * 60)
    print("Stage 1: DS3M Regime Detection")
    print("-" * 60)

    if ds3m_checkpoint is None:
        # Find latest checkpoint
        ds3m_dir = Path(__file__).parent / "outputs" / "ds3m"
        pattern = f"{country}_TARGET_mv_d{d_dim}_*"
        matches = list(ds3m_dir.glob(pattern))
        if matches:
            matches.sort(key=lambda x: x.name, reverse=True)
            ds3m_checkpoint = str(matches[0] / "checkpoints" / "best.tar")
        else:
            raise FileNotFoundError(f"No DS3M checkpoint found for {country} d_dim={d_dim}")

    print(f"Loading DS3M from: {ds3m_checkpoint}")
    model, checkpoint = load_ds3m_checkpoint(ds3m_checkpoint, config, device)
    model.eval()

    # Get regime assignments for test set
    testX = data['testX'].to(device)
    testY = data['testY'].to(device)
    test_regimes = get_regime_assignments(model, testX, testY, d_dim)

    # Get regime assignments for train set
    trainX = data['trainX'].to(device)
    trainY = data['trainY'].to(device)
    train_regimes = get_regime_assignments(model, trainX, trainY, d_dim)

    # Get full regime sequence for visualization
    test_regime_sequence = get_full_regime_sequence(model, testX, testY, d_dim)
    train_regime_sequence = get_full_regime_sequence(model, trainX, trainY, d_dim)

    # Report regime distributions
    print("\nRegime Distribution (Test Set):")
    for r in range(d_dim):
        count = (test_regimes == r).sum()
        print(f"  Regime {r}: {count} samples ({100*count/len(test_regimes):.1f}%)")

    print("\nRegime Distribution (Train Set):")
    for r in range(d_dim):
        count = (train_regimes == r).sum()
        print(f"  Regime {r}: {count} samples ({100*count/len(train_regimes):.1f}%)")

    all_regimes = np.concatenate([train_regimes, test_regimes])
    print("\nRegime Distribution (Full Dataset):")
    for r in range(d_dim):
        count = (all_regimes == r).sum()
        print(f"  Regime {r}: {count} samples ({100*count/len(all_regimes):.1f}%)")

    # ===== STAGE 2: Per-Regime DAG Discovery =====
    print("\n" + "-" * 60)
    print("Stage 2: Per-Regime Causal Discovery")
    print("-" * 60)

    # Prepare data for FANTOM (continuous time series with lag)
    # Filter raw data by country
    if country != 'ALL':
        df_filtered = df[df['COUNTRY'] == country].copy()
    else:
        df_filtered = df.copy()

    df_filtered = df_filtered.sort_values('DAY_ID').reset_index(drop=True)

    # Prepare FANTOM format data
    X_fantom, fantom_feature_names = prepare_fantom_data(df_filtered, feature_cols, lag=1)
    print(f"\nFANTOM data shape: {X_fantom.shape}")
    print(f"Features for FANTOM: {fantom_feature_names[:5]}... ({len(fantom_feature_names)} total)")

    # Map regime assignments back to raw data indices
    # This is complex because DS3M uses sliding windows
    # For simplicity, we'll use the full-dataset approach: split by regime in raw data
    # based on the regime assignments from last timestep of each window

    # Combine train and test indices
    n_train = data['trainX'].shape[1]
    n_test = data['testX'].shape[1]

    regime_dags = {}
    regime_parents = {}
    regime_models = {}

    for regime_id in range(d_dim):
        regime_count_train = (train_regimes == regime_id).sum()
        regime_count_test = (test_regimes == regime_id).sum()
        regime_count_total = regime_count_train + regime_count_test

        print(f"\nRegime {regime_id}: {regime_count_total} samples (train: {regime_count_train}, test: {regime_count_test})")

        if regime_count_total < min_regime_samples:
            print(f"  Skipping DAG learning (< {min_regime_samples} samples)")
            print("  Using correlation analysis instead...")

            # Get indices for this regime
            train_mask = train_regimes == regime_id
            test_mask = test_regimes == regime_id

            # Extract data for correlation analysis
            X_train_regime = data['trainX'][:, train_mask, :].cpu().numpy()
            X_test_regime = data['testX'][:, test_mask, :].cpu().numpy()

            # Use last timestep features for correlation
            if X_train_regime.shape[1] > 0:
                X_regime_flat = X_train_regime[-1]  # [n_samples, features]
            else:
                X_regime_flat = X_test_regime[-1]

            # Add TARGET
            if X_train_regime.shape[1] > 0:
                Y_regime = data['trainY'][-1, train_mask, :].cpu().numpy()
            else:
                Y_regime = data['testY'][-1, test_mask, :].cpu().numpy()

            X_with_target = np.concatenate([X_regime_flat, Y_regime], axis=1)
            all_names = list(feature_cols) + ['TARGET']

            corr_results = correlation_analysis(X_with_target, all_names)
            regime_parents[regime_id] = {
                'type': 'correlation',
                'n_samples': int(regime_count_total),
                **corr_results
            }

            print(f"  Top correlations with TARGET:")
            for name, corr in corr_results['top_correlations'][:5]:
                print(f"    {name}: {corr:.3f}")
            continue

        if skip_fantom:
            print("  Skipping FANTOM (--skip_fantom flag set)")
            regime_parents[regime_id] = {
                'type': 'skipped',
                'n_samples': int(regime_count_total)
            }
            continue

        # For FANTOM, we need contiguous time series data
        # We'll use the raw data indices corresponding to this regime
        # This is approximate - we use the regime assignment from the last timestep

        # Get indices for this regime from the combined dataset
        train_indices = np.where(train_regimes == regime_id)[0]
        test_indices = np.where(test_regimes == regime_id)[0] + n_train
        regime_indices = np.concatenate([train_indices, test_indices])

        # Map sliding window indices back to approximate raw data indices
        # Each window index i roughly corresponds to raw data at position timestep + i - 1
        timestep = config['timestep']
        raw_indices = regime_indices + timestep - 1
        raw_indices = raw_indices[raw_indices < len(X_fantom)]

        if len(raw_indices) < min_regime_samples:
            print(f"  Insufficient contiguous samples for DAG learning ({len(raw_indices)} valid)")
            # Fall back to correlation
            regime_parents[regime_id] = {
                'type': 'correlation_fallback',
                'n_samples': int(len(raw_indices)),
                **correlation_analysis(X_fantom[raw_indices], fantom_feature_names)
            }
            continue

        # Extract regime-specific FANTOM data
        X_regime_fantom = X_fantom[raw_indices]
        print(f"  Extracted {len(X_regime_fantom)} samples for FANTOM")

        # Fit FANTOM
        print(f"  Fitting FANTOM DAG...")
        try:
            fantom_model = fit_fantom_dag(X_regime_fantom, device, verbose=True)
            regime_models[regime_id] = fantom_model

            # Extract DAG
            A = fantom_model.get_adj_matrix(samples=1, most_likely_graph=True, squeeze=True)
            regime_dags[regime_id] = A

            # Get causal parents of TARGET
            parents = fantom_model.get_causal_parents(fantom_feature_names, threshold=0.5)
            regime_parents[regime_id] = {
                'type': 'dag',
                'n_samples': int(len(X_regime_fantom)),
                'parents': {
                    'instantaneous': [(n, l, float(w)) for n, l, w in parents['instantaneous']],
                    'lagged': [(n, l, float(w)) for n, l, w in parents['lagged']]
                }
            }

            print(f"  DAG learned successfully")
            print(f"    Instantaneous parents: {len(parents['instantaneous'])}")
            print(f"    Lagged parents: {len(parents['lagged'])}")

            # Show top parents
            if parents['instantaneous']:
                print(f"    Top instantaneous: {parents['instantaneous'][0][0]} ({parents['instantaneous'][0][2]:.3f})")
            if parents['lagged']:
                print(f"    Top lagged: {parents['lagged'][0][0]} lag={parents['lagged'][0][1]} ({parents['lagged'][0][2]:.3f})")

        except Exception as e:
            print(f"  FANTOM fitting failed: {e}")
            regime_parents[regime_id] = {
                'type': 'error',
                'n_samples': int(len(X_regime_fantom)),
                'error': str(e)
            }

    # ===== Save Results =====
    print("\n" + "-" * 60)
    print("Saving Results")
    print("-" * 60)

    results = {
        'country': country,
        'd_dim': d_dim,
        'seed': seed,
        'ds3m_checkpoint': ds3m_checkpoint,
        'regime_distribution_test': {
            str(r): int((test_regimes == r).sum()) for r in range(d_dim)
        },
        'regime_distribution_train': {
            str(r): int((train_regimes == r).sum()) for r in range(d_dim)
        },
        'regime_distribution_full': {
            str(r): int((all_regimes == r).sum()) for r in range(d_dim)
        },
        'regime_analysis': {str(k): v for k, v in regime_parents.items()},
        'feature_cols': list(feature_cols),
        'fantom_feature_cols': fantom_feature_names
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save regime assignments
    np.save(output_dir / 'test_regimes.npy', test_regimes)
    np.save(output_dir / 'train_regimes.npy', train_regimes)
    np.save(output_dir / 'full_regimes.npy', all_regimes)

    # Save regime sequences (for time series visualization)
    np.save(output_dir / 'test_regime_sequence.npy', test_regime_sequence)
    np.save(output_dir / 'train_regime_sequence.npy', train_regime_sequence)

    # Save DAGs
    for r, dag in regime_dags.items():
        np.save(output_dir / f'dag_regime_{r}.npy', dag)

    # Save FANTOM models
    for r, model in regime_models.items():
        torch.save(model.state_dict(), output_dir / f'fantom_regime_{r}.pt')

    print(f"\nResults saved to: {output_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in range(d_dim):
        info = regime_parents.get(r, {})
        analysis_type = info.get('type', 'unknown')
        n_samples = info.get('n_samples', 0)
        print(f"\nRegime {r}: {n_samples} samples ({analysis_type})")

        if analysis_type == 'dag':
            parents = info.get('parents', {})
            inst = parents.get('instantaneous', [])
            lagged = parents.get('lagged', [])
            if inst:
                print(f"  Top instantaneous cause: {inst[0][0]} (weight={inst[0][2]:.3f})")
            if lagged:
                print(f"  Top lagged cause: {lagged[0][0]} lag={lagged[0][1]} (weight={lagged[0][2]:.3f})")
        elif analysis_type in ('correlation', 'correlation_fallback'):
            top_corr = info.get('top_correlations', [])
            if top_corr:
                print(f"  Top correlation: {top_corr[0][0]} (r={top_corr[0][1]:.3f})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DS3M + Per-Regime DAG Discovery")
    parser.add_argument('--country', type=str, default='ALL',
                        choices=['ALL', 'FR', 'DE'],
                        help='Country code')
    parser.add_argument('--d_dim', type=int, default=2,
                        help='Number of regimes')
    parser.add_argument('--ds3m_checkpoint', type=str, default=None,
                        help='Path to DS3M checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--min_regime_samples', type=int, default=30,
                        help='Minimum samples for FANTOM DAG learning')
    parser.add_argument('--skip_fantom', action='store_true',
                        help='Skip FANTOM fitting (for quick testing)')

    args = parser.parse_args()

    run_ds3m_with_dag(
        country=args.country,
        d_dim=args.d_dim,
        ds3m_checkpoint=args.ds3m_checkpoint,
        output_dir=args.output_dir,
        seed=args.seed,
        min_regime_samples=args.min_regime_samples,
        skip_fantom=args.skip_fantom
    )
