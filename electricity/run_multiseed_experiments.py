#!/usr/bin/env python3
"""
Multi-Seed Experiment Runner for Robust Results

Runs experiments with multiple seeds and reports mean ± std for Spearman correlation.

Usage:
    python run_multiseed_experiments.py --model ds3m_uv --dataset DE_FR --seeds 5
    python run_multiseed_experiments.py --model fantom --dataset DE_FR --seeds 5
    python run_multiseed_experiments.py --model ds3m_mv --dataset DE_FR --d_dim 2 --seeds 5
"""

import sys
import os
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from scipy.stats import spearmanr
import random

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
from paths import DS3M_DIR, FANTOM_CODE_DIR
FANTOM_PATH = str(FANTOM_CODE_DIR)
DS3M_PATH = str(DS3M_DIR)
sys.path.insert(0, FANTOM_PATH)
sys.path.insert(0, DS3M_PATH)
sys.path.insert(0, os.path.join(DS3M_PATH, "src"))


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


def ds3m_predict(model, X: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
    """
    Predict using DS3M model WITHOUT data leakage.

    This function performs proper forecasting by:
    1. Running the forward RNN on X only (no Y dependency)
    2. Sampling from the prior for d (regime) and z (latent state)
    3. Computing emission mean from sampled states

    Unlike the standard forward() method which uses a backward RNN conditioned
    on actual Y values, this function only uses information available at
    prediction time.

    Args:
        model: DSSSM model
        X: Input features of shape (timestep, batch, x_dim)
        n_samples: Number of Monte Carlo samples for prediction

    Returns:
        predictions: Shape (timestep, batch, y_dim) - mean predictions
    """
    model.eval()
    device = model.device
    timestep, batch_size, x_dim = X.shape

    with torch.no_grad():
        all_predictions = []

        for _ in range(n_samples):
            # Initialize hidden states
            h0 = torch.zeros((model.n_layers, batch_size, model.h_dim), device=device)

            # Run forward RNN on X only
            output_forward, _ = model.rnn_forward(X, h0)

            # Initialize regime and latent state
            d_t = torch.ones((batch_size, model.d_dim), device=device) / model.d_dim
            z_t = torch.zeros((batch_size, model.z_dim), device=device)

            # Get transition matrix
            Transition = model.TransitionMatrix()

            predictions = []

            for t in range(timestep):
                # Sample regime from prior (transition from previous regime)
                d_prior = torch.mm(d_t, Transition)
                d_samples = torch.distributions.Categorical(d_prior).sample()
                d_t = torch.eye(model.d_dim, device=device)[d_samples]

                # Compute z prior for each regime and weight by regime
                z_prior_mean = torch.zeros((batch_size, model.z_dim), device=device)
                z_prior_std = torch.zeros((batch_size, model.z_dim), device=device)

                for i in range(model.d_dim):
                    z_prior_hidden = model.ztrainsition_list[i](
                        torch.cat([output_forward[t], z_t], dim=1)
                    )
                    z_prior_mean_i = model.ztrainsition_mean_list[i](z_prior_hidden)
                    z_prior_std_i = model.ztrainsition_std_list[i](z_prior_hidden)

                    z_prior_mean += z_prior_mean_i * d_t[:, i:i+1]
                    z_prior_std += z_prior_std_i * d_t[:, i:i+1]

                # Sample z from prior
                z_t = torch.distributions.Normal(z_prior_mean, z_prior_std + 1e-6).sample()

                # Compute emission (prediction) for each regime and weight
                y_emission_mean = torch.zeros((batch_size, model.y_dim), device=device)

                for i in range(model.d_dim):
                    y_hidden = model.yemission_list[i](
                        torch.cat([output_forward[t], z_t], dim=1)
                    )
                    y_mean_i = model.yemission_mean_list[i](y_hidden)
                    y_emission_mean += y_mean_i * d_t[:, i:i+1]

                predictions.append(y_emission_mean)

            # Stack predictions: (timestep, batch, y_dim)
            predictions = torch.stack(predictions, dim=0)
            all_predictions.append(predictions)

        # Average over Monte Carlo samples
        all_predictions = torch.stack(all_predictions, dim=0)  # (n_samples, T, B, y_dim)
        mean_predictions = all_predictions.mean(dim=0)  # (T, B, y_dim)

    return mean_predictions


def run_ds3m_univariate(
    dataset: str,
    d_dim: int,
    seed: int,
    use_regularization: bool = True,
    device: str = 'cuda',
    verbose: bool = True,
    task_type: str = 'prediction'
) -> Dict:
    """Run DS3M univariate experiment (baseline).

    Args:
        task_type: 'prediction' or 'estimation'
            - 'prediction': Forecast Y[t+1] using X[0:t] and Y[0:t]
            - 'estimation': Estimate Y[t] using X[0:t] and Y[0:t-1]
    """
    from unified_data_loader import prepare_unified_ds3m_data
    from DSSSMCode import DSSSM
    from ds3m_fantom.training.regime_regularization import RegimeRegularizer

    set_seed(seed)

    # Load data - univariate uses only target-related features
    data = prepare_unified_ds3m_data(
        dataset,
        timestep=14,
        feature_groups=['spread'] if dataset == 'DE_FR' else ['price'],
        target_col='price_spread_change_pct' if dataset == 'DE_FR' else 'price_change_pct',
        task_type=task_type
    )

    # Model config - univariate uses target history only
    x_dim = data['trainX'].shape[-1]
    model = DSSSM(
        x_dim=x_dim,
        y_dim=1,
        h_dim=30,
        z_dim=8,
        d_dim=d_dim,
        n_layers=1,
        device=torch.device(device)
    ).to(device)

    # Regime regularizer to prevent collapse
    regularizer = None
    if use_regularization and d_dim > 1:
        regularizer = RegimeRegularizer(
            d_dim=d_dim,
            entropy_weight=2.0,      # Stronger entropy to encourage diversity
            min_usage_weight=1.0,    # Penalize unused regimes
            smoothness_weight=0.1,   # Allow regime switching
            kl_weight=1.0,
            min_usage_ratio=0.15,    # At least 15% per regime
            annealing_start=20,
            annealing_end=80
        )

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainX = data['trainX'].to(device)
    trainY = data['trainY'].to(device)
    valX = data['valX'].to(device)
    valY = data['valY'].to(device)
    testX = data['testX'].to(device)
    testY = data['testY'].to(device)

    best_val_spearman = -np.inf
    patience_counter = 0
    patience = 15
    best_state = None
    batch_size = min(64, trainX.size(1))

    for epoch in range(150):
        model.train()

        # Annealing schedule (from original DS3M code)
        if epoch < 75:
            annealing = 0.01
        else:
            annealing = min(1.0, 0.01 + epoch / 150)

        # Mini-batch training
        for batch in range(0, trainX.size(1), batch_size):
            batchX = trainX[:, batch:(batch+batch_size), :]
            batchY = trainY[:, batch:(batch+batch_size), :]

            optimizer.zero_grad()
            # DS3M forward returns tuple: (kld_gauss, kld_cat, nll, z_stats, y_stats, d_plot, z_samp, d_post, d_samp)
            kld_gauss, kld_cat, nll, _, _, d_plot, _, d_post, _ = model(batchX, batchY)

            # Compute loss with annealing
            size = batchX.size(1) * batchX.size(0)
            loss = annealing * (kld_gauss + kld_cat) / size + nll / size

            # Add regime regularization to prevent collapse
            if regularizer is not None and d_post is not None:
                # d_post is a list of [batch, d_dim] tensors (one per timestep)
                # Stack to [timestep, batch, d_dim]
                regime_posteriors = torch.stack(d_post, dim=0)
                reg_loss = regularizer(regime_posteriors)
                loss = loss + reg_loss / size

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        # Step the regularizer's annealing schedule
        if regularizer is not None:
            regularizer.step()

        # Validation
        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                # FIX: Use ds3m_predict() to avoid data leakage (don't pass actual Y)
                val_predictions = ds3m_predict(model, valX, n_samples=10)

                # Take last timestep prediction
                val_pred = val_predictions[-1, :, 0].cpu().numpy()
                val_true = valY[-1, :, 0].cpu().numpy()

                # Denormalize
                y_mean = data['Y_moments'][0].item() if hasattr(data['Y_moments'][0], 'item') else data['Y_moments'][0]
                y_std = data['Y_moments'][1].item() if hasattr(data['Y_moments'][1], 'item') else data['Y_moments'][1]
                val_pred_denorm = val_pred * y_std + y_mean
                val_true_denorm = val_true * y_std + y_mean

                val_spearman, _ = spearmanr(val_true_denorm, val_pred_denorm)

                if verbose:
                    # Get loss from forward pass (still needed for training diagnostics)
                    kld_g, kld_c, nll, _, _, _, _, _, _ = model(valX, valY)
                    val_loss = (kld_g + kld_c + nll).item() / (valX.size(1) * valX.size(0))
                    print(f"  Epoch {epoch+1}: Loss={val_loss:.4f}, Val Spearman={val_spearman:.4f}")

                if val_spearman > best_val_spearman:
                    best_val_spearman = val_spearman
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch+1}")
                        break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Test evaluation
    model.eval()
    with torch.no_grad():
        # FIX: Use ds3m_predict() to avoid data leakage (don't pass actual Y)
        test_predictions = ds3m_predict(model, testX, n_samples=50)

        # Get predictions
        test_pred = test_predictions[-1, :, 0].cpu().numpy()
        test_true = testY[-1, :, 0].cpu().numpy()

        # Denormalize
        y_mean = data['Y_moments'][0].item() if hasattr(data['Y_moments'][0], 'item') else data['Y_moments'][0]
        y_std = data['Y_moments'][1].item() if hasattr(data['Y_moments'][1], 'item') else data['Y_moments'][1]
        test_pred_denorm = test_pred * y_std + y_mean
        test_true_denorm = test_true * y_std + y_mean

        test_spearman, test_pval = spearmanr(test_true_denorm, test_pred_denorm)
        rmse = np.sqrt(np.mean((test_true_denorm - test_pred_denorm) ** 2))

        # Directional accuracy: percentage of correct sign predictions
        correct_direction = ((test_pred_denorm > 0) == (test_true_denorm > 0))
        directional_accuracy = float(correct_direction.mean())

        # Get regime assignments from a forward pass (for diagnostic purposes only)
        _, _, _, _, _, d_plot, _, d_post, _ = model(testX, testY)
        d_assignments = torch.stack(d_plot)[-1].argmax(dim=-1).cpu().numpy()
        regime_counts = {}
        for r in range(d_dim):
            regime_counts[r] = int((d_assignments == r).sum())

    return {
        'model': 'DS3M_Univariate',
        'dataset': dataset,
        'd_dim': d_dim,
        'seed': seed,
        'task_type': task_type,
        'test_spearman': test_spearman,
        'test_pval': test_pval,
        'test_rmse': rmse,
        'test_directional_accuracy': directional_accuracy,
        'best_val_spearman': best_val_spearman,
        'regime_distribution': regime_counts
    }


def run_ds3m_multivariate(
    dataset: str,
    d_dim: int,
    seed: int,
    feature_groups: List[str] = None,
    use_regularization: bool = True,
    device: str = 'cuda',
    verbose: bool = True,
    task_type: str = 'prediction'
) -> Dict:
    """Run DS3M multivariate experiment.

    Args:
        task_type: 'prediction' or 'estimation'
            - 'prediction': Forecast Y[t+1] using X[0:t] and Y[0:t]
            - 'estimation': Estimate Y[t] using X[0:t] and Y[0:t-1]
    """
    from unified_data_loader import prepare_unified_ds3m_data
    from DSSSMCode import DSSSM
    from ds3m_fantom.training.regime_regularization import RegimeRegularizer

    set_seed(seed)

    # Default feature groups for spread prediction
    if feature_groups is None:
        if dataset == 'DE_FR':
            feature_groups = ['spread', 'price_de', 'price_fr', 'calendar', 'spgci']
        else:
            feature_groups = ['price', 'calendar', 'load', 'weather']

    # Load data
    data = prepare_unified_ds3m_data(
        dataset,
        timestep=14,
        feature_groups=feature_groups,
        target_col='price_spread_change_pct' if dataset == 'DE_FR' else 'price_change_pct',
        task_type=task_type
    )

    x_dim = data['trainX'].shape[-1]

    # Model config
    model = DSSSM(
        x_dim=x_dim,
        y_dim=1,
        h_dim=32,
        z_dim=8,
        d_dim=d_dim,
        n_layers=1,
        device=torch.device(device)
    ).to(device)

    # Regime regularizer to prevent collapse
    regularizer = None
    if use_regularization and d_dim > 1:
        regularizer = RegimeRegularizer(
            d_dim=d_dim,
            entropy_weight=2.0,      # Stronger entropy to encourage diversity
            min_usage_weight=1.0,    # Penalize unused regimes
            smoothness_weight=0.1,   # Allow regime switching
            kl_weight=1.0,
            min_usage_ratio=0.15,    # At least 15% per regime
            annealing_start=20,
            annealing_end=80
        )

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainX = data['trainX'].to(device)
    trainY = data['trainY'].to(device)
    valX = data['valX'].to(device)
    valY = data['valY'].to(device)
    testX = data['testX'].to(device)
    testY = data['testY'].to(device)

    best_val_spearman = -np.inf
    patience_counter = 0
    patience = 15
    best_state = None
    batch_size = min(64, trainX.size(1))

    for epoch in range(150):
        model.train()

        # Annealing schedule (from original DS3M code)
        if epoch < 75:
            annealing = 0.01
        else:
            annealing = min(1.0, 0.01 + epoch / 150)

        # Mini-batch training
        for batch in range(0, trainX.size(1), batch_size):
            batchX = trainX[:, batch:(batch+batch_size), :]
            batchY = trainY[:, batch:(batch+batch_size), :]

            optimizer.zero_grad()
            # DS3M forward returns tuple: (kld_gauss, kld_cat, nll, z_stats, y_stats, d_plot, z_samp, d_post, d_samp)
            kld_gauss, kld_cat, nll, _, _, d_plot, _, d_post, _ = model(batchX, batchY)

            # Compute loss with annealing
            size = batchX.size(1) * batchX.size(0)
            loss = annealing * (kld_gauss + kld_cat) / size + nll / size

            # Add regime regularization to prevent collapse
            if regularizer is not None and d_post is not None:
                # d_post is a list of [batch, d_dim] tensors (one per timestep)
                # Stack to [timestep, batch, d_dim]
                regime_posteriors = torch.stack(d_post, dim=0)
                reg_loss = regularizer(regime_posteriors)
                loss = loss + reg_loss / size

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        # Step the regularizer's annealing schedule
        if regularizer is not None:
            regularizer.step()

        # Validation
        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                # FIX: Use ds3m_predict() to avoid data leakage (don't pass actual Y)
                val_predictions = ds3m_predict(model, valX, n_samples=10)

                # Get predictions
                val_pred = val_predictions[-1, :, 0].cpu().numpy()
                val_true = valY[-1, :, 0].cpu().numpy()

                # Denormalize
                y_mean = data['Y_moments'][0].item() if hasattr(data['Y_moments'][0], 'item') else data['Y_moments'][0]
                y_std = data['Y_moments'][1].item() if hasattr(data['Y_moments'][1], 'item') else data['Y_moments'][1]
                val_pred_denorm = val_pred * y_std + y_mean
                val_true_denorm = val_true * y_std + y_mean

                val_spearman, _ = spearmanr(val_true_denorm, val_pred_denorm)

                if verbose:
                    # Get loss from forward pass (still needed for training diagnostics)
                    kld_g, kld_c, nll, _, _, _, _, _, _ = model(valX, valY)
                    val_loss = (kld_g + kld_c + nll).item() / (valX.size(1) * valX.size(0))
                    print(f"  Epoch {epoch+1}: Loss={val_loss:.4f}, Val Spearman={val_spearman:.4f}")

                if val_spearman > best_val_spearman:
                    best_val_spearman = val_spearman
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch+1}")
                        break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Test evaluation
    model.eval()
    with torch.no_grad():
        # FIX: Use ds3m_predict() to avoid data leakage (don't pass actual Y)
        test_predictions = ds3m_predict(model, testX, n_samples=50)

        # Get predictions
        test_pred = test_predictions[-1, :, 0].cpu().numpy()
        test_true = testY[-1, :, 0].cpu().numpy()

        # Denormalize
        y_mean = data['Y_moments'][0].item() if hasattr(data['Y_moments'][0], 'item') else data['Y_moments'][0]
        y_std = data['Y_moments'][1].item() if hasattr(data['Y_moments'][1], 'item') else data['Y_moments'][1]
        test_pred_denorm = test_pred * y_std + y_mean
        test_true_denorm = test_true * y_std + y_mean

        test_spearman, test_pval = spearmanr(test_true_denorm, test_pred_denorm)
        rmse = np.sqrt(np.mean((test_true_denorm - test_pred_denorm) ** 2))

        # Directional accuracy: percentage of correct sign predictions
        correct_direction = ((test_pred_denorm > 0) == (test_true_denorm > 0))
        directional_accuracy = float(correct_direction.mean())

        # Get regime assignments from a forward pass (for diagnostic purposes only)
        _, _, _, _, _, d_plot, _, d_post, _ = model(testX, testY)
        d_assignments = torch.stack(d_plot)[-1].argmax(dim=-1).cpu().numpy()
        regime_counts = {}
        for r in range(d_dim):
            regime_counts[r] = int((d_assignments == r).sum())

    return {
        'model': 'DS3M_Multivariate',
        'dataset': dataset,
        'd_dim': d_dim,
        'n_features': x_dim,
        'seed': seed,
        'task_type': task_type,
        'test_spearman': test_spearman,
        'test_pval': test_pval,
        'test_rmse': rmse,
        'test_directional_accuracy': directional_accuracy,
        'best_val_spearman': best_val_spearman,
        'regime_distribution': regime_counts
    }


def run_fantom_spread(
    dataset: str,
    seed: int,
    n_regimes: int = 2,
    use_bem: bool = True,
    feature_groups: List[str] = None,
    max_features: int = 30,
    device: str = 'cuda',
    verbose: bool = True,
    task_type: str = 'estimation'
) -> Dict:
    """
    Run FANTOM experiment with optional BEM for per-regime DAG learning.

    Args:
        dataset: 'DE', 'FR', or 'DE_FR'
        seed: Random seed
        n_regimes: Number of regimes for BEM (if use_bem=True)
        use_bem: If True, use BEM for per-regime DAG learning
        feature_groups: Feature groups to include
        max_features: Maximum number of features
        device: Device to use
        verbose: Print progress
        task_type: 'estimation' or 'prediction'
            - 'estimation': Estimate Y[t] using X[0:t]
            - 'prediction': Forecast Y[t+1] using X[0:t]
    """
    from unified_data_loader import prepare_unified_fantom_data

    set_seed(seed)

    # Default feature groups
    if feature_groups is None:
        if dataset == 'DE_FR':
            feature_groups = ['spread', 'price_de', 'price_fr', 'calendar']
        else:
            feature_groups = ['price', 'calendar', 'load']

    # Load data
    data = prepare_unified_fantom_data(
        dataset,
        lag=1,
        feature_groups=feature_groups,
        target_col='price_spread_change_pct' if dataset == 'DE_FR' else 'price_change_pct',
        max_features=max_features,
        task_type=task_type
    )

    # Note: prepare_unified_fantom_data returns 'train', 'val', 'test' keys
    num_nodes = data['train'].shape[-1]
    target_idx = data['target_idx']

    if use_bem:
        # Use FANTOM with BEM for per-regime DAG learning
        from fantom_regime import FANTOMRegime

        model = FANTOMRegime(
            n_nodes=num_nodes,
            target_idx=target_idx,
            lag=1,
            device=device,
            initial_n_regimes=n_regimes,
            window_size=min(300, len(data['train']) // n_regimes),
            min_regime_size=30,  # Reduced from 50 to allow smaller regimes
            max_iterations=3,
            model_config={
                'lambda_dag': 100.0,
                'lambda_sparse': 0.01,  # Light sparsity to allow edge learning
                'heteroscedastic': True,
                'encoder_layer_sizes': [32, 32],
                'decoder_layer_sizes': [32, 32],
            },
            # FIX: New parameters to prevent regime collapse
            use_soft_assignments=True,      # Use probabilistic assignments
            entropy_weight=0.5,             # Encourage diverse regime usage
            init_method='kmeans',           # Better initialization via clustering
            prune_after_iteration=2         # Don't prune early (allow regimes to develop)
        )

        # Train - FANTOMRegime.fit() expects numpy arrays, not tensors
        train_np = data['train'].numpy() if isinstance(data['train'], torch.Tensor) else data['train']
        if verbose:
            print(f"  Training FANTOM with BEM ({n_regimes} regimes)...")
        model.fit(train_np, verbose=verbose)

        # Predict - FANTOMRegime.predict_target() also expects numpy arrays
        test_np = data['test'].numpy() if isinstance(data['test'], torch.Tensor) else data['test']
        test_pred = model.predict_target(test_np)

        # Get regime info
        regime_assignments = model.get_regime_assignments()
        n_regimes_final = model.n_regimes

        # Get per-regime DAGs
        dags = model.get_regime_dags()

    else:
        # Use single FANTOM model (stationary)
        from fantom_electricity import FANTOMElectricity, create_model
        from torch.utils.data import DataLoader

        model = create_model(
            num_nodes=num_nodes,
            target_idx=target_idx,
            lag=1,
            device=device,
            model_config={
                'lambda_dag': 100.0,
                'lambda_sparse': 0.01,  # Light sparsity to allow edge learning
                'heteroscedastic': True,
            }
        )

        # Train - data['train'] is already a tensor
        train_tensor = data['train'] if isinstance(data['train'], torch.Tensor) else torch.tensor(data['train'], dtype=torch.float32)
        dataloader = DataLoader(train_tensor, batch_size=32, shuffle=True)

        training_params = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'max_steps_auglag': 10,
            'max_auglag_inner_epochs': 100,
            'rho': 1.0,
            'alpha': 0.0,
            'progress_rate': 0.9,
            'safety_rho': 1e9,
            'safety_alpha': 1e9,
            'tol_dag': 1e-6,
        }

        if verbose:
            print(f"  Training FANTOM (stationary)...")
        model.run_train(dataloader, len(train_tensor), training_params)

        # Predict - data['test'] is already a tensor
        test_tensor = data['test'] if isinstance(data['test'], torch.Tensor) else torch.tensor(data['test'], dtype=torch.float32)
        test_tensor = test_tensor.to(device)
        test_pred = model.predict_target(test_tensor).cpu().numpy()

        n_regimes_final = 1
        regime_assignments = None
        dags = [{'regime': 0, 'adjacency_matrix': model.get_adj_matrix(samples=1, most_likely_graph=True, squeeze=True)}]

    # Evaluate - data['test'] is [N, lag+1, num_nodes], target is at index target_idx
    test_data_np = data['test'].numpy() if isinstance(data['test'], torch.Tensor) else data['test']
    test_true = test_data_np[:, -1, target_idx]  # Last timestep, target column

    # Denormalize using 'moments' key (not 'Y_moments')
    moments = data['moments']
    test_pred_denorm = test_pred * moments[target_idx, 1] + moments[target_idx, 0]
    test_true_denorm = test_true * moments[target_idx, 1] + moments[target_idx, 0]

    test_spearman, test_pval = spearmanr(test_true_denorm, test_pred_denorm)
    rmse = np.sqrt(np.mean((test_true_denorm - test_pred_denorm) ** 2))

    # Directional accuracy: percentage of correct sign predictions
    correct_direction = ((test_pred_denorm > 0) == (test_true_denorm > 0))
    directional_accuracy = float(correct_direction.mean())

    return {
        'model': 'FANTOM_BEM' if use_bem else 'FANTOM_Stationary',
        'dataset': dataset,
        'n_regimes_initial': n_regimes if use_bem else 1,
        'n_regimes_final': n_regimes_final,
        'n_features': num_nodes,
        'seed': seed,
        'task_type': task_type,
        'test_spearman': test_spearman,
        'test_pval': test_pval,
        'test_rmse': rmse,
        'test_directional_accuracy': directional_accuracy,
        'dags': [{'regime': d['regime'],
                  'n_edges': int((d['adjacency_matrix'] > 0.3).sum()) if d['adjacency_matrix'] is not None else 0,
                  'n_edges_high': int((d['adjacency_matrix'] > 0.5).sum()) if d['adjacency_matrix'] is not None else 0,
                  'edge_prob_sum': round(float(d['adjacency_matrix'].sum()), 2) if d['adjacency_matrix'] is not None else 0}
                 for d in dags]
    }


def run_ds3m_causal(
    dataset: str,
    d_dim: int,
    seed: int,
    feature_groups: List[str] = None,
    sharing_mode: str = 'independent',
    use_regularization: bool = True,
    device: str = 'cuda',
    verbose: bool = True,
    task_type: str = 'prediction'
) -> Dict:
    """
    Run DS3MCausal hybrid experiment (DS3M regime switching + FANTOM per-regime DAGs).

    This model learns a separate causal DAG for each regime, combining:
    - DS3M's discrete regime switching (d_t via Markov transition)
    - FANTOM's causal graph discovery (A^(d) per regime)

    Args:
        dataset: 'DE', 'FR', or 'DE_FR'
        d_dim: Number of regimes
        seed: Random seed
        feature_groups: Feature groups to include
        sharing_mode: 'independent' (separate DAG per regime) or 'shared_backbone'
        use_regularization: Use regime regularization to prevent collapse
        device: Device to use
        verbose: Print progress
        task_type: 'prediction' or 'estimation'
            - 'prediction': Forecast Y[t+1] using X[0:t] and Y[0:t]
            - 'estimation': Estimate Y[t] using X[0:t] and Y[0:t-1]
    """
    from unified_data_loader import prepare_unified_ds3m_data
    from ds3m_fantom.models.ds3m_causal import DS3MCausal
    from ds3m_fantom.training.regime_regularization import (
        RegimeRegularizer, diagnose_regime_collapse
    )

    set_seed(seed)

    # Default feature groups
    if feature_groups is None:
        if dataset == 'DE_FR':
            feature_groups = ['spread', 'price_de', 'price_fr', 'calendar', 'spgci']
        else:
            feature_groups = ['price', 'calendar', 'load', 'weather']

    # Load data
    data = prepare_unified_ds3m_data(
        dataset,
        timestep=14,
        feature_groups=feature_groups,
        target_col='price_spread_change_pct' if dataset == 'DE_FR' else 'price_change_pct',
        task_type=task_type
    )

    x_dim = data['trainX'].shape[-1]
    device_obj = torch.device(device)

    # Create DS3MCausal model (learns per-regime DAGs)
    model = DS3MCausal(
        x_dim=x_dim,
        y_dim=1,
        h_dim=32,
        z_dim=8,
        d_dim=d_dim,
        device=device_obj,
        n_layers=1,
        num_nodes=x_dim,
        lag=1,
        sharing_mode=sharing_mode,  # 'independent' = separate DAG per regime
        tau_gumbel=1.0,
        lambda_dag=100.0,
        lambda_sparse=0.01,  # Light sparsity to allow edge learning
        lambda_kl=1.0,
    ).to(device)

    # Regime regularizer (prevents collapse)
    regularizer = None
    if use_regularization:
        regularizer = RegimeRegularizer(
            d_dim=d_dim,
            entropy_weight=1.0,
            min_usage_weight=0.5,
            smoothness_weight=0.1,
            kl_weight=1.0,
            min_usage_ratio=0.1,
            annealing_start=10,
            annealing_end=50
        )

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainX = data['trainX'].to(device)
    trainY = data['trainY'].to(device)
    testX = data['testX'].to(device)
    testY = data['testY'].to(device)

    best_val_spearman = -np.inf
    patience_counter = 0
    patience = 20
    best_state = None

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        results = model(trainX, trainY)

        # Use model's computed loss (includes proper lambda_sparse weighting)
        loss = results['loss']

        # Add regime regularization
        if regularizer is not None and 'regime_posteriors' in results:
            reg_loss = regularizer(results['regime_posteriors'])
            loss += reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if regularizer is not None:
            regularizer.step()

        # Validation
        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                # FIX: Use model.predict() to avoid data leakage (don't pass actual Y)
                val_pred_result = model.predict(data['valX'].to(device), n_samples=10)
                # predictions has shape [timestep, batch, y_dim] - take last timestep
                val_pred = val_pred_result['predictions'][-1, :, 0].cpu().numpy()  # [batch]
                val_true = data['valY'][-1, :, 0].cpu().numpy()  # [batch]

                # Denormalize
                val_pred_denorm = val_pred * data['Y_moments'][1] + data['Y_moments'][0]
                val_true_denorm = val_true * data['Y_moments'][1] + data['Y_moments'][0]

                val_spearman, _ = spearmanr(val_true_denorm, val_pred_denorm)

                # Check for regime collapse using a forward pass (training context only)
                val_result = model(data['valX'].to(device), data['valY'].to(device))
                if 'regime_posteriors' in val_result:
                    collapse_info = diagnose_regime_collapse(val_result['regime_posteriors'])
                    if verbose and epoch % 20 == 19:
                        print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Val Spearman={val_spearman:.4f}, "
                              f"Regimes={collapse_info['effective_regimes']}, Collapsed={collapse_info['collapsed']}")
                elif verbose:
                    print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Val Spearman={val_spearman:.4f}")

                if val_spearman > best_val_spearman:
                    best_val_spearman = val_spearman
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch+1}")
                        break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Test evaluation
    model.eval()
    with torch.no_grad():
        # FIX: Use model.predict() to avoid data leakage (don't pass actual Y)
        test_pred_result = model.predict(testX, n_samples=50)  # More samples for final evaluation
        # predictions has shape [timestep, batch, y_dim] - take last timestep
        test_pred = test_pred_result['predictions'][-1, :, 0].cpu().numpy()  # [batch]
        test_true = testY[-1, :, 0].cpu().numpy()  # [batch]

        # Denormalize
        test_pred_denorm = test_pred * data['Y_moments'][1] + data['Y_moments'][0]
        test_true_denorm = test_true * data['Y_moments'][1] + data['Y_moments'][0]

        test_spearman, test_pval = spearmanr(test_true_denorm, test_pred_denorm)
        rmse = np.sqrt(np.mean((test_true_denorm - test_pred_denorm) ** 2))

        # Directional accuracy: percentage of correct sign predictions
        correct_direction = ((test_pred_denorm > 0) == (test_true_denorm > 0))
        directional_accuracy = float(correct_direction.mean())

        # Get regime info from predict result
        regime_assignments = test_pred_result.get('regimes', None)
        regime_counts = {}
        collapse_info = {}
        if regime_assignments is not None:
            # Count regime assignments
            for r in range(d_dim):
                regime_counts[r] = int((regime_assignments == r).sum().item())
            # Check for collapse (>90% in one regime)
            total = sum(regime_counts.values())
            max_count = max(regime_counts.values()) if regime_counts else 0
            collapsed = (max_count / total) > 0.9 if total > 0 else False
            effective_regimes = sum(1 for c in regime_counts.values() if c > 0.05 * total)
            collapse_info = {
                'collapsed': collapsed,
                'effective_regimes': effective_regimes,
                'regime_counts': regime_counts
            }

        # Get per-regime DAGs
        dags_info = []
        try:
            dags = model.get_causal_graphs()  # Returns list of adjacency matrices per regime
            for r, adj in enumerate(dags):
                if isinstance(adj, torch.Tensor):
                    adj = adj.cpu().numpy()
                # Count edges at different thresholds for better diagnostics
                n_edges_high = int((np.abs(adj) > 0.5).sum())  # High confidence edges
                n_edges_low = int((np.abs(adj) > 0.3).sum())   # Medium confidence edges
                edge_prob_sum = float(np.abs(adj).sum())       # Total edge probability
                dags_info.append({
                    'regime': r,
                    'n_edges': n_edges_low,  # Use lower threshold for main count
                    'n_edges_high': n_edges_high,
                    'edge_prob_sum': round(edge_prob_sum, 2)
                })
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not extract DAGs: {e}")

    return {
        'model': 'DS3MCausal',
        'dataset': dataset,
        'd_dim': d_dim,
        'sharing_mode': sharing_mode,
        'n_features': x_dim,
        'seed': seed,
        'task_type': task_type,
        'test_spearman': test_spearman,
        'test_pval': test_pval,
        'test_rmse': rmse,
        'test_directional_accuracy': directional_accuracy,
        'best_val_spearman': best_val_spearman,
        'regime_distribution': regime_counts,
        'regime_collapsed': collapse_info.get('collapsed', None),
        'effective_regimes': collapse_info.get('effective_regimes', d_dim),
        'dags': dags_info
    }


def run_multiseed_experiment(
    model_type: str,
    dataset: str,
    n_seeds: int = 5,
    d_dim: int = 2,
    use_bem: bool = True,
    device: str = 'cuda',
    verbose: bool = True,
    task_type: str = 'prediction'
) -> Dict:
    """
    Run experiment with multiple seeds and aggregate results.

    Args:
        model_type: 'ds3m_uv', 'ds3m_mv', 'fantom', 'fantom_bem', or 'ds3m_causal'
        dataset: 'DE', 'FR', or 'DE_FR'
        n_seeds: Number of random seeds
        d_dim: Number of regimes
        use_bem: Use BEM for FANTOM
        device: Device to use
        verbose: Print progress
        task_type: 'prediction' or 'estimation'
            - 'prediction': Forecast Y[t+1] using X[0:t] and Y[0:t]
            - 'estimation': Estimate Y[t] using X[0:t] and Y[0:t-1]

    Returns:
        Dictionary with mean, std, and individual results
    """
    seeds = [42, 123, 456, 789, 1000][:n_seeds]

    results = []
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {model_type} on {dataset} (seed {seed}, {i+1}/{n_seeds})")
            print(f"{'='*60}")

        try:
            if model_type == 'ds3m_uv':
                result = run_ds3m_univariate(dataset, d_dim, seed, use_regularization=True, device=device, verbose=verbose, task_type=task_type)
            elif model_type == 'ds3m_mv':
                result = run_ds3m_multivariate(dataset, d_dim, seed, device=device, verbose=verbose, task_type=task_type)
            elif model_type in ['fantom', 'fantom_bem']:
                result = run_fantom_spread(dataset, seed, n_regimes=d_dim, use_bem=use_bem or model_type=='fantom_bem', device=device, verbose=verbose, task_type=task_type)
            elif model_type == 'ds3m_causal':
                result = run_ds3m_causal(dataset, d_dim, seed, device=device, verbose=verbose, task_type=task_type)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            results.append(result)

            if verbose:
                print(f"  Result: Spearman={result['test_spearman']:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'seed': seed, 'error': str(e), 'test_spearman': np.nan})

    # Aggregate
    spearman_values = [r['test_spearman'] for r in results if not np.isnan(r.get('test_spearman', np.nan))]
    rmse_values = [r.get('test_rmse', np.nan) for r in results if not np.isnan(r.get('test_rmse', np.nan))]
    accuracy_values = [r.get('test_directional_accuracy', np.nan) for r in results if not np.isnan(r.get('test_directional_accuracy', np.nan))]

    summary = {
        'model': model_type,
        'dataset': dataset,
        'd_dim': d_dim,
        'task_type': task_type,
        'n_seeds': n_seeds,
        'spearman_mean': np.mean(spearman_values) if spearman_values else np.nan,
        'spearman_std': np.std(spearman_values) if len(spearman_values) > 1 else 0.0,
        'spearman_min': np.min(spearman_values) if spearman_values else np.nan,
        'spearman_max': np.max(spearman_values) if spearman_values else np.nan,
        'rmse_mean': np.mean(rmse_values) if rmse_values else np.nan,
        'rmse_std': np.std(rmse_values) if len(rmse_values) > 1 else 0.0,
        'accuracy_mean': np.mean(accuracy_values) if accuracy_values else np.nan,
        'accuracy_std': np.std(accuracy_values) if len(accuracy_values) > 1 else 0.0,
        'n_successful': len(spearman_values),
        'individual_results': results
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument('--model', type=str, required=True,
                        choices=['ds3m_uv', 'ds3m_mv', 'fantom', 'fantom_bem', 'ds3m_causal'],
                        help='Model type: ds3m_uv (univariate), ds3m_mv (multivariate), '
                             'fantom (stationary), fantom_bem (per-regime DAG), '
                             'ds3m_causal (hybrid: DS3M regimes + per-regime DAGs)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['DE', 'FR', 'DE_FR'],
                        help='Dataset')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--d_dim', type=int, default=2, help='Number of regimes')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--task_type', type=str, default='prediction',
                        choices=['prediction', 'estimation'],
                        help='Task type: prediction (forecast Y[t+1]) or estimation (estimate Y[t] using Y[0:t-1])')

    args = parser.parse_args()

    # Run experiment
    print(f"\n{'#'*70}")
    print(f"# Multi-Seed Experiment: {args.model} on {args.dataset}")
    print(f"# Seeds: {args.seeds}, Regimes: {args.d_dim}, Task: {args.task_type}")
    print(f"{'#'*70}")

    summary = run_multiseed_experiment(
        model_type=args.model,
        dataset=args.dataset,
        n_seeds=args.seeds,
        d_dim=args.d_dim,
        device=args.device,
        task_type=args.task_type
    )

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {args.model} on {args.dataset}")
    print(f"{'='*70}")
    print(f"  Spearman: {summary['spearman_mean']:.4f} ± {summary['spearman_std']:.4f}")
    print(f"  RMSE:     {summary['rmse_mean']:.2f} ± {summary['rmse_std']:.2f}")
    print(f"  Accuracy: {summary['accuracy_mean']:.2%} ± {summary['accuracy_std']:.2%}")
    print(f"  Range:    [{summary['spearman_min']:.4f}, {summary['spearman_max']:.4f}]")
    print(f"  Success:  {summary['n_successful']}/{args.seeds}")

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "outputs" / "multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{args.model}_{args.dataset}_d{args.d_dim}_{args.task_type}_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
