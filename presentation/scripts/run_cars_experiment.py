#!/usr/bin/env python3
"""
CaRS (Causal Regime-Switching State Space Model) experiment runner.

This script runs CaRS experiments on the unified electricity dataset with:
- Multiple lag configurations (lag=1, 2, 3, ...)
- Sparsity parameter sweeps
- Different numbers of regimes (d=2, 3, 4)

Usage:
    python run_cars_experiment.py --dataset DE --n_regimes 2 --lag 1 --lambda_sparse 5.0 --seed 42
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add the FANTOM code directory to path
FANTOM_CODE_DIR = os.path.join(os.path.dirname(__file__), '../../FANTOM_supplementary/fantom_code')
sys.path.insert(0, FANTOM_CODE_DIR)

# Add CASTOR electricity directory for unified data loader
CASTOR_ELECTRICITY_DIR = '/lustre/home/dthumm/CASTOR/electricity'
sys.path.insert(0, CASTOR_ELECTRICITY_DIR)

# Import from FANTOM code
import fantom

# Import unified data loader from CASTOR
from unified_data_loader import load_unified_dataset, get_feature_columns

# Import DS3MCausal model from shared_backbone
FANTOM_ROOT = str(Path(__file__).parent.parent.parent)
if FANTOM_ROOT not in sys.path:
    sys.path.insert(0, FANTOM_ROOT)

try:
    from shared_backbone.models.ds3m_causal import DS3MCausal
    from shared_backbone.training.train_e2e import AugmentedLagrangianTrainer
    DS3M_CAUSAL_AVAILABLE = True
except ImportError as e:
    DS3M_CAUSAL_AVAILABLE = False
    print(f"Warning: DS3MCausal not available ({e}). Only FANTOM model will be usable.")
from torch.utils.data import DataLoader
import torch.nn as nn

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def get_model_config(n_nodes: int, lag: int, lambda_sparse: float = 5.0,
                     lambda_dag: float = 100.0, tau_gumbel: float = 1.0) -> dict:
    """Get CaRS model configuration.

    Args:
        n_nodes: Number of features/nodes
        lag: Number of time lags to consider
        lambda_sparse: L1 sparsity regularization parameter
        lambda_dag: DAG constraint penalty
        tau_gumbel: Gumbel-Softmax temperature
    """
    # Adaptive configuration based on feature count
    # Large datasets (>50 features) need more conservative settings
    is_large_dataset = n_nodes > 50

    if is_large_dataset:
        # Conservative settings for 30-50 features
        learning_rate = 0.003
        encoder_layers = [64, 64]
        decoder_layers = [64, 64]
        spline_bins = 64
        batch_size = 128
        base_dist_type = 'conditional_spline'
        print(f"  Using large dataset config (n_nodes={n_nodes}): lr={learning_rate}")
    else:
        # Standard settings for <30 features
        learning_rate = 0.005
        encoder_layers = [32, 32]
        decoder_layers = [32, 32]
        spline_bins = 128
        batch_size = 256
        base_dist_type = 'conditional_spline'

    config = {
        'model_config': {
            'tau_gumbel': tau_gumbel,
            'lambda_dag': lambda_dag,
            'lambda_sparse': lambda_sparse,
            'lambda_sparse_l2': 0.0,
            'l2_group_mode': 'column',
            'spline_bins': spline_bins,
            'encoder_layer_sizes': encoder_layers,
            'decoder_layer_sizes': decoder_layers,
            'var_dist_A_mode': 'temporal_three',
            'heteroscedastic': True,
            'norm_layers': True,
            'res_connection': True,
            'base_distribution_type': base_dist_type,
        },
        'training_params': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'stardardize_data_mean': False,
            'stardardize_data_std': False,
            'rho': 1.0,
            'safety_rho': 1e13,
            'alpha': 0.0,
            'safety_alpha': 1e13,
            'tol_dag': 1e-4,
            'progress_rate': 0.65,
            'max_steps_auglag': 3,
            'max_auglag_inner_epochs': 2000,
            'max_p_train_dropout': 0.0,
            'reconstruction_loss_factor': 1.0,
            'anneal_entropy': 'noanneal',
        }
    }

    return config


def load_electricity_data(dataset: str, data_dir: str = None, smooth_window: int = 0,
                          use_returns: bool = False) -> tuple:
    """Load unified electricity dataset using CASTOR data loader.

    Uses curated feature groups for causal discovery (~20-30 features)
    rather than all available features, since FANTOM's ICGNN becomes
    numerically unstable with >50 features.

    Args:
        dataset: Dataset name ('DE', 'FR', or 'DE_FR')
        data_dir: Directory containing the data files (ignored, uses CASTOR unified data)
        smooth_window: Rolling window size for smoothing (0 = no smoothing).
                       For DE_FR, applies to price spread features to reduce rapid regime switching.
        use_returns: If True, use price_change_pct instead of raw Day_Ahead_Price.
                     This makes FANTOM results comparable to CaRS/DS3M which predict returns.

    Returns:
        Tuple of (data array, feature names, timestamps)
    """
    # Load unified dataset from CASTOR
    df = load_unified_dataset(dataset, clean=True)

    # Select feature groups for causal discovery
    # Using key groups rather than all features for numerical stability
    if dataset == 'DE_FR':
        groups = ['price_de', 'price_fr', 'generation_de', 'generation_fr',
                  'load_de', 'load_fr', 'weather_de', 'weather_fr',
                  'flow', 'commodity']
    else:
        groups = ['price', 'generation', 'load', 'weather', 'flow', 'commodity']

    feature_cols = get_feature_columns(df, groups=groups, exclude_target=True, country=dataset)

    # Remove derived price columns that are targets or redundant
    if use_returns:
        # When using returns: include price_change_pct, exclude raw prices
        # This makes FANTOM comparable to CaRS which predicts percentage returns
        if dataset == 'DE_FR':
            # For DE_FR: use price_spread_change_pct as the price feature
            # Exclude: raw prices, lags, rolling features, all other price change variants
            exclude_patterns = ['Day_Ahead_Price', 'price_lag', 'Price_Change', 'Price_Return',
                                'price_direction', '_rolling_', 'price_change', 'price_spread_lag',
                                'price_spread_rolling']
            # Filter out excluded patterns
            feature_cols = [c for c in feature_cols if not any(p in c for p in exclude_patterns)]
            # Add spread change pct as the first feature (the target equivalent)
            if 'price_spread_change_pct' in df.columns:
                feature_cols = ['price_spread_change_pct'] + feature_cols
        else:
            # For DE/FR: use price_change_pct as the price feature
            exclude_patterns = ['Day_Ahead_Price', 'price_lag', 'Price_Change', 'Price_Return',
                                'price_direction', '_rolling_', 'price_change']
            # Filter out excluded patterns (but keep price_change_pct since we add it explicitly)
            feature_cols = [c for c in feature_cols
                            if not any(p in c for p in exclude_patterns) or c == 'price_change_pct']
            # Remove price_change_pct if it got included, then add it as first feature
            feature_cols = [c for c in feature_cols if c != 'price_change_pct']
            if 'price_change_pct' in df.columns:
                feature_cols = ['price_change_pct'] + feature_cols
        print(f"  Using returns mode: price feature is {'price_spread_change_pct' if dataset == 'DE_FR' else 'price_change_pct'}")
    else:
        # Original behavior: exclude all derived price columns, keep raw prices
        exclude_patterns = ['price_change', 'price_pct', 'Price_Change', 'Price_Return',
                            'price_direction', '_rolling_', '_lag']
        feature_cols = [c for c in feature_cols
                        if not any(p in c for p in exclude_patterns)]

    print(f"  Selected {len(feature_cols)} features for causal discovery")

    # Apply smoothing if requested (particularly useful for DE_FR to reduce rapid switching)
    if smooth_window > 0:
        print(f"  Applying rolling average smoothing (window={smooth_window})...")
        for col in feature_cols:
            # Apply centered rolling mean, filling edges with original values
            smoothed = df[col].rolling(window=smooth_window, center=True, min_periods=1).mean()
            df[col] = smoothed
        print(f"  Smoothing applied to {len(feature_cols)} features")

    # Extract data and timestamps
    data = df[feature_cols].values.astype(np.float32)
    timestamps = df.index.values

    # Handle any NaN values
    data = np.nan_to_num(data, nan=0.0)

    # Robust preprocessing for stability
    # 1. Clip extreme outliers (beyond 5 std from mean)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    clip_threshold = 5.0
    data = np.clip(data, mean - clip_threshold * std, mean + clip_threshold * std)

    # 2. Standardize to zero mean, unit variance
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

    # 3. Final clip to [-5, 5] for numerical stability
    data = np.clip(data, -5.0, 5.0)

    print(f"  Preprocessed data: min={data.min():.2f}, max={data.max():.2f}, std={data.std():.2f}")

    return data, feature_cols, timestamps


def prepare_temporal_data(data: np.ndarray, lag: int) -> np.ndarray:
    """Prepare data in temporal format for FANTOM model.

    Args:
        data: Raw data array of shape (T, n_features)
        lag: Number of time lags

    Returns:
        Temporal data array of shape (T-lag, lag+1, n_features)
    """
    T, n_features = data.shape
    n_samples = T - lag

    # Create temporal windows
    temporal_data = np.zeros((n_samples, lag + 1, n_features), dtype=np.float32)

    for i in range(n_samples):
        for l in range(lag + 1):
            temporal_data[i, l, :] = data[i + l, :]

    return temporal_data


class RegimePriorNetwork(nn.Module):
    """Regime prior network for Bayesian EM."""
    def __init__(self, n_regimes: int):
        super().__init__()
        self.n_regimes = n_regimes
        self.linear = nn.Linear(1, n_regimes)

    def forward(self, t):
        return self.linear(t)


def train_regime_prior(model, data, gamma, num_epochs=500, lr=0.001):
    """Train the regime prior network."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    softmax = nn.Softmax(dim=-1)

    for _ in range(num_epochs):
        y_pred = model(data)
        loss = criterion(y_pred, gamma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return softmax(y_pred), loss.item(), model


def train_ds3m_causal(model: nn.Module, data: np.ndarray, device: torch.device,
                      training_params: dict, n_epochs: int = 500) -> None:
    """Train DS3MCausal model using standard PyTorch training loop.

    Args:
        model: DS3MCausal model instance
        data: Training data of shape (n_samples, lag+1, n_features)
        device: PyTorch device
        training_params: Training parameters dict
        n_epochs: Number of training epochs
    """
    model.train()
    model.to(device)

    lr = training_params.get('learning_rate', 0.001)
    batch_size = training_params.get('batch_size', 64)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert data to tensor
    # DS3MCausal expects [timestep, batch, features] format
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)

    n_samples = data_tensor.shape[0]
    n_features = data_tensor.shape[2]

    for epoch in range(n_epochs):
        # Shuffle data
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples - batch_size, batch_size):
            batch_idx = perm[i:i + batch_size]
            batch_data = data_tensor[batch_idx]  # [batch, lag+1, features]

            # Transpose to [lag+1, batch, features] for DS3MCausal
            x = batch_data.permute(1, 0, 2)  # [lag+1, batch, features]

            # Target = Price (first column); all features are input to causal graph
            y = x[:, :, 0:1].clone()  # [lag+1, batch, 1]

            # Forward pass (all features including Price)
            optimizer.zero_grad()
            result = model(x, y)
            loss = result['loss']

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"    DS3MCausal epoch {epoch + 1}/{n_epochs}, loss: {avg_loss:.4f}")


def compute_ds3m_causal_log_prob(model: nn.Module, data: np.ndarray,
                                  device: torch.device) -> np.ndarray:
    """Compute log probability for DS3MCausal model.

    Args:
        model: Trained DS3MCausal model
        data: Data of shape (n_samples, lag+1, n_features)
        device: PyTorch device

    Returns:
        Log probabilities of shape (n_samples,)
    """
    model.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)

    with torch.no_grad():
        n_samples = data_tensor.shape[0]
        log_probs = np.zeros(n_samples)

        # Process in batches to avoid memory issues
        batch_size = 64
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_data = data_tensor[i:end_idx]  # [batch, lag+1, features]

            # Transpose to [lag+1, batch, features]
            x = batch_data.permute(1, 0, 2)
            y = x[:, :, 0:1].clone()  # Target = Price (first column)

            # Forward pass (all features as input, including Price)
            result = model(x, y)

            # Use negative NLL as log probability (normalized by timesteps)
            nll = result['nll'].item()
            timesteps = x.shape[0]
            batch_log_prob = -nll / timesteps

            log_probs[i:end_idx] = batch_log_prob

    return log_probs


def train_ds3m_causal_native(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    training_params: dict,
    n_epochs: int = 1000,
) -> None:
    """Train DS3MCausal with native Markov regime switching using Augmented Lagrangian.

    Uses the AugmentedLagrangianTrainer for proper DAG constraint enforcement,
    temperature annealing, and validation-based early stopping.

    Args:
        model: DS3MCausal model instance with d_dim=n_regimes
        X: Training data of shape (n_samples, lag+1, n_features)
        device: PyTorch device
        training_params: Training parameters dict
        n_epochs: Number of training epochs (used for max_auglag_steps)
    """
    model.train()
    model.to(device)

    lr = training_params.get('learning_rate', 0.003)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert data to tensor and split into train/test
    data_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    n_samples = data_tensor.shape[0]

    # 80/20 train/test split (chronological)
    split_idx = int(n_samples * 0.8)
    train_data = data_tensor[:split_idx]
    test_data = data_tensor[split_idx:]

    # Prepare data in [timestep, batch, features] format
    # Use all train samples as a single batch (timestep=lag+1, batch=n_train)
    # All features (including Price at index 0) go into x; Price is also the target y.
    # The causal graph is built over ALL features so we can see edges TO/FROM Price.
    train_x = train_data.permute(1, 0, 2)  # [lag+1, n_train, features]
    trainX = train_x              # All features including Price
    trainY = train_x[:, :, 0:1]  # Target = Price (first column)

    test_x = test_data.permute(1, 0, 2)  # [lag+1, n_test, features]
    testX = test_x
    testY = test_x[:, :, 0:1]

    # Target variable index (Day_Ahead_Price is typically first feature = index 0)
    target_idx = 0

    # Create Augmented Lagrangian trainer with proper DAG enforcement
    trainer = AugmentedLagrangianTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        # Augmented Lagrangian parameters
        alpha_init=0.0,
        rho_init=1.0,
        rho_max=1e9,
        progress_rate=0.9,
        tol_dag=1e-6,
        # Training parameters
        max_auglag_steps=50,
        max_inner_epochs=50,
        patience_dag=5,
        patience_rho=3,
        # Early stopping on validation
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_metric='directional_accuracy',
        # Target constraint: encourage meaningful edges to Price
        target_idx=target_idx,
        lambda_target=10.0,
        # Temperature annealing: push edges toward binary (0/1)
        tau_init=1.0,
        tau_final=0.1,
        tau_anneal_steps=50,
        # Regime differentiation: encourage distinct DAGs per regime
        lambda_regime_diff=1.0,
        verbose=True,
    )

    print(f"  Training with Augmented Lagrangian (max 50 outer steps x 50 inner epochs)")
    print(f"  Train: {trainX.shape[1]} samples, Test: {testX.shape[1]} samples")

    # Run full augmented Lagrangian training
    history = trainer.train(trainX, trainY, testX, testY)

    print(f"  Training complete. Final DAG penalty: {history['dag_penalty'][-1]:.8f}"
          if history['dag_penalty'] else "  Training complete.")


def run_cars_native_markov(
    X: np.ndarray,
    n_regimes: int,
    n_nodes: int,
    lag: int,
    device: torch.device,
    model_config: dict,
    training_params: dict,
    n_epochs: int = 1000,
) -> dict:
    """Run CaRS using DS3MCausal's native Markov regime switching.

    Uses a single DS3MCausal model with d_dim=n_regimes.
    The model learns regime transitions via its internal Markov chain.

    Args:
        X: Temporal data of shape (T, lag+1, n_nodes)
        n_regimes: Number of regimes
        n_nodes: Number of features/nodes
        lag: Number of time lags
        device: PyTorch device
        model_config: Model configuration
        training_params: Training parameters
        n_epochs: Number of training epochs

    Returns:
        Dictionary with results including gamma_hat, models, adj_matrices
    """
    if not DS3M_CAUSAL_AVAILABLE:
        raise RuntimeError("DS3MCausal model not available. Check import errors.")

    T = X.shape[0]
    x_dim = n_nodes  # All features including Price (causal graph over all nodes)

    print(f"  Native Markov: Creating DS3MCausal with d_dim={n_regimes}")

    # Single model with K regimes
    model = DS3MCausal(
        x_dim=x_dim,
        y_dim=1,
        h_dim=32,
        z_dim=8,
        d_dim=n_regimes,  # Native multi-regime
        device=device,
        num_nodes=x_dim,
        lag=lag,
        lambda_dag=model_config.get('lambda_dag', 100.0),
        lambda_sparse=model_config.get('lambda_sparse', 5.0),
    )

    # Train end-to-end with native Markov
    print(f"  Training with native Markov regime switching...")
    train_ds3m_causal_native(model, X, device, training_params, n_epochs=n_epochs)

    # Extract regime assignments from model's d_t posteriors
    print(f"  Extracting regime assignments...")
    regime_assignments = model.get_regime_assignments(X)

    # Get soft posteriors for gamma_hat
    regime_posteriors = model.get_regime_posteriors(X)

    # Construct gamma_hat (one-hot from assignments)
    gamma_hat = np.zeros((T, n_regimes))
    gamma_hat[np.arange(T), regime_assignments] = 1.0

    # Get weighted causal DAG per regime (W * A = actual causal coefficients)
    adj_matrices = model.get_weighted_causal_graphs()

    # Count regimes
    regime_counts = np.sum(gamma_hat, axis=0)
    n_regimes_final = np.sum(regime_counts > 0)

    print(f"  Native Markov: {n_regimes_final} regimes active, counts: {regime_counts}")

    return {
        'gamma_hat': gamma_hat,
        'models': [model],
        'adj_matrices': adj_matrices,
        'n_regimes_final': int(n_regimes_final),
        'regime_assignments': regime_assignments,
        'regime_posteriors': regime_posteriors,
    }


def run_cars(X: np.ndarray, n_regimes: int, n_nodes: int, lag: int,
             device: torch.device, model_config: dict, training_params: dict,
             max_iterations: int = 3, zeta_factor: float = 0.3,
             prior_thresh: float = 0.85, model_type: str = 'fantom') -> dict:
    """Run CaRS algorithm with Bayesian EM for regime detection.

    Args:
        X: Temporal data of shape (T, lag+1, n_nodes)
        n_regimes: Number of regimes
        n_nodes: Number of features/nodes
        lag: Number of time lags
        device: PyTorch device
        model_config: Model configuration dict
        training_params: Training parameters dict
        max_iterations: Maximum BEM iterations
        zeta_factor: Regime pruning threshold factor
        prior_thresh: Prior training loss threshold
        model_type: Model type ('fantom' or 'ds3m_causal')

    Returns:
        Dict containing regime assignments, models, and adjacency matrices
    """
    T = X.shape[0]
    window = T // n_regimes
    zeta = int(window * zeta_factor)

    # Standardize data
    X_std = np.zeros_like(X)
    for l in range(lag + 1):
        mean = X[:, l, :].mean(axis=0)
        std = X[:, l, :].std(axis=0) + 1e-8
        X_std[:, l, :] = (X[:, l, :] - mean) / std

    N_regime = n_regimes

    for iteration in range(max_iterations):
        print(f"\nBEM Iteration {iteration + 1}/{max_iterations}")

        # Adjust training params for later iterations
        if iteration >= 2:
            training_params['encoder_layer_sizes'] = [32, 32]
            training_params['decoder_layer_sizes'] = [32, 32]
            training_params['max_auglag_inner_epochs'] = 3000
            training_params['max_steps_auglag'] = 3

        # Initialize regime-specific models
        if model_type == 'fantom':
            models = [
                fantom.FANTOM_stationary(
                    n_nodes, device, lag=lag, allow_instantaneous=True,
                    **model_config
                ) for _ in range(N_regime)
            ]
        elif model_type == 'ds3m_causal':
            if not DS3M_CAUSAL_AVAILABLE:
                raise RuntimeError("DS3MCausal model not available. Check import errors.")
            # DS3MCausal: x_dim = all input features (including Price)
            # num_nodes = nodes in causal graph = all features
            # Price (first column) is the target Y; causal graph covers all features
            x_dim_ds3m = n_nodes
            models = [
                DS3MCausal(
                    x_dim=x_dim_ds3m,        # All features including Price
                    y_dim=1,                  # Target dimension
                    h_dim=32,
                    z_dim=8,
                    d_dim=1,  # Each instance handles 1 regime in BEM
                    device=device,
                    num_nodes=x_dim_ds3m,     # Causal graph over all features
                    lag=lag,
                    lambda_dag=model_config.get('lambda_dag', 100.0),
                    lambda_sparse=model_config.get('lambda_sparse', 5.0),
                ) for _ in range(N_regime)
            ]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        log_pdf_emission = np.zeros((T, N_regime))

        # E-step: Initialize or update regime assignments
        if iteration == 0:
            # Initial assignment based on time windows
            p = np.zeros((T, N_regime))
            for c in range(N_regime):
                if c == N_regime - 1:
                    start_idx = c * window
                    p[start_idx:, c] = 1.0
                    regime_data = X_std[start_idx:, :, :]
                else:
                    start_idx = c * window
                    end_idx = (c + 1) * window
                    p[start_idx:end_idx, c] = 1.0
                    regime_data = X_std[start_idx:end_idx, :, :]

                # Standardize regime data
                regime_data_norm = np.zeros_like(regime_data)
                for l in range(lag + 1):
                    mean = regime_data[:, l, :].mean(axis=0)
                    std = regime_data[:, l, :].std(axis=0) + 1e-8
                    regime_data_norm[:, l, :] = (regime_data[:, l, :] - mean) / std

                # Train model for this regime
                if model_type == 'fantom':
                    dataloader = DataLoader(regime_data_norm, training_params['batch_size'])
                    models[c].run_train(dataloader, regime_data_norm.shape[0], training_params)
                    # Compute emission probabilities
                    log_pdf_emission[:, c] = np.exp(
                        models[c].log_prob(torch.tensor(X_std), 1)
                    )
                else:  # ds3m_causal
                    train_ds3m_causal(models[c], regime_data_norm, device, training_params, n_epochs=500)
                    # Compute emission probabilities
                    log_pdf_emission[:, c] = np.exp(
                        compute_ds3m_causal_log_prob(models[c], X_std, device)
                    )
        else:
            # Use gamma_hat for soft assignment
            for c in range(N_regime):
                gamma = gamma_hat[:, c]
                masked_data = gamma.reshape((T, 1, 1)) * X
                nonzero_mask = ~np.all(masked_data == 0, axis=2)
                regime_data = masked_data[nonzero_mask.any(axis=1)]

                if len(regime_data) > 0:
                    regime_data = regime_data.reshape(-1, lag + 1, n_nodes)

                    # Standardize
                    for l in range(lag + 1):
                        mean = regime_data[:, l, :].mean(axis=0)
                        std = regime_data[:, l, :].std(axis=0) + 1e-8
                        regime_data[:, l, :] = (regime_data[:, l, :] - mean) / std

                    if model_type == 'fantom':
                        dataloader = DataLoader(regime_data, training_params['batch_size'])
                        models[c].run_train(dataloader, regime_data.shape[0], training_params)
                    else:  # ds3m_causal
                        train_ds3m_causal(models[c], regime_data, device, training_params, n_epochs=500)

                if model_type == 'fantom':
                    log_pdf_emission[:, c] = np.exp(
                        models[c].log_prob(torch.tensor(X_std), 1)
                    )
                else:  # ds3m_causal
                    log_pdf_emission[:, c] = np.exp(
                        compute_ds3m_causal_log_prob(models[c], X_std, device)
                    )

        # Compute posterior
        pall = np.sum(p * log_pdf_emission, axis=1)
        gamma_hat = (p * log_pdf_emission) / (pall.reshape((T, 1)) + 1e-8)

        # Soft-to-hard assignment: keep soft assignments for intermediate iterations
        # to prevent premature regime collapse
        is_last_iteration = (iteration >= max_iterations - 1)
        if is_last_iteration:
            # Hard assignment only on last iteration
            idx = np.argmax(gamma_hat, axis=-1)
            gamma_hat = np.zeros_like(gamma_hat)
            gamma_hat[np.arange(T), idx] = 1.0
        else:
            # Keep soft assignments but sharpen them slightly
            # Temperature annealing: start soft (temp=2), end hard (temp=0.5)
            temp = 2.0 - (iteration + 1) * 0.5
            temp = max(temp, 0.5)
            gamma_hat = np.power(gamma_hat, 1.0 / temp)
            gamma_hat = gamma_hat / (gamma_hat.sum(axis=1, keepdims=True) + 1e-8)

        # Train prior network
        t = torch.tensor(np.linspace(0, 20 * N_regime, T).reshape((T, 1)))
        prior_model = RegimePriorNetwork(N_regime)
        p, loss, prior_model = train_regime_prior(
            prior_model, t.float(), torch.tensor(gamma_hat).float()
        )
        p = p.detach().numpy()

        # Continue training until convergence
        while loss >= prior_thresh:
            p, loss, prior_model = train_regime_prior(
                prior_model, t.float(), torch.tensor(gamma_hat).float(),
                num_epochs=100
            )
            p = p.detach().numpy()

        # Update posterior with learned prior
        if iteration == 0:
            pall = np.sum(p * log_pdf_emission, axis=1)
            gamma_hat = (p * log_pdf_emission) / (pall.reshape((T, 1)) + 1e-8)
            # Only do hard assignment if this is also the last iteration
            if is_last_iteration:
                idx = np.argmax(gamma_hat, axis=-1)
                gamma_hat = np.zeros_like(gamma_hat)
                gamma_hat[np.arange(T), idx] = 1.0
            else:
                # Keep soft with temperature
                gamma_hat = np.power(gamma_hat, 1.0 / temp)
                gamma_hat = gamma_hat / (gamma_hat.sum(axis=1, keepdims=True) + 1e-8)

        # Prune small regimes
        gamma_sum = np.sum(gamma_hat, axis=0)
        gamma_sum[gamma_sum < zeta] = 0
        valid_regimes = np.where(gamma_sum != 0)[0]
        gamma_hat = gamma_hat[:, valid_regimes]
        p = p[:, valid_regimes]
        models = [models[i] for i in valid_regimes]
        N_regime = len(valid_regimes)

        print(f"  Regime sizes: {np.sum(gamma_hat, axis=0)}, N_regime: {N_regime}")

    # Extract adjacency matrices
    adj_matrices = []
    for model in models:
        if model_type == 'fantom':
            adj = model.get_adj_matrix(samples=1, most_likely_graph=True, squeeze=True)
            adj_matrices.append(adj)
        else:  # ds3m_causal
            # DS3MCausal returns list of numpy arrays from get_causal_graphs()
            graphs = model.get_causal_graphs()
            # get_causal_graphs returns a list per regime, but we have 1 regime per model (d_dim=1)
            adj_matrices.append(graphs[0] if len(graphs) == 1 else np.stack(graphs))

    return {
        'gamma_hat': gamma_hat,
        'models': models,
        'adj_matrices': adj_matrices,
        'n_regimes_final': N_regime,
        'regime_assignments': np.argmax(gamma_hat, axis=-1),
    }


def compute_edge_statistics(adj_matrices: list, feature_names: list, lag: int) -> dict:
    """Compute edge statistics from adjacency matrices.

    Args:
        adj_matrices: List of adjacency matrices per regime
        feature_names: List of feature names
        lag: Number of time lags

    Returns:
        Dict with edge statistics
    """
    n_regimes = len(adj_matrices)
    n_nodes = len(feature_names)

    stats = {
        'n_regimes': n_regimes,
        'n_nodes': n_nodes,
        'lag': lag,
        'regimes': []
    }

    for r, adj in enumerate(adj_matrices):
        # adj has shape (lag+1, n_nodes, n_nodes)
        # adj[0] = instantaneous edges
        # adj[1:] = lagged edges

        regime_stats = {
            'regime': r,
            'instantaneous': {
                'n_edges': int(np.sum(adj[0] > 0)),
                'avg_weight': float(np.mean(np.abs(adj[0][adj[0] != 0]))) if np.any(adj[0] != 0) else 0,
                'max_weight': float(np.max(np.abs(adj[0]))) if np.any(adj[0] != 0) else 0,
            },
            'lagged': {},
            'total_edges': int(np.sum(adj > 0)),
        }

        for l in range(1, lag + 1):
            regime_stats['lagged'][f'lag_{l}'] = {
                'n_edges': int(np.sum(adj[l] > 0)),
                'avg_weight': float(np.mean(np.abs(adj[l][adj[l] != 0]))) if np.any(adj[l] != 0) else 0,
                'max_weight': float(np.max(np.abs(adj[l]))) if np.any(adj[l] != 0) else 0,
            }

        # Feature importance (sum of incoming edge weights)
        feature_importance = {}
        for i, fname in enumerate(feature_names):
            # Sum of all edges pointing to this node
            incoming = np.sum(np.abs(adj[:, :, i]))
            outgoing = np.sum(np.abs(adj[:, i, :]))
            feature_importance[fname] = {
                'incoming': float(incoming),
                'outgoing': float(outgoing),
                'total': float(incoming + outgoing),
            }
        regime_stats['feature_importance'] = feature_importance

        stats['regimes'].append(regime_stats)

    return stats


def run_experiment(args):
    """Run a single CaRS experiment."""
    print("=" * 80)
    print(f"CaRS Experiment")
    print(f"model={args.model}, dataset={args.dataset}, n_regimes={args.n_regimes}, lag={args.lag}")
    print(f"lambda_sparse={args.lambda_sparse}, seed={args.seed}, use_returns={args.use_returns}")
    print("=" * 80)

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    try:
        data, feature_names, timestamps = load_electricity_data(
            args.dataset, args.data_dir, smooth_window=args.smooth_window,
            use_returns=args.use_returns
        )
        print(f"  Data shape: {data.shape}")
        print(f"  Features: {feature_names}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        # Create synthetic data for testing
        print("\nUsing synthetic data for testing...")
        n_nodes = 10
        T = 5000
        data = np.random.randn(T, n_nodes).astype(np.float32)
        feature_names = [f'feature_{i}' for i in range(n_nodes)]
        timestamps = None

    n_nodes = data.shape[1]

    # Prepare temporal data
    print("\nPreparing temporal data...")
    temporal_data = prepare_temporal_data(data, args.lag)
    print(f"  Temporal data shape: {temporal_data.shape}")

    # Get model configuration
    config = get_model_config(
        n_nodes=n_nodes,
        lag=args.lag,
        lambda_sparse=args.lambda_sparse,
        lambda_dag=args.lambda_dag,
        tau_gumbel=args.tau_gumbel,
    )

    # Run CaRS
    print(f"\nRunning CaRS with model={args.model}, regime_method={args.regime_method}...")

    if args.model == 'ds3m_causal' and args.regime_method == 'native_markov':
        # Use native Markov regime switching (single model with d_dim=n_regimes)
        results = run_cars_native_markov(
            X=temporal_data,
            n_regimes=args.n_regimes,
            n_nodes=n_nodes,
            lag=args.lag,
            device=device,
            model_config=config['model_config'],
            training_params=config['training_params'],
            n_epochs=1000,
        )
    else:
        # Use BEM-based regime detection (default for FANTOM, optional for DS3MCausal)
        results = run_cars(
            X=temporal_data,
            n_regimes=args.n_regimes,
            n_nodes=n_nodes,
            lag=args.lag,
            device=device,
            model_config=config['model_config'],
            training_params=config['training_params'],
            max_iterations=args.max_iterations,
            model_type=args.model,
        )

    # Compute edge statistics
    print("\nComputing edge statistics...")
    # DS3MCausal causal graph now covers all features (including Price)
    edge_feature_names = feature_names
    edge_stats = compute_edge_statistics(
        results['adj_matrices'],
        edge_feature_names,
        args.lag
    )

    # Compute regime statistics
    regime_counts = np.bincount(results['regime_assignments'], minlength=results['n_regimes_final'])
    regime_collapsed = results['n_regimes_final'] < args.n_regimes

    # Compile results
    output = {
        'config': {
            'model': args.model,
            'regime_method': args.regime_method,
            'dataset': args.dataset,
            'n_regimes_requested': args.n_regimes,
            'n_regimes_final': results['n_regimes_final'],
            'lag': args.lag,
            'lambda_sparse': args.lambda_sparse,
            'lambda_dag': args.lambda_dag,
            'tau_gumbel': args.tau_gumbel,
            'seed': args.seed,
            'n_samples': len(results['regime_assignments']),
            'n_features': n_nodes,
            'smooth_window': args.smooth_window,
            'use_returns': args.use_returns,
        },
        'regime_statistics': {
            'regime_counts': regime_counts.tolist(),
            'regime_collapsed': regime_collapsed,
        },
        'edge_statistics': edge_stats,
        'regime_assignments': results['regime_assignments'].tolist(),
        'adjacency_matrices': [adj.tolist() for adj in results['adj_matrices']],
        'feature_names': feature_names,
        'timestamp': datetime.now().isoformat(),
    }

    # Print summary
    print("\n" + "=" * 80)
    print("Results Summary:")
    print(f"  Final regimes: {results['n_regimes_final']} (requested: {args.n_regimes})")
    print(f"  Regime collapsed: {regime_collapsed}")
    print(f"  Regime counts: {regime_counts}")
    for r_stats in edge_stats['regimes']:
        print(f"  Regime {r_stats['regime']}: {r_stats['total_edges']} total edges")
    print("=" * 80)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    # Include model type and regime method in filename
    model_suffix = f"_{args.model}" if args.model != 'fantom' else ""
    method_suffix = f"_{args.regime_method}" if args.model == 'ds3m_causal' else ""
    smooth_suffix = f"_smooth{args.smooth_window}" if args.smooth_window > 0 else ""
    returns_suffix = "_returns" if args.use_returns else ""
    filename = f"cars_{args.dataset.lower()}_d{args.n_regimes}_lag{args.lag}_ls{args.lambda_sparse}_seed{args.seed}{model_suffix}{method_suffix}{smooth_suffix}{returns_suffix}.json"
    output_path = os.path.join(args.output_dir, filename)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Run CaRS experiment")
    parser.add_argument("--model", type=str, default="fantom",
                        choices=['fantom', 'ds3m_causal'],
                        help="Model type: fantom (FANTOM_stationary) or ds3m_causal (DS3MCausal)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['DE', 'FR', 'DE_FR'],
                        help="Dataset to use (DE, FR, or DE_FR)")
    parser.add_argument("--n_regimes", type=int, required=True,
                        help="Number of regimes (2, 3, or 4)")
    parser.add_argument("--lag", type=int, default=1,
                        help="Number of time lags (default: 1)")
    parser.add_argument("--lambda_sparse", type=float, default=5.0,
                        help="L1 sparsity regularization (default: 5.0)")
    parser.add_argument("--lambda_dag", type=float, default=100.0,
                        help="DAG constraint penalty (default: 100.0)")
    parser.add_argument("--tau_gumbel", type=float, default=1.0,
                        help="Gumbel-Softmax temperature (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Maximum BEM iterations (default: 3)")
    parser.add_argument("--regime_method", type=str, default="native_markov",
                        choices=['native_markov', 'bem'],
                        help="Regime learning method: native_markov (DS3M internal) or bem (BEM wrapper)")
    parser.add_argument("--output_dir", type=str,
                        default="presentation/results/cars/",
                        help="Output directory for results")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing unified electricity data (ignored, uses CASTOR)")
    parser.add_argument("--smooth_window", type=int, default=0,
                        help="Rolling window for smoothing (0=none). Recommended: 7 for DE_FR to reduce rapid switching.")
    parser.add_argument("--use_returns", action="store_true",
                        help="Use price_change_pct instead of raw Day_Ahead_Price. "
                             "Makes FANTOM results comparable to CaRS/DS3M which predict returns.")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
