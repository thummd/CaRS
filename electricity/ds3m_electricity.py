"""
DS3M (Deep Switching State Space Model) for electricity price prediction.

Adapts DS3M for the QRT electricity dataset, supporting:
- France, Germany, and Combined country settings
- TARGET prediction (price change)
- Consumption prediction (for comparison with original DS3M)
- UNIVARIATE mode: Uses only TARGET history (original DS3M approach)
- MULTIVARIATE mode: Uses all 32 features as input, predicts TARGET only

Usage:
    # Univariate mode (original DS3M approach)
    python ds3m_electricity.py --country FR --target TARGET --train
    python ds3m_electricity.py --country DE --target TARGET --train

    # Multivariate mode (improved approach with all features)
    python ds3m_electricity.py --country FR --mode multivariate --train
    python ds3m_electricity.py --country FR --mode multivariate --d_dim 1 --train
    python ds3m_electricity.py --country FR --mode multivariate --d_dim 2 --train

    # Quick debug run
    python ds3m_electricity.py --country FR --mode multivariate --train --debug
"""

import sys
import os
from pathlib import Path

from paths import DS3M_DIR
# Add DS3M code to path
DS3M_PATH = str(DS3M_DIR)
sys.path.insert(0, DS3M_PATH)
sys.path.insert(0, os.path.join(DS3M_PATH, "src"))

import argparse
import json
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import yaml
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Import DS3M components
from DSSSMCode import DSSSM, train, test, EarlyStopping

# Import local adapter
from ds3m_adapter import (
    load_qrt_data, qrt_to_ds3m_format, prepare_train_test_split,
    prepare_official_test_data, normalize_invert, evaluation,
    get_country_sample_counts,
    # Multivariate functions
    get_feature_columns, qrt_to_ds3m_multivariate,
    prepare_multivariate_train_test_split, normalize_features_invert
)


# Default configurations for QRT data (smaller than original due to limited data)
QRT_CONFIG = {
    'FR': {
        'timestep': 14,
        'predict_dim': 1,
        'h_dim': 20,
        'z_dim': 5,
        'd_dim': 2,
        'n_layers': 1,
        'n_epochs': 200,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'clip': 10,
        'bidirection': False,
    },
    'DE': {
        'timestep': 14,
        'predict_dim': 1,
        'h_dim': 20,
        'z_dim': 5,
        'd_dim': 2,
        'n_layers': 1,
        'n_epochs': 200,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'clip': 10,
        'bidirection': False,
    },
    'ALL': {
        'timestep': 14,
        'predict_dim': 1,
        'h_dim': 30,
        'z_dim': 8,
        'd_dim': 2,
        'n_layers': 1,
        'n_epochs': 200,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'clip': 10,
        'bidirection': False,
    }
}

# Multivariate configurations: all features as input, TARGET as output
# These configs use more capacity to handle higher-dimensional input
# Note: d_dim can be overridden via --d_dim command line argument
QRT_MULTIVARIATE_CONFIG = {
    'FR': {
        'timestep': 14,
        'x_dim': 32,           # Number of input features (all except ID, DAY_ID, COUNTRY, TARGET)
        'y_dim': 1,            # Predict TARGET only
        'h_dim': 32,           # Larger hidden dim for more features
        'z_dim': 8,            # Larger latent dim
        'd_dim': 2,            # Default to 2 regimes (can be set to 1 via --d_dim)
        'n_layers': 1,
        'n_epochs': 200,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'clip': 10,
        'bidirection': False,
        'dropout': 0.1,        # Add regularization for smaller dataset
    },
    'DE': {
        'timestep': 14,
        'x_dim': 32,
        'y_dim': 1,
        'h_dim': 32,
        'z_dim': 8,
        'd_dim': 2,
        'n_layers': 1,
        'n_epochs': 200,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'clip': 10,
        'bidirection': False,
        'dropout': 0.1,
    },
    'ALL': {
        'timestep': 14,
        'x_dim': 32,
        'y_dim': 1,
        'h_dim': 48,           # Even larger for combined dataset
        'z_dim': 12,
        'd_dim': 2,
        'n_layers': 1,
        'n_epochs': 200,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'clip': 10,
        'bidirection': False,
        'dropout': 0.1,
    }
}


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


def forecast(
    model: DSSSM,
    testX: torch.Tensor,
    testY: torch.Tensor,
    moments: np.ndarray,
    d_dim: int,
    forecaststep: int = 1,
    MC_S: int = 200
):
    """
    Generate forecasts and regime assignments.

    Args:
        model: Trained DSSSM model
        testX: Test input (timestep, batch, features)
        testY: Test target (timestep, batch, features)
        moments: Normalization moments
        d_dim: Number of discrete states
        forecaststep: Forecast horizon
        MC_S: Monte Carlo samples

    Returns:
        Dictionary with predictions, regimes, metrics
    """
    model.eval()

    # Get forecasts via Monte Carlo
    forecast_MC, forecast_d_MC, forecast_z_MC = model._forecastingMultiStep(
        testX, testY, forecaststep, MC_S
    )

    # Process forecasts
    if forecaststep == 1:
        all_testForecast = normalize_invert(
            forecast_MC.squeeze(1).transpose(1, 0, 2), moments
        )
    else:
        all_testForecast = normalize_invert(
            forecast_MC.squeeze(2).transpose(1, 0, 2), moments
        )

    # Original values
    testY_inversed = normalize_invert(
        testY.cpu().numpy().transpose(1, 0, 2), moments
    )

    # Regime assignments (mode of MC samples)
    forecast_d_MC_argmax = []
    for i in range(d_dim):
        forecast_d_MC_argmax.append(
            np.sum(forecast_d_MC[:, -1, :, :] == i, axis=0)
        )
    forecast_d_MC_argmax = np.argmax(
        np.array(forecast_d_MC_argmax), axis=0
    ).reshape(-1)

    # Compute statistics
    testForecast_mean = np.mean(all_testForecast, axis=1)
    testForecast_uq = np.quantile(all_testForecast, 0.95, axis=1)
    testForecast_lq = np.quantile(all_testForecast, 0.05, axis=1)

    # Get original values for comparison
    testOriginal = testY_inversed[:, -1, :].reshape(-1)

    # Evaluation metrics
    res = evaluation(testForecast_mean.T, testOriginal.reshape(-1, 1).T)

    # Spearman correlation (QRT challenge metric)
    spearman_corr, spearman_pval = spearmanr(testOriginal, testForecast_mean.reshape(-1))
    res['spearman'] = spearman_corr
    res['spearman_pval'] = spearman_pval

    return {
        'predictions': testForecast_mean,
        'predictions_upper': testForecast_uq,
        'predictions_lower': testForecast_lq,
        'original': testOriginal,
        'regime_assignments': forecast_d_MC_argmax,
        'metrics': res,
        'size': len(testOriginal)
    }


def forecast_multivariate(
    model: DSSSM,
    testX: torch.Tensor,
    testY: torch.Tensor,
    y_moments: np.ndarray,
    d_dim: int,
    MC_S: int = 200
):
    """
    Generate forecasts for multivariate input model.

    For multivariate models (x_dim != y_dim), we cannot use the standard
    autoregressive forecasting. Instead, we use the model's forward pass
    to get predictions at each timestep using actual features.

    Args:
        model: Trained DSSSM model
        testX: Test input (timestep, batch, n_features)
        testY: Test target (timestep, batch, 1)
        y_moments: Normalization moments for target
        d_dim: Number of discrete states
        MC_S: Monte Carlo samples for uncertainty estimation

    Returns:
        Dictionary with predictions, regimes, metrics
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Forward pass to get predictions and regime assignments
        # DS3M forward returns:
        # kld_g, kld_c, nll,
        # (all_z_posterior_mean, all_z_posterior_std),
        # (all_y_emission_mean, all_y_emission_std),
        # all_d_t_sampled_plot, all_z_t_sampled,
        # all_d_posterior, all_d_t_sampled
        outputs = model(testX, testY)
        kld_g = outputs[0]
        kld_c = outputs[1]
        nll = outputs[2]
        z_posterior = outputs[3]  # tuple: (mean_list, std_list)
        y_emission = outputs[4]   # tuple: (mean_list, std_list)
        all_d_sampled_plot = outputs[5]  # list of regime samples for plotting
        all_z_sampled = outputs[6]
        all_d_posterior = outputs[7]  # list of (batch, d_dim)
        all_d_sampled = outputs[8]

        # Get predictions (mean of emission distribution)
        all_y_mean, all_y_std = y_emission
        # all_y_mean is a list of tensors, each (batch, y_dim)
        predictions_norm = torch.stack(all_y_mean, dim=0)  # (timestep, batch, y_dim)

        # Get regime assignments from d_posterior
        # all_d_posterior is a list of tensors, each (batch, d_dim)
        d_posteriors = torch.stack(all_d_posterior, dim=0)  # (timestep, batch, d_dim)
        regime_assignments = d_posteriors[-1].argmax(dim=1).cpu().numpy()  # Last timestep

        # Convert predictions to numpy - use last timestep prediction
        predictions_last = predictions_norm[-1].cpu().numpy()  # (batch, y_dim)

        # Get actual values
        testY_norm = testY[-1].cpu().numpy()  # (batch, 1) - last timestep

        # Uncertainty estimation via multiple forward passes
        all_predictions = [predictions_last]
        for _ in range(MC_S - 1):
            outputs_mc = model(testX, testY)
            y_emission_mc = outputs_mc[4]
            all_y_mean_mc, _ = y_emission_mc
            pred_mc = torch.stack(all_y_mean_mc, dim=0)[-1].cpu().numpy()
            all_predictions.append(pred_mc)

        all_predictions = np.array(all_predictions)  # (MC_S, batch, y_dim)

        # Compute statistics
        predictions_mean = np.mean(all_predictions, axis=0)  # (batch, y_dim)
        predictions_upper = np.quantile(all_predictions, 0.95, axis=0)
        predictions_lower = np.quantile(all_predictions, 0.05, axis=0)

        # Invert normalization
        def invert_norm(data, moments):
            return data * moments[1] + moments[0]

        predictions_mean_inv = invert_norm(predictions_mean, y_moments)
        predictions_upper_inv = invert_norm(predictions_upper, y_moments)
        predictions_lower_inv = invert_norm(predictions_lower, y_moments)
        testY_inv = invert_norm(testY_norm, y_moments)

        # Flatten for metrics
        pred_flat = predictions_mean_inv.reshape(-1)
        actual_flat = testY_inv.reshape(-1)

        # Evaluation metrics
        res = evaluation(pred_flat.reshape(1, -1), actual_flat.reshape(1, -1))

        # Spearman correlation
        spearman_corr, spearman_pval = spearmanr(actual_flat, pred_flat)
        res['spearman'] = spearman_corr
        res['spearman_pval'] = spearman_pval

    return {
        'predictions': predictions_mean_inv,
        'predictions_upper': predictions_upper_inv,
        'predictions_lower': predictions_lower_inv,
        'original': testY_inv.reshape(-1),
        'regime_assignments': regime_assignments,
        'metrics': res,
        'size': len(actual_flat)
    }


def plot_predictions(
    results: dict,
    output_path: str,
    title: str = "DS3M Predictions"
):
    """Plot predictions with confidence intervals and regimes."""
    size = results['size']
    d_dim = len(np.unique(results['regime_assignments']))
    cmap = plt.get_cmap('RdBu', d_dim)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        sharex=True, gridspec_kw={'height_ratios': [3, 0.5]}
    )

    # Predictions plot
    ax1.plot(results['original'], label='Original', color='blue', alpha=0.8)
    ax1.plot(results['predictions'].reshape(-1), label='Forecast Mean', color='red')
    ax1.plot(results['predictions_upper'].reshape(-1), color='grey', alpha=0.5)
    ax1.plot(results['predictions_lower'].reshape(-1), color='grey', alpha=0.5)
    ax1.fill_between(
        np.arange(size),
        results['predictions_upper'].reshape(-1),
        results['predictions_lower'].reshape(-1),
        color='grey', alpha=0.2, label='90% CI'
    )
    ax1.set_ylabel('TARGET')
    ax1.set_title(f"{title}\nSpearman: {results['metrics']['spearman']:.4f}, RMSE: {results['metrics']['rmse']:.4f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Regime heatmap
    sns.heatmap(
        results['regime_assignments'].reshape(1, -1),
        linewidth=0, cbar=False, alpha=1, cmap=cmap,
        vmin=0, vmax=d_dim - 1, ax=ax2
    )
    ax2.set_ylabel('Regime')
    ax2.set_xlabel('Time')
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction plot to: {output_path}")


def plot_regime_timeline(
    results: dict,
    output_path: str,
    title: str = "Regime Assignments"
):
    """Plot regime assignments over time."""
    regime_assignments = results['regime_assignments']
    n_regimes = len(np.unique(regime_assignments))
    colors = plt.cm.Set2(np.linspace(0, 1, n_regimes))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [2, 1]})

    # Target with regime colors
    x = np.arange(len(results['original']))
    current_regime = regime_assignments[0]
    start_idx = 0
    legend_added = set()

    for i in range(1, len(regime_assignments) + 1):
        if i == len(regime_assignments) or regime_assignments[i] != current_regime:
            end_idx = i
            label = f'Regime {current_regime}' if current_regime not in legend_added else None
            ax1.axvspan(start_idx, end_idx, alpha=0.3, color=colors[current_regime], label=label)
            legend_added.add(current_regime)
            if i < len(regime_assignments):
                current_regime = regime_assignments[i]
                start_idx = i

    ax1.plot(x, results['original'], 'k-', linewidth=1, alpha=0.8)
    ax1.set_ylabel('TARGET')
    ax1.set_title(title)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Regime step plot
    ax2.step(x, regime_assignments, where='post', color='navy', linewidth=1.5)
    ax2.set_ylabel('Regime')
    ax2.set_xlabel('Time')
    ax2.set_ylim(-0.5, n_regimes - 0.5)
    ax2.set_yticks(range(n_regimes))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved regime timeline to: {output_path}")


def run_experiment(
    country: str,
    target_col: str,
    config: dict,
    output_dir: Path,
    train_model: bool = True,
    seed: int = 42,
    verbose: bool = True
):
    """
    Run DS3M experiment.

    Args:
        country: 'FR', 'DE', or 'ALL'
        target_col: Target column name
        config: Hyperparameter configuration
        output_dir: Output directory
        train_model: Whether to train (True) or load checkpoint (False)
        seed: Random seed
        verbose: Print progress

    Returns:
        Results dictionary
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data for {country} with target {target_col}...")
    df = load_qrt_data()

    country_filter = None if country == 'ALL' else country
    data = prepare_train_test_split(
        df,
        country=country_filter,
        target_col=target_col,
        timestep=config['timestep'],
        test_ratio=0.2
    )

    trainX = data['trainX'].to(device)
    trainY = data['trainY'].to(device)
    testX = data['testX'].to(device)
    testY = data['testY'].to(device)
    moments = data['moments']

    print(f"Train shape: {trainX.shape}")
    print(f"Test shape: {testX.shape}")

    # Create model
    model = DSSSM(
        x_dim=config['predict_dim'],
        y_dim=config['predict_dim'],
        h_dim=config['h_dim'],
        z_dim=config['z_dim'],
        d_dim=config['d_dim'],
        n_layers=config['n_layers'],
        device=device,
        bidirection=config['bidirection']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params}")

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if train_model:
        print("\n--- Training ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        early_stopping = EarlyStopping(patience=20, verbose=True)

        best_validation = float('inf')
        loss_history = {'train': [], 'test': []}
        start_time = time.time()

        for epoch in range(1, config['n_epochs'] + 1):
            # Training
            _, _, loss_train, _, _ = train(
                model, optimizer, trainX, trainY, epoch,
                config['batch_size'], config['n_epochs']
            )

            # Testing
            _, _, loss_test, _, _ = test(model, testX, testY, epoch, "test")

            loss_history['train'].append(loss_train)
            loss_history['test'].append(loss_test)

            # Save best model
            if loss_test < best_validation:
                best_validation = loss_test
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train,
                    'config': config,
                    'moments': moments,
                }, checkpoint_dir / 'best.tar')

            scheduler.step(loss_test)
            if verbose and epoch % 10 == 0:
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

            # Early stopping
            early_stopping(loss_test, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f}s")

        # Plot loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history['train'], label='Train Loss')
        plt.plot(loss_history['test'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig(figures_dir / 'loss_history.png', dpi=150)
        plt.close()

    # Load best model
    checkpoint = torch.load(checkpoint_dir / 'best.tar', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")

    # Generate forecasts
    print("\n--- Forecasting ---")
    results = forecast(
        model, testX, testY, moments, config['d_dim'],
        forecaststep=1, MC_S=200
    )

    print(f"\nResults:")
    print(f"  RMSE: {results['metrics']['rmse']:.4f}")
    print(f"  MAPE: {results['metrics']['mape']:.4f}")
    print(f"  Spearman: {results['metrics']['spearman']:.4f}")

    # Generate plots
    plot_predictions(
        results,
        str(figures_dir / 'predictions.png'),
        title=f"DS3M Predictions ({country}, {target_col})"
    )

    plot_regime_timeline(
        results,
        str(figures_dir / 'regime_timeline.png'),
        title=f"Regime Assignments ({country}, {target_col})"
    )

    # Save results
    results_summary = {
        'country': country,
        'target': target_col,
        'config': config,
        'metrics': {k: float(v) for k, v in results['metrics'].items()},
        'n_regimes_detected': len(np.unique(results['regime_assignments'])),
        'regime_distribution': {
            int(k): int(v) for k, v in
            zip(*np.unique(results['regime_assignments'], return_counts=True))
        },
        'training_samples': int(trainX.shape[1]),
        'test_samples': int(testX.shape[1]),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        'original': results['original'].reshape(-1),
        'prediction': results['predictions'].reshape(-1),
        'regime': results['regime_assignments']
    })
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)

    # Save regime assignments
    np.save(output_dir / 'regime_assignments.npy', results['regime_assignments'])

    print(f"\nResults saved to: {output_dir}")

    return results


def run_multivariate_experiment(
    country: str,
    config: dict,
    output_dir: Path,
    train_model: bool = True,
    seed: int = 42,
    verbose: bool = True
):
    """
    Run DS3M experiment in MULTIVARIATE mode.

    Uses all features as input (x_dim=32) and predicts TARGET only (y_dim=1).
    This addresses the key limitation of univariate DS3M which ignores
    cross-variable relationships.

    Args:
        country: 'FR', 'DE', or 'ALL'
        config: Hyperparameter configuration (should have x_dim, y_dim)
        output_dir: Output directory
        train_model: Whether to train (True) or load checkpoint (False)
        seed: Random seed
        verbose: Print progress

    Returns:
        Results dictionary
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print(f"MULTIVARIATE DS3M EXPERIMENT")
    print(f"{'='*60}")

    # Load data using multivariate adapter
    print(f"\nLoading MULTIVARIATE data for {country}...")
    df = load_qrt_data()

    country_filter = None if country == 'ALL' else country
    data = prepare_multivariate_train_test_split(
        df,
        country=country_filter,
        feature_cols=None,  # Use all features
        target_col='TARGET',
        timestep=config['timestep'],
        test_ratio=0.2
    )

    trainX = data['trainX'].to(device)
    trainY = data['trainY'].to(device)
    testX = data['testX'].to(device)
    testY = data['testY'].to(device)
    x_moments = data['X_moments']
    y_moments = data['Y_moments']
    feature_cols = data['feature_cols']
    n_features = data['n_features']

    print(f"Train X shape: {trainX.shape} (timestep, batch, {n_features} features)")
    print(f"Train Y shape: {trainY.shape} (timestep, batch, TARGET)")
    print(f"Test X shape: {testX.shape}")
    print(f"Test Y shape: {testY.shape}")
    print(f"Features: {feature_cols[:5]}... ({n_features} total)")

    # Update config with actual x_dim if not set
    if 'x_dim' not in config or config['x_dim'] != n_features:
        print(f"Updating x_dim from {config.get('x_dim', 'unset')} to {n_features}")
        config['x_dim'] = n_features

    # Create model with different x_dim and y_dim
    model = DSSSM(
        x_dim=config['x_dim'],      # 32 features as input
        y_dim=config['y_dim'],      # 1 (TARGET) as output
        h_dim=config['h_dim'],
        z_dim=config['z_dim'],
        d_dim=config['d_dim'],
        n_layers=config['n_layers'],
        device=device,
        bidirection=config.get('bidirection', False)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params}")

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if train_model:
        print("\n--- Training ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        early_stopping = EarlyStopping(patience=20, verbose=True)

        best_validation = float('inf')
        loss_history = {'train': [], 'test': []}
        start_time = time.time()

        for epoch in range(1, config['n_epochs'] + 1):
            # Training
            _, _, loss_train, _, _ = train(
                model, optimizer, trainX, trainY, epoch,
                config['batch_size'], config['n_epochs']
            )

            # Testing
            _, _, loss_test, _, _ = test(model, testX, testY, epoch, "test")

            loss_history['train'].append(loss_train)
            loss_history['test'].append(loss_test)

            # Save best model
            if loss_test < best_validation:
                best_validation = loss_test
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train,
                    'config': config,
                    'x_moments': x_moments,
                    'y_moments': y_moments,
                    'feature_cols': feature_cols,
                }, checkpoint_dir / 'best.tar')

            scheduler.step(loss_test)
            if verbose and epoch % 10 == 0:
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

            # Early stopping
            early_stopping(loss_test, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f}s")

        # Plot loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history['train'], label='Train Loss')
        plt.plot(loss_history['test'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History (Multivariate)')
        plt.legend()
        plt.savefig(figures_dir / 'loss_history.png', dpi=150)
        plt.close()

    # Load best model
    checkpoint = torch.load(checkpoint_dir / 'best.tar', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")

    # Generate forecasts using multivariate forecast function
    print("\n--- Forecasting (Multivariate) ---")
    results = forecast_multivariate(
        model, testX, testY, y_moments, config['d_dim'],
        MC_S=200
    )

    print(f"\nResults:")
    print(f"  RMSE: {results['metrics']['rmse']:.4f}")
    print(f"  MAPE: {results['metrics']['mape']:.4f}")
    print(f"  Spearman: {results['metrics']['spearman']:.4f}")

    # Generate plots
    plot_predictions(
        results,
        str(figures_dir / 'predictions.png'),
        title=f"Multivariate DS3M Predictions ({country})"
    )

    plot_regime_timeline(
        results,
        str(figures_dir / 'regime_timeline.png'),
        title=f"Regime Assignments ({country}, Multivariate)"
    )

    # Save results
    results_summary = {
        'country': country,
        'mode': 'multivariate',
        'target': 'TARGET',
        'n_features': n_features,
        'feature_cols': feature_cols,
        'config': config,
        'metrics': {k: float(v) for k, v in results['metrics'].items()},
        'n_regimes_detected': len(np.unique(results['regime_assignments'])),
        'regime_distribution': {
            int(k): int(v) for k, v in
            zip(*np.unique(results['regime_assignments'], return_counts=True))
        },
        'training_samples': int(trainX.shape[1]),
        'test_samples': int(testX.shape[1]),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        'original': results['original'].reshape(-1),
        'prediction': results['predictions'].reshape(-1),
        'regime': results['regime_assignments']
    })
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)

    # Save regime assignments
    np.save(output_dir / 'regime_assignments.npy', results['regime_assignments'])

    print(f"\nResults saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="DS3M for electricity price prediction")
    parser.add_argument('--country', type=str, required=True,
                        choices=['FR', 'DE', 'ALL'],
                        help='Country to model')
    parser.add_argument('--target', type=str, default='TARGET',
                        help='Target column (TARGET, FR_CONSUMPTION, DE_CONSUMPTION)')
    parser.add_argument('--mode', type=str, default='univariate',
                        choices=['univariate', 'multivariate'],
                        help='Mode: univariate (TARGET only) or multivariate (all features)')
    parser.add_argument('--d_dim', type=int, default=None,
                        help='Number of discrete regimes (overrides config)')
    parser.add_argument('--train', action='store_true',
                        help='Train model (otherwise load checkpoint)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (fewer epochs)')

    args = parser.parse_args()

    # Load config based on mode
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # Extract relevant config section
            if 'model_config' in config:
                base_config = config['model_config'].copy()
                base_config.update(config.get('training_params', {}))
            else:
                base_config = config
    else:
        # Select config based on mode
        if args.mode == 'multivariate':
            base_config = QRT_MULTIVARIATE_CONFIG.get(args.country, QRT_MULTIVARIATE_CONFIG['FR']).copy()
        else:
            base_config = QRT_CONFIG.get(args.country, QRT_CONFIG['FR']).copy()

    # Override with command line args
    if args.epochs:
        base_config['n_epochs'] = args.epochs
    if args.d_dim is not None:
        base_config['d_dim'] = args.d_dim
    if args.debug:
        base_config['n_epochs'] = 10

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_suffix = args.target.replace('_', '')
        mode_suffix = 'mv' if args.mode == 'multivariate' else 'uv'
        d_dim = base_config.get('d_dim', 2)
        output_dir = Path(__file__).parent / "outputs" / "ds3m" / f"{args.country}_{target_suffix}_{mode_suffix}_d{d_dim}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(base_config, f)

    # Run experiment based on mode
    if args.mode == 'multivariate':
        results = run_multivariate_experiment(
            country=args.country,
            config=base_config,
            output_dir=output_dir,
            train_model=args.train,
            seed=args.seed,
            verbose=True
        )
    else:
        results = run_experiment(
            country=args.country,
            target_col=args.target,
            config=base_config,
            output_dir=output_dir,
            train_model=args.train,
            seed=args.seed,
            verbose=True
        )

    print("\n" + "=" * 60)
    print("DS3M Experiment Complete")
    print("=" * 60)
    print(f"Country: {args.country}")
    print(f"Mode: {args.mode}")
    print(f"Target: {args.target}")
    print(f"d_dim (regimes): {base_config.get('d_dim', 2)}")
    print(f"Spearman Correlation: {results['metrics']['spearman']:.4f}")
    print(f"RMSE: {results['metrics']['rmse']:.4f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
