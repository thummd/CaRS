#!/usr/bin/env python3
"""
DS3M-FANTOM Hybrid Experiment Runner

Runs experiments combining DS3M regime switching with FANTOM causal discovery.

Usage:
    # Single experiment
    python run_experiments.py --country FR --sharing_mode independent --train

    # Debug mode (fewer epochs)
    python run_experiments.py --country FR --train --debug

    # Load and evaluate
    python run_experiments.py --country FR --checkpoint /path/to/checkpoint
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
from paths import DS3M_DIR
DS3M_PATH = str(DS3M_DIR)
sys.path.insert(0, DS3M_PATH)
sys.path.insert(0, os.path.join(DS3M_PATH, "src"))

import argparse
import json
import yaml
import random
import time
from datetime import datetime
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Import our modules
from ds3m_fantom.models.ds3m_causal import DS3MCausal
from ds3m_fantom.training.train_e2e import train_end_to_end
from ds3m_fantom.training.train_2stage import train_two_stage

# Import data adapters
from ds3m_adapter import (
    load_qrt_data, prepare_multivariate_train_test_split, normalize_invert
)

# Import unified data loader for self-sourced data
try:
    from unified_data_loader import prepare_unified_ds3m_data
    UNIFIED_DATA_AVAILABLE = True
except ImportError:
    UNIFIED_DATA_AVAILABLE = False
    print("WARNING: unified_data_loader not available for self-sourced data")


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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def plot_predictions(
    predictions: np.ndarray,
    predictions_std: np.ndarray,
    actuals: np.ndarray,
    regimes: np.ndarray,
    output_path: str,
    title: str = "DS3M-Causal Predictions"
):
    """Plot predictions with confidence intervals and regime assignments."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        sharex=True, gridspec_kw={'height_ratios': [3, 0.5]}
    )

    n_samples = len(actuals)
    x = np.arange(n_samples)

    # Predictions
    ax1.plot(x, actuals, 'b-', label='Actual', alpha=0.8)
    ax1.plot(x, predictions, 'r-', label='Predicted')
    ax1.fill_between(
        x,
        predictions - 1.96 * predictions_std,
        predictions + 1.96 * predictions_std,
        color='red', alpha=0.2, label='95% CI'
    )
    ax1.set_ylabel('TARGET')
    ax1.legend()
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Regimes
    n_regimes = len(np.unique(regimes))
    cmap = plt.get_cmap('Set2', n_regimes)
    sns.heatmap(
        regimes.reshape(1, -1),
        linewidth=0, cbar=False, alpha=1, cmap=cmap,
        vmin=0, vmax=n_regimes - 1, ax=ax2
    )
    ax2.set_ylabel('Regime')
    ax2.set_xlabel('Time')
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_causal_graph(
    adjacency: np.ndarray,
    feature_names: list,
    output_path: str,
    regime_id: int = 0,
    threshold: float = 0.1
):
    """Plot causal graph as heatmap."""
    # Use instantaneous adjacency (lag=0)
    if adjacency.ndim == 3:
        adj = adjacency[0]  # Instantaneous
    else:
        adj = adjacency

    # Threshold small values
    adj_plot = adj.copy()
    adj_plot[np.abs(adj_plot) < threshold] = 0

    fig, ax = plt.subplots(figsize=(12, 10))

    # Limit to first 15 features for readability
    n_show = min(15, len(feature_names))
    adj_show = adj_plot[:n_show, :n_show]
    names_show = feature_names[:n_show]

    sns.heatmap(
        adj_show,
        xticklabels=names_show,
        yticklabels=names_show,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        ax=ax,
        cbar_kws={'label': 'Edge Weight'}
    )
    ax.set_title(f'Causal Graph - Regime {regime_id}')
    ax.set_xlabel('Effect')
    ax.set_ylabel('Cause')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_experiment(
    country: str,
    sharing_mode: str,
    config: dict,
    output_dir: Path,
    seed: int = 42,
    train: bool = True,
    checkpoint_path: str = None,
    training_mode: str = 'end_to_end',
    data_source: str = 'qrt',
    verbose: bool = True
):
    """
    Run a single DS3M-Causal experiment.

    Args:
        country: 'FR', 'DE', 'DE_FR', or 'ALL'
        sharing_mode: 'independent' or 'shared_backbone'
        config: Training configuration
        output_dir: Output directory
        seed: Random seed
        train: Whether to train
        checkpoint_path: Path to load checkpoint
        training_mode: 'end_to_end' or 'two_stage'
        data_source: 'qrt' (QRT challenge data) or 'unified' (self-sourced ENTSO-E data)
        verbose: Print progress
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"DS3M-Causal Experiment")
    print(f"{'='*60}")
    print(f"Country: {country}")
    print(f"Sharing mode: {sharing_mode}")
    print(f"Training mode: {training_mode}")
    print(f"Data source: {data_source}")
    print(f"Device: {device}")

    # Load data based on source
    print("\nLoading data...")

    if data_source == 'unified':
        # Use self-sourced unified data (ENTSO-E + weather + calendar)
        if not UNIFIED_DATA_AVAILABLE:
            raise RuntimeError("unified_data_loader not available. Ensure it's in the Python path.")

        # Prepare unified data
        data = prepare_unified_ds3m_data(
            country=country,
            timestep=config['data']['timestep'],
            test_ratio=config['data']['test_ratio'],
            val_ratio=config['data'].get('val_ratio', 0.1),
            feature_groups=config['data'].get('feature_groups', None)
        )

        trainX = data['trainX'].to(device)
        trainY = data['trainY'].to(device)
        testX = data['testX'].to(device)
        testY = data['testY'].to(device)
        y_moments = data['Y_moments']
        feature_cols = data['feature_cols']

        # For unified data, use validation set for early stopping
        valX = data['valX'].to(device) if 'valX' in data else None
        valY = data['valY'].to(device) if 'valY' in data else None

        print(f"Data source: Unified (ENTSO-E + weather + calendar)")
        print(f"Target: {data['target_col']}")
    else:
        # Use QRT challenge data (default)
        df = load_qrt_data()
        country_filter = None if country == 'ALL' else country

        data = prepare_multivariate_train_test_split(
            df,
            country=country_filter,
            timestep=config['data']['timestep'],
            test_ratio=config['data']['test_ratio']
        )

        trainX = data['trainX'].to(device)
        trainY = data['trainY'].to(device)
        testX = data['testX'].to(device)
        testY = data['testY'].to(device)
        y_moments = data['Y_moments']
        feature_cols = data['feature_cols']
        valX = None
        valY = None

        print(f"Data source: QRT Challenge")

    print(f"Train X: {trainX.shape}")
    print(f"Train Y: {trainY.shape}")
    print(f"Test X: {testX.shape}")
    print(f"Test Y: {testY.shape}")

    # Create model
    model_config = config['model']
    x_dim = len(feature_cols)

    model = DS3MCausal(
        x_dim=x_dim,
        y_dim=1,
        h_dim=model_config['h_dim'],
        z_dim=model_config['z_dim'],
        d_dim=model_config['d_dim'],
        device=device,
        n_layers=model_config['n_layers'],
        num_nodes=x_dim,
        lag=model_config['lag'],
        sharing_mode=sharing_mode,
        tau_gumbel=model_config['tau_gumbel'],
        init_logits=model_config['init_logits'],
        lambda_dag=config['training']['lambda_dag'],
        lambda_sparse=config['training']['lambda_sparse'],
        lambda_kl=config['training']['lambda_kl'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Train or load
    if train:
        print(f"\nStarting {training_mode} training...")
        if training_mode == 'two_stage':
            results = train_two_stage(
                model=model,
                trainX=trainX,
                trainY=trainY,
                testX=testX,
                testY=testY,
                config=config['training'],
                output_dir=output_dir,
                verbose=verbose
            )
        else:  # end_to_end
            results = train_end_to_end(
                model=model,
                trainX=trainX,
                trainY=trainY,
                testX=testX,
                testY=testY,
                config=config['training'],
                output_dir=output_dir,
                verbose=verbose
            )
    else:
        if checkpoint_path:
            print(f"\nLoading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Must provide checkpoint_path when train=False")

    # Evaluate
    print("\nEvaluating...")
    model.eval()
    with torch.no_grad():
        pred_result = model.predict(testX, n_samples=100)

    # Get predictions for last timestep
    predictions = pred_result['predictions'][-1].cpu().numpy().flatten()
    predictions_std = pred_result['predictions_std'][-1].cpu().numpy().flatten()
    actuals = testY[-1].cpu().numpy().flatten()
    regimes = pred_result['regimes'][-1].cpu().numpy()

    # Denormalize predictions
    predictions_denorm = predictions * y_moments[1] + y_moments[0]
    actuals_denorm = actuals * y_moments[1] + y_moments[0]
    predictions_std_denorm = predictions_std * y_moments[1]

    # Compute metrics
    spearman_corr, spearman_pval = spearmanr(actuals_denorm, predictions_denorm)
    rmse = np.sqrt(np.mean((predictions_denorm - actuals_denorm) ** 2))

    print(f"\nResults:")
    print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_pval:.4e})")
    print(f"  RMSE: {rmse:.4f}")

    # Save results
    results_summary = {
        'country': country,
        'sharing_mode': sharing_mode,
        'spearman': float(spearman_corr),
        'spearman_pval': float(spearman_pval),
        'rmse': float(rmse),
        'n_features': x_dim,
        'model_params': total_params,
        'n_regimes': model_config['d_dim'],
        'regime_distribution': {
            int(k): int(v) for k, v in
            zip(*np.unique(regimes, return_counts=True))
        },
        'config': config,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    # Plot predictions
    plot_predictions(
        predictions_denorm, predictions_std_denorm, actuals_denorm, regimes,
        str(figures_dir / 'predictions.png'),
        f"DS3M-Causal Predictions ({country}, {sharing_mode})\nSpearman: {spearman_corr:.4f}"
    )

    # Plot causal graphs
    graphs = model.get_causal_graphs()
    for d, g in enumerate(graphs):
        plot_causal_graph(
            g, feature_cols,
            str(figures_dir / f'causal_graph_regime{d}.png'),
            regime_id=d
        )
        # Save raw adjacency
        np.save(figures_dir / f'adjacency_regime{d}.npy', g)

    # Save predictions
    import pandas as pd
    pred_df = pd.DataFrame({
        'actual': actuals_denorm,
        'prediction': predictions_denorm,
        'prediction_std': predictions_std_denorm,
        'regime': regimes
    })
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)

    print(f"\nResults saved to: {output_dir}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="DS3M-FANTOM Hybrid Experiments")
    parser.add_argument('--country', type=str, default='FR',
                        choices=['FR', 'DE', 'DE_FR', 'ALL'],
                        help='Country to model (DE_FR for spread prediction)')
    parser.add_argument('--sharing_mode', type=str, default='independent',
                        choices=['independent', 'shared_backbone'],
                        help='DAG sharing mode')
    parser.add_argument('--training_mode', type=str, default='end_to_end',
                        choices=['end_to_end', 'two_stage'],
                        help='Training mode: end_to_end or two_stage')
    parser.add_argument('--data_source', type=str, default='qrt',
                        choices=['qrt', 'unified'],
                        help='Data source: qrt (QRT challenge) or unified (self-sourced ENTSO-E)')
    parser.add_argument('--d_dim', type=int, default=None,
                        help='Number of regimes (override config)')
    parser.add_argument('--train', action='store_true',
                        help='Train model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for loading')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (fewer epochs)')
    parser.add_argument('--feature_groups', type=str, nargs='+', default=None,
                        help='Feature groups for unified data (e.g., price load weather calendar)')

    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        default_config = Path(__file__).parent / "config" / "ds3m_causal.yaml"
        config = load_config(str(default_config))

    # Override with command line args
    if args.d_dim:
        config['model']['d_dim'] = args.d_dim

    if args.feature_groups:
        if 'data' not in config:
            config['data'] = {}
        config['data']['feature_groups'] = args.feature_groups

    if args.debug:
        config['training']['max_auglag_steps'] = 3
        config['training']['max_inner_epochs'] = 5
        config['training']['epochs_stage1'] = 10
        config['training']['epochs_stage2'] = 10

    # Validate country/data_source combination
    if args.country == 'DE_FR' and args.data_source != 'unified':
        print("WARNING: DE_FR country requires unified data source. Switching to unified.")
        args.data_source = 'unified'

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_short = 'ind' if args.sharing_mode == 'independent' else 'shared'
        train_short = 'e2e' if args.training_mode == 'end_to_end' else '2stg'
        data_short = 'uni' if args.data_source == 'unified' else 'qrt'
        output_dir = Path(__file__).parent.parent / "outputs" / "ds3m_causal" / \
                     f"{args.country}_{data_short}_{mode_short}_{train_short}_d{config['model']['d_dim']}_{timestamp}"

    # Run experiment
    results = run_experiment(
        country=args.country,
        sharing_mode=args.sharing_mode,
        config=config,
        output_dir=output_dir,
        seed=args.seed,
        train=args.train,
        checkpoint_path=args.checkpoint,
        training_mode=args.training_mode,
        data_source=args.data_source,
        verbose=True
    )

    print("\n" + "="*60)
    print("Experiment Complete")
    print("="*60)
    print(f"Spearman: {results['spearman']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")


if __name__ == "__main__":
    main()
