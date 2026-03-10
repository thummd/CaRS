#!/usr/bin/env python3
"""
Run Shared Backbone Experiment on Unified Electricity Data.

This script trains DS3M-Causal with shared backbone mode on the unified
electricity price datasets (DE, FR, DE_FR).

Usage:
    python run_shared_backbone.py --market DE --d_dim 2 --sharing_mode shared_backbone
    python run_shared_backbone.py --market FR --d_dim 3 --sharing_mode independent
    python run_shared_backbone.py --market DE_FR --d_dim 2 --seed 42
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import prepare_unified_ds3m_data
from models.ds3m_causal import DS3MCausal
from training.train_e2e import train_end_to_end


def parse_args():
    parser = argparse.ArgumentParser(description='Run Shared Backbone Experiment')

    # Data arguments
    parser.add_argument('--market', type=str, default='DE',
                        choices=['DE', 'FR', 'DE_FR'],
                        help='Market to run experiment on')
    parser.add_argument('--timestep', type=int, default=14,
                        help='Lookback window size')
    parser.add_argument('--task_type', type=str, default='prediction',
                        choices=['prediction', 'estimation'],
                        help='Task type')

    # Model arguments
    parser.add_argument('--d_dim', type=int, default=2,
                        help='Number of regimes')
    parser.add_argument('--sharing_mode', type=str, default='shared_backbone',
                        choices=['shared_backbone', 'independent'],
                        help='DAG sharing mode')
    parser.add_argument('--h_dim', type=int, default=32,
                        help='GRU hidden dimension')
    parser.add_argument('--z_dim', type=int, default=8,
                        help='Latent dimension')
    parser.add_argument('--lag', type=int, default=1,
                        help='Temporal lag for causal structure')

    # Loss weights
    parser.add_argument('--lambda_dag', type=float, default=100.0,
                        help='DAG constraint weight')
    parser.add_argument('--lambda_sparse', type=float, default=0.001,
                        help='Sparsity penalty weight (reduced to allow edge learning)')
    parser.add_argument('--lambda_target', type=float, default=10.0,
                        help='Target constraint weight (encourages edges TO Price)')
    parser.add_argument('--target_idx', type=int, default=0,
                        help='Index of target variable (Day_Ahead_Price)')

    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--max_auglag_steps', type=int, default=200,
                        help='Maximum augmented Lagrangian steps (increased for DAG convergence)')
    parser.add_argument('--max_inner_epochs', type=int, default=50,
                        help='Maximum inner epochs per step')
    parser.add_argument('--early_stopping_patience', type=int, default=30,
                        help='Early stopping patience (increased to allow DAG convergence)')
    parser.add_argument('--early_stopping_metric', type=str, default='directional_accuracy',
                        choices=['directional_accuracy', 'spearman'],
                        help='Metric for early stopping')

    # Temperature annealing for sparse edges
    parser.add_argument('--tau_init', type=float, default=1.0,
                        help='Initial Gumbel-Softmax temperature')
    parser.add_argument('--tau_final', type=float, default=0.1,
                        help='Final temperature (lower = more sparse/binary edges)')
    parser.add_argument('--tau_anneal_steps', type=int, default=100,
                        help='Steps over which to anneal temperature')

    # Regime differentiation
    parser.add_argument('--lambda_regime_diff', type=float, default=1.0,
                        help='Regime differentiation penalty (encourages different DAGs per regime)')
    parser.add_argument('--regime_noise_std', type=float, default=0.0,
                        help='Noise std for regime deviation initialization (breaks symmetry)')

    # Execution arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated if not provided)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print progress')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Set seed
    set_seed(args.seed)

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"Shared Backbone Experiment")
    print(f"{'='*60}")
    print(f"Market: {args.market}")
    print(f"Sharing mode: {args.sharing_mode}")
    print(f"Regimes: {args.d_dim}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Lambda sparse: {args.lambda_sparse}")
    print(f"Lambda target: {args.lambda_target}")
    print(f"Lambda regime diff: {args.lambda_regime_diff}")
    print(f"Max AugLag steps: {args.max_auglag_steps}")
    print(f"Early stop metric: {args.early_stopping_metric}")
    print(f"Tau: {args.tau_init} -> {args.tau_final} over {args.tau_anneal_steps} steps")
    print(f"{'='*60}\n")

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent / 'results' / args.market / f'{args.sharing_mode}_d{args.d_dim}_seed{args.seed}_{timestamp}'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    feature_groups = config.get('data', {}).get('feature_groups',
                                                ['price', 'generation', 'load', 'weather', 'calendar'])

    data = prepare_unified_ds3m_data(
        country=args.market,
        timestep=args.timestep,
        feature_groups=feature_groups,
        task_type=args.task_type,
    )

    x_dim = data['trainX'].shape[-1]
    print(f"Features: {x_dim}")
    print(f"Train samples: {data['trainX'].shape[1]}")
    print(f"Test samples: {data['testX'].shape[1]}")

    # Create model
    print("\nCreating model...")
    model = DS3MCausal(
        x_dim=x_dim,
        y_dim=1,
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        d_dim=args.d_dim,
        device=device,
        num_nodes=x_dim,
        lag=args.lag,
        sharing_mode=args.sharing_mode,
        tau_gumbel=1.0,
        init_logits=[-0.5, -0.5],
        lambda_dag=args.lambda_dag,
        lambda_sparse=args.lambda_sparse,
        lambda_kl=1.0,
        regime_noise_std=args.regime_noise_std,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # Training config
    train_config = {
        'learning_rate': args.learning_rate,
        'max_auglag_steps': args.max_auglag_steps,
        'max_inner_epochs': args.max_inner_epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_min_delta': 0.001,
        'early_stopping_metric': args.early_stopping_metric,
        'alpha_init': 0.0,
        'rho_init': 1.0,
        'rho_max': 1e9,
        'progress_rate': 0.9,
        'tol_dag': 1e-6,
        # Target constraint parameters
        'target_idx': args.target_idx,
        'lambda_target': args.lambda_target,
        # Temperature annealing
        'tau_init': args.tau_init,
        'tau_final': args.tau_final,
        'tau_anneal_steps': args.tau_anneal_steps,
        # Regime differentiation
        'lambda_regime_diff': args.lambda_regime_diff,
    }

    # Save experiment config
    experiment_config = {
        'args': vars(args),
        'train_config': train_config,
        'data': {
            'market': args.market,
            'x_dim': x_dim,
            'timestep': args.timestep,
            'feature_groups': feature_groups,
            'feature_cols': data['feature_cols'],
            'target_col': data['target_col'],
            'task_type': args.task_type,
        },
        'model': {
            'h_dim': args.h_dim,
            'z_dim': args.z_dim,
            'd_dim': args.d_dim,
            'lag': args.lag,
            'sharing_mode': args.sharing_mode,
            'total_params': total_params,
        }
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(experiment_config, f, indent=2, default=str)

    # Train
    print("\nTraining...")
    start_time = time.time()

    results = train_end_to_end(
        model=model,
        trainX=data['trainX'],
        trainY=data['trainY'],
        testX=data['testX'],
        testY=data['testY'],
        config=train_config,
        output_dir=output_dir,
        verbose=args.verbose,
    )

    total_time = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Spearman correlation: {results['spearman']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Directional accuracy: {results['directional_accuracy']:.4f}")
    print(f"Final DAG penalty: {results['final_dag_penalty']:.8f}")
    print(f"Training time: {total_time:.2f}s")

    # Get and save graphs
    graphs = model.get_causal_graphs()
    print(f"\nLearned causal graphs:")
    for d, g in enumerate(graphs):
        edges = (np.abs(g) > 0.5).sum()
        print(f"  Regime {d}: {edges} edges (threshold 0.5)")

    # Save graphs as numpy
    np.savez(output_dir / 'graphs.npz', **{f'regime_{d}': g for d, g in enumerate(graphs)})

    # If shared_backbone mode, also save shared vs regime-specific edges
    if args.sharing_mode == 'shared_backbone':
        shared_edges = model.dag_dist.get_shared_edges()
        if shared_edges is not None:
            shared_edges_np = shared_edges.cpu().detach().numpy()
            np.save(output_dir / 'shared_edges.npy', shared_edges_np)
            print(f"  Shared edges: {(np.abs(shared_edges_np) > 0.5).sum()}")

        for d in range(args.d_dim):
            regime_specific = model.dag_dist.get_regime_specific_edges(d)
            if regime_specific is not None:
                regime_specific_np = regime_specific.cpu().detach().numpy()
                np.save(output_dir / f'regime_{d}_specific_edges.npy', regime_specific_np)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == '__main__':
    main()
