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
                        help='Market to run experiment on (e.g. DE, FR, AT, DE_FR)')
    parser.add_argument('--timestep', type=int, default=14,
                        help='Lookback window size')
    parser.add_argument('--task_type', type=str, default='prediction',
                        choices=['prediction', 'estimation'],
                        help='Task type')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Forecast horizon in hours (1, 6, 12, 24)')
    parser.add_argument('--feature_groups', type=str, default=None,
                        help='Comma-separated feature groups (overrides config)')
    parser.add_argument('--target_col', type=str, default=None,
                        help='Target column (default: auto-select price_return)')

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
    parser.add_argument('--lambda_dag', type=float, default=0.0,
                        help='DAG constraint weight in base loss (0 = let augmented Lagrangian handle DAG enforcement exclusively)')
    parser.add_argument('--lambda_sparse', type=float, default=0.001,
                        help='Sparsity penalty weight (reduced to allow edge learning)')
    parser.add_argument('--lambda_target', type=float, default=10.0,
                        help='Target constraint weight (encourages edges TO Price)')
    parser.add_argument('--lambda_var_reg', type=float, default=0.01,
                        help='Variance regularization weight (prevents variance blowup)')
    parser.add_argument('--lambda_kl', type=float, default=1.0,
                        help='KL divergence weight (latent z + regime d). Lower '
                             'values (e.g. 0.05) mitigate posterior collapse on '
                             'small/noisy datasets such as the daily resolution.')
    parser.add_argument('--lambda_w_aux', type=float, default=0.0,
                        help='W-routed auxiliary loss weight. Adds '
                             'MSE(W-only prediction, target) to force the causal '
                             'weights W onto the prediction path rather than the '
                             'latent conditioning bypass. 0 = off.')
    parser.add_argument('--w_init_scale', type=float, default=0.01,
                        help='ICGNN W initialization scale (higher = stronger initial edges)')
    parser.add_argument('--aggregation_mode', type=str, default='linear',
                        choices=['linear', 'dual_channel', 'cam', 'gat', 'cam_gat'],
                        help='ICGNN aggregation: linear (default), dual_channel, cam, gat, or cam_gat (CAM MLPs + GATv2 attention combined)')
    parser.add_argument('--cam_hidden_dim', type=int, default=32,
                        help='Hidden dimension for CAM per-parent MLPs')
    parser.add_argument('--emission_embedding_size', type=int, default=32,
                        help='CARGO node-embedding dim (default 32). Small values '
                             '(e.g. 4) constrain the decoder so the structural W '
                             'must carry the causal signal (W-identifiability prototype).')
    parser.add_argument('--emission_decoder_layers', type=str, default='64,64',
                        help='Comma-separated hidden sizes of the CARGO decoder MLP '
                             '(default "64,64"). Empty string "" = a LINEAR readout, '
                             'which removes the decoder capacity that fits the target '
                             'from a random-W projection and forces W to be identified.')
    parser.add_argument('--emission_encoder_layers', type=str, default='64,64',
                        help='Comma-separated hidden sizes of the CARGO g-encoder MLP '
                             '(default "64,64"). Empty string "" = a LINEAR encoder, '
                             'shrinking per-node embedding capacity so the W-weighted '
                             'aggregation (not a rich encoder) carries the signal. '
                             'Combine with --no_attention --aggregation_mode linear '
                             'for a near-linear SEM where W must bear the load.')
    parser.add_argument('--dual_channel', action='store_true',
                        help='[Deprecated] Use --aggregation_mode dual_channel instead')
    parser.add_argument('--elastic_threshold', type=float, default=0.0,
                        help='Elastic-net: min edge magnitude to reward (0 = off, 0.05 typical)')
    parser.add_argument('--elastic_weight', type=float, default=0.0,
                        help='Elastic-net: reverse penalty strength (0.1-0.5 typical)')
    parser.add_argument('--dag_constraint', type=str, default='notears',
                        choices=['notears', 'dagma'],
                        help='DAG acyclicity constraint: "notears" (tr(exp(A))-d) or '
                             '"dagma" (logdet-based, ~1.5x faster, smoother gradients)')
    parser.add_argument('--lambda_entropy', type=float, default=0.0,
                        help='Regime entropy regularization (0=off, 1.0 recommended for K>2)')
    parser.add_argument('--target_idx', type=int, default=0,
                        help='Index of target variable (Day_Ahead_Price)')
    # Physical-interconnect prior on cross-border CARGO emission rows
    parser.add_argument('--physical_prior_mode', type=str, default='off',
                        choices=['off', 'hard', 'soft'],
                        help='Physical-interconnect prior on the CARGO'
                             ' emission. "hard" zeros cross-border W rows'
                             ' on non-physical market pairs; "soft"'
                             ' attenuates them through a learnable sigmoid'
                             ' scalar. Requires --market to be a 2-letter'
                             ' code and feature_cols to include'
                             ' `{SRC}_price_lag*` columns for cross-border'
                             ' identification.')
    parser.add_argument('--physical_prior_alpha_init', type=float, default=0.05,
                        help='Initial pass-through fraction for the soft prior'
                             ' (0.05 = 5%% initial leak on forbidden edges).')
    parser.add_argument('--frequency', type=str, default='H', choices=['H', 'D'],
                        help='Frequency of the unified dataset to load:'
                             ' H = hourly (default, ~390k rows per market),'
                             ' D = daily (~4k rows per market). Daily mode'
                             ' is much faster per epoch and is the dataset'
                             ' used for the multi-seed bootstrap.')
    parser.add_argument('--start_date', type=str, default=None,
                        help='If set (YYYY-MM-DD), restrict the dataset to dates '
                             '>= start_date before splitting, so the causal graph '
                             'is fit on a single structural-break era (e.g. '
                             'post-2024 = post German nuclear phase-out).')
    parser.add_argument('--end_date', type=str, default=None,
                        help='If set (YYYY-MM-DD), restrict the dataset to dates '
                             '< end_date (exclusive); pair with --start_date to '
                             'isolate a single era.')

    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--max_auglag_steps', type=int, default=40,
                        help='Maximum augmented Lagrangian steps')
    parser.add_argument('--max_inner_epochs', type=int, default=50,
                        help='Maximum inner epochs per step')
    parser.add_argument('--patience_dag', type=int, default=5,
                        help='Adaptive AugLag stop: number of consecutive steps with '
                             'dag_penalty < tol_dag before halting outer loop')
    parser.add_argument('--use_amp', action='store_true',
                        help='Enable bf16 mixed-precision autocast for forward/backward. '
                             '~1.5-2x speedup on A100/H100 with negligible accuracy loss.')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--early_stopping_metric', type=str, default='spearman',
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

    # Mini-batching
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Mini-batch size (0 = full batch)')

    # Attention aggregation
    parser.add_argument('--no_attention', action='store_true',
                        help='Disable attention-weighted aggregation in ICGNN')
    parser.add_argument('--pure_scm_readout', action='store_true',
                        help='Emit the target as the bare linear structural '
                             'equation y = sum_lag,parent W[parent->y]*x_parent, '
                             'with NO learnable readout between W and the output '
                             '(W IS the regression coefficient). Strongest forcing '
                             'of W; removes the random-features escape that a '
                             'learnable decoder gives even in --aggregation_mode '
                             'linear. Pair with --lambda_w_aux.')
    parser.add_argument('--warmstart_ckpt', type=str, default=None,
                        help='Path to a checkpoint (e.g. a controls_daily run with '
                             'good regime occupancy) whose regime backbone '
                             '(rnn_forward/rnn_backward GRU encoder + d_posterior_nets '
                             '+ d_transition) is loaded into this model and FROZEN, '
                             'so long pure-SCM training identifies the per-regime W on '
                             'a fixed, balanced regime split instead of re-collapsing. '
                             'Requires matching feature set / x_dim.')

    # Data options
    parser.add_argument('--spillover', action='store_true',
                        help='Include lagged cross-country prices and flows')
    parser.add_argument('--resampled', action='store_true',
                        help='Use hourly resampled datasets (common period)')

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
    print(f"Forecast horizon: {args.horizon}")
    print(f"Tau: {args.tau_init} -> {args.tau_final} over {args.tau_anneal_steps} steps")
    print(f"{'='*60}\n")

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = '_spillover' if args.spillover else ('_resampled' if args.resampled else '')
        h_str = f'_h{args.horizon}' if args.horizon > 1 else ''
        output_dir = Path(__file__).parent / 'results' / args.market / f'{args.sharing_mode}_d{args.d_dim}_seed{args.seed}{suffix}{h_str}_{timestamp}'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    # CLI --feature_groups overrides YAML config
    if args.feature_groups:
        feature_groups = args.feature_groups.split(',')
    else:
        feature_groups = config.get('data', {}).get('feature_groups',
                                                    ['price', 'generation', 'load', 'weather', 'calendar'])

    data = prepare_unified_ds3m_data(
        country=args.market,
        timestep=args.timestep,
        feature_groups=feature_groups,
        target_col=args.target_col,
        task_type=args.task_type,
        resampled=args.resampled,
        spillover=args.spillover,
        horizon=args.horizon,
        frequency=args.frequency,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    x_dim = data['trainX'].shape[-1]
    print(f"Features: {x_dim}")
    print(f"Train samples: {data['trainX'].shape[1]}")
    print(f"Test samples: {data['testX'].shape[1]}")

    # Build the physical-interconnect mask if the prior is enabled.
    # `data['feature_cols']` is the list of features actually consumed by
    # this checkpoint; we match cross-border features by their canonical
    # `{SRC}_price_lag*` / `{SRC}_flow_lag*` prefix and forbid rows whose
    # (SRC, args.market) pair is not in the physical interconnect set.
    physical_mask = None
    if args.physical_prior_mode != "off":
        from modules.physical_prior import build_physical_mask, summarize_mask
        feature_cols = data.get("feature_cols")
        if feature_cols is None:
            raise RuntimeError(
                "--physical_prior_mode requires data['feature_cols'] to be"
                " present in the prepared dataset; current data dict has"
                f" keys {list(data.keys())}.")
        physical_mask = build_physical_mask(
            feature_cols, target_country=args.market, lag=args.lag)
        diag = summarize_mask(physical_mask, feature_cols)
        print(f"\nCARGO physical-interconnect prior ({args.physical_prior_mode}):"
              f" {diag['n_forbidden_features']}/{diag['n_features']} CARGO input"
              f" features forbidden ({diag['fraction_blocked_edges']:.1%} of edges).")
        if diag['forbidden_features']:
            print(f"  forbidden features: {', '.join(diag['forbidden_features'])}")

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
        lambda_kl=args.lambda_kl,
        lambda_w_aux=args.lambda_w_aux,
        lambda_var_reg=args.lambda_var_reg,
        regime_noise_std=args.regime_noise_std,
        use_attention=not args.no_attention,
        w_init_scale=args.w_init_scale,
        aggregation_mode=args.aggregation_mode,
        cam_hidden_dim=args.cam_hidden_dim,
        emission_embedding_size=args.emission_embedding_size,
        emission_decoder_layers=tuple(
            int(x) for x in args.emission_decoder_layers.split(',') if x.strip()),
        emission_encoder_layers=tuple(
            int(x) for x in args.emission_encoder_layers.split(',') if x.strip()),
        dual_channel=args.dual_channel,
        elastic_threshold=args.elastic_threshold,
        elastic_weight=args.elastic_weight,
        dag_constraint=args.dag_constraint,
        physical_mask=physical_mask,
        physical_prior_mode=args.physical_prior_mode,
        physical_prior_alpha_init=args.physical_prior_alpha_init,
        pure_scm_readout=args.pure_scm_readout,
    ).to(device)

    # Warm-start + freeze the regime backbone (GRU encoder + regime-posterior
    # nets + transition) from a checkpoint with good regime occupancy, so long
    # pure-SCM training identifies the per-regime W on a FIXED, balanced regime
    # split rather than re-collapsing to a single dominant regime.
    if args.warmstart_ckpt:
        _FREEZE = ('rnn_forward', 'rnn_backward', 'd_posterior_nets', 'd_transition')
        _ws = torch.load(args.warmstart_ckpt, map_location=device, weights_only=False)
        _ws_sd = _ws.get('model_state_dict', _ws) if isinstance(_ws, dict) else _ws
        _msd = model.state_dict()
        _loaded, _skipped = 0, 0
        for _k, _v in _ws_sd.items():
            if any(_k.startswith(p) for p in _FREEZE):
                if _k in _msd and _msd[_k].shape == _v.shape:
                    _msd[_k] = _v; _loaded += 1
                else:
                    _skipped += 1
        model.load_state_dict(_msd, strict=False)
        _nfroz = 0
        for _name, _p in model.named_parameters():
            if any(_name.startswith(p) for p in _FREEZE):
                _p.requires_grad = False; _nfroz += 1
        print(f"Warm-start: loaded {_loaded} regime-backbone tensors "
              f"(skipped {_skipped} shape-mismatch) from {args.warmstart_ckpt}; "
              f"froze {_nfroz} param tensors (regime backbone fixed).")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters (trainable): {total_params:,}")

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
        'patience_dag': args.patience_dag,
        # Target constraint parameters
        'target_idx': args.target_idx,
        'lambda_target': args.lambda_target,
        # Temperature annealing
        'tau_init': args.tau_init,
        'tau_final': args.tau_final,
        'tau_anneal_steps': args.tau_anneal_steps,
        # Regime differentiation
        'lambda_regime_diff': args.lambda_regime_diff,
        # Mini-batching
        'batch_size': args.batch_size,
        # Forecast horizon
        'horizon': args.horizon,
        # Regime entropy regularization
        'lambda_entropy': args.lambda_entropy,
        # W-routed auxiliary loss (force target prediction through W)
        'lambda_w_aux': args.lambda_w_aux,
        # Mixed precision
        'use_amp': args.use_amp,
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
        Y_moments=data.get('Y_moments'),
        verbose=args.verbose,
    )

    total_time = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Directional accuracy: {results['directional_accuracy']:.4f}")
    print(f"Spearman correlation: {results['spearman']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"sMAPE: {results['smape']:.2f}")
    if results.get('crps') is not None:
        print(f"CRPS: {results['crps']:.4f}")
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
