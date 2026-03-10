#!/usr/bin/env python3
"""
Run Target-Constrained Shared Backbone Experiment.

This script runs the improved training with target constraint, which encourages
edges TO the Price node to be learned. This addresses the issue where previous
training runs resulted in no meaningful edges to the target variable.

Key improvements over default run_shared_backbone.py:
- Target constraint loss (lambda_target=10.0)
- Longer training (max_auglag_steps=200)
- Reduced sparsity penalty (lambda_sparse=0.001)
- More patient early stopping (patience=25)

Usage:
    python run_target_constrained.py --market DE --seed 42
    python run_target_constrained.py --market FR --seed 42
    python run_target_constrained.py --market DE_FR --seed 42

Results are saved to:
    results/<MARKET>/target_constrained_d<D>_seed<SEED>/
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Run Target-Constrained Experiment')
    parser.add_argument('--market', type=str, default='DE',
                        choices=['DE', 'FR', 'DE_FR'],
                        help='Market to run experiment on')
    parser.add_argument('--d_dim', type=int, default=2,
                        help='Number of regimes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--lambda_target', type=float, default=10.0,
                        help='Target constraint weight')
    parser.add_argument('--lambda_sparse', type=float, default=0.001,
                        help='Sparsity penalty weight')
    parser.add_argument('--max_auglag_steps', type=int, default=200,
                        help='Maximum augmented Lagrangian steps')
    parser.add_argument('--early_stopping_patience', type=int, default=25,
                        help='Early stopping patience')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print command without executing')

    args = parser.parse_args()

    # Output directory
    output_dir = Path(__file__).parent / 'results' / args.market / f'target_constrained_d{args.d_dim}_seed{args.seed}'

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'run_shared_backbone.py'),
        '--market', args.market,
        '--d_dim', str(args.d_dim),
        '--seed', str(args.seed),
        '--sharing_mode', 'shared_backbone',
        '--lambda_target', str(args.lambda_target),
        '--lambda_sparse', str(args.lambda_sparse),
        '--max_auglag_steps', str(args.max_auglag_steps),
        '--early_stopping_patience', str(args.early_stopping_patience),
        '--output_dir', str(output_dir),
    ]

    print("=" * 70)
    print("Target-Constrained Shared Backbone Experiment")
    print("=" * 70)
    print(f"Market: {args.market}")
    print(f"Regimes: {args.d_dim}")
    print(f"Seed: {args.seed}")
    print(f"Lambda target: {args.lambda_target}")
    print(f"Lambda sparse: {args.lambda_sparse}")
    print(f"Max AugLag steps: {args.max_auglag_steps}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    print()

    if args.dry_run:
        print("Dry run - would execute:")
        print(' '.join(cmd))
        return

    # Run the experiment
    print("Running experiment...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nExperiment failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print("\n" + "=" * 70)
    print("Experiment completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    # Print summary of learned edges
    graphs_path = output_dir / 'graphs.npz'
    if graphs_path.exists():
        import numpy as np
        graphs = np.load(graphs_path)
        print("\nLearned causal graphs summary:")
        for key in graphs.keys():
            g = graphs[key]
            # Check edges TO first variable (Price)
            edges_to_target = g[:, :, 0]
            n_edges_to_target = (np.abs(edges_to_target) > 0.1).sum()
            print(f"  {key}: {n_edges_to_target} edges TO Price (threshold 0.1)")


if __name__ == '__main__':
    main()
