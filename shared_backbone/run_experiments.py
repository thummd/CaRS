#!/usr/bin/env python3
"""
Run Multi-Seed Experiments for Shared Backbone Comparison.

Runs experiments across multiple seeds, markets, and configurations
to compare shared_backbone vs independent DAG learning modes.

Usage:
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --markets DE FR    # Only DE and FR
    python run_experiments.py --sharing_modes shared_backbone  # Only shared backbone
    python run_experiments.py --dry_run          # Preview without running
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd

# Default experiment configuration
DEFAULT_MARKETS = ['DE', 'FR', 'DE_FR']
DEFAULT_SHARING_MODES = ['shared_backbone', 'independent']
DEFAULT_D_DIMS = [2, 3]
DEFAULT_SEEDS = [42, 123, 456, 789, 1011]


def parse_args():
    parser = argparse.ArgumentParser(description='Run Multi-Seed Experiments')

    parser.add_argument('--markets', nargs='+', default=DEFAULT_MARKETS,
                        help='Markets to test')
    parser.add_argument('--sharing_modes', nargs='+', default=DEFAULT_SHARING_MODES,
                        help='Sharing modes to test')
    parser.add_argument('--d_dims', nargs='+', type=int, default=DEFAULT_D_DIMS,
                        help='Number of regimes to test')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS,
                        help='Random seeds')

    parser.add_argument('--output_base', type=str, default=None,
                        help='Base output directory')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel jobs (1 for sequential)')

    return parser.parse_args()


def generate_experiment_configs(args) -> List[Dict]:
    """Generate all experiment configurations."""
    configs = []

    for market in args.markets:
        for sharing_mode in args.sharing_modes:
            for d_dim in args.d_dims:
                for seed in args.seeds:
                    configs.append({
                        'market': market,
                        'sharing_mode': sharing_mode,
                        'd_dim': d_dim,
                        'seed': seed,
                    })

    return configs


def run_experiment(config: Dict, output_base: Path, device: str, dry_run: bool) -> Dict:
    """Run a single experiment."""
    script_path = Path(__file__).parent / 'run_shared_backbone.py'

    cmd = [
        sys.executable, str(script_path),
        '--market', config['market'],
        '--sharing_mode', config['sharing_mode'],
        '--d_dim', str(config['d_dim']),
        '--seed', str(config['seed']),
        '--device', device,
    ]

    if output_base:
        output_dir = output_base / config['market'] / f"{config['sharing_mode']}_d{config['d_dim']}_seed{config['seed']}"
        cmd.extend(['--output_dir', str(output_dir)])

    print(f"\nRunning: {config}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        return {'status': 'dry_run', **config}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            # Try to load results
            results_file = output_dir / 'results.json' if output_base else None
            metrics = {}
            if results_file and results_file.exists():
                with open(results_file) as f:
                    metrics = json.load(f)

            return {'status': 'success', **config, **metrics}
        else:
            return {'status': 'failed', 'error': result.stderr[-500:], **config}

    except subprocess.TimeoutExpired:
        return {'status': 'timeout', **config}
    except Exception as e:
        return {'status': 'error', 'error': str(e), **config}


def aggregate_results(results: List[Dict]) -> pd.DataFrame:
    """Aggregate results into a DataFrame."""
    df = pd.DataFrame(results)

    # Filter successful runs
    success_df = df[df['status'] == 'success'].copy()

    if len(success_df) == 0:
        print("No successful runs to aggregate.")
        return df

    # Group by configuration and compute mean/std
    group_cols = ['market', 'sharing_mode', 'd_dim']
    metric_cols = ['spearman', 'rmse']

    available_metrics = [c for c in metric_cols if c in success_df.columns]

    if available_metrics:
        summary = success_df.groupby(group_cols)[available_metrics].agg(['mean', 'std'])
        print("\n" + "="*60)
        print("Results Summary (mean ± std)")
        print("="*60)
        print(summary.round(4))

    return df


def main():
    args = parse_args()

    # Setup output directory
    if args.output_base:
        output_base = Path(args.output_base)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_base = Path(__file__).parent / 'results' / f'experiment_{timestamp}'

    output_base.mkdir(parents=True, exist_ok=True)

    # Generate experiment configurations
    configs = generate_experiment_configs(args)

    print(f"\n{'='*60}")
    print(f"Multi-Seed Experiment Runner")
    print(f"{'='*60}")
    print(f"Total experiments: {len(configs)}")
    print(f"Markets: {args.markets}")
    print(f"Sharing modes: {args.sharing_modes}")
    print(f"Regimes (d_dim): {args.d_dims}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {output_base}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*60}")

    # Save experiment plan
    plan = {
        'configs': configs,
        'args': vars(args),
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_base / 'experiment_plan.json', 'w') as f:
        json.dump(plan, f, indent=2)

    # Run experiments
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running experiment...")
        result = run_experiment(config, output_base, args.device, args.dry_run)
        results.append(result)

        # Save intermediate results
        with open(output_base / 'results_partial.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Aggregate and save final results
    df = aggregate_results(results)
    df.to_csv(output_base / 'results_all.csv', index=False)

    # Save final results as JSON
    with open(output_base / 'results_final.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiments completed!")
    print(f"Results saved to: {output_base}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
