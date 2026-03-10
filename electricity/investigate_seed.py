"""
Investigate Seed 43: Why does it produce optimal regime splits?

Analyzes what makes certain seeds (like 43) achieve good performance:
1. Initial weight distributions
2. Learned transition matrices
3. Regime assignment patterns (temporal)
4. Latent space structure
5. Training dynamics

Usage:
    python investigate_seed.py --seeds 42 43 44 --checkpoint_dir outputs/ds3m
"""

import sys
import os
from pathlib import Path

from paths import DS3M_DIR, OUTPUT_DIR
# Add DS3M code to path
DS3M_PATH = str(DS3M_DIR)
sys.path.insert(0, DS3M_PATH)
sys.path.insert(0, os.path.join(DS3M_PATH, "src"))

import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from DSSSMCode import DSSSM
from ds3m_adapter import load_qrt_data


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def analyze_initial_weights(seeds: list, config: dict, device='cpu'):
    """Compare initial weight distributions across seeds."""
    print("\n=== Initial Weight Analysis ===")

    results = {}

    for seed in seeds:
        set_seed(seed)

        # Create model with this seed
        model = DSSSM(
            x_dim=config.get('predict_dim', 1),
            y_dim=config.get('predict_dim', 1),
            h_dim=config['h_dim'],
            z_dim=config['z_dim'],
            d_dim=config['d_dim'],
            n_layers=config['n_layers'],
            device=device,
            bidirection=config.get('bidirection', False)
        ).to(device)

        # Extract initial weights
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item()
            }

        # Initial transition matrix
        transition = model.TransitionMatrix().detach().cpu().numpy()

        results[seed] = {
            'weights': weights,
            'transition_matrix': transition
        }

        print(f"\nSeed {seed}:")
        print(f"  Initial transition matrix:\n{transition}")

    return results


def analyze_learned_models(seeds: list, checkpoint_dir: Path, device='cpu'):
    """Compare learned models across seeds."""
    print("\n=== Learned Model Analysis ===")

    results = {}

    for seed in seeds:
        # Find checkpoint for this seed
        pattern = f"*seed{seed}*" if seed != 42 else "*seed42*"
        matches = list(checkpoint_dir.glob(pattern))

        if not matches:
            print(f"  Seed {seed}: No checkpoint found")
            continue

        model_dir = matches[0]
        checkpoint_path = model_dir / 'checkpoints' / 'best.tar'

        if not checkpoint_path.exists():
            print(f"  Seed {seed}: No best.tar found")
            continue

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']

        # Create model
        x_dim = config.get('x_dim', config.get('predict_dim', 1))
        y_dim = config.get('y_dim', config.get('predict_dim', 1))

        model = DSSSM(
            x_dim=x_dim,
            y_dim=y_dim,
            h_dim=config['h_dim'],
            z_dim=config['z_dim'],
            d_dim=config['d_dim'],
            n_layers=config['n_layers'],
            device=device,
            bidirection=config.get('bidirection', False)
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Get learned transition matrix
        transition = model.TransitionMatrix().detach().cpu().numpy()

        # Load results
        results_path = model_dir / 'results.json'
        with open(results_path) as f:
            model_results = json.load(f)

        # Load predictions for regime analysis
        pred_df = pd.read_csv(model_dir / 'predictions.csv')

        results[seed] = {
            'model_dir': str(model_dir),
            'transition_matrix': transition,
            'spearman': model_results['metrics']['spearman'],
            'regime_distribution': model_results.get('regime_distribution', {}),
            'predictions': pred_df['prediction'].values,
            'original': pred_df['original'].values,
            'regimes': pred_df['regime'].values
        }

        print(f"\nSeed {seed} ({model_dir.name}):")
        print(f"  Spearman: {model_results['metrics']['spearman']:.4f}")
        print(f"  Regime distribution: {model_results.get('regime_distribution', {})}")
        print(f"  Learned transition matrix:\n{transition}")

    return results


def analyze_regime_patterns(results: dict, output_dir: Path = None):
    """Analyze temporal patterns in regime assignments."""
    print("\n=== Regime Pattern Analysis ===")

    # Load QRT data to get time information
    df = load_qrt_data()
    test_size = len(list(results.values())[0]['regimes'])

    # Get day IDs for test set
    day_ids = df['DAY_ID'].values[-test_size:]

    for seed, data in results.items():
        regimes = data['regimes']
        spearman = data['spearman']

        # Regime statistics
        n_regime0 = sum(1 for r in regimes if r == 0)
        n_regime1 = sum(1 for r in regimes if r == 1)
        ratio = n_regime0 / len(regimes)

        print(f"\nSeed {seed} (Spearman={spearman:.4f}):")
        print(f"  Regime 0: {n_regime0} samples ({ratio*100:.1f}%)")
        print(f"  Regime 1: {n_regime1} samples ({(1-ratio)*100:.1f}%)")

        # Check regime 0 positions (are they clustered or scattered?)
        regime0_indices = np.where(regimes == 0)[0]
        if len(regime0_indices) > 1:
            gaps = np.diff(regime0_indices)
            print(f"  Regime 0 indices: first={regime0_indices[0]}, last={regime0_indices[-1]}")
            print(f"  Gaps between regime 0: mean={gaps.mean():.1f}, max={gaps.max()}")

            # Check if regime 0 is clustered (consecutive) or scattered
            n_clusters = sum(1 for g in gaps if g > 5) + 1
            print(f"  Number of regime 0 clusters: {n_clusters}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot regime assignments over time
        fig, axes = plt.subplots(len(results), 1, figsize=(14, 3*len(results)), sharex=True)
        if len(results) == 1:
            axes = [axes]

        for ax, (seed, data) in zip(axes, results.items()):
            regimes = data['regimes']
            spearman = data['spearman']

            # Create time series plot
            ax.fill_between(range(len(regimes)), 0, 1,
                           where=np.array(regimes)==0, alpha=0.3, color='red', label='Regime 0')
            ax.fill_between(range(len(regimes)), 0, 1,
                           where=np.array(regimes)==1, alpha=0.3, color='blue', label='Regime 1')

            # Overlay original values
            ax2 = ax.twinx()
            ax2.plot(data['original'], 'k-', alpha=0.5, linewidth=0.5)
            ax2.set_ylabel('TARGET')

            ax.set_ylabel('Regime')
            ax.set_title(f'Seed {seed}: Spearman={spearman:.4f}, Regime 0={sum(regimes==0)}/{len(regimes)}')
            ax.legend(loc='upper right')
            ax.set_ylim(0, 1)

        axes[-1].set_xlabel('Test Sample Index')
        plt.tight_layout()
        plt.savefig(output_dir / 'regime_patterns.png', dpi=150)
        plt.close()

        print(f"\nPlot saved to: {output_dir / 'regime_patterns.png'}")


def analyze_prediction_quality(results: dict, output_dir: Path = None):
    """Analyze prediction quality per regime."""
    print("\n=== Per-Regime Prediction Analysis ===")

    for seed, data in results.items():
        regimes = data['regimes']
        predictions = data['predictions']
        original = data['original']

        print(f"\nSeed {seed}:")

        for regime in [0, 1]:
            mask = regimes == regime
            if mask.sum() == 0:
                continue

            pred_r = predictions[mask]
            orig_r = original[mask]

            if len(pred_r) > 2:
                corr, _ = spearmanr(orig_r, pred_r)
                rmse = np.sqrt(np.mean((pred_r - orig_r)**2))
                print(f"  Regime {regime} ({mask.sum()} samples): Spearman={corr:.4f}, RMSE={rmse:.4f}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Scatter plot per regime
        n_seeds = len(results)
        fig, axes = plt.subplots(n_seeds, 2, figsize=(12, 4*n_seeds))
        if n_seeds == 1:
            axes = axes.reshape(1, -1)

        for row, (seed, data) in enumerate(results.items()):
            for col, regime in enumerate([0, 1]):
                ax = axes[row, col]
                mask = data['regimes'] == regime

                if mask.sum() > 0:
                    ax.scatter(data['original'][mask], data['predictions'][mask],
                              alpha=0.5, s=20)
                    if mask.sum() > 2:
                        corr, _ = spearmanr(data['original'][mask], data['predictions'][mask])
                        ax.set_title(f'Seed {seed}, Regime {regime}: Spearman={corr:.4f}')
                    else:
                        ax.set_title(f'Seed {seed}, Regime {regime}: n={mask.sum()}')
                else:
                    ax.set_title(f'Seed {seed}, Regime {regime}: No samples')

                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')

                # Add diagonal
                lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                       max(ax.get_xlim()[1], ax.get_ylim()[1])]
                ax.plot(lims, lims, 'r--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_by_regime.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Investigate seed differences in DS3M")
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44],
                       help='Seeds to analyze')
    parser.add_argument('--checkpoint_dir', type=str,
                       default=str(OUTPUT_DIR) + '/ds3m',
                       help='Directory containing model checkpoints')
    parser.add_argument('--pattern', type=str, default='ALL_TARGET_uv_d2_*',
                       help='Pattern to match model directories')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots')

    args = parser.parse_args()

    print("=" * 60)
    print("SEED INVESTIGATION")
    print("=" * 60)
    print(f"Seeds: {args.seeds}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / 'seed_investigation'

    # Default config for initial weight analysis
    config = {
        'h_dim': 30,
        'z_dim': 8,
        'd_dim': 2,
        'n_layers': 1,
        'predict_dim': 1,
        'bidirection': False
    }

    # 1. Analyze initial weights
    initial_results = analyze_initial_weights(args.seeds, config)

    # 2. Analyze learned models
    learned_results = analyze_learned_models(args.seeds, checkpoint_dir)

    if learned_results:
        # 3. Analyze regime patterns
        analyze_regime_patterns(learned_results, output_dir)

        # 4. Analyze prediction quality
        analyze_prediction_quality(learned_results, output_dir)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        # Sort by Spearman
        sorted_seeds = sorted(learned_results.items(), key=lambda x: x[1]['spearman'], reverse=True)

        print("\nRanking by Spearman:")
        for seed, data in sorted_seeds:
            regime_dist = data['regime_distribution']
            print(f"  Seed {seed}: Spearman={data['spearman']:.4f}, Regimes={regime_dist}")

        # Identify pattern
        best_seed, best_data = sorted_seeds[0]
        worst_seed, worst_data = sorted_seeds[-1]

        print(f"\nBest seed {best_seed} characteristics:")
        n0 = sum(1 for r in best_data['regimes'] if r == 0)
        n1 = sum(1 for r in best_data['regimes'] if r == 1)
        print(f"  Regime split: {n0}/{n1} ({n0/(n0+n1)*100:.1f}% / {n1/(n0+n1)*100:.1f}%)")

        print(f"\nWorst seed {worst_seed} characteristics:")
        n0 = sum(1 for r in worst_data['regimes'] if r == 0)
        n1 = sum(1 for r in worst_data['regimes'] if r == 1)
        if n0 + n1 > 0:
            print(f"  Regime split: {n0}/{n1} ({n0/(n0+n1)*100:.1f}% / {n1/(n0+n1)*100:.1f}%)")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
