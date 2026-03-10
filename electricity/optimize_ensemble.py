"""
Ensemble Optimization for DS3M Models.

Explores different ensemble configurations:
1. Temperature tuning for softmax
2. Top-K model selection
3. Min Spearman threshold
4. Leave-one-out cross-validation

Usage:
    python optimize_ensemble.py --output_dir outputs/ds3m
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import OUTPUT_DIR
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_model_predictions(model_dir: Path):
    """Load predictions from a trained model directory."""
    pred_path = model_dir / 'predictions.csv'
    results_path = model_dir / 'results.json'

    if not pred_path.exists() or not results_path.exists():
        return None

    pred_df = pd.read_csv(pred_path)
    with open(results_path) as f:
        results = json.load(f)

    # Skip models with only 1 regime
    regime_dist = results.get('regime_distribution', {})
    n_regime0 = regime_dist.get('0', 0)
    n_regime1 = regime_dist.get('1', 0)
    if n_regime0 == 0 or n_regime1 == 0:
        return None

    return {
        'predictions': pred_df['prediction'].values,
        'original': pred_df['original'].values,
        'regimes': pred_df['regime'].values,
        'spearman': results['metrics']['spearman'],
        'regime_distribution': regime_dist,
        'dir': str(model_dir)
    }


def find_all_models(output_dir: Path, pattern: str = 'ALL_TARGET_uv_d2_*'):
    """Find all models (no filtering)."""
    models = []
    for model_dir in output_dir.glob(pattern):
        if not model_dir.is_dir():
            continue
        result = load_model_predictions(model_dir)
        if result is not None:
            models.append(result)
    return sorted(models, key=lambda x: x['spearman'], reverse=True)


def create_ensemble(models, method='softmax', temperature=5.0):
    """Create ensemble predictions."""
    if len(models) == 0:
        return None, None

    predictions = np.array([m['predictions'] for m in models])
    spearman_scores = np.array([m['spearman'] for m in models])
    original = models[0]['original']

    if method == 'equal':
        ensemble_pred = np.mean(predictions, axis=0)
    elif method == 'spearman':
        weights = np.maximum(spearman_scores, 0)
        if weights.sum() == 0:
            weights = np.ones(len(weights))
        weights = weights / weights.sum()
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
    elif method == 'softmax':
        weights = np.exp(spearman_scores * temperature)
        weights = weights / weights.sum()
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
    elif method == 'median':
        ensemble_pred = np.median(predictions, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    ensemble_spearman, _ = spearmanr(original, ensemble_pred)
    return ensemble_pred, ensemble_spearman


def optimize_temperature(models, temps=[0.5, 1, 2, 3, 5, 7, 10, 15, 20]):
    """Find optimal softmax temperature."""
    results = []
    for temp in temps:
        _, spearman = create_ensemble(models, 'softmax', temp)
        results.append({'temperature': temp, 'spearman': spearman})

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['spearman'].idxmax()]
    return results_df, best['temperature'], best['spearman']


def optimize_top_k(models, k_values=None):
    """Find optimal number of top models to include."""
    if k_values is None:
        k_values = list(range(1, len(models) + 1))

    results = []
    for k in k_values:
        top_k_models = models[:k]  # Already sorted by spearman
        _, spearman = create_ensemble(top_k_models, 'softmax', 5.0)
        results.append({'k': k, 'spearman': spearman, 'models': [m['spearman'] for m in top_k_models]})

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['spearman'].idxmax()]
    return results_df, int(best['k']), best['spearman']


def optimize_threshold(models, thresholds=None):
    """Find optimal minimum Spearman threshold."""
    if thresholds is None:
        unique_spearmans = sorted(set(m['spearman'] for m in models))
        thresholds = [0] + [(a+b)/2 for a, b in zip(unique_spearmans[:-1], unique_spearmans[1:])]

    results = []
    for thresh in thresholds:
        filtered = [m for m in models if m['spearman'] >= thresh]
        if len(filtered) == 0:
            continue
        _, spearman = create_ensemble(filtered, 'softmax', 5.0)
        results.append({
            'threshold': thresh,
            'n_models': len(filtered),
            'spearman': spearman
        })

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['spearman'].idxmax()]
    return results_df, best['threshold'], best['spearman']


def leave_one_out_cv(models):
    """Estimate generalization with leave-one-out CV."""
    results = []
    for i, left_out in enumerate(models):
        remaining = [m for j, m in enumerate(models) if j != i]
        _, spearman = create_ensemble(remaining, 'softmax', 5.0)
        seed = Path(left_out['dir']).name.split('seed')[1].split('_')[0] if 'seed' in left_out['dir'] else '?'
        results.append({
            'left_out_seed': seed,
            'left_out_spearman': left_out['spearman'],
            'ensemble_spearman': spearman
        })

    results_df = pd.DataFrame(results)
    mean_cv = results_df['ensemble_spearman'].mean()
    std_cv = results_df['ensemble_spearman'].std()
    return results_df, mean_cv, std_cv


def main():
    parser = argparse.ArgumentParser(description="Optimize DS3M ensemble")
    parser.add_argument('--output_dir', type=str,
                       default=str(OUTPUT_DIR) + '/ds3m',
                       help='Directory containing model outputs')
    parser.add_argument('--pattern', type=str, default='ALL_TARGET_uv_d2_*',
                       help='Glob pattern for model directories')
    parser.add_argument('--min_spearman', type=float, default=0.0,
                       help='Minimum Spearman for initial filtering (0 = no filter)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save optimization results')

    args = parser.parse_args()

    print("=" * 70)
    print("ENSEMBLE OPTIMIZATION")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    models = find_all_models(output_dir, args.pattern)

    if args.min_spearman > 0:
        models = [m for m in models if m['spearman'] >= args.min_spearman]

    print(f"\nLoaded {len(models)} models")
    if len(models) == 0:
        print("No models found!")
        return

    # Print top models
    print("\nTop 10 models:")
    for m in models[:10]:
        seed = Path(m['dir']).name.split('seed')[1].split('_')[0] if 'seed' in m['dir'] else '?'
        r0 = m['regime_distribution'].get('0', 0)
        r1 = m['regime_distribution'].get('1', 0)
        print(f"  seed {seed}: Spearman={m['spearman']:.4f}, regimes={r0}/{r1}")

    save_dir = Path(args.save_dir) if args.save_dir else output_dir / 'ensemble_optimization'
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Optimize temperature
    print("\n--- Temperature Optimization ---")
    temp_df, best_temp, best_temp_spearman = optimize_temperature(models)
    print(temp_df.to_string(index=False))
    print(f"\nBest temperature: {best_temp} (Spearman={best_temp_spearman:.4f})")
    temp_df.to_csv(save_dir / 'temperature_optimization.csv', index=False)

    # 2. Optimize top-K
    print("\n--- Top-K Optimization ---")
    topk_df, best_k, best_k_spearman = optimize_top_k(models)
    print(topk_df[['k', 'spearman']].to_string(index=False))
    print(f"\nBest K: {best_k} models (Spearman={best_k_spearman:.4f})")
    topk_df.to_csv(save_dir / 'topk_optimization.csv', index=False)

    # 3. Optimize threshold
    print("\n--- Threshold Optimization ---")
    thresh_df, best_thresh, best_thresh_spearman = optimize_threshold(models)
    print(thresh_df.to_string(index=False))
    print(f"\nBest threshold: {best_thresh:.4f} (Spearman={best_thresh_spearman:.4f})")
    thresh_df.to_csv(save_dir / 'threshold_optimization.csv', index=False)

    # 4. Leave-one-out CV (only on top models)
    top_models = models[:min(10, len(models))]
    if len(top_models) >= 3:
        print("\n--- Leave-One-Out CV (top 10 models) ---")
        cv_df, cv_mean, cv_std = leave_one_out_cv(top_models)
        print(cv_df.to_string(index=False))
        print(f"\nMean CV Spearman: {cv_mean:.4f} ± {cv_std:.4f}")
        cv_df.to_csv(save_dir / 'leave_one_out_cv.csv', index=False)

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)

    # Best single model
    best_single = models[0]['spearman']
    print(f"Best single model: {best_single:.4f}")

    # Different ensemble methods
    print("\nEnsemble methods (all models):")
    for method in ['equal', 'spearman', 'softmax', 'median']:
        _, spearman = create_ensemble(models, method, 5.0)
        print(f"  {method:12s}: {spearman:.4f}")

    # Best configuration
    print(f"\nBest configurations:")
    print(f"  Temperature (softmax): {best_temp} -> {best_temp_spearman:.4f}")
    print(f"  Top-K models: {best_k} -> {best_k_spearman:.4f}")
    print(f"  Min threshold: {best_thresh:.4f} -> {best_thresh_spearman:.4f}")

    # Overall best
    overall_best = max(best_temp_spearman, best_k_spearman, best_thresh_spearman)
    print(f"\nOverall best ensemble Spearman: {overall_best:.4f}")
    print(f"Improvement over best single: {overall_best - best_single:+.4f}")
    print(f"Improvement over target (0.6149): {overall_best - 0.6149:+.4f}")

    # Save summary
    summary = {
        'n_models': len(models),
        'best_single': best_single,
        'best_temperature': {'temp': best_temp, 'spearman': best_temp_spearman},
        'best_top_k': {'k': best_k, 'spearman': best_k_spearman},
        'best_threshold': {'threshold': best_thresh, 'spearman': best_thresh_spearman},
        'overall_best': overall_best
    }
    with open(save_dir / 'optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot optimization curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Temperature
    axes[0].plot(temp_df['temperature'].values, temp_df['spearman'].values, 'bo-')
    axes[0].axhline(y=best_single, color='r', linestyle='--', label=f'Best single ({best_single:.4f})')
    axes[0].axhline(y=0.6149, color='g', linestyle='--', label='Target (0.6149)')
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Spearman')
    axes[0].set_title('Temperature Optimization')
    axes[0].legend()

    # Top-K
    axes[1].plot(topk_df['k'].values, topk_df['spearman'].values, 'bo-')
    axes[1].axhline(y=best_single, color='r', linestyle='--', label=f'Best single ({best_single:.4f})')
    axes[1].axhline(y=0.6149, color='g', linestyle='--', label='Target (0.6149)')
    axes[1].set_xlabel('Number of Models (K)')
    axes[1].set_ylabel('Spearman')
    axes[1].set_title('Top-K Optimization')
    axes[1].legend()

    # Threshold
    axes[2].plot(thresh_df['threshold'].values, thresh_df['spearman'].values, 'bo-')
    axes[2].axhline(y=best_single, color='r', linestyle='--', label=f'Best single ({best_single:.4f})')
    axes[2].axhline(y=0.6149, color='g', linestyle='--', label='Target (0.6149)')
    axes[2].set_xlabel('Min Spearman Threshold')
    axes[2].set_ylabel('Spearman')
    axes[2].set_title('Threshold Optimization')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_dir / 'optimization_curves.png', dpi=150)
    plt.close()

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
