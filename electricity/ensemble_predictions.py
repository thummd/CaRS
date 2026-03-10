"""
Ensemble Predictions for DS3M Models.

Combines predictions from multiple trained models to reduce variance
and potentially exceed single-model performance.

Usage:
    python ensemble_predictions.py --output_dir outputs/ds3m --min_spearman 0.3
    python ensemble_predictions.py --model_dirs dir1 dir2 dir3 --weights spearman
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_model_results(model_dir: Path):
    """Load predictions and results from a model directory."""
    pred_path = model_dir / 'predictions.csv'
    results_path = model_dir / 'results.json'

    if not pred_path.exists() or not results_path.exists():
        return None

    pred_df = pd.read_csv(pred_path)
    with open(results_path) as f:
        results = json.load(f)

    return {
        'predictions': pred_df['prediction'].values,
        'original': pred_df['original'].values,
        'regimes': pred_df['regime'].values,
        'spearman': results['metrics']['spearman'],
        'regime_distribution': results.get('regime_distribution', {}),
        'dir': str(model_dir)
    }


def find_models(output_dir: Path, pattern: str = '*', min_spearman: float = None):
    """Find all model directories matching pattern."""
    models = []

    for model_dir in output_dir.glob(pattern):
        if not model_dir.is_dir():
            continue

        result = load_model_results(model_dir)
        if result is None:
            continue

        # Filter by minimum Spearman if specified
        if min_spearman is not None and result['spearman'] < min_spearman:
            continue

        # Skip models that use only 1 regime
        n_regimes = len(result['regime_distribution'])
        if n_regimes < 2:
            continue

        models.append(result)

    return models


def create_ensemble(models, weights='spearman'):
    """
    Create ensemble predictions from multiple models.

    Args:
        models: List of model result dicts
        weights: 'equal', 'spearman', or array of weights

    Returns:
        ensemble_pred: Combined predictions
        spearman: Ensemble Spearman correlation
    """
    if len(models) == 0:
        return None, None

    predictions = np.array([m['predictions'] for m in models])  # (n_models, n_samples)
    spearman_scores = np.array([m['spearman'] for m in models])
    original = models[0]['original']

    print(f"\nCreating ensemble from {len(models)} models")
    print(f"Individual Spearman scores: {spearman_scores}")

    if weights == 'equal':
        ensemble_pred = np.mean(predictions, axis=0)
        print("Using equal weights")

    elif weights == 'spearman':
        # Use Spearman scores as weights (only positive)
        w = np.maximum(spearman_scores, 0)
        if w.sum() == 0:
            w = np.ones(len(w))
        w = w / w.sum()
        ensemble_pred = np.average(predictions, axis=0, weights=w)
        print(f"Using Spearman weights: {w}")

    elif weights == 'softmax':
        # Softmax of Spearman scores
        w = np.exp(spearman_scores * 5)  # Temperature scaling
        w = w / w.sum()
        ensemble_pred = np.average(predictions, axis=0, weights=w)
        print(f"Using softmax weights: {w}")

    elif weights == 'median':
        ensemble_pred = np.median(predictions, axis=0)
        print("Using median ensemble")

    else:
        # Custom weights
        w = np.array(weights)
        w = w / w.sum()
        ensemble_pred = np.average(predictions, axis=0, weights=w)

    # Evaluate ensemble
    spearman, pval = spearmanr(original, ensemble_pred)

    return ensemble_pred, spearman


def analyze_ensemble(models, output_dir: Path = None):
    """Analyze ensemble performance with different weighting schemes."""
    results = {}

    # Try different weighting schemes
    for weight_scheme in ['equal', 'spearman', 'softmax', 'median']:
        pred, spearman = create_ensemble(models, weights=weight_scheme)
        results[weight_scheme] = {
            'spearman': spearman,
            'predictions': pred
        }
        print(f"  {weight_scheme:12s}: Spearman = {spearman:.4f}")

    # Best single model
    best_single = max(models, key=lambda m: m['spearman'])
    results['best_single'] = {
        'spearman': best_single['spearman'],
        'model': best_single['dir']
    }
    print(f"  {'best_single':12s}: Spearman = {best_single['spearman']:.4f} ({Path(best_single['dir']).name})")

    # Find best ensemble
    best_ensemble_key = max(
        [k for k in results if k != 'best_single'],
        key=lambda k: results[k]['spearman']
    )
    best_ensemble_spearman = results[best_ensemble_key]['spearman']

    print(f"\n  Best ensemble: {best_ensemble_key} (Spearman = {best_ensemble_spearman:.4f})")
    improvement = best_ensemble_spearman - best_single['spearman']
    print(f"  Improvement over best single: {improvement:+.4f}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        summary = {
            'n_models': len(models),
            'models': [m['dir'] for m in models],
            'individual_spearman': [m['spearman'] for m in models],
            'ensemble_results': {k: v['spearman'] for k, v in results.items() if 'spearman' in v},
            'best_single': best_single['spearman'],
            'best_ensemble': best_ensemble_spearman,
            'best_ensemble_method': best_ensemble_key,
            'improvement': improvement
        }

        with open(output_dir / 'ensemble_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Save best ensemble predictions
        best_pred = results[best_ensemble_key]['predictions']
        pd.DataFrame({
            'original': models[0]['original'],
            'prediction': best_pred
        }).to_csv(output_dir / 'ensemble_predictions.csv', index=False)

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart of Spearman scores
        methods = list(results.keys())
        spearman_values = [results[m]['spearman'] if 'spearman' in results[m] else results[m]['spearman'] for m in methods]

        ax1 = axes[0]
        colors = ['steelblue'] * (len(methods) - 1) + ['darkorange']  # Orange for best_single
        bars = ax1.bar(methods, spearman_values, color=colors)
        ax1.axhline(y=0.6149, color='red', linestyle='--', label='Target (0.6149)')
        ax1.set_ylabel('Spearman Correlation')
        ax1.set_title('Ensemble Methods Comparison')
        ax1.legend()
        ax1.set_ylim(0, max(spearman_values) * 1.1)

        # Add value labels
        for bar, val in zip(bars, spearman_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        # Scatter plot: predictions vs original
        ax2 = axes[1]
        ax2.scatter(models[0]['original'], best_pred, alpha=0.5, s=20)
        ax2.plot([models[0]['original'].min(), models[0]['original'].max()],
                [models[0]['original'].min(), models[0]['original'].max()],
                'r--', label='Perfect prediction')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Ensemble Prediction')
        ax2.set_title(f'Best Ensemble ({best_ensemble_key}): Spearman = {best_ensemble_spearman:.4f}')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'ensemble_comparison.png', dpi=150)
        plt.close()

        print(f"\nResults saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ensemble DS3M predictions")
    parser.add_argument('--output_dir', type=str, default='outputs/ds3m',
                       help='Directory containing model outputs')
    parser.add_argument('--model_dirs', nargs='+', default=None,
                       help='Specific model directories to ensemble')
    parser.add_argument('--pattern', type=str, default='ALL_*',
                       help='Glob pattern for model directories')
    parser.add_argument('--min_spearman', type=float, default=0.3,
                       help='Minimum Spearman to include model')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save ensemble results')

    args = parser.parse_args()

    print("=" * 60)
    print("ENSEMBLE PREDICTIONS")
    print("=" * 60)

    if args.model_dirs:
        # Load specific models
        models = []
        for dir_path in args.model_dirs:
            result = load_model_results(Path(dir_path))
            if result:
                models.append(result)
    else:
        # Find models matching pattern
        output_dir = Path(args.output_dir)
        models = find_models(output_dir, args.pattern, args.min_spearman)

    print(f"\nFound {len(models)} models with Spearman > {args.min_spearman}")

    if len(models) == 0:
        print("No models found!")
        return

    # Sort by Spearman
    models = sorted(models, key=lambda m: m['spearman'], reverse=True)

    print("\nModels included:")
    for m in models:
        n_regime0 = sum(1 for r in m['regimes'] if r == 0)
        n_regime1 = sum(1 for r in m['regimes'] if r == 1)
        print(f"  {Path(m['dir']).name}: Spearman={m['spearman']:.4f}, Regimes={n_regime0}/{n_regime1}")

    # Analyze ensemble
    save_dir = Path(args.save_dir) if args.save_dir else Path(args.output_dir) / 'ensemble'
    results = analyze_ensemble(models, save_dir)

    print("\n" + "=" * 60)
    print("ENSEMBLE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
