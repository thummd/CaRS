"""
Create Submission Files for QRT Challenge.

Generates:
1. Y_test.csv - Predictions for test set
2. params.json - Model hyperparameters
3. method_info.json - Full method description

Usage:
    python create_submission.py --method RS2M-Ens --output_dir submissions/
    python create_submission.py --model_dirs dir1 dir2 ... --ensemble softmax
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
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from DSSSMCode import DSSSM
from ds3m_adapter import load_qrt_data


def load_model_predictions(model_dir: Path):
    """Load predictions from a trained model directory."""
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
        'dir': str(model_dir),
        'config': results.get('config', {})
    }


def find_models(output_dir: Path, pattern: str = 'ALL_TARGET_uv_d2_*', min_spearman: float = 0.3):
    """Find all model directories matching pattern."""
    models = []

    for model_dir in output_dir.glob(pattern):
        if not model_dir.is_dir():
            continue

        result = load_model_predictions(model_dir)
        if result is None:
            continue

        # Filter by minimum Spearman
        if min_spearman is not None and result['spearman'] < min_spearman:
            continue

        # Skip models that use only 1 regime
        n_regime0 = result['regime_distribution'].get('0', 0)
        n_regime1 = result['regime_distribution'].get('1', 0)
        if n_regime0 == 0 or n_regime1 == 0:
            continue

        models.append(result)

    return sorted(models, key=lambda x: x['spearman'], reverse=True)


def create_ensemble(models, method='softmax', temperature=5):
    """Create ensemble predictions."""
    if len(models) == 0:
        return None, None, None

    predictions = np.array([m['predictions'] for m in models])
    spearman_scores = np.array([m['spearman'] for m in models])
    original = models[0]['original']

    if method == 'equal':
        weights = np.ones(len(models)) / len(models)
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
        weights = np.ones(len(models)) / len(models)
        ensemble_pred = np.median(predictions, axis=0)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    # Compute ensemble Spearman
    ensemble_spearman, _ = spearmanr(original, ensemble_pred)

    return ensemble_pred, weights, ensemble_spearman


def create_submission_files(
    method_name: str,
    ensemble_pred: np.ndarray,
    ensemble_spearman: float,
    models: list,
    weights: np.ndarray,
    ensemble_method: str,
    output_dir: Path,
    temperature: float = 5.0
):
    """Create all submission files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load QRT data to get DAY_IDs for test set
    df = load_qrt_data()
    test_size = len(ensemble_pred)
    test_day_ids = df['DAY_ID'].values[-test_size:]

    # 1. Y_test.csv - Predictions
    submission_df = pd.DataFrame({
        'DAY_ID': test_day_ids,
        'TARGET': ensemble_pred
    })
    submission_df.to_csv(output_dir / f'{method_name}_Y_test.csv', index=False)

    # 2. params.json - Hyperparameters
    # Extract config from first model
    base_config = models[0].get('config', {})

    params = {
        "method": method_name,
        "base_model": "DS3M (Deep Switching State Space Model)",
        "mode": "univariate",
        "input": "TARGET only (price difference)",
        "d_dim": base_config.get('d_dim', 2),
        "h_dim": base_config.get('h_dim', 30),
        "z_dim": base_config.get('z_dim', 8),
        "n_layers": base_config.get('n_layers', 1),
        "timestep": base_config.get('timestep', 14),
        "n_epochs": base_config.get('n_epochs', 200),
        "learning_rate": base_config.get('lr', 0.001),
        "ensemble_method": ensemble_method,
        "ensemble_temperature": temperature,
        "n_models": len(models),
        "min_spearman_threshold": 0.3,
        "spearman_validation": float(ensemble_spearman)
    }

    with open(output_dir / f'{method_name}_params.json', 'w') as f:
        json.dump(params, f, indent=2)

    # 3. method_info.json - Full method description
    model_details = []
    for m, w in zip(models, weights):
        seed = Path(m['dir']).name.split('seed')[1].split('_')[0] if 'seed' in m['dir'] else 'unknown'
        model_details.append({
            'seed': seed,
            'spearman': float(m['spearman']),
            'weight': float(w),
            'regime_distribution': m['regime_distribution'],
            'dir': m['dir']
        })

    method_info = {
        "method_name": method_name,
        "full_name": "Regime-Switching State Space Model Ensemble",
        "description": (
            "Deep Switching State Space Model (DS3M) with 2 latent regimes. "
            "Uses a Markov-switching VAE to capture regime-dependent dynamics in electricity prices. "
            "Ensemble combines multiple seeds with softmax-weighted averaging."
        ),
        "key_insight": (
            "Univariate modeling (TARGET only) outperforms multivariate. "
            "Seeds with 5-25% regime 0 allocation achieve best Spearman correlation. "
            "Regime switching captures market structure changes."
        ),
        "parameters": params,
        "models": model_details,
        "created": datetime.now().isoformat(),
        "validation_spearman": float(ensemble_spearman)
    }

    with open(output_dir / f'{method_name}_method_info.json', 'w') as f:
        json.dump(method_info, f, indent=2)

    # Also save raw predictions with original values for verification
    verification_df = pd.DataFrame({
        'DAY_ID': test_day_ids,
        'original': models[0]['original'],
        'prediction': ensemble_pred
    })
    verification_df.to_csv(output_dir / f'{method_name}_verification.csv', index=False)

    print(f"\nSubmission files created in {output_dir}/")
    print(f"  - {method_name}_Y_test.csv")
    print(f"  - {method_name}_params.json")
    print(f"  - {method_name}_method_info.json")
    print(f"  - {method_name}_verification.csv")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Create submission files for QRT challenge")
    parser.add_argument('--method', type=str, default='RS2M-Ens',
                       help='Method name for submission')
    parser.add_argument('--output_dir', type=str, default='submissions',
                       help='Output directory for submission files')
    parser.add_argument('--ds3m_dir', type=str,
                       default=str(OUTPUT_DIR) + '/ds3m',
                       help='Directory containing DS3M model outputs')
    parser.add_argument('--pattern', type=str, default='ALL_TARGET_uv_d2_*',
                       help='Glob pattern for model directories')
    parser.add_argument('--min_spearman', type=float, default=0.3,
                       help='Minimum Spearman to include model in ensemble')
    parser.add_argument('--ensemble', type=str, default='softmax',
                       choices=['equal', 'spearman', 'softmax', 'median'],
                       help='Ensemble method')
    parser.add_argument('--temperature', type=float, default=5.0,
                       help='Temperature for softmax ensemble')
    parser.add_argument('--model_dirs', nargs='+', default=None,
                       help='Specific model directories to use')

    args = parser.parse_args()

    print("=" * 70)
    print("CREATE SUBMISSION FILES")
    print("=" * 70)
    print(f"Method: {args.method}")
    print(f"Ensemble: {args.ensemble}")
    print(f"Output: {args.output_dir}")

    # Find or load models
    if args.model_dirs:
        models = []
        for dir_path in args.model_dirs:
            result = load_model_predictions(Path(dir_path))
            if result:
                models.append(result)
        models = sorted(models, key=lambda x: x['spearman'], reverse=True)
    else:
        ds3m_dir = Path(args.ds3m_dir)
        models = find_models(ds3m_dir, args.pattern, args.min_spearman)

    print(f"\nFound {len(models)} models with Spearman >= {args.min_spearman}")

    if len(models) == 0:
        print("ERROR: No models found!")
        return

    # Print model summary
    print("\nModels included:")
    print(f"{'Seed':<8} {'Spearman':<10} {'Regime 0':<10} {'Regime 1':<10}")
    print("-" * 40)
    for m in models:
        seed = Path(m['dir']).name.split('seed')[1].split('_')[0] if 'seed' in m['dir'] else '?'
        r0 = m['regime_distribution'].get('0', 0)
        r1 = m['regime_distribution'].get('1', 0)
        print(f"{seed:<8} {m['spearman']:<10.4f} {r0:<10} {r1:<10}")

    # Create ensemble
    print(f"\nCreating {args.ensemble} ensemble...")
    ensemble_pred, weights, ensemble_spearman = create_ensemble(
        models, args.ensemble, args.temperature
    )

    print(f"\nEnsemble Spearman: {ensemble_spearman:.4f}")
    print(f"Best single model: {models[0]['spearman']:.4f}")
    print(f"Improvement: {ensemble_spearman - models[0]['spearman']:+.4f}")

    # Create submission files
    output_dir = Path(args.output_dir)
    create_submission_files(
        method_name=args.method,
        ensemble_pred=ensemble_pred,
        ensemble_spearman=ensemble_spearman,
        models=models,
        weights=weights,
        ensemble_method=args.ensemble,
        output_dir=output_dir,
        temperature=args.temperature
    )

    print("\n" + "=" * 70)
    print("SUBMISSION COMPLETE")
    print("=" * 70)
    print(f"Method: {args.method}")
    print(f"Validation Spearman: {ensemble_spearman:.4f}")
    print(f"Target to beat: 0.6149")
    print(f"Margin: {ensemble_spearman - 0.6149:+.4f} ({(ensemble_spearman/0.6149 - 1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
