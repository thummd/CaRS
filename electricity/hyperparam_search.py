"""
Hyperparameter search for FANTOM electricity model.

Uses temporal cross-validation and grid search to find optimal parameters.

Usage:
    python hyperparam_search.py --country DE --n_trials 20
    python hyperparam_search.py --country FR --n_trials 20
    python hyperparam_search.py --country DE --full_grid  # Full grid search
"""

import argparse
import os
import sys
import json
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from itertools import product
import random
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import ElectricityDataset
from fantom_electricity import FANTOMElectricity, create_model


# Hyperparameter search space
PARAM_GRID = {
    'lambda_sparse': [0.1, 1.0, 5.0, 10.0],
    'lambda_dag': [50.0, 100.0, 200.0],
    'encoder_layer_sizes': [[16, 16], [32, 32], [64, 64]],
    'spline_bins': [4, 8, 16],
    'learning_rate': [0.0005, 0.001, 0.002],
    'batch_size': [16, 32, 64],
}

# Reduced grid for quick experiments
QUICK_PARAM_GRID = {
    'lambda_sparse': [0.5, 1.0, 5.0],
    'lambda_dag': [100.0],
    'encoder_layer_sizes': [[32, 32]],
    'spline_bins': [8],
    'learning_rate': [0.001],
    'batch_size': [32],
}


class TemporalCV:
    """
    Temporal cross-validation splitter.

    For time series data, we train on earlier periods and validate on later.
    """

    def __init__(self, n_splits: int = 5, test_ratio: float = 0.2):
        self.n_splits = n_splits
        self.test_ratio = test_ratio

    def split(self, X: np.ndarray, day_ids: np.ndarray):
        """
        Generate train/val indices for temporal CV.

        Args:
            X: Data array
            day_ids: Day identifiers for temporal ordering

        Yields:
            train_idx, val_idx for each fold
        """
        n = len(X)
        sorted_idx = np.argsort(day_ids)

        # For temporal CV, we use expanding window
        val_size = int(n * self.test_ratio)

        for fold in range(self.n_splits):
            # Calculate split point for this fold
            split_point = int(n * (0.5 + fold * 0.1))  # Start at 50%, expand
            split_point = min(split_point, n - val_size)

            train_idx = sorted_idx[:split_point]
            val_idx = sorted_idx[split_point:split_point + val_size]

            yield train_idx, val_idx


def train_and_evaluate(
    train_data: np.ndarray,
    val_data: np.ndarray,
    train_target: np.ndarray,
    val_target: np.ndarray,
    num_nodes: int,
    target_idx: int,
    lag: int,
    model_config: Dict,
    training_params: Dict,
    device: torch.device,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Train model and evaluate on validation set.

    Returns:
        Dictionary with validation metrics
    """
    # Create model
    model = create_model(
        num_nodes=num_nodes,
        target_idx=target_idx,
        lag=lag,
        device=str(device),
        model_config=model_config
    )

    # Create dataloader
    X_train = torch.tensor(train_data, dtype=torch.float32)
    dataloader = DataLoader(X_train, batch_size=training_params['batch_size'], shuffle=True)

    # Train
    model.train()

    # Reduce verbosity during hyperparam search
    original_print = print
    if not verbose:
        import builtins
        builtins.print = lambda *args, **kwargs: None

    try:
        model.run_train(
            dataloader=dataloader,
            num_samples=len(X_train),
            train_config_dict=training_params
        )
    finally:
        if not verbose:
            import builtins
            builtins.print = original_print

    # Evaluate
    model.eval()
    X_val = torch.tensor(val_data, dtype=torch.float32)
    metrics = model.evaluate_predictions(X_val, val_target)

    return metrics


def run_cv_experiment(
    dataset: ElectricityDataset,
    model_config: Dict,
    training_params: Dict,
    device: torch.device,
    n_folds: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run cross-validation experiment with given hyperparameters.

    Returns:
        Dictionary with mean and std of metrics across folds
    """
    cv = TemporalCV(n_splits=n_folds, test_ratio=0.2)

    fold_metrics = []

    # Get day_ids for temporal ordering
    day_ids = dataset.df['DAY_ID'].values[dataset.lag:]

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(dataset.X, day_ids)):
        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_folds}: train={len(train_idx)}, val={len(val_idx)}")

        train_data = dataset.X[train_idx]
        val_data = dataset.X[val_idx]
        train_target = dataset.target[train_idx]
        val_target = dataset.target[val_idx]

        try:
            metrics = train_and_evaluate(
                train_data=train_data,
                val_data=val_data,
                train_target=train_target,
                val_target=val_target,
                num_nodes=dataset.get_num_nodes(),
                target_idx=dataset.get_target_idx(),
                lag=dataset.X.shape[1] - 1,
                model_config=model_config,
                training_params=training_params,
                device=device,
                verbose=verbose
            )
            fold_metrics.append(metrics)
        except Exception as e:
            print(f"  Fold {fold_idx + 1} failed: {e}")
            fold_metrics.append({'spearman': np.nan, 'mse': np.nan, 'mae': np.nan, 'r2': np.nan})

    # Aggregate metrics
    result = {
        'spearman_mean': np.nanmean([m['spearman'] for m in fold_metrics]),
        'spearman_std': np.nanstd([m['spearman'] for m in fold_metrics]),
        'mse_mean': np.nanmean([m['mse'] for m in fold_metrics]),
        'r2_mean': np.nanmean([m['r2'] for m in fold_metrics]),
        'fold_metrics': fold_metrics,
        'n_successful_folds': sum(1 for m in fold_metrics if not np.isnan(m['spearman']))
    }

    return result


def grid_search(
    country: str,
    param_grid: Dict,
    n_folds: int = 3,
    device: torch.device = None,
    output_dir: Path = None,
    verbose: bool = False
) -> List[Dict]:
    """
    Perform grid search over hyperparameters.

    Args:
        country: 'DE' or 'FR'
        param_grid: Dictionary of hyperparameter lists
        n_folds: Number of CV folds
        device: PyTorch device
        output_dir: Directory to save results

    Returns:
        List of experiment results sorted by validation Spearman
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Grid Search for {country}")
    print(f"{'='*60}")

    # Load base config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Load dataset
    dataset = ElectricityDataset(country=country, lag=1, imputation='mean')
    print(f"Dataset: {dataset.X.shape}")

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    print(f"Total configurations: {len(combinations)}")

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))

        print(f"\nConfig {i+1}/{len(combinations)}: {params}")

        # Build model config
        model_config = base_config['model_config'].copy()
        for key in ['lambda_sparse', 'lambda_dag', 'encoder_layer_sizes', 'spline_bins']:
            if key in params:
                model_config[key] = params[key]
                if key == 'encoder_layer_sizes':
                    model_config['decoder_layer_sizes'] = params[key]

        # Build training params
        training_params = base_config['training_params'].copy()
        for key in ['learning_rate', 'batch_size']:
            if key in params:
                training_params[key] = params[key]

        # Reduce training for hyperparameter search
        training_params['max_steps_auglag'] = 8
        training_params['max_auglag_inner_epochs'] = 800

        # Run CV
        cv_result = run_cv_experiment(
            dataset=dataset,
            model_config=model_config,
            training_params=training_params,
            device=device,
            n_folds=n_folds,
            verbose=verbose
        )

        result = {
            'params': params,
            'model_config': model_config,
            'training_params': training_params,
            **cv_result
        }
        results.append(result)

        print(f"  Spearman: {cv_result['spearman_mean']:.4f} +/- {cv_result['spearman_std']:.4f}")

        # Save intermediate results
        if output_dir:
            with open(output_dir / "results_partial.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)

    # Sort by validation Spearman (descending)
    results.sort(key=lambda x: x['spearman_mean'], reverse=True)

    return results


def random_search(
    country: str,
    param_grid: Dict,
    n_trials: int = 20,
    n_folds: int = 3,
    device: torch.device = None,
    output_dir: Path = None,
    verbose: bool = False
) -> List[Dict]:
    """
    Perform random search over hyperparameters.

    More efficient than grid search for large parameter spaces.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Random Search for {country} ({n_trials} trials)")
    print(f"{'='*60}")

    # Load base config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Load dataset
    dataset = ElectricityDataset(country=country, lag=1, imputation='mean')
    print(f"Dataset: {dataset.X.shape}")

    results = []

    for i in range(n_trials):
        # Sample random parameters
        params = {key: random.choice(values) for key, values in param_grid.items()}

        print(f"\nTrial {i+1}/{n_trials}: {params}")

        # Build model config
        model_config = base_config['model_config'].copy()
        for key in ['lambda_sparse', 'lambda_dag', 'encoder_layer_sizes', 'spline_bins']:
            if key in params:
                model_config[key] = params[key]
                if key == 'encoder_layer_sizes':
                    model_config['decoder_layer_sizes'] = params[key]

        # Build training params
        training_params = base_config['training_params'].copy()
        for key in ['learning_rate', 'batch_size']:
            if key in params:
                training_params[key] = params[key]

        # Reduce training for hyperparameter search
        training_params['max_steps_auglag'] = 8
        training_params['max_auglag_inner_epochs'] = 800

        # Run CV
        cv_result = run_cv_experiment(
            dataset=dataset,
            model_config=model_config,
            training_params=training_params,
            device=device,
            n_folds=n_folds,
            verbose=verbose
        )

        result = {
            'trial': i + 1,
            'params': params,
            'model_config': model_config,
            'training_params': training_params,
            **cv_result
        }
        results.append(result)

        print(f"  Spearman: {cv_result['spearman_mean']:.4f} +/- {cv_result['spearman_std']:.4f}")

        # Save intermediate results
        if output_dir:
            with open(output_dir / "results_partial.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)

    # Sort by validation Spearman (descending)
    results.sort(key=lambda x: x['spearman_mean'], reverse=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for FANTOM electricity model")
    parser.add_argument('--country', type=str, required=True, choices=['DE', 'FR'],
                        help='Country to optimize for')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of random search trials')
    parser.add_argument('--n_folds', type=int, default=3,
                        help='Number of CV folds')
    parser.add_argument('--full_grid', action='store_true',
                        help='Use full grid search instead of random search')
    parser.add_argument('--quick', action='store_true',
                        help='Use reduced parameter grid for quick testing')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed training output')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "hyperparam_results" / f"{args.country}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Select parameter grid
    if args.quick:
        param_grid = QUICK_PARAM_GRID
    else:
        param_grid = PARAM_GRID

    # Run search
    if args.full_grid:
        results = grid_search(
            country=args.country,
            param_grid=param_grid,
            n_folds=args.n_folds,
            device=device,
            output_dir=output_dir,
            verbose=args.verbose
        )
    else:
        results = random_search(
            country=args.country,
            param_grid=param_grid,
            n_trials=args.n_trials,
            n_folds=args.n_folds,
            device=device,
            output_dir=output_dir,
            verbose=args.verbose
        )

    # Save final results
    with open(output_dir / "results_final.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*60)
    print(f"\nTop 5 configurations for {args.country}:")
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. Spearman: {result['spearman_mean']:.4f} +/- {result['spearman_std']:.4f}")
        print(f"   Params: {result['params']}")

    # Save best config
    best_config = {
        'country': args.country,
        'best_params': results[0]['params'],
        'best_model_config': results[0]['model_config'],
        'best_training_params': results[0]['training_params'],
        'best_spearman': results[0]['spearman_mean'],
    }
    with open(output_dir / "best_config.json", 'w') as f:
        json.dump(best_config, f, indent=2, default=str)

    print(f"\nBest configuration saved to: {output_dir / 'best_config.json'}")


if __name__ == "__main__":
    main()
