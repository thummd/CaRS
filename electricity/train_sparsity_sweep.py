"""
Sparsity sweep experiment for electricity price prediction.

Usage:
    python train_sparsity_sweep.py --country DE --lambda_sparse 1.0 --seed 0
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add local modules
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import ElectricityDataset
from fantom_electricity import FANTOMElectricity, create_model, get_default_training_params


def run_experiment(
    country: str,
    lambda_sparse: float,
    lambda_sparse_l2: float = 0.0,
    l2_group_mode: str = "column",
    seed: int = 0,
    output_dir: str = "sparsity_results"
) -> dict:
    """
    Run a single sparsity experiment.

    Args:
        country: 'DE' or 'FR'
        lambda_sparse: L1 sparsity regularization parameter
        lambda_sparse_l2: L2 group sparsity penalty (0 = disabled)
        l2_group_mode: Grouping mode for L2 penalty ('column', 'row', 'lag', 'frobenius')
        seed: Random seed
        output_dir: Directory to save results

    Returns:
        Dictionary with results
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Country: {country}, lambda_sparse: {lambda_sparse}, lambda_sparse_l2: {lambda_sparse_l2}, "
          f"l2_group_mode: {l2_group_mode}, seed: {seed}")

    # Load data
    dataset = ElectricityDataset(country=country, lag=1, standardize=True)

    # Split into train/test
    train_ds, test_ds = dataset.train_test_split(test_ratio=0.2, random=False)

    # Model config
    model_config = {
        'lambda_dag': 100.0,
        'lambda_sparse': lambda_sparse,
        'lambda_sparse_l2': lambda_sparse_l2,
        'l2_group_mode': l2_group_mode,
        'tau_gumbel': 1.0,
        'base_distribution_type': 'spline',
        'spline_bins': 8,
        'norm_layers': True,
        'res_connection': True,
        'encoder_layer_sizes': [32, 32],
        'decoder_layer_sizes': [32, 32],
        'heteroscedastic': True,
        'allow_instantaneous': True,
        'constrain_target': True,
    }

    # Training params
    training_params = get_default_training_params()
    training_params['rho'] = 1.0
    training_params['max_steps_auglag'] = 15
    training_params['max_auglag_inner_epochs'] = 1000

    # Create model
    model = create_model(
        num_nodes=dataset.get_num_nodes(),
        target_idx=dataset.get_target_idx(),
        lag=1,
        device=str(device),
        model_config=model_config
    )

    # Prepare data
    X_train = torch.tensor(train_ds.X, dtype=torch.float32)

    print(f"Training samples: {len(X_train)}")
    print(f"Number of nodes: {dataset.get_num_nodes()}")

    # Train
    from torch.utils.data import DataLoader
    dataloader = DataLoader(X_train, batch_size=training_params['batch_size'], shuffle=True)

    model.train()
    model.run_train(
        dataloader=dataloader,
        num_samples=len(X_train),
        train_config_dict=training_params
    )

    # Evaluate on test set
    X_test = torch.tensor(test_ds.X, dtype=torch.float32)
    y_test = test_ds.target

    metrics = model.evaluate_predictions(X_test, y_test)

    # Get causal structure
    feature_names = dataset.get_feature_names()
    parents = model.get_causal_parents(feature_names, threshold=0.5)
    important = model.get_important_features(feature_names, importance_threshold=0.01)

    # Count edges
    A = model.get_adj_matrix(samples=1, most_likely_graph=True, squeeze=True)
    n_instantaneous_edges = int(np.sum(A[0] > 0.5))
    n_lagged_edges = int(np.sum(A[1:] > 0.5))

    # Results
    results = {
        'config': {
            'country': country,
            'lambda_sparse': lambda_sparse,
            'lambda_sparse_l2': lambda_sparse_l2,
            'l2_group_mode': l2_group_mode,
            'seed': seed,
            'rho': training_params['rho'],
        },
        'metrics': {
            'spearman': float(metrics['spearman']),
            'mse': float(metrics['mse']),
            'mae': float(metrics['mae']),
            'r2': float(metrics['r2']),
        },
        'structure': {
            'n_instantaneous_edges': n_instantaneous_edges,
            'n_lagged_edges': n_lagged_edges,
            'total_edges': n_instantaneous_edges + n_lagged_edges,
            'instantaneous_parents': len(parents['instantaneous']),
            'lagged_parents': len(parents['lagged']),
        },
        'causal_structure': {
            'instantaneous': parents['instantaneous'][:10],  # Top 10
            'lagged': parents['lagged'][:10],  # Top 10
        },
        'timestamp': datetime.now().isoformat(),
    }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build filename with L2 info if L2 is enabled
    if lambda_sparse_l2 > 0:
        filename = f"sparsity_{country}_l1_{lambda_sparse}_l2_{lambda_sparse_l2}_{l2_group_mode}_seed{seed}.json"
    else:
        filename = f"sparsity_{country}_lambda{lambda_sparse}_seed{seed}.json"
    with open(output_path / filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("Results:")
    print(f"  Spearman: {metrics['spearman']:.4f}")
    print(f"  Instantaneous edges: {n_instantaneous_edges}")
    print(f"  Lagged edges: {n_lagged_edges}")
    print(f"  Saved to: {output_path / filename}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Electricity sparsity sweep")
    parser.add_argument("--country", type=str, required=True, choices=['DE', 'FR'])
    parser.add_argument("--lambda_sparse", type=float, required=True,
                        help="L1 sparsity regularization")
    parser.add_argument("--lambda_sparse_l2", type=float, default=0.0,
                        help="L2 group sparsity penalty (0 = disabled)")
    parser.add_argument("--l2_group_mode", type=str, default="column",
                        choices=["column", "row", "lag", "frobenius"],
                        help="Grouping mode for L2 penalty")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="sparsity_results")

    args = parser.parse_args()

    run_experiment(
        country=args.country,
        lambda_sparse=args.lambda_sparse,
        lambda_sparse_l2=args.lambda_sparse_l2,
        l2_group_mode=args.l2_group_mode,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
