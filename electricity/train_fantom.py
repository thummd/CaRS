"""
Training script for FANTOM electricity price model.

Usage:
    python train_fantom.py --experiment joint
    python train_fantom.py --experiment germany
    python train_fantom.py --experiment france
    python train_fantom.py --experiment joint_instantaneous
"""

import argparse
import os
import sys
import yaml
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add local modules
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import ElectricityDataset
from fantom_electricity import FANTOMElectricity, create_model, get_default_training_params


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_dir(base_dir: str, experiment_name: str) -> Path:
    """Create output directory for experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def train_model(
    dataset: ElectricityDataset,
    model_config: Dict,
    training_params: Dict,
    device: torch.device,
    output_dir: Optional[Path] = None
) -> FANTOMElectricity:
    """
    Train FANTOM model on electricity data.

    Args:
        dataset: ElectricityDataset instance
        model_config: Model configuration dictionary
        training_params: Training parameters dictionary
        device: PyTorch device
        output_dir: Optional directory to save outputs

    Returns:
        Trained FANTOMElectricity model
    """
    print("\n" + "="*60)
    print("Creating FANTOM model...")
    print("="*60)

    # Get data dimensions
    num_nodes = dataset.get_num_nodes()
    target_idx = dataset.get_target_idx()
    lag = dataset.X.shape[1] - 1  # Infer lag from data shape

    print(f"Number of nodes: {num_nodes}")
    print(f"Target index: {target_idx}")
    print(f"Lag: {lag}")
    print(f"Training samples: {len(dataset.X)}")

    # Create model
    model = create_model(
        num_nodes=num_nodes,
        target_idx=target_idx,
        lag=lag,
        device=str(device),
        model_config=model_config
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare data
    X = torch.tensor(dataset.X, dtype=torch.float32)
    dataloader = DataLoader(
        X,
        batch_size=training_params['batch_size'],
        shuffle=True
    )

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    # Train model
    model.train()
    model.run_train(
        dataloader=dataloader,
        num_samples=len(X),
        train_config_dict=training_params
    )

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    # Save model if output_dir provided
    if output_dir:
        model_path = output_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    return model


def evaluate_model(
    model: FANTOMElectricity,
    dataset: ElectricityDataset,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluate trained model.

    Args:
        model: Trained FANTOMElectricity model
        dataset: Dataset to evaluate on
        output_dir: Optional directory to save results

    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("Evaluating model...")
    print("="*60)

    X = torch.tensor(dataset.X, dtype=torch.float32)
    y_true = dataset.target

    # Get metrics
    metrics = model.evaluate_predictions(X, y_true)

    print(f"\nEvaluation Results:")
    print(f"  Spearman correlation: {metrics['spearman']:.4f} (p={metrics['spearman_pval']:.4e})")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")

    # Get causal parents
    feature_names = dataset.get_feature_names()
    parents = model.get_causal_parents(feature_names, threshold=0.5)

    print(f"\nCausal parents of TARGET (threshold=0.5):")
    print(f"  Instantaneous ({len(parents['instantaneous'])} edges):")
    for name, lag, weight in parents['instantaneous'][:10]:  # Top 10
        print(f"    {name}: weight={weight:.4f}")

    print(f"  Lagged ({len(parents['lagged'])} edges):")
    for name, lag, weight in parents['lagged'][:10]:  # Top 10
        print(f"    {name} (lag={lag}): weight={weight:.4f}")

    # Save results
    if output_dir:
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) if not np.isnan(v) else None
                      for k, v in metrics.items()}, f, indent=2)

        # Save causal structure
        structure_path = output_dir / "causal_structure.json"
        with open(structure_path, 'w') as f:
            json.dump({
                'instantaneous': [(n, l, float(w)) for n, l, w in parents['instantaneous']],
                'lagged': [(n, l, float(w)) for n, l, w in parents['lagged']]
            }, f, indent=2)

        # Save adjacency matrix
        adj_matrix = model.get_adj_matrix(samples=1, most_likely_graph=True, squeeze=True)
        np.save(output_dir / "adjacency_matrix.npy", adj_matrix)

        # Save predictions
        predictions = model.predict_target(X).cpu().numpy()
        np.save(output_dir / "predictions.npy", predictions)
        np.save(output_dir / "y_true.npy", y_true)
        np.save(output_dir / "ids.npy", dataset.ids)

        print(f"\nResults saved to: {output_dir}")

    return metrics


def run_experiment(
    experiment_name: str,
    config: Dict,
    device: torch.device
) -> Dict[str, Any]:
    """
    Run a single experiment.

    Args:
        experiment_name: Name of experiment from config
        config: Full configuration dictionary
        device: PyTorch device

    Returns:
        Dictionary with experiment results
    """
    print("\n" + "#"*60)
    print(f"# Running experiment: {experiment_name}")
    print("#"*60)

    # Get experiment-specific config
    exp_config = config['experiments'].get(experiment_name, {})

    # Dataset config
    dataset_kwargs = {
        'country': exp_config.get('country', None),
        'lag': exp_config.get('lag', config['dataset_config']['lag']),
        'imputation': config['dataset_config']['imputation'],
        'use_temporal': exp_config.get('use_temporal', config['dataset_config']['use_temporal']),
        'standardize': config['dataset_config']['standardize'],
    }

    print(f"\nDataset configuration:")
    for k, v in dataset_kwargs.items():
        print(f"  {k}: {v}")

    # Load dataset
    dataset = ElectricityDataset(**dataset_kwargs)

    print(f"\nData shape: {dataset.X.shape}")
    print(f"Features: {dataset.get_feature_names()}")

    # Setup output directory
    output_dir = setup_output_dir(
        config['output']['save_dir'],
        experiment_name
    )

    # Save config
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump({
            'experiment': experiment_name,
            'dataset_kwargs': dataset_kwargs,
            'model_config': config['model_config'],
            'training_params': config['training_params']
        }, f)

    # Train model
    model = train_model(
        dataset=dataset,
        model_config=config['model_config'],
        training_params=config['training_params'],
        device=device,
        output_dir=output_dir
    )

    # Evaluate
    metrics = evaluate_model(
        model=model,
        dataset=dataset,
        output_dir=output_dir
    )

    return {
        'experiment': experiment_name,
        'metrics': metrics,
        'output_dir': str(output_dir)
    }


def main():
    parser = argparse.ArgumentParser(description="Train FANTOM electricity model")
    parser.add_argument(
        '--experiment',
        type=str,
        default='joint',
        choices=['joint', 'germany', 'france', 'joint_instantaneous', 'all'],
        help='Experiment to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu or cuda)'
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    Path(config['output']['save_dir']).mkdir(parents=True, exist_ok=True)

    # Run experiment(s)
    if args.experiment == 'all':
        experiments = ['joint', 'germany', 'france', 'joint_instantaneous']
    else:
        experiments = [args.experiment]

    all_results = []
    for exp_name in experiments:
        try:
            results = run_experiment(exp_name, config, device)
            all_results.append(results)
        except Exception as e:
            print(f"Error in experiment {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for result in all_results:
        metrics = result['metrics']
        print(f"\n{result['experiment']}:")
        print(f"  Spearman: {metrics['spearman']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  Output: {result['output_dir']}")


if __name__ == "__main__":
    main()
