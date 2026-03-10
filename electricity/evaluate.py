"""
Evaluation and visualization script for trained FANTOM electricity models.

Usage:
    python evaluate.py --model_dir outputs/joint_20240115_120000
    python evaluate.py --compare outputs/joint_* outputs/germany_*
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import ElectricityDataset, load_raw_data, merge_data, impute_missing
from fantom_electricity import FANTOMElectricity, create_model


def load_model_from_dir(model_dir: Path, device: torch.device) -> Tuple[FANTOMElectricity, Dict]:
    """Load a trained model from directory."""
    import yaml

    # Load config
    config_path = model_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Recreate dataset to get dimensions
    dataset = ElectricityDataset(**config['dataset_kwargs'])

    # Create model
    model = create_model(
        num_nodes=dataset.get_num_nodes(),
        target_idx=dataset.get_target_idx(),
        lag=dataset.X.shape[1] - 1,
        device=str(device),
        model_config=config['model_config']
    )

    # Load weights
    model_path = model_dir / "model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, config


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Predictions vs True Values"
):
    """Plot predicted vs true values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            'r--', label='Perfect prediction')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{title}\nSpearman: {spearmanr(y_true, y_pred)[0]:.4f}')
    ax.legend()

    # Residuals
    ax = axes[1]
    residuals = y_pred - y_true
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_xlabel('Residual (Predicted - True)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residual Distribution\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved predictions plot to: {save_path}")
    plt.close()


def plot_adjacency_matrix(
    adj_matrix: np.ndarray,
    feature_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "Learned Causal Structure"
):
    """Plot adjacency matrix as heatmap."""
    # For temporal adjacency [lag+1, num_nodes, num_nodes],
    # we plot each lag level separately

    n_lags = adj_matrix.shape[0]
    fig, axes = plt.subplots(1, n_lags, figsize=(8*n_lags, 7))

    if n_lags == 1:
        axes = [axes]

    for lag_idx, ax in enumerate(axes):
        lag_name = "Instantaneous" if lag_idx == 0 else f"Lag {lag_idx}"

        sns.heatmap(
            adj_matrix[lag_idx],
            ax=ax,
            xticklabels=feature_names,
            yticklabels=feature_names,
            cmap='Blues',
            vmin=0, vmax=1,
            annot=False,
            square=True
        )
        ax.set_title(f'{title}\n{lag_name}')
        ax.set_xlabel('To')
        ax.set_ylabel('From')

        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved adjacency plot to: {save_path}")
    plt.close()


def plot_target_parents(
    parents: Dict,
    save_path: Optional[Path] = None,
    title: str = "Causal Parents of TARGET",
    top_n: int = 15
):
    """Plot bar chart of TARGET's causal parents."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (parent_type, ax) in enumerate(zip(['instantaneous', 'lagged'], axes)):
        parent_list = parents[parent_type][:top_n]

        if not parent_list:
            ax.text(0.5, 0.5, 'No parents found', ha='center', va='center')
            ax.set_title(f'{parent_type.capitalize()} Parents')
            continue

        names = [f"{p[0]} (lag={p[1]})" if p[1] > 0 else p[0] for p in parent_list]
        weights = [p[2] for p in parent_list]
        colors = ['green' if w > 0 else 'red' for w in weights]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, weights, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Edge Weight')
        ax.set_title(f'{parent_type.capitalize()} Parents (top {len(parent_list)})')
        ax.invert_yaxis()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved parents plot to: {save_path}")
    plt.close()


def compare_models(
    model_dirs: List[Path],
    device: torch.device,
    output_dir: Optional[Path] = None
):
    """Compare multiple trained models."""
    results = []

    for model_dir in model_dirs:
        model_dir = Path(model_dir)

        # Load metrics
        metrics_path = model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}

        # Get experiment name from directory
        exp_name = model_dir.name.rsplit('_', 2)[0]  # Remove timestamp

        results.append({
            'experiment': exp_name,
            'directory': str(model_dir),
            **metrics
        })

    # Create comparison DataFrame
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(df.to_string(index=False))

    if output_dir:
        df.to_csv(output_dir / "comparison.csv", index=False)

    # Plot comparison
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(results))
        metrics_to_plot = ['spearman', 'r2']

        width = 0.35
        for i, metric in enumerate(metrics_to_plot):
            values = [r.get(metric, 0) for r in results]
            ax.bar(x + i*width, values, width, label=metric.capitalize())

        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels([r['experiment'] for r in results], rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()

        if output_dir:
            plt.savefig(output_dir / "comparison.png", dpi=150)
        plt.close()

    return results


def generate_visualizations(model_dir: Path, device: torch.device):
    """Generate all visualizations for a trained model."""
    print(f"\nGenerating visualizations for: {model_dir}")

    import yaml

    # Load config and data
    config_path = model_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset = ElectricityDataset(**config['dataset_kwargs'])
    feature_names = dataset.get_feature_names()

    # Load saved predictions and adjacency
    predictions = np.load(model_dir / "predictions.npy")
    y_true = np.load(model_dir / "y_true.npy")
    adj_matrix = np.load(model_dir / "adjacency_matrix.npy")

    # Load causal structure
    with open(model_dir / "causal_structure.json", 'r') as f:
        parents = json.load(f)

    viz_dir = model_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_predictions(
        y_true, predictions,
        save_path=viz_dir / "predictions.png",
        title=f"FANTOM Predictions - {config['experiment']}"
    )

    plot_adjacency_matrix(
        adj_matrix, feature_names,
        save_path=viz_dir / "adjacency.png",
        title=f"Learned Causal Structure - {config['experiment']}"
    )

    plot_target_parents(
        parents,
        save_path=viz_dir / "target_parents.png",
        title=f"Causal Parents of TARGET - {config['experiment']}"
    )

    print(f"Visualizations saved to: {viz_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FANTOM electricity models")
    parser.add_argument(
        '--model_dir',
        type=str,
        help='Directory containing trained model'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        type=str,
        help='Model directories to compare'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use'
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    if args.compare:
        # Compare multiple models
        import glob
        model_dirs = []
        for pattern in args.compare:
            model_dirs.extend(glob.glob(pattern))
        model_dirs = [Path(d) for d in model_dirs if Path(d).is_dir()]

        output_dir = Path(args.output_dir) if args.output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        compare_models(model_dirs, device, output_dir)

    elif args.model_dir:
        model_dir = Path(args.model_dir)

        if args.visualize:
            generate_visualizations(model_dir, device)
        else:
            # Just print metrics
            metrics_path = model_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                print("\nMetrics:")
                for k, v in metrics.items():
                    print(f"  {k}: {v}")
    else:
        print("Please specify --model_dir or --compare")
        parser.print_help()


if __name__ == "__main__":
    main()
