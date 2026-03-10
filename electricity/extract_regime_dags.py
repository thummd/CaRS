#!/usr/bin/env python3
"""
Extract and visualize per-regime DAGs from DS3M Causal model.

This script trains a DS3M Causal model and saves:
1. Adjacency matrices for each regime (as .npy files)
2. Causal graph visualizations (as .png files)
3. Feature importance rankings per regime

Usage:
    python extract_regime_dags.py --dataset DE --d_dim 3 --seed 42
    python extract_regime_dags.py --dataset FR --d_dim 4 --seed 42
    python extract_regime_dags.py --dataset DE_FR --d_dim 2 --seed 42
"""

import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
from paths import DS3M_DIR, FANTOM_CODE_DIR
FANTOM_PATH = str(FANTOM_CODE_DIR)
DS3M_PATH = str(DS3M_DIR)
sys.path.insert(0, FANTOM_PATH)
sys.path.insert(0, DS3M_PATH)
sys.path.insert(0, os.path.join(DS3M_PATH, "src"))


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_causal_graph(
    adj_matrix: np.ndarray,
    feature_names: list,
    target_idx: int,
    title: str,
    save_path: Path,
    threshold: float = 0.3
):
    """
    Plot causal graph from adjacency matrix.

    Args:
        adj_matrix: Adjacency matrix [num_nodes, num_nodes] or [lag+1, num_nodes, num_nodes]
        feature_names: List of feature names
        target_idx: Index of target variable
        title: Plot title
        save_path: Path to save figure
        threshold: Edge threshold for visualization
    """
    # Use instantaneous effects if temporal
    if adj_matrix.ndim == 3:
        adj = adj_matrix[0]  # Instantaneous
    else:
        adj = adj_matrix

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for i, name in enumerate(feature_names):
        G.add_node(i, label=name)

    # Add edges above threshold
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] > threshold:
                G.add_edge(i, j, weight=adj[i, j])

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes
    node_colors = ['lightcoral' if i == target_idx else 'lightblue' for i in range(n)]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)

    # Draw edges with width based on weight
    edges = G.edges(data=True)
    if edges:
        weights = [d['weight'] * 3 for _, _, d in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7,
                               edge_color='gray', arrows=True,
                               arrowsize=15, ax=ax)

    # Draw labels
    labels = {i: name[:15] for i, name in enumerate(feature_names)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    ax.set_title(title, fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved graph: {save_path}")


def get_top_parents(adj_matrix: np.ndarray, target_idx: int, feature_names: list, top_k: int = 10):
    """Get top-k parents of target variable."""
    if adj_matrix.ndim == 3:
        adj = adj_matrix[0]  # Instantaneous
    else:
        adj = adj_matrix

    # Get incoming edges to target
    incoming = adj[:, target_idx]

    # Sort by weight
    sorted_idx = np.argsort(incoming)[::-1]

    parents = []
    for i in sorted_idx[:top_k]:
        if incoming[i] > 0.1:  # Only include significant edges
            parents.append({
                'feature': feature_names[i],
                'weight': float(incoming[i])
            })

    return parents


def get_top_parents_by_W(W_matrix: np.ndarray, target_idx: int, feature_names: list, top_k: int = 10):
    """
    Get top-k parents of target variable based on ICGNN W weights.

    Unlike edge probabilities (A), W weights can be negative (inhibitory).
    We rank by absolute value to find most influential parents.
    """
    if W_matrix.ndim == 3:
        # Combine instantaneous and lagged weights
        W_inst = W_matrix[0]  # Instantaneous
        W_lag = W_matrix[1] if W_matrix.shape[0] > 1 else np.zeros_like(W_inst)
    else:
        W_inst = W_matrix
        W_lag = np.zeros_like(W_inst)

    # Get incoming edges to target (both instantaneous and lagged)
    incoming_inst = W_inst[:, target_idx]
    incoming_lag = W_lag[:, target_idx]

    parents = []
    for i in range(len(incoming_inst)):
        # Combine instantaneous and lagged influence
        total_influence = abs(incoming_inst[i]) + abs(incoming_lag[i])
        if total_influence > 0.01:  # Threshold for significance
            parents.append({
                'feature': feature_names[i],
                'W_instantaneous': float(incoming_inst[i]),
                'W_lagged': float(incoming_lag[i]),
                'total_influence': float(total_influence),
                'sign': 'positive' if (incoming_inst[i] + incoming_lag[i]) > 0 else 'negative'
            })

    # Sort by total influence
    parents.sort(key=lambda x: x['total_influence'], reverse=True)
    return parents[:top_k]


def analyze_W_weights(W_matrix: np.ndarray):
    """
    Analyze the ICGNN W weight matrix to understand learned causal structure.

    Returns statistics that help understand if W is learning differentiated structure.
    """
    if W_matrix.ndim == 3:
        W_inst = W_matrix[0]
        W_lag = W_matrix[1] if W_matrix.shape[0] > 1 else None
    else:
        W_inst = W_matrix
        W_lag = None

    stats = {
        'instantaneous': {
            'mean': float(W_inst.mean()),
            'std': float(W_inst.std()),
            'min': float(W_inst.min()),
            'max': float(W_inst.max()),
            'abs_mean': float(np.abs(W_inst).mean()),
            'n_positive': int((W_inst > 0.01).sum()),
            'n_negative': int((W_inst < -0.01).sum()),
            'n_near_zero': int((np.abs(W_inst) <= 0.01).sum()),
        }
    }

    if W_lag is not None:
        stats['lagged'] = {
            'mean': float(W_lag.mean()),
            'std': float(W_lag.std()),
            'min': float(W_lag.min()),
            'max': float(W_lag.max()),
            'abs_mean': float(np.abs(W_lag).mean()),
            'n_positive': int((W_lag > 0.01).sum()),
            'n_negative': int((W_lag < -0.01).sum()),
            'n_near_zero': int((np.abs(W_lag) <= 0.01).sum()),
        }

    return stats


def train_and_extract_dags(
    dataset: str,
    d_dim: int,
    seed: int,
    device: str = 'cuda',
    output_dir: Path = None,
    lambda_sparse: float = 10.0,
    init_logits: float = -0.5,
    tau_gumbel: float = 1.0,
    edge_threshold: float = 0.3
):
    """
    Train DS3M Causal model and extract per-regime DAGs.
    """
    from unified_data_loader import prepare_unified_ds3m_data
    from ds3m_fantom.models.ds3m_causal import DS3MCausal
    from ds3m_fantom.training.regime_regularization import (
        RegimeRegularizer, diagnose_regime_collapse
    )

    set_seed(seed)

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include key hyperparameters in directory name
        config_str = f"sparse{int(lambda_sparse)}_init{abs(init_logits)}_tau{tau_gumbel}_thresh{edge_threshold}"
        output_dir = Path(__file__).parent / "outputs" / "regime_dags" / f"{dataset}_d{d_dim}_{config_str}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training DS3M Causal: {dataset}, d_dim={d_dim}, seed={seed}")
    print(f"Sparsity penalty (lambda_sparse): {lambda_sparse}")
    print(f"Init logits: {init_logits} (more negative = sparser prior)")
    print(f"Gumbel temperature (tau): {tau_gumbel} (lower = sharper)")
    print(f"Edge threshold: {edge_threshold}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Feature groups
    if dataset == 'DE_FR':
        feature_groups = ['spread', 'price_de', 'price_fr', 'calendar', 'spgci']
    else:
        feature_groups = ['price', 'calendar', 'load', 'weather']

    # Load data
    data = prepare_unified_ds3m_data(
        dataset,
        timestep=14,
        feature_groups=feature_groups,
        target_col='price_spread_change_pct' if dataset == 'DE_FR' else 'price_change_pct'
    )

    x_dim = data['trainX'].shape[-1]
    # Get actual feature names from data loader (key is 'feature_cols')
    feature_names = data.get('feature_cols', data.get('feature_names', [f'feature_{i}' for i in range(x_dim)]))
    if not isinstance(feature_names, list) or len(feature_names) != x_dim:
        feature_names = [f'feature_{i}' for i in range(x_dim)]
    target_name = 'price_spread_change_pct' if dataset == 'DE_FR' else 'price_change_pct'

    print(f"Features ({x_dim}): {feature_names[:5]}...")

    device_obj = torch.device(device)

    # Create model with configurable sparsity parameters
    model = DS3MCausal(
        x_dim=x_dim,
        y_dim=1,
        h_dim=32,
        z_dim=8,
        d_dim=d_dim,
        device=device_obj,
        n_layers=1,
        num_nodes=x_dim,
        lag=1,
        sharing_mode='independent',
        tau_gumbel=tau_gumbel,  # Lower = sharper edge selection
        init_logits=[init_logits, init_logits],  # More negative = sparser prior
        lambda_dag=100.0,
        lambda_sparse=lambda_sparse,  # Sparsity penalty weight
        lambda_kl=1.0,
    ).to(device)

    # Regularizer
    regularizer = RegimeRegularizer(
        d_dim=d_dim,
        entropy_weight=1.0,
        min_usage_weight=0.5,
        smoothness_weight=0.1,
        kl_weight=1.0,
        min_usage_ratio=0.1,
        annealing_start=10,
        annealing_end=50
    )

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainX = data['trainX'].to(device)
    trainY = data['trainY'].to(device)
    testX = data['testX'].to(device)
    testY = data['testY'].to(device)

    best_val_spearman = -np.inf
    patience_counter = 0
    patience = 20
    best_state = None

    print("Training...")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        results = model(trainX, trainY)

        loss = results['nll'] + results['kl_z'] + results['kl_d']
        loss += results['dag_penalty'] + results['sparse_penalty']

        if 'regime_posteriors' in results:
            reg_loss = regularizer(results['regime_posteriors'])
            loss += reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        regularizer.step()

        # Validation
        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                val_pred_result = model.predict(data['valX'].to(device), n_samples=10)
                val_pred = val_pred_result['predictions'][-1, :, 0].cpu().numpy()
                val_true = data['valY'][-1, :, 0].cpu().numpy()

                val_pred_denorm = val_pred * data['Y_moments'][1] + data['Y_moments'][0]
                val_true_denorm = val_true * data['Y_moments'][1] + data['Y_moments'][0]

                val_spearman, _ = spearmanr(val_true_denorm, val_pred_denorm)

                collapse_info = diagnose_regime_collapse(results['regime_posteriors'])
                print(f"  Epoch {epoch+1}: Loss={loss.item():.2f}, Val Spearman={val_spearman:.4f}, "
                      f"Regimes={collapse_info['effective_regimes']}")

                if val_spearman > best_val_spearman:
                    best_val_spearman = val_spearman
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_pred_result = model.predict(testX, n_samples=50)
        test_pred = test_pred_result['predictions'][-1, :, 0].cpu().numpy()
        test_true = testY[-1, :, 0].cpu().numpy()

        test_pred_denorm = test_pred * data['Y_moments'][1] + data['Y_moments'][0]
        test_true_denorm = test_true * data['Y_moments'][1] + data['Y_moments'][0]

        test_spearman, _ = spearmanr(test_true_denorm, test_pred_denorm)

        # Get regime assignments
        regime_assignments = test_pred_result['regimes']
        regime_counts = {}
        for r in range(d_dim):
            regime_counts[r] = int((regime_assignments == r).sum().item())

    print(f"\nTest Spearman: {test_spearman:.4f}")
    print(f"Regime distribution: {regime_counts}")

    # Extract and save DAGs and ICGNN W weights
    print("\nExtracting per-regime DAGs and ICGNN W weights...")
    dags = model.get_causal_graphs()

    dag_info = []
    w_weight_info = []

    for r, adj in enumerate(dags):
        if isinstance(adj, torch.Tensor):
            adj = adj.cpu().numpy()

        # Save raw adjacency matrix (soft probabilities)
        np.save(output_dir / f"adjacency_regime{r}_soft.npy", adj)

        # Apply hard thresholding for interpretable DAG
        adj_hard = (adj > edge_threshold).astype(np.float32)
        np.save(output_dir / f"adjacency_regime{r}.npy", adj_hard)

        # Extract and save ICGNN W weights for this regime
        W = model.causal_emissions[r].icgnn.get_weighted_adjacency()
        if isinstance(W, torch.Tensor):
            W = W.detach().cpu().numpy()
        np.save(output_dir / f"W_weights_regime{r}.npy", W)

        # Analyze W weights
        W_stats = analyze_W_weights(W)
        w_weight_info.append({
            'regime': r,
            'W_stats': W_stats
        })

        # Count edges using the threshold
        n_edges_soft = int((np.abs(adj) > edge_threshold).sum())
        n_edges_hard = int(adj_hard.sum())
        sparsity = 1 - (n_edges_hard / adj_hard.size)

        # Get top parents of target (last variable) - by edge probability
        target_idx = x_dim - 1  # Assuming target is last
        top_parents = get_top_parents(adj, target_idx, feature_names, top_k=10)

        # Get top parents by W weights (the actual learned causal strength)
        top_parents_W = get_top_parents_by_W(W, target_idx, feature_names, top_k=10)

        dag_info.append({
            'regime': r,
            'n_samples': regime_counts.get(r, 0),
            'n_edges': n_edges_hard,
            'n_edges_soft': n_edges_soft,
            'sparsity': sparsity,
            'top_parents_by_prob': top_parents,
            'top_parents_by_W': top_parents_W,
            'edge_threshold': edge_threshold,
            'W_stats': W_stats
        })

        print(f"\n  Regime {r} ({regime_counts.get(r, 0)} samples):")
        print(f"    Edge probabilities (A): {n_edges_hard} edges (>{edge_threshold}), Sparsity: {sparsity:.2%}")

        # Show W weight statistics
        print(f"    ICGNN W weights (learned causal strength):")
        if 'instantaneous' in W_stats:
            ws = W_stats['instantaneous']
            print(f"      Instantaneous: mean={ws['mean']:.4f}, std={ws['std']:.4f}, range=[{ws['min']:.4f}, {ws['max']:.4f}]")
            print(f"        {ws['n_positive']} positive, {ws['n_negative']} negative, {ws['n_near_zero']} near-zero")
        if 'lagged' in W_stats:
            ws = W_stats['lagged']
            print(f"      Lagged: mean={ws['mean']:.4f}, std={ws['std']:.4f}, range=[{ws['min']:.4f}, {ws['max']:.4f}]")
            print(f"        {ws['n_positive']} positive, {ws['n_negative']} negative, {ws['n_near_zero']} near-zero")

        print(f"    Top parents of {target_name} (by W weights):")
        for p in top_parents_W[:5]:
            sign = '+' if p['W_instantaneous'] + p['W_lagged'] > 0 else '-'
            print(f"      {sign} {p['feature']}: inst={p['W_instantaneous']:.3f}, lag={p['W_lagged']:.3f}")

        # Plot causal graph with hard thresholding
        plot_causal_graph(
            adj_hard,  # Use hard-thresholded adjacency
            feature_names,
            target_idx,
            title=f"Regime {r} Causal Graph ({regime_counts.get(r, 0)} samples, thresh={edge_threshold})",
            save_path=output_dir / f"causal_graph_regime{r}.png",
            threshold=0.5  # For hard adjacency, anything > 0.5 is an edge
        )

        # Also plot causal graph based on W weights (using absolute value for edge strength)
        W_combined = np.abs(W[0]) + np.abs(W[1]) if W.ndim == 3 and W.shape[0] > 1 else np.abs(W[0] if W.ndim == 3 else W)
        plot_causal_graph(
            W_combined,
            feature_names,
            target_idx,
            title=f"Regime {r} W-weighted Graph ({regime_counts.get(r, 0)} samples)",
            save_path=output_dir / f"W_graph_regime{r}.png",
            threshold=0.05  # Lower threshold for W weights
        )

    # Save summary
    summary = {
        'dataset': dataset,
        'd_dim': d_dim,
        'seed': seed,
        'hyperparameters': {
            'lambda_sparse': lambda_sparse,
            'init_logits': init_logits,
            'tau_gumbel': tau_gumbel,
            'edge_threshold': edge_threshold
        },
        'test_spearman': float(test_spearman),
        'best_val_spearman': float(best_val_spearman),
        'n_features': x_dim,
        'feature_names': feature_names,
        'target_name': target_name,
        'regime_distribution': regime_counts,
        'dag_info': dag_info,
        'note': 'Edge probabilities (A) may be uniform. Check W_weights for actual learned causal structure.'
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\nSaved files per regime:")
    print(f"  - adjacency_regime{{r}}_soft.npy: Edge probabilities A (may be uniform)")
    print(f"  - adjacency_regime{{r}}.npy: Hard-thresholded A")
    print(f"  - W_weights_regime{{r}}.npy: ICGNN learned weights (actual causal strength)")
    print(f"  - causal_graph_regime{{r}}.png: Graph based on A")
    print(f"  - W_graph_regime{{r}}.png: Graph based on W weights")
    print(f"\nNote: If edge probabilities are uniform, look at W_weights for differentiation.")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract per-regime DAGs from DS3M Causal")
    parser.add_argument('--dataset', type=str, required=True, choices=['DE', 'FR', 'DE_FR'])
    parser.add_argument('--d_dim', type=int, required=True, help='Number of regimes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--lambda_sparse', type=float, default=10.0,
                        help='Sparsity penalty weight (higher = sparser DAGs)')
    parser.add_argument('--init_logits', type=float, default=-0.5,
                        help='Initial logits for edge prior (more negative = sparser prior, e.g., -2.0)')
    parser.add_argument('--tau_gumbel', type=float, default=1.0,
                        help='Gumbel-Softmax temperature (lower = sharper sampling, e.g., 0.1)')
    parser.add_argument('--edge_threshold', type=float, default=0.3,
                        help='Threshold for hard edge selection (default 0.3)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    train_and_extract_dags(
        dataset=args.dataset,
        d_dim=args.d_dim,
        seed=args.seed,
        device=args.device,
        output_dir=output_dir,
        lambda_sparse=args.lambda_sparse,
        init_logits=args.init_logits,
        tau_gumbel=args.tau_gumbel,
        edge_threshold=args.edge_threshold
    )


if __name__ == "__main__":
    main()
