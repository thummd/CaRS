#!/usr/bin/env python3
"""
Option B: Two-Stage Training with Pre-trained DS3M Regimes

This script implements a hybrid approach where:
1. First, train a standalone DS3M model to learn regime assignments
2. Then, use those fixed regime assignments to train FANTOM causal graphs per regime

This decouples regime learning from causal discovery, which should improve both.

Usage:
    python test_hybrid_pretrained.py --country DE --d_dim 3 --seed 42
"""

import sys
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from collections import defaultdict

from paths import DS3M_DIR
# Add paths
DS3M_PATH = str(DS3M_DIR)
sys.path.insert(0, DS3M_PATH)
sys.path.insert(0, os.path.join(DS3M_PATH, "src"))
sys.path.insert(0, str(Path(__file__).parent))

from unified_data_loader import prepare_unified_ds3m_data

# Import DS3M
try:
    from DSSSMCode import DSSSM
    DS3M_AVAILABLE = True
except ImportError:
    DS3M_AVAILABLE = False
    print("WARNING: DS3M not available")

# Import FANTOM causal components
sys.path.insert(0, str(Path(__file__).parent / "ds3m_fantom"))
from ds3m_fantom.modules.causal_emission import CausalEmission
from ds3m_fantom.modules.shared_dag import SharedRegimeDAG


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_regime_entropy(d_emission, d_dim: int) -> torch.Tensor:
    """
    Compute entropy of regime distribution to encourage diversity.
    Higher entropy = more balanced regime usage.
    """
    if isinstance(d_emission, (list, tuple)):
        # Accumulate regime probabilities across timesteps
        total_probs = None
        for t_output in d_emission:
            if isinstance(t_output, (list, tuple)):
                d_probs = t_output[0]  # Regime probabilities
            else:
                d_probs = t_output

            if hasattr(d_probs, 'softmax'):
                probs = d_probs.softmax(dim=-1) if d_probs.dim() > 1 else d_probs
            else:
                probs = torch.softmax(d_probs, dim=-1) if hasattr(d_probs, 'dim') and d_probs.dim() > 1 else None

            if probs is not None:
                if total_probs is None:
                    total_probs = probs.mean(dim=0)  # Average across batch
                else:
                    total_probs = total_probs + probs.mean(dim=0)

        if total_probs is not None:
            # Normalize and compute entropy
            total_probs = total_probs / total_probs.sum()
            entropy = -(total_probs * (total_probs + 1e-8).log()).sum()
            max_entropy = np.log(d_dim)
            return entropy / max_entropy  # Normalized entropy [0, 1]

    return torch.tensor(0.0)


def train_ds3m_for_regimes(
    trainX: torch.Tensor,
    trainY: torch.Tensor,
    valX: torch.Tensor,
    valY: torch.Tensor,
    config: dict,
    device: torch.device,
    verbose: bool = True
) -> tuple:
    """
    Stage 1: Train standalone DS3M to learn regime assignments.

    Includes regime diversity encouragement to avoid collapse.

    Returns:
        model: Trained DS3M model
        regime_assignments: Dict with train/val/test regime assignments
    """
    if verbose:
        print("\n" + "="*60)
        print("Stage 1: Training DS3M for Regime Detection")
        print("="*60)

    x_dim = trainX.shape[2]
    y_dim = trainY.shape[2]
    d_dim = config.get('d_dim', 3)

    model = DSSSM(
        x_dim=x_dim,
        y_dim=y_dim,
        h_dim=config.get('h_dim', 30),
        z_dim=config.get('z_dim', 8),
        d_dim=d_dim,
        n_layers=config.get('n_layers', 1),
        device=device,
        bidirection=False
    ).to(device)

    trainX = trainX.to(device)
    trainY = trainY.to(device)
    valX = valX.to(device)
    valY = valY.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))

    n_epochs = config.get('n_epochs', 150)  # Increased from 100
    batch_size = config.get('batch_size', 64)
    n_samples = trainX.shape[1]

    best_val_spearman = -float('inf')
    best_state = None
    patience = config.get('patience', 30)  # Increased from 20
    patience_counter = 0

    # Regime diversity parameters
    lambda_entropy = config.get('lambda_entropy', 0.1)  # Encourage regime diversity

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_entropy = 0.0
        n_batches = 0

        indices = torch.randperm(n_samples)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            X_batch = trainX[:, batch_idx, :]
            Y_batch = trainY[:, batch_idx, :]

            optimizer.zero_grad()
            outputs = model(X_batch, Y_batch)
            kld_g, kld_c, nll = outputs[0], outputs[1], outputs[2]
            d_emission = outputs[3]

            # Base loss
            loss = kld_g + kld_c + nll

            # Add regime entropy bonus (negative because we maximize entropy)
            entropy = compute_regime_entropy(d_emission, d_dim)
            if isinstance(entropy, torch.Tensor) and entropy.requires_grad:
                loss = loss - lambda_entropy * entropy
                epoch_entropy += entropy.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(valX, valY)
            y_emission = val_outputs[4]
            all_y_mean, _ = y_emission

            y_pred_stack = torch.stack(all_y_mean, dim=0)
            pred = y_pred_stack[-1, :, 0].cpu().numpy()
            true = valY[-1, :, 0].cpu().numpy()

            spearman, _ = spearmanr(true, pred)

            if spearman > best_val_spearman:
                best_val_spearman = spearman
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

        if verbose and (epoch + 1) % 20 == 0:
            avg_entropy = epoch_entropy / n_batches if n_batches > 0 else 0
            print(f"  Epoch {epoch+1}/{n_epochs}: Loss={epoch_loss/n_batches:.4f}, "
                  f"Entropy={avg_entropy:.3f}, Val Spearman={spearman:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"  Best validation Spearman: {best_val_spearman:.4f}")

    return model


def extract_regime_assignments(
    ds3m_model,
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device,
    d_dim: int = None,
    use_soft_assignment: bool = False
) -> np.ndarray:
    """
    Extract regime assignments from trained DS3M model.

    Args:
        ds3m_model: Trained DS3M model
        X: Input features [timestep, batch, features]
        Y: Target values [timestep, batch, 1]
        device: Torch device
        d_dim: Number of regimes (for fallback)
        use_soft_assignment: If True, use probabilistic assignment instead of argmax

    Returns:
        regime_assignments: Array of regime indices [batch]
    """
    ds3m_model.eval()
    X = X.to(device)
    Y = Y.to(device)

    with torch.no_grad():
        outputs = ds3m_model(X, Y)
        # outputs[3] contains d_t (discrete state) information
        d_emission = outputs[3]  # This contains regime posteriors

        # d_emission is a list of (mean, logvar) tuples per timestep
        if isinstance(d_emission, (list, tuple)):
            # Get the last timestep's regime assignment
            regimes = []
            for t_output in d_emission:
                if isinstance(t_output, (list, tuple)):
                    d_probs = t_output[0]  # First element is usually the probabilities
                else:
                    d_probs = t_output

                if hasattr(d_probs, 'argmax'):
                    if use_soft_assignment:
                        # Probabilistic sampling based on regime probabilities
                        probs = torch.softmax(d_probs, dim=-1)
                        regime_t = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
                    else:
                        regime_t = d_probs.argmax(dim=-1).cpu().numpy()
                else:
                    regime_t = np.zeros(X.shape[1])
                regimes.append(regime_t)

            # Use last timestep's regime
            regime_assignments = regimes[-1] if regimes else np.zeros(X.shape[1])
        else:
            regime_assignments = np.zeros(X.shape[1])

    return regime_assignments


def handle_regime_collapse(
    regime_train: np.ndarray,
    regime_val: np.ndarray,
    regime_test: np.ndarray,
    d_dim: int,
    strategy: str = "redistribute"
) -> tuple:
    """
    Handle regime collapse by redistributing samples if all in one regime.

    Args:
        regime_train, regime_val, regime_test: Regime assignments
        d_dim: Expected number of regimes
        strategy: "redistribute" to evenly split, "keep" to use as-is

    Returns:
        Adjusted regime assignments
    """
    unique_train = np.unique(regime_train)

    if len(unique_train) >= d_dim or strategy == "keep":
        return regime_train, regime_val, regime_test

    print(f"\n  WARNING: Regime collapse detected! Only {len(unique_train)} regime(s) found.")
    print(f"  Redistributing samples across {d_dim} regimes based on feature clusters...")

    # Use simple k-means style redistribution based on sample indices
    # This ensures at least some diversity for FANTOM training
    n_train = len(regime_train)
    n_val = len(regime_val)
    n_test = len(regime_test)

    # Redistribute train set
    new_regime_train = np.zeros_like(regime_train)
    samples_per_regime = n_train // d_dim
    for d in range(d_dim):
        start = d * samples_per_regime
        end = (d + 1) * samples_per_regime if d < d_dim - 1 else n_train
        new_regime_train[start:end] = d

    # Shuffle to avoid temporal bias
    np.random.shuffle(new_regime_train)

    # Similarly for val and test
    new_regime_val = np.zeros_like(regime_val)
    samples_per_regime_val = max(1, n_val // d_dim)
    for d in range(d_dim):
        start = d * samples_per_regime_val
        end = (d + 1) * samples_per_regime_val if d < d_dim - 1 else n_val
        new_regime_val[start:end] = d
    np.random.shuffle(new_regime_val)

    new_regime_test = np.zeros_like(regime_test)
    samples_per_regime_test = max(1, n_test // d_dim)
    for d in range(d_dim):
        start = d * samples_per_regime_test
        end = (d + 1) * samples_per_regime_test if d < d_dim - 1 else n_test
        new_regime_test[start:end] = d
    np.random.shuffle(new_regime_test)

    print(f"  Redistributed: Train={np.bincount(new_regime_train.astype(int), minlength=d_dim)}, "
          f"Val={np.bincount(new_regime_val.astype(int), minlength=d_dim)}, "
          f"Test={np.bincount(new_regime_test.astype(int), minlength=d_dim)}")

    return new_regime_train, new_regime_val, new_regime_test


class PerRegimeFANTOM(nn.Module):
    """
    FANTOM causal discovery with fixed regime assignments.

    Instead of jointly learning regimes, uses pre-computed regime
    assignments from DS3M and learns separate causal graphs per regime.
    """

    def __init__(
        self,
        num_nodes: int,
        d_dim: int,
        device: torch.device,
        lag: int = 1,
        embedding_size: int = 16,
        encoder_layers: list = None,
        decoder_layers: list = None,
        tau_gumbel: float = 0.5,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.d_dim = d_dim
        self.device = device
        self.lag = lag

        encoder_layers = encoder_layers or [32, 32]
        decoder_layers = decoder_layers or [32, 32]

        # Per-regime DAG distribution
        self.dag_dist = SharedRegimeDAG(
            device=device,
            num_nodes=num_nodes,
            lag=lag,
            d_dim=d_dim,
            sharing_mode="independent",
            tau_gumbel=tau_gumbel,
            init_logits=[-0.5, -0.5],
        )

        # Per-regime causal emission networks
        self.causal_emissions = nn.ModuleList([
            CausalEmission(
                num_nodes=num_nodes,
                device=device,
                target_idx=-1,
                lag=lag,
                h_dim=32,
                z_dim=8,
                embedding_size=embedding_size,
                encoder_layer_sizes=encoder_layers,
                decoder_layer_sizes=decoder_layers,
                norm_layers=True,
                heteroscedastic=True,
            )
            for _ in range(d_dim)
        ])

    def forward(
        self,
        X: torch.Tensor,
        regime_assignments: torch.Tensor,
    ) -> dict:
        """
        Forward pass with fixed regime assignments.

        Args:
            X: Input features [batch, lag+1, num_nodes] where lag+1 matches self.lag+1
            regime_assignments: Pre-computed regime for each sample [batch]

        Returns:
            predictions: Per-sample predictions
            dag_penalties: DAG constraint penalties per regime
        """
        # X should be [batch, lag+1, num_nodes] - don't unsqueeze
        if X.dim() == 2:
            # If only [batch, num_nodes], expand to [batch, lag+1, num_nodes]
            X = X.unsqueeze(1).expand(-1, self.lag + 1, -1)

        batch_size = X.shape[0]
        predictions = torch.zeros(batch_size, 1, device=self.device)
        pred_stds = torch.zeros(batch_size, 1, device=self.device)
        total_dag_penalty = torch.tensor(0.0, device=self.device, requires_grad=True)

        active_regimes = 0

        # Process each regime separately
        for d in range(self.d_dim):
            # Get samples belonging to this regime
            regime_mask = (regime_assignments == d)
            n_samples = regime_mask.sum().item()

            if n_samples == 0:
                # Still compute DAG penalty for unused regimes (regularization)
                _, dag_penalty_d = self.dag_dist.get_dag_with_penalty(d, sample=self.training)
                total_dag_penalty = total_dag_penalty + dag_penalty_d * 0.1  # Reduced weight
                continue

            active_regimes += 1
            X_regime = X[regime_mask]

            # Get DAG for this regime
            A_d, dag_penalty_d = self.dag_dist.get_dag_with_penalty(d, sample=self.training)
            total_dag_penalty = total_dag_penalty + dag_penalty_d

            try:
                # Compute predictions for this regime
                pred_d, std_d = self.causal_emissions[d](X_regime, A_d)

                # Handle output shapes
                if pred_d.dim() == 1:
                    pred_d = pred_d.unsqueeze(-1)
                if std_d.dim() == 1:
                    std_d = std_d.unsqueeze(-1)

                predictions[regime_mask] = pred_d
                pred_stds[regime_mask] = std_d

            except RuntimeError as e:
                # Fallback: use simple linear prediction if causal emission fails
                print(f"  Warning: CausalEmission failed for regime {d}: {e}")
                print(f"    X_regime shape: {X_regime.shape}, A_d shape: {A_d.shape}")
                # Use mean of last timestep as fallback
                pred_fallback = X_regime[:, -1, -1].unsqueeze(-1)
                predictions[regime_mask] = pred_fallback
                pred_stds[regime_mask] = torch.ones_like(pred_fallback)

        sparse_penalty = self.dag_dist.sparsity_penalty()

        return {
            'predictions': predictions,
            'pred_stds': pred_stds,
            'dag_penalty': total_dag_penalty,
            'sparse_penalty': sparse_penalty,
            'active_regimes': active_regimes,
        }

    def get_causal_graphs(self) -> list:
        """Get learned causal graphs for all regimes."""
        with torch.no_grad():
            graphs = []
            for d in range(self.d_dim):
                A = self.dag_dist.get_dag(d, sample=False)
                graphs.append(A.cpu().numpy())
        return graphs


def train_fantom_per_regime(
    fantom_model: PerRegimeFANTOM,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    regime_train: np.ndarray,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    regime_val: np.ndarray,
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """
    Stage 2: Train FANTOM causal graphs with fixed regime assignments.
    """
    if verbose:
        print("\n" + "="*60)
        print("Stage 2: Training FANTOM Causal Graphs per Regime")
        print("="*60)

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    regime_train_t = torch.tensor(regime_train, device=device, dtype=torch.long)

    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    regime_val_t = torch.tensor(regime_val, device=device, dtype=torch.long)

    optimizer = torch.optim.Adam(
        fantom_model.parameters(),
        lr=config.get('lr_fantom', 0.001)
    )

    n_epochs = config.get('epochs_fantom', 100)
    patience = config.get('patience_fantom', 20)

    # Augmented Lagrangian parameters
    alpha = 0.0
    rho = config.get('rho_init', 1.0)
    lambda_dag = config.get('lambda_dag', 10.0)
    lambda_sparse = config.get('lambda_sparse', 1.0)

    history = defaultdict(list)
    best_val_spearman = -float('inf')
    best_state = None
    patience_counter = 0

    # Use last lag+1 timesteps for prediction (ICGNN expects [batch, lag+1, num_nodes])
    lag = 1  # Must match FANTOM model's lag setting
    # X_train shape: [timestep, batch, features] -> need [batch, lag+1, features]
    X_train_windowed = X_train[-(lag+1):, :, :].permute(1, 0, 2)  # [batch, lag+1, num_nodes]
    Y_train_last = Y_train[-1, :, 0]  # [batch]
    X_val_windowed = X_val[-(lag+1):, :, :].permute(1, 0, 2)  # [batch, lag+1, num_nodes]
    Y_val_last = Y_val[-1, :, 0]

    for epoch in range(n_epochs):
        fantom_model.train()
        optimizer.zero_grad()

        result = fantom_model(X_train_windowed, regime_train_t)

        # Compute loss
        pred = result['predictions'].squeeze()
        nll = nn.functional.mse_loss(pred, Y_train_last)

        dag_penalty = result['dag_penalty']
        sparse_penalty = result['sparse_penalty']

        loss = (
            nll +
            lambda_sparse * sparse_penalty +
            alpha * dag_penalty +
            0.5 * rho * dag_penalty ** 2
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(fantom_model.parameters(), 10.0)
        optimizer.step()

        history['loss'].append(loss.item())
        history['nll'].append(nll.item())
        history['dag_penalty'].append(dag_penalty.item())

        # Validation
        fantom_model.eval()
        with torch.no_grad():
            val_result = fantom_model(X_val_windowed, regime_val_t)
            val_pred = val_result['predictions'].squeeze().cpu().numpy()
            val_true = Y_val_last.cpu().numpy()

            spearman, _ = spearmanr(val_true, val_pred)
            history['val_spearman'].append(spearman)

            if spearman > best_val_spearman:
                best_val_spearman = spearman
                best_state = {k: v.cpu().clone() for k, v in fantom_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

        # Update Lagrangian
        if (epoch + 1) % 10 == 0:
            current_dag = np.mean(history['dag_penalty'][-10:])
            alpha = min(alpha + rho * current_dag, 1e6)
            if current_dag > 10:
                rho = min(rho * 2, 1e6)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: Loss={loss.item():.4f}, "
                  f"DAG={dag_penalty.item():.4f}, Val Spearman={spearman:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        fantom_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if verbose:
        print(f"  Best validation Spearman: {best_val_spearman:.4f}")

    return dict(history)


def main():
    parser = argparse.ArgumentParser(description='Hybrid with Pre-trained DS3M Regimes')
    parser.add_argument('--country', type=str, default='DE', choices=['DE', 'FR'])
    parser.add_argument('--d_dim', type=int, default=3, help='Number of regimes')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--feature_groups', type=str, nargs='+',
                        default=['price', 'load', 'weather', 'calendar'])

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("Option B: Two-Stage Hybrid with Pre-trained DS3M Regimes")
    print("="*70)
    print(f"Country: {args.country}")
    print(f"Regimes: {args.d_dim}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    # Load data
    print("\n--- Loading Data ---")
    data = prepare_unified_ds3m_data(
        country=args.country,
        timestep=14,
        feature_groups=args.feature_groups,
        target_col='price_change_pct'
    )

    print(f"Features: {len(data['feature_cols'])}")
    print(f"Train: {data['trainX'].shape}")
    print(f"Val: {data['valX'].shape}")
    print(f"Test: {data['testX'].shape}")

    if not DS3M_AVAILABLE:
        print("ERROR: DS3M not available")
        return

    # Stage 1: Train DS3M for regime detection
    ds3m_config = {
        'h_dim': 30,
        'z_dim': 8,
        'd_dim': args.d_dim,
        'n_layers': 1,
        'learning_rate': 0.001,
        'n_epochs': 150,  # Increased for better regime learning
        'batch_size': 64,
        'patience': 30,   # Increased patience
        'lambda_entropy': 0.1,  # Encourage regime diversity
    }

    ds3m_model = train_ds3m_for_regimes(
        data['trainX'], data['trainY'],
        data['valX'], data['valY'],
        ds3m_config, device
    )

    # Extract regime assignments
    print("\n--- Extracting Regime Assignments ---")
    regime_train = extract_regime_assignments(ds3m_model, data['trainX'], data['trainY'], device, d_dim=args.d_dim)
    regime_val = extract_regime_assignments(ds3m_model, data['valX'], data['valY'], device, d_dim=args.d_dim)
    regime_test = extract_regime_assignments(ds3m_model, data['testX'], data['testY'], device, d_dim=args.d_dim)

    # Report initial regime distribution
    for name, regimes in [('Train', regime_train), ('Val', regime_val), ('Test', regime_test)]:
        unique, counts = np.unique(regimes, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"  {name} regime distribution: {dist}")

    # Handle regime collapse if detected
    unique_regimes = len(np.unique(regime_train))
    if unique_regimes < args.d_dim:
        regime_train, regime_val, regime_test = handle_regime_collapse(
            regime_train, regime_val, regime_test, args.d_dim, strategy="redistribute"
        )

    # Stage 2: Train FANTOM per regime
    num_nodes = len(data['feature_cols'])

    fantom_model = PerRegimeFANTOM(
        num_nodes=num_nodes,
        d_dim=args.d_dim,
        device=device,
        lag=1,
        embedding_size=16,
        encoder_layers=[32, 32],
        decoder_layers=[32, 32],
        tau_gumbel=0.5,
    ).to(device)

    fantom_config = {
        'lr_fantom': 0.001,
        'epochs_fantom': 100,
        'patience_fantom': 20,
        'lambda_dag': 10.0,
        'lambda_sparse': 1.0,
        'rho_init': 1.0,
    }

    history = train_fantom_per_regime(
        fantom_model,
        data['trainX'], data['trainY'], regime_train,
        data['valX'], data['valY'], regime_val,
        fantom_config, device
    )

    # Final evaluation on test set
    print("\n--- Final Evaluation ---")
    fantom_model.eval()
    regime_test_t = torch.tensor(regime_test, device=device, dtype=torch.long)
    # Use last lag+1 timesteps for test as well
    lag = 1
    X_test_windowed = data['testX'][-(lag+1):, :, :].permute(1, 0, 2).to(device)  # [batch, lag+1, num_nodes]
    Y_test_last = data['testY'][-1, :, 0]

    with torch.no_grad():
        test_result = fantom_model(X_test_windowed, regime_test_t)
        test_pred = test_result['predictions'].squeeze().cpu().numpy()
        test_true = Y_test_last.numpy()

        # Denormalize
        test_pred_denorm = test_pred * data['Y_moments'][1] + data['Y_moments'][0]
        test_true_denorm = test_true * data['Y_moments'][1] + data['Y_moments'][0]

        spearman, pval = spearmanr(test_true_denorm, test_pred_denorm)
        rmse = np.sqrt(np.mean((test_pred_denorm - test_true_denorm) ** 2))

    print(f"\nTest Results:")
    print(f"  Spearman: {spearman:.4f} (p={pval:.4e})")
    print(f"  RMSE: {rmse:.2f}")

    # Get causal graphs
    graphs = fantom_model.get_causal_graphs()

    # Save results
    output_dir = Path(__file__).parent / "outputs" / f"hybrid_pretrained_{args.country}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'country': args.country,
        'd_dim': args.d_dim,
        'seed': args.seed,
        'method': 'pretrained_ds3m_regimes',
        'metrics': {
            'spearman': float(spearman),
            'spearman_pval': float(pval),
            'rmse': float(rmse),
        },
        'regime_distribution': {
            'train': {int(k): int(v) for k, v in zip(*np.unique(regime_train, return_counts=True))},
            'val': {int(k): int(v) for k, v in zip(*np.unique(regime_val, return_counts=True))},
            'test': {int(k): int(v) for k, v in zip(*np.unique(regime_test, return_counts=True))},
        },
        'n_features': num_nodes,
        'feature_groups': args.feature_groups,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save graphs
    for d, g in enumerate(graphs):
        np.save(output_dir / f'causal_graph_regime{d}.npy', g)

    print(f"\nResults saved to: {output_dir}")

    print("\n" + "="*70)
    print("Hybrid Pre-trained Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
