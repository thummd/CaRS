"""
Two-Stage Training for DS3M-Causal Hybrid

Stage 1: Train DS3M encoder for regime detection (freeze causal components)
Stage 2: Train FANTOM causal graphs per regime (freeze DS3M encoder)

This approach is more stable than end-to-end training because:
1. Regime detection is established first with clean gradients
2. Causal discovery operates on stable regime assignments
3. Each stage has simpler optimization landscape
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import time
from collections import defaultdict
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr

from paths import DS3M_DIR
# Add paths
DS3M_PATH = str(DS3M_DIR)
if DS3M_PATH not in sys.path:
    sys.path.insert(0, DS3M_PATH)
    sys.path.insert(0, os.path.join(DS3M_PATH, "src"))

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ds3m_fantom.models.ds3m_causal import DS3MCausal
from ds3m_fantom.modules.shared_dag import dagness_factor
from ds3m_fantom.training.train_e2e import EarlyStopping


class TwoStageTrainer:
    """
    Two-stage trainer for DS3M-Causal hybrid.

    Stage 1: Train DS3M encoder and regime detection
             - Freeze causal emission parameters
             - Use simple reconstruction loss for DS3M

    Stage 2: Train FANTOM causal graphs
             - Freeze DS3M encoder parameters
             - Use augmented Lagrangian for DAG constraint
    """

    def __init__(
        self,
        model: DS3MCausal,
        device: torch.device,
        # Stage 1 parameters
        lr_stage1: float = 1e-3,
        epochs_stage1: int = 100,
        patience_stage1: int = 15,
        # Stage 2 parameters
        lr_stage2: float = 5e-4,
        epochs_stage2: int = 100,
        rho_init: float = 1.0,
        rho_max: float = 1e9,
        alpha_max: float = 1e9,
        patience_stage2: int = 10,
        # General
        verbose: bool = True
    ):
        self.model = model
        self.device = device

        self.lr_stage1 = lr_stage1
        self.epochs_stage1 = epochs_stage1
        self.patience_stage1 = patience_stage1

        self.lr_stage2 = lr_stage2
        self.epochs_stage2 = epochs_stage2
        self.rho_init = rho_init
        self.rho_max = rho_max
        self.alpha_max = alpha_max
        self.patience_stage2 = patience_stage2

        self.verbose = verbose
        self.history = defaultdict(list)

    def _freeze_causal_components(self):
        """Freeze FANTOM causal emission parameters."""
        for name, param in self.model.named_parameters():
            if 'causal_emission' in name or 'shared_dag' in name:
                param.requires_grad = False

    def _unfreeze_causal_components(self):
        """Unfreeze FANTOM causal emission parameters."""
        for name, param in self.model.named_parameters():
            if 'causal_emission' in name or 'shared_dag' in name:
                param.requires_grad = True

    def _freeze_ds3m_components(self):
        """Freeze DS3M encoder parameters."""
        for name, param in self.model.named_parameters():
            if any(comp in name for comp in ['rnn_forward', 'rnn_backward',
                                              'dposterior', 'zposterior', 'dtransition']):
                param.requires_grad = False

    def _unfreeze_ds3m_components(self):
        """Unfreeze DS3M encoder parameters."""
        for name, param in self.model.named_parameters():
            if any(comp in name for comp in ['rnn_forward', 'rnn_backward',
                                              'dposterior', 'zposterior', 'dtransition']):
                param.requires_grad = True

    def compute_validation_spearman(
        self,
        testX: torch.Tensor,
        testY: torch.Tensor
    ) -> float:
        """Compute Spearman correlation on validation set."""
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            pred_result = self.model.predict(testX, n_samples=50)
            predictions = pred_result['predictions'][-1].cpu().numpy()
            actuals = testY[-1].cpu().numpy()
            spearman, _ = spearmanr(actuals.flatten(), predictions.flatten())
        # Restore training mode
        if was_training:
            self.model.train()
        return float(spearman) if not np.isnan(spearman) else 0.0

    def train_stage1(
        self,
        trainX: torch.Tensor,
        trainY: torch.Tensor,
        testX: torch.Tensor,
        testY: torch.Tensor
    ) -> Dict:
        """
        Stage 1: Train DS3M encoder for regime detection.

        Uses simplified loss without DAG constraint:
        L = NLL + KL_z + KL_d
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("Stage 1: Training DS3M Encoder")
            print(f"{'='*60}")

        # Freeze causal components
        self._freeze_causal_components()

        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.lr_stage1)

        n_trainable = sum(p.numel() for p in trainable_params)
        if self.verbose:
            print(f"Trainable parameters: {n_trainable:,}")

        early_stopping = EarlyStopping(
            patience=self.patience_stage1,
            min_delta=0.001,
            mode='max',
            verbose=self.verbose
        )

        tracker = defaultdict(list)

        for epoch in range(self.epochs_stage1):
            # Ensure model is in training mode at start of each epoch
            self.model.train()

            optimizer.zero_grad()

            # Forward pass
            result = self.model(trainX, trainY)

            # Stage 1 loss: only DS3M components (no DAG penalty)
            loss = result['nll'] + result['kl_z'] + result['kl_d']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 10.0)
            optimizer.step()

            tracker['loss'].append(loss.item())
            tracker['nll'].append(result['nll'].item())
            tracker['kl_z'].append(result['kl_z'].item())
            tracker['kl_d'].append(result['kl_d'].item())

            # Validation (this will temporarily set eval mode)
            val_spearman = self.compute_validation_spearman(testX, testY)
            tracker['val_spearman'].append(val_spearman)

            if epoch % 10 == 0 and self.verbose:
                print(f"  Epoch {epoch}: loss={loss.item():.4f}, "
                      f"val_spearman={val_spearman:.4f}")

            # Early stopping
            if early_stopping(val_spearman, self.model.state_dict()):
                print(f"  Early stopping at epoch {epoch} "
                      f"(best Spearman: {early_stopping.best_score:.4f})")
                early_stopping.restore_best(self.model)
                break

        # Unfreeze for next stage
        self._unfreeze_causal_components()

        return dict(tracker)

    def train_stage2(
        self,
        trainX: torch.Tensor,
        trainY: torch.Tensor,
        testX: torch.Tensor,
        testY: torch.Tensor
    ) -> Dict:
        """
        Stage 2: Train FANTOM causal graphs with frozen DS3M encoder.

        Uses augmented Lagrangian for DAG constraint.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("Stage 2: Training Causal Graphs")
            print(f"{'='*60}")

        # Freeze DS3M components
        self._freeze_ds3m_components()

        # Get trainable parameters (only causal components)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.lr_stage2)

        n_trainable = sum(p.numel() for p in trainable_params)
        if self.verbose:
            print(f"Trainable parameters: {n_trainable:,}")

        early_stopping = EarlyStopping(
            patience=self.patience_stage2,
            min_delta=0.001,
            mode='max',
            verbose=self.verbose
        )

        # Augmented Lagrangian parameters
        alpha = 0.0
        rho = self.rho_init

        tracker = defaultdict(list)
        best_val_spearman = -float('inf')

        for epoch in range(self.epochs_stage2):
            # Ensure model is in training mode at start of each epoch
            self.model.train()

            optimizer.zero_grad()

            # Forward pass
            result = self.model(trainX, trainY)

            # Stage 2 loss with DAG constraint
            dag_penalty = result['dag_penalty']
            loss = (
                result['nll'] +
                result['sparse_penalty'] +
                alpha * dag_penalty +
                0.5 * rho * dag_penalty ** 2
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 10.0)
            optimizer.step()

            tracker['loss'].append(loss.item())
            tracker['nll'].append(result['nll'].item())
            tracker['dag_penalty'].append(dag_penalty.item())
            tracker['sparse_penalty'].append(result['sparse_penalty'].item())

            # Validation
            val_spearman = self.compute_validation_spearman(testX, testY)
            tracker['val_spearman'].append(val_spearman)

            if val_spearman > best_val_spearman:
                best_val_spearman = val_spearman

            if epoch % 10 == 0 and self.verbose:
                print(f"  Epoch {epoch}: loss={loss.item():.4f}, "
                      f"dag={dag_penalty.item():.4f}, "
                      f"val_spearman={val_spearman:.4f}")

            # Update Lagrangian multipliers every 10 epochs
            if (epoch + 1) % 10 == 0:
                current_dag = np.mean(tracker['dag_penalty'][-10:])
                alpha = min(alpha + rho * current_dag, self.alpha_max)
                if current_dag > 100:  # DAG constraint not satisfied
                    rho = min(rho * 2, self.rho_max)

            # Early stopping
            if early_stopping(val_spearman, self.model.state_dict()):
                print(f"  Early stopping at epoch {epoch} "
                      f"(best Spearman: {early_stopping.best_score:.4f})")
                early_stopping.restore_best(self.model)
                break

        # Unfreeze all
        self._unfreeze_ds3m_components()

        return dict(tracker)

    def train(
        self,
        trainX: torch.Tensor,
        trainY: torch.Tensor,
        testX: torch.Tensor,
        testY: torch.Tensor
    ) -> Dict:
        """Run full two-stage training."""
        start_time = time.time()

        # Stage 1: DS3M encoder
        stage1_tracker = self.train_stage1(trainX, trainY, testX, testY)
        self.history['stage1'] = stage1_tracker

        # Stage 2: Causal graphs
        stage2_tracker = self.train_stage2(trainX, trainY, testX, testY)
        self.history['stage2'] = stage2_tracker

        self.history['total_time'] = time.time() - start_time

        return dict(self.history)


def train_two_stage(
    model: DS3MCausal,
    trainX: torch.Tensor,
    trainY: torch.Tensor,
    testX: torch.Tensor,
    testY: torch.Tensor,
    config: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train DS3M-Causal model in two stages.

    Args:
        model: DS3MCausal model
        trainX, trainY: Training data [timestep, batch, dim]
        testX, testY: Test data
        config: Training configuration
        output_dir: Output directory for checkpoints
        verbose: Print progress

    Returns:
        Training results dictionary
    """
    device = next(model.parameters()).device

    # Move data to device
    trainX = trainX.to(device)
    trainY = trainY.to(device)
    testX = testX.to(device)
    testY = testY.to(device)

    # Create trainer
    trainer = TwoStageTrainer(
        model=model,
        device=device,
        lr_stage1=float(config.get('lr_stage1', 1e-3)),
        epochs_stage1=int(config.get('epochs_stage1', 100)),
        patience_stage1=int(config.get('patience_stage1', 15)),
        lr_stage2=float(config.get('lr_stage2', 5e-4)),
        epochs_stage2=int(config.get('epochs_stage2', 100)),
        patience_stage2=int(config.get('patience_stage2', 10)),
        rho_init=float(config.get('rho_init', 1.0)),
        rho_max=float(config.get('rho_max', 1e9)),
        verbose=verbose
    )

    # Train
    history = trainer.train(trainX, trainY, testX, testY)
    training_time = history['total_time']

    # Save checkpoint
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, checkpoint_dir / 'final.tar')

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_result = model.predict(testX, n_samples=100)

    # Compute metrics
    predictions = pred_result['predictions'][-1].cpu().numpy()
    actuals = testY[-1].cpu().numpy()

    spearman_corr, p_value = spearmanr(actuals.flatten(), predictions.flatten())
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # Get causal graphs
    graphs = model.get_causal_graphs()

    # Get final DAG penalty
    final_dag = None
    if 'dag_penalty' in history.get('stage2', {}):
        final_dag = history['stage2']['dag_penalty'][-1] if history['stage2']['dag_penalty'] else None

    results = {
        'training_time': training_time,
        'training_mode': 'two_stage',
        'final_dag_penalty': float(final_dag) if final_dag else None,
        'spearman': float(spearman_corr),
        'p_value': float(p_value),
        'rmse': float(rmse),
        'history': history,
        'graphs': [g.tolist() for g in graphs],
        'regime_distribution': np.unique(
            pred_result['regimes'][-1].cpu().numpy(), return_counts=True
        )
    }

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        results_json = {k: v for k, v in results.items()
                       if k not in ['graphs', 'history', 'regime_distribution']}
        results_json['regime_counts'] = dict(zip(
            results['regime_distribution'][0].tolist(),
            results['regime_distribution'][1].tolist()
        ))
        json.dump(results_json, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print("Two-Stage Training Complete")
        print(f"{'='*60}")
        print(f"Training time: {training_time:.2f}s")
        print(f"Spearman correlation: {spearman_corr:.4f} (p={p_value:.4e})")
        print(f"RMSE: {rmse:.4f}")
        if final_dag:
            print(f"Final DAG penalty: {final_dag:.4f}")

    return results


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    model = DS3MCausal(
        x_dim=10,
        y_dim=1,
        h_dim=16,
        z_dim=4,
        d_dim=2,
        device=device,
    ).to(device)

    timestep = 14
    batch_size = 32
    trainX = torch.randn(timestep, batch_size, 10, device=device)
    trainY = torch.randn(timestep, batch_size, 1, device=device)
    testX = torch.randn(timestep, batch_size // 2, 10, device=device)
    testY = torch.randn(timestep, batch_size // 2, 1, device=device)

    config = {
        'epochs_stage1': 10,
        'epochs_stage2': 10,
        'patience_stage1': 5,
        'patience_stage2': 5,
    }

    output_dir = Path("/tmp/ds3m_causal_2stage_test")
    output_dir.mkdir(exist_ok=True)

    results = train_two_stage(
        model, trainX, trainY, testX, testY, config, output_dir, verbose=True
    )

    print("\nTwo-stage training test passed!")
