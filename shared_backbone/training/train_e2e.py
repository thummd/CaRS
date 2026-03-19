"""
End-to-End Training for DS3M-Causal Hybrid

Implements joint training of:
- DS3M encoder and regime detection
- FANTOM causal graph discovery
- Augmented Lagrangian method for DAG constraint

Training follows FANTOM's augmented Lagrangian approach:
1. Inner loop: Optimize ELBO for fixed (rho, alpha)
2. Outer loop: Update Lagrangian multipliers when DAG constraint satisfied
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ds3m_causal import DS3MCausal
from modules.shared_dag import dagness_factor


class EarlyStopping:
    """Early stopping to prevent overfitting based on validation metric."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',  # 'max' for Spearman (higher is better), 'min' for loss
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, score: float, model_state: dict = None) -> bool:
        if self.best_score is None:
            self.best_score = score
            if model_state is not None:
                self.best_state = {k: v.cpu().clone() for k, v in model_state.items()}
        else:
            improved = (
                (self.mode == 'max' and score > self.best_score + self.min_delta) or
                (self.mode == 'min' and score < self.best_score - self.min_delta)
            )
            if improved:
                self.best_score = score
                self.counter = 0
                if model_state is not None:
                    self.best_state = {k: v.cpu().clone() for k, v in model_state.items()}
            else:
                self.counter += 1
                if self.verbose:
                    print(f"  EarlyStopping: {self.counter}/{self.patience} (best={self.best_score:.4f})")
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop

    def restore_best(self, model: nn.Module) -> None:
        """Restore best model state."""
        if self.best_state is not None:
            device = next(model.parameters()).device
            model.load_state_dict({k: v.to(device) for k, v in self.best_state.items()})


class AugmentedLagrangianTrainer:
    """
    Augmented Lagrangian trainer for DAG-constrained optimization.

    Uses the AL method from NOTEARS/DECI:
    L_AL = ELBO + alpha * h(A) + (rho/2) * h(A)^2

    where h(A) = tr(exp(A)) - d is the DAGness constraint.

    Training proceeds in outer and inner loops:
    - Inner loop: Optimize with fixed (alpha, rho)
    - Outer loop: Update alpha <- alpha + rho * h(A)
                  Increase rho if progress is slow
    """

    def __init__(
        self,
        model: DS3MCausal,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        # Augmented Lagrangian parameters
        alpha_init: float = 0.0,
        rho_init: float = 1.0,
        rho_max: float = 1e9,
        alpha_max: float = 1e9,
        progress_rate: float = 0.9,
        tol_dag: float = 1e-6,
        # Training parameters
        max_auglag_steps: int = 50,
        max_inner_epochs: int = 50,
        patience_dag: int = 5,
        patience_rho: int = 3,
        # Early stopping on validation Spearman
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.001,
        # Target constraint parameters
        target_idx: int = 0,           # Index of target variable (Day_Ahead_Price)
        lambda_target: float = 10.0,   # Weight for target constraint
        # Temperature annealing for sparse edges
        tau_init: float = 1.0,         # Initial Gumbel-Softmax temperature
        tau_final: float = 0.1,        # Final temperature (lower = more sparse/binary)
        tau_anneal_steps: int = 100,   # Steps over which to anneal temperature
        # Regime differentiation
        lambda_regime_diff: float = 1.0,  # Weight for regime differentiation penalty
        # Early stopping metric
        early_stopping_metric: str = 'directional_accuracy',  # 'directional_accuracy' or 'spearman'
        # Logging
        verbose: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.alpha = alpha_init
        self.rho = rho_init
        self.rho_max = rho_max
        self.alpha_max = alpha_max
        self.progress_rate = progress_rate
        self.tol_dag = tol_dag

        self.max_auglag_steps = max_auglag_steps
        self.max_inner_epochs = max_inner_epochs
        self.patience_dag = patience_dag
        self.patience_rho = patience_rho

        self.verbose = verbose

        # Target constraint parameters
        self.target_idx = target_idx
        self.lambda_target = lambda_target

        # Temperature annealing parameters
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.tau_anneal_steps = tau_anneal_steps
        self.current_tau = tau_init

        # Regime differentiation
        self.lambda_regime_diff = lambda_regime_diff

        # Early stopping metric
        self.early_stopping_metric = early_stopping_metric

        # Early stopping based on validation metric
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode='max',  # Higher is better for both directional_accuracy and spearman
            verbose=verbose
        )

        # History tracking
        self.history = defaultdict(list)

    def update_temperature(self, step: int) -> float:
        """
        Anneal Gumbel-Softmax temperature for sparse edge convergence.

        Lower temperature forces edges toward binary (0/1) values.
        Anneals from tau_init to tau_final over tau_anneal_steps.

        Args:
            step: Current augmented Lagrangian step

        Returns:
            Current temperature value
        """
        if self.tau_anneal_steps <= 0:
            return self.tau_init

        progress = min(1.0, step / self.tau_anneal_steps)
        # Exponential annealing: tau = tau_init * (tau_final/tau_init)^progress
        tau = self.tau_init * (self.tau_final / self.tau_init) ** progress
        self.current_tau = tau

        # Update tau_gumbel in model's DAG distribution
        if hasattr(self.model, 'dag_dist'):
            dag_dist = self.model.dag_dist
            # Update shared backbone tau
            if dag_dist.var_dist_A_shared is not None:
                dag_dist.var_dist_A_shared.tau_gumbel = tau
            # Update regime-specific tau
            for var_dist in dag_dist.var_dist_A_regime:
                var_dist.tau_gumbel = tau

        return tau

    def compute_augmented_loss(
        self,
        result: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute augmented Lagrangian loss with target and regime differentiation constraints.

        L = ELBO + alpha * h(A) + (rho/2) * h(A)^2 + lambda_target * target_loss
            + lambda_regime_diff * regime_diff_loss

        The target constraint encourages edges TO the target variable (Price).
        The regime differentiation constraint penalizes similar DAGs across regimes.

        Args:
            result: Forward pass result containing dag_penalty and adj_matrices

        Returns:
            Tuple of (augmented loss, target loss, regime diff loss for logging)
        """
        elbo = result['nll'] + result['kl_z'] + result['kl_d'] + result['sparse_penalty']
        dag_penalty = result['dag_penalty']

        # Target constraint: encourage edges TO the target variable
        adj_matrices = result.get('adj_matrices', None)
        target_loss = torch.tensor(0.0, device=self.device)
        regime_diff_loss = torch.tensor(0.0, device=self.device)

        if adj_matrices is not None:
            # Target constraint
            if self.lambda_target > 0:
                for A in adj_matrices:  # For each regime
                    # A has shape [lag+1, num_nodes, num_nodes]
                    # Sum of edges TO target (column target_idx) from all lags
                    edges_to_target = A[:, :, self.target_idx].sum()
                    # Penalize if few edges to target (negative log encourages higher values)
                    # Adding 0.1 for numerical stability
                    target_loss += -torch.log(edges_to_target + 0.1)

            # Regime differentiation: penalize if DAGs are too similar
            # Using negative L2 distance (maximize distance = minimize negative distance)
            if len(adj_matrices) >= 2 and self.lambda_regime_diff > 0:
                # Compare all pairs of regime DAGs
                for i in range(len(adj_matrices)):
                    for j in range(i + 1, len(adj_matrices)):
                        A_i = adj_matrices[i]
                        A_j = adj_matrices[j]
                        # L2 distance between DAGs (negative because we want to maximize it)
                        l2_dist = torch.sqrt(((A_i - A_j) ** 2).sum() + 1e-8)
                        # Penalize small distance (want DAGs to be different)
                        # Using negative distance so minimizing loss = maximizing distance
                        regime_diff_loss = regime_diff_loss - l2_dist

        augmented = (
            elbo +
            self.alpha * dag_penalty +
            0.5 * self.rho * dag_penalty ** 2 +
            self.lambda_target * target_loss +
            self.lambda_regime_diff * regime_diff_loss
        )

        return augmented, target_loss, regime_diff_loss

    def train_inner(
        self,
        trainX: torch.Tensor,
        trainY: torch.Tensor,
        testX: Optional[torch.Tensor] = None,
        testY: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, Dict]:
        """
        Run inner optimization loop for fixed (alpha, rho).

        Args:
            trainX: Training input [timestep, batch, x_dim]
            trainY: Training target [timestep, batch, y_dim]
            testX: Optional test input
            testY: Optional test target

        Returns:
            done: Whether inner optimization converged
            tracker: Loss terms history
        """
        self.model.train()
        tracker = defaultdict(list)

        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        for epoch in range(self.max_inner_epochs):
            self.optimizer.zero_grad()

            # Forward pass
            result = self.model(trainX, trainY)

            # Compute augmented loss (includes target and regime diff constraints)
            loss, target_loss, regime_diff_loss = self.compute_augmented_loss(result)

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

            self.optimizer.step()

            # Track losses
            tracker['loss'].append(loss.item())
            tracker['nll'].append(result['nll'].item())
            tracker['kl_z'].append(result['kl_z'].item())
            tracker['kl_d'].append(result['kl_d'].item())
            tracker['dag_penalty'].append(result['dag_penalty'].item())
            tracker['sparse_penalty'].append(result['sparse_penalty'].item())
            tracker['target_loss'].append(target_loss.item())
            tracker['regime_diff_loss'].append(regime_diff_loss.item())

            # Test loss
            if testX is not None and testY is not None:
                with torch.no_grad():
                    test_result = self.model(testX, testY)
                    test_loss, _, _ = self.compute_augmented_loss(test_result)
                    tracker['test_loss'].append(test_loss.item())

            # Check convergence
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 10:
                break

            if epoch % 10 == 0 and self.verbose:
                print(f"    Inner epoch {epoch}: loss={loss.item():.4f}, "
                      f"dag={result['dag_penalty'].item():.6f}, "
                      f"target={target_loss.item():.4f}, "
                      f"regime_diff={regime_diff_loss.item():.4f}, "
                      f"tau={self.current_tau:.4f}")

        done = patience_counter >= 10 or epoch >= self.max_inner_epochs - 1

        return done, dict(tracker)

    def compute_validation_metrics(
        self,
        testX: torch.Tensor,
        testY: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute validation metrics: RMSE, MAE, sMAPE, Spearman, directional
        accuracy, and CRPS (if prediction uncertainty available).

        Args:
            testX: Test input tensor
            testY: Test target tensor

        Returns:
            Dict with metric names as keys
        """
        from electricity.evaluation.metrics import compute_all_metrics

        self.model.eval()
        with torch.no_grad():
            pred_result = self.model.predict(testX, n_samples=50)
            predictions = pred_result['predictions'][-1].cpu().numpy()
            predictions_std = pred_result['predictions_std'][-1].cpu().numpy()
            actuals = testY[-1].cpu().numpy()
            prev_actuals = testY[-2].cpu().numpy()

        self.model.train()
        return compute_all_metrics(
            y_true=actuals,
            y_pred=predictions,
            y_prev=prev_actuals,
            y_pred_std=predictions_std
        )

    def train(
        self,
        trainX: torch.Tensor,
        trainY: torch.Tensor,
        testX: Optional[torch.Tensor] = None,
        testY: Optional[torch.Tensor] = None,
    ) -> Dict[str, List]:
        """
        Run full augmented Lagrangian training.

        Args:
            trainX, trainY: Training data
            testX, testY: Test data (optional)

        Returns:
            History dictionary with all loss terms
        """
        dag_penalty_prev = float('inf')
        num_below_tol = 0
        num_max_rho = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print("Augmented Lagrangian Training")
            print(f"{'='*60}")

        for step in range(self.max_auglag_steps):
            # Update Gumbel-Softmax temperature for sparse edge convergence
            tau = self.update_temperature(step)

            if self.verbose:
                print(f"\nAugLag Step {step}: alpha={self.alpha:.4f}, rho={self.rho:.4f}, tau={tau:.4f}")

            # Check stopping conditions
            if num_below_tol >= self.patience_dag:
                print(f"DAG penalty below tolerance for {self.patience_dag} steps. Stopping.")
                break
            if num_max_rho >= self.patience_rho:
                print(f"At max rho for {self.patience_rho} steps. Stopping.")
                break

            # Inner optimization
            done_inner, tracker = self.train_inner(trainX, trainY, testX, testY)

            dag_penalty = np.mean(tracker['dag_penalty'][-10:])

            # Compute validation metrics and check early stopping
            if testX is not None and testY is not None:
                val_metrics = self.compute_validation_metrics(testX, testY)
                self.history['val_spearman'].append(val_metrics['spearman'])
                self.history['val_directional_accuracy'].append(val_metrics.get('directional_accuracy', 0.0))
                self.history['val_rmse'].append(val_metrics.get('rmse', 0.0))
                self.history['val_mae'].append(val_metrics.get('mae', 0.0))
                self.history['val_crps'].append(val_metrics.get('crps', 0.0))
                self.history['tau'].append(tau)

                if self.verbose:
                    parts = [f"Spearman={val_metrics['spearman']:.4f}"]
                    if 'rmse' in val_metrics:
                        parts.append(f"RMSE={val_metrics['rmse']:.4f}")
                    if 'directional_accuracy' in val_metrics:
                        parts.append(f"DirAcc={val_metrics['directional_accuracy']:.4f}")
                    if 'crps' in val_metrics:
                        parts.append(f"CRPS={val_metrics['crps']:.4f}")
                    print(f"  Validation: {', '.join(parts)}")

                # Early stopping based on selected metric
                early_stop_metric = val_metrics[self.early_stopping_metric]
                if self.early_stopping(early_stop_metric, self.model.state_dict()):
                    print(f"Early stopping triggered at step {step} "
                          f"(best {self.early_stopping_metric}: {self.early_stopping.best_score:.4f})")
                    # Restore best model
                    self.early_stopping.restore_best(self.model)
                    break

            # Update history
            self.history['auglag_step'].append(step)
            self.history['alpha'].append(self.alpha)
            self.history['rho'].append(self.rho)
            self.history['dag_penalty'].append(dag_penalty)
            self.history['loss'].extend(tracker['loss'])

            if self.verbose:
                print(f"  Final dag_penalty: {dag_penalty:.8f}")

            # Check DAG tolerance
            if dag_penalty < self.tol_dag:
                num_below_tol += 1
            else:
                num_below_tol = 0

            if self.rho >= self.rho_max:
                num_max_rho += 1

            # Update Lagrangian multipliers
            if done_inner:
                if dag_penalty > dag_penalty_prev * self.progress_rate:
                    # Not enough progress, increase rho
                    if self.verbose:
                        print(f"  Increasing rho: {self.rho} -> {self.rho * 10}")
                    self.rho = min(self.rho * 10, self.rho_max)
                else:
                    # Good progress, update alpha
                    if self.verbose:
                        print(f"  Updating alpha: {self.alpha} -> {self.alpha + self.rho * dag_penalty}")
                    dag_penalty_prev = dag_penalty
                    self.alpha = min(self.alpha + self.rho * dag_penalty, self.alpha_max)

        return dict(self.history)


def train_end_to_end(
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
    Train DS3M-Causal model end-to-end.

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

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 0.0)
    )

    # Create trainer (ensure float conversion for scientific notation from YAML)
    trainer = AugmentedLagrangianTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        alpha_init=float(config.get('alpha_init', 0.0)),
        rho_init=float(config.get('rho_init', 1.0)),
        rho_max=float(config.get('rho_max', 1e9)),
        progress_rate=float(config.get('progress_rate', 0.9)),
        tol_dag=float(config.get('tol_dag', 1e-6)),
        max_auglag_steps=int(config.get('max_auglag_steps', 50)),
        max_inner_epochs=int(config.get('max_inner_epochs', 50)),
        # Early stopping parameters
        early_stopping_patience=int(config.get('early_stopping_patience', 10)),
        early_stopping_min_delta=float(config.get('early_stopping_min_delta', 0.001)),
        # Target constraint parameters
        target_idx=int(config.get('target_idx', 0)),
        lambda_target=float(config.get('lambda_target', 10.0)),
        # Temperature annealing
        tau_init=float(config.get('tau_init', 1.0)),
        tau_final=float(config.get('tau_final', 0.1)),
        tau_anneal_steps=int(config.get('tau_anneal_steps', 100)),
        # Regime differentiation
        lambda_regime_diff=float(config.get('lambda_regime_diff', 1.0)),
        # Early stopping metric
        early_stopping_metric=config.get('early_stopping_metric', 'directional_accuracy'),
        verbose=verbose,
    )

    # Train
    start_time = time.time()
    history = trainer.train(trainX, trainY, testX, testY)
    training_time = time.time() - start_time

    # Save checkpoint
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'alpha': trainer.alpha,
        'rho': trainer.rho,
        'config': config,
    }, checkpoint_dir / 'final.tar')

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_result = model.predict(testX, n_samples=100)

    # Compute metrics
    predictions = pred_result['predictions'][-1].cpu().numpy()  # Last timestep
    actuals = testY[-1].cpu().numpy()  # Last timestep

    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(actuals.flatten(), predictions.flatten())
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # Compute directional accuracy
    # Compare predicted direction vs actual direction relative to previous timestep
    prev_actuals = testY[-2].cpu().numpy()  # Second to last timestep
    actual_direction = np.sign(actuals - prev_actuals)  # +1 for up, -1 for down, 0 for no change
    pred_direction = np.sign(predictions - prev_actuals)
    # Only count samples where there was actual movement
    valid_mask = actual_direction.flatten() != 0
    if valid_mask.sum() > 0:
        directional_accuracy = float(np.mean(actual_direction.flatten()[valid_mask] == pred_direction.flatten()[valid_mask]))
    else:
        directional_accuracy = 0.5  # Default if no movement

    # Get causal graphs
    graphs = model.get_causal_graphs()

    results = {
        'training_time': training_time,
        'final_alpha': trainer.alpha,
        'final_rho': trainer.rho,
        'final_dag_penalty': float(history['dag_penalty'][-1]) if history['dag_penalty'] else None,
        'spearman': float(spearman_corr),
        'rmse': float(rmse),
        'directional_accuracy': directional_accuracy,
        'history': history,
        'graphs': [g.tolist() for g in graphs],
        'regime_distribution': np.unique(
            pred_result['regimes'][-1].cpu().numpy(), return_counts=True
        )
    }

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        # Convert numpy arrays for JSON
        results_json = {k: v for k, v in results.items() if k not in ['graphs', 'history', 'regime_distribution']}
        results_json['regime_counts'] = dict(zip(
            results['regime_distribution'][0].tolist(),
            results['regime_distribution'][1].tolist()
        ))
        json.dump(results_json, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print("Training Complete")
        print(f"{'='*60}")
        print(f"Training time: {training_time:.2f}s")
        print(f"Spearman correlation: {spearman_corr:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Directional accuracy: {directional_accuracy:.4f}")
        print(f"Final DAG penalty: {results['final_dag_penalty']:.8f}")

    return results


if __name__ == "__main__":
    # Quick test of training pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Create model
    model = DS3MCausal(
        x_dim=10,
        y_dim=1,
        h_dim=16,
        z_dim=4,
        d_dim=2,
        device=device,
    ).to(device)

    # Create dummy data
    timestep = 14
    batch_size = 32
    trainX = torch.randn(timestep, batch_size, 10, device=device)
    trainY = torch.randn(timestep, batch_size, 1, device=device)
    testX = torch.randn(timestep, batch_size // 2, 10, device=device)
    testY = torch.randn(timestep, batch_size // 2, 1, device=device)

    # Config
    config = {
        'learning_rate': 1e-3,
        'max_auglag_steps': 3,
        'max_inner_epochs': 5,
        'rho_init': 1.0,
    }

    # Train
    output_dir = Path("/tmp/ds3m_causal_test")
    output_dir.mkdir(exist_ok=True)

    results = train_end_to_end(
        model, trainX, trainY, testX, testY, config, output_dir, verbose=True
    )

    print("\nTraining test passed!")
