"""
FANTOM with regime detection for electricity price prediction.

Implements the EM-style regime detection from the original FANTOM paper,
adapted for electricity market data.

Usage:
    python fantom_regime.py --country DE --n_regimes 3
"""

import sys
import os
from pathlib import Path

from paths import FANTOM_CODE_DIR
# Add FANTOM code to path
FANTOM_PATH = str(FANTOM_CODE_DIR)
sys.path.insert(0, FANTOM_PATH)
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import yaml

from fantom import FANTOM_stationary
from data_loader import ElectricityDataset
from fantom_electricity import FANTOMElectricity, create_model
from sklearn.cluster import KMeans


class RegimePriorNetwork(nn.Module):
    """
    Neural network for predicting regime probabilities over time.

    Maps time index to regime probability distribution.
    """

    def __init__(self, n_regimes: int, hidden_dim: int = 32):
        super().__init__()
        self.n_regimes = n_regimes
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Predict regime probabilities.

        Args:
            t: Time indices of shape [N, 1]

        Returns:
            Regime probabilities of shape [N, n_regimes]
        """
        logits = self.network(t)
        return self.softmax(logits)


def train_prior_network(
    model: RegimePriorNetwork,
    t: torch.Tensor,
    gamma: torch.Tensor,
    n_epochs: int = 500,
    lr: float = 0.001,
    threshold: float = 0.85
) -> Tuple[torch.Tensor, float, RegimePriorNetwork]:
    """
    Train the prior network to predict regime assignments.

    Args:
        model: Prior network
        t: Time indices [N, 1]
        gamma: Regime assignments [N, n_regimes] (one-hot or soft)
        n_epochs: Number of training epochs
        lr: Learning rate
        threshold: Loss threshold for convergence

    Returns:
        p: Predicted probabilities
        loss: Final loss value
        model: Trained model
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        p = model(t)
        loss = criterion(p, gamma)
        loss.backward()
        optimizer.step()

    # Continue training if loss is above threshold (with max iterations to prevent infinite loops)
    max_extra_iterations = 50  # Cap to prevent infinite loops with high n_regimes
    iteration = 0
    while loss.item() >= threshold and iteration < max_extra_iterations:
        for _ in range(100):
            optimizer.zero_grad()
            p = model(t)
            loss = criterion(p, gamma)
            loss.backward()
            optimizer.step()

        iteration += 1
        if loss.item() < threshold:
            break

    if iteration >= max_extra_iterations and loss.item() >= threshold:
        import warnings
        warnings.warn(f"Prior network did not converge below threshold {threshold} "
                      f"after {max_extra_iterations} extra iterations (final loss: {loss.item():.4f})")

    model.eval()
    with torch.no_grad():
        p = model(t)

    return p, loss.item(), model


class FANTOMRegime:
    """
    FANTOM with regime detection.

    Uses EM algorithm to:
    1. Detect different regimes (structural changes) in the data
    2. Learn separate causal structures per regime
    3. Combine regime-specific predictions
    """

    def __init__(
        self,
        n_nodes: int,
        target_idx: int,
        lag: int = 1,
        device: str = "cpu",
        initial_n_regimes: int = 3,
        window_size: int = 200,
        min_regime_size: int = 100,
        max_iterations: int = 3,
        prior_threshold: float = 0.85,
        model_config: Optional[Dict] = None,
        training_params: Optional[Dict] = None,
        use_soft_assignments: bool = True,
        entropy_weight: float = 0.5,
        init_method: str = 'kmeans',
        prune_after_iteration: int = 2
    ):
        """
        Initialize FANTOM with regime detection.

        Args:
            n_nodes: Number of variables
            target_idx: Index of TARGET variable
            lag: Temporal lag
            device: PyTorch device
            initial_n_regimes: Starting number of regimes
            window_size: Initial window size for regime initialization
            min_regime_size: Minimum samples per regime (zeta)
            max_iterations: Number of EM iterations
            prior_threshold: Loss threshold for prior network
            model_config: FANTOM model configuration
            training_params: Training parameters
            use_soft_assignments: If True, use soft (probabilistic) assignments in E-step
            entropy_weight: Weight for entropy regularization (encourages diverse regimes)
            init_method: 'kmeans' for feature-based init, 'temporal' for time-window init
            prune_after_iteration: Only prune small regimes after this iteration
        """
        self.n_nodes = n_nodes
        self.target_idx = target_idx
        self.lag = lag
        self.device = torch.device(device)
        self.initial_n_regimes = initial_n_regimes
        self.window_size = window_size
        self.min_regime_size = min_regime_size
        self.max_iterations = max_iterations
        self.prior_threshold = prior_threshold
        self.use_soft_assignments = use_soft_assignments
        self.entropy_weight = entropy_weight
        self.init_method = init_method
        self.prune_after_iteration = prune_after_iteration

        # Default configurations
        self.model_config = model_config or {
            'lambda_dag': 100.0,
            'lambda_sparse': 1.0,
            'lambda_sparse_l2': 0.0,
            'l2_group_mode': 'column',
            'tau_gumbel': 1.0,
            'base_distribution_type': 'spline',
            'spline_bins': 8,
            'encoder_layer_sizes': [32, 32],
            'decoder_layer_sizes': [32, 32],
            'heteroscedastic': True,
            'allow_instantaneous': True,
            'constrain_target': True,
        }

        self.training_params = training_params or {
            'batch_size': 32,
            'learning_rate': 0.001,
            'max_steps_auglag': 10,
            'max_auglag_inner_epochs': 1000,
            'rho': 1.0,
            'alpha': 0.0,
            'progress_rate': 0.9,
            'safety_rho': 1e9,
            'safety_alpha': 1e9,
            'tol_dag': 1e-6,
            'anneal_entropy': 'noanneal',
            'reconstruction_loss_factor': 1.0,
        }

        # Will be set during training
        self.regime_models: List[FANTOMElectricity] = []
        self.prior_network: Optional[RegimePriorNetwork] = None
        self.gamma_hat: Optional[np.ndarray] = None
        self.n_regimes: int = initial_n_regimes

    def _standardize_per_regime(
        self,
        X: np.ndarray,
        gamma: np.ndarray,
        regime_idx: int
    ) -> np.ndarray:
        """Standardize data for a specific regime."""
        mask = gamma[:, regime_idx] > 0.5
        if mask.sum() == 0:
            return X

        X_regime = X[mask].copy()
        for t in range(X_regime.shape[1]):
            mean = X_regime[:, t, :].mean(axis=0)
            std = X_regime[:, t, :].std(axis=0)
            std[std == 0] = 1
            X_regime[:, t, :] = (X_regime[:, t, :] - mean) / std

        return X_regime

    def fit(
        self,
        X: np.ndarray,
        verbose: bool = True,
        checkpoint_dir: Path = None
    ) -> 'FANTOMRegime':
        """
        Fit FANTOM with regime detection.

        Args:
            X: Data of shape [N, lag+1, n_nodes]
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints (optional)

        Returns:
            self
        """
        m = X.shape[0]
        n_regimes = self.initial_n_regimes

        # Standardized data for emission computation
        X_std = X.copy()
        for t in range(X_std.shape[1]):
            mean = X_std[:, t, :].mean(axis=0)
            std = X_std[:, t, :].std(axis=0)
            std[std == 0] = 1
            X_std[:, t, :] = (X_std[:, t, :] - mean) / std

        # Initialize regime probabilities
        p = np.zeros((m, n_regimes))

        if self.init_method == 'kmeans':
            # Use k-means clustering on features for better initialization
            # Flatten temporal dimension for clustering
            X_flat = X_std.reshape(m, -1)  # [N, (lag+1)*n_nodes]

            if verbose:
                print(f"Initializing regimes with k-means clustering...")

            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_flat)

            # Convert to soft probabilities (add some uncertainty)
            for i in range(m):
                # Compute distance to each cluster center
                distances = np.linalg.norm(X_flat[i] - kmeans.cluster_centers_, axis=1)
                # Convert to probabilities using softmax with temperature
                temperature = 1.0
                exp_neg_dist = np.exp(-distances / temperature)
                p[i] = exp_neg_dist / exp_neg_dist.sum()

            if verbose:
                print(f"  K-means cluster sizes: {np.bincount(labels)}")

        else:
            # Initialize regime probabilities based on time windows (original method)
            for c in range(n_regimes):
                if c == n_regimes - 1:
                    p[c * self.window_size:, c] = 1.0
                else:
                    p[c * self.window_size:(c + 1) * self.window_size, c] = 1.0

        # EM iterations
        for it in range(self.max_iterations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"EM Iteration {it + 1}/{self.max_iterations}")
                print(f"{'='*60}")

            # Create models for each regime
            self.regime_models = []
            log_pdf_emission = np.zeros((m, n_regimes))

            for c in range(n_regimes):
                if verbose:
                    print(f"\nTraining regime {c + 1}/{n_regimes}...")

                # Create model for this regime
                model = create_model(
                    num_nodes=self.n_nodes,
                    target_idx=self.target_idx,
                    lag=self.lag,
                    device=str(self.device),
                    model_config=self.model_config
                )

                if it == 0:
                    # Initial training on time windows
                    if c == n_regimes - 1:
                        regime_data = X[c * self.window_size:]
                    else:
                        regime_data = X[c * self.window_size:(c + 1) * self.window_size]

                    # Standardize regime data
                    regime_data_std = regime_data.copy()
                    for t in range(regime_data.shape[1]):
                        mean = regime_data[:, t, :].mean(axis=0)
                        std = regime_data[:, t, :].std(axis=0)
                        std[std == 0] = 1
                        regime_data_std[:, t, :] = (regime_data[:, t, :] - mean) / std

                else:
                    # Training on regime-assigned data
                    gamma = gamma_hat[:, c]

                    if self.use_soft_assignments:
                        # FIX: With soft assignments, use weighted sampling instead of hard mask
                        # Sample indices proportionally to their regime probability
                        # Ensure minimum number of high-probability samples
                        high_prob_mask = gamma > 0.3  # Lower threshold for soft assignments
                        n_high_prob = high_prob_mask.sum()

                        if n_high_prob < 10:
                            if verbose:
                                print(f"  Regime {c} has too few high-prob samples ({n_high_prob}), skipping")
                            self.regime_models.append(None)
                            continue

                        # Use weighted sampling to select training samples
                        # Higher gamma = higher probability of selection
                        sample_weights = gamma / gamma.sum()
                        n_samples = min(int(gamma.sum()), len(X))  # Effective sample size
                        n_samples = max(n_samples, 50)  # Minimum samples

                        # Sample with replacement based on weights
                        np.random.seed(42 + c)  # Reproducible sampling
                        selected_indices = np.random.choice(
                            len(X), size=n_samples, replace=True, p=sample_weights
                        )
                        regime_data = X[selected_indices]
                    else:
                        # Original hard assignment behavior
                        mask = gamma > 0.5
                        regime_data = X[mask]

                        if len(regime_data) < 10:
                            if verbose:
                                print(f"  Regime {c} has too few samples ({len(regime_data)}), skipping")
                            self.regime_models.append(None)
                            continue

                    # Standardize
                    regime_data_std = regime_data.copy()
                    for t in range(regime_data.shape[1]):
                        mean = regime_data[:, t, :].mean(axis=0)
                        std = regime_data[:, t, :].std(axis=0)
                        std[std == 0] = 1
                        regime_data_std[:, t, :] = (regime_data[:, t, :] - mean) / std

                # Train model with error handling
                dataloader = DataLoader(
                    torch.tensor(regime_data_std, dtype=torch.float32),
                    batch_size=self.training_params['batch_size'],
                    shuffle=True
                )

                try:
                    model.train()
                    model.run_train(
                        dataloader=dataloader,
                        num_samples=len(regime_data),
                        train_config_dict=self.training_params
                    )

                    self.regime_models.append(model)

                    # Compute emission probabilities
                    model.eval()
                    with torch.no_grad():
                        log_prob = model.log_prob(
                            torch.tensor(X_std, dtype=torch.float32),
                            Nsamples_per_graph=1
                        )
                        log_pdf_emission[:, c] = np.exp(log_prob)

                except Exception as e:
                    print(f"  Training failed for regime {c}: {e}")
                    self.regime_models.append(None)
                    if checkpoint_dir:
                        self._save_partial_results(checkpoint_dir, c, gamma_hat if 'gamma_hat' in dir() else None)
                    continue

            # E-step: Compute regime assignments with entropy regularization
            # Add entropy bonus to encourage diverse regime assignments
            if self.entropy_weight > 0:
                # Compute entropy of emission distribution per sample
                eps = 1e-10
                emission_probs = log_pdf_emission / (log_pdf_emission.sum(axis=1, keepdims=True) + eps)
                # Add entropy bonus (higher entropy = more uniform = better)
                entropy_bonus = -np.sum(emission_probs * np.log(emission_probs + eps), axis=1, keepdims=True)
                # Scale and add to emissions
                log_pdf_emission_reg = log_pdf_emission * (1 + self.entropy_weight * entropy_bonus)
            else:
                log_pdf_emission_reg = log_pdf_emission

            pall = np.sum(p * log_pdf_emission_reg, axis=1, keepdims=True)
            pall[pall == 0] = 1e-10  # Avoid division by zero

            gamma_hat = (p * log_pdf_emission_reg) / pall
            gamma_hat = gamma_hat / gamma_hat.sum(axis=1, keepdims=True)

            # FIX: Use soft or hard assignments based on setting
            if self.use_soft_assignments:
                # Keep soft (probabilistic) assignments - prevents winner-takes-all collapse
                # Add small temperature to avoid completely deterministic assignments
                temperature = 0.5
                gamma_hat = np.exp(np.log(gamma_hat + 1e-10) / temperature)
                gamma_hat = gamma_hat / gamma_hat.sum(axis=1, keepdims=True)
                if verbose:
                    print(f"  Using soft assignments (avg entropy: {-np.mean(np.sum(gamma_hat * np.log(gamma_hat + 1e-10), axis=1)):.3f})")
            else:
                # Hard assignment (original behavior)
                idx = np.argmax(gamma_hat, axis=-1)
                gamma_hat = np.zeros_like(gamma_hat)
                gamma_hat[np.arange(m), idx] = 1

            # Train prior network
            t = torch.tensor(
                np.linspace(0, 20 * n_regimes, m).reshape((m, 1)),
                dtype=torch.float32
            )
            self.prior_network = RegimePriorNetwork(n_regimes)
            p, loss, self.prior_network = train_prior_network(
                self.prior_network,
                t,
                torch.tensor(gamma_hat, dtype=torch.float32),
                threshold=self.prior_threshold
            )
            p = p.detach().numpy()

            # FIX: Only prune small regimes after specified iteration
            # This prevents premature pruning before regimes have a chance to develop
            if it >= self.prune_after_iteration:
                gamma_sum = np.sum(gamma_hat, axis=0)
                valid_regimes = gamma_sum >= self.min_regime_size

                # Make sure we handle the case where regime_models might have different length
                # due to failed training attempts
                if len(self.regime_models) == len(valid_regimes):
                    self.regime_models = [m for i, m in enumerate(self.regime_models) if valid_regimes[i]]
                else:
                    # Models list doesn't match - keep only models that have valid index
                    new_models = []
                    for i, valid in enumerate(valid_regimes):
                        if valid and i < len(self.regime_models):
                            new_models.append(self.regime_models[i])
                    self.regime_models = new_models

                gamma_hat = gamma_hat[:, valid_regimes]
                p = p[:, valid_regimes]
                n_regimes = int(valid_regimes.sum())

                if verbose:
                    print(f"  Pruned to {n_regimes} regimes (min_size={self.min_regime_size})")
            else:
                if verbose:
                    gamma_sum = np.sum(gamma_hat, axis=0)
                    print(f"  Skipping pruning (iteration {it+1} < {self.prune_after_iteration})")
                    print(f"  Current regime sizes: {gamma_sum}")

            if verbose:
                print(f"\nIteration {it + 1} results:")
                print(f"  Active regimes: {n_regimes}")
                print(f"  Samples per regime: {np.sum(gamma_hat, axis=0)}")
                print(f"  Prior network loss: {loss:.4f}")

            # Save checkpoint after each iteration
            self.gamma_hat = gamma_hat
            self.n_regimes = n_regimes
            if checkpoint_dir:
                self._save_checkpoint(it + 1, checkpoint_dir)

        self.gamma_hat = gamma_hat
        self.n_regimes = n_regimes

        return self

    def predict_target(
        self,
        X: np.ndarray,
        use_regime_weights: bool = True
    ) -> np.ndarray:
        """
        Predict TARGET using regime-specific models.

        Args:
            X: Data of shape [N, lag+1, n_nodes]
            use_regime_weights: Weight predictions by regime probabilities

        Returns:
            Predictions of shape [N]
        """
        m = X.shape[0]
        predictions = np.zeros((m, self.n_regimes))

        # Get regime probabilities
        if use_regime_weights and self.prior_network is not None:
            t = torch.tensor(
                np.linspace(0, 20 * self.n_regimes, m).reshape((m, 1)),
                dtype=torch.float32
            )
            with torch.no_grad():
                regime_probs = self.prior_network(t).numpy()
        else:
            regime_probs = self.gamma_hat if self.gamma_hat is not None else np.ones((m, self.n_regimes)) / self.n_regimes

        # NOTE: Input X is already normalized by prepare_unified_fantom_data()
        # Do NOT apply per-batch standardization here - it causes double-normalization
        # which corrupts the scale and leads to inconsistent RMSE across seeds
        # (while Spearman stays reasonable since it's rank-based)

        # Get predictions from each regime model
        for c, model in enumerate(self.regime_models):
            if model is not None:
                model.eval()
                with torch.no_grad():
                    pred = model.predict_target(
                        torch.tensor(X, dtype=torch.float32)
                    ).cpu().numpy()
                    predictions[:, c] = pred

        # Weighted combination
        if use_regime_weights:
            final_pred = np.sum(predictions * regime_probs, axis=1)
        else:
            # Use hard assignment
            regime_idx = np.argmax(self.gamma_hat, axis=1) if self.gamma_hat is not None else np.zeros(m, dtype=int)
            final_pred = predictions[np.arange(m), regime_idx]

        return final_pred

    def get_regime_assignments(self) -> np.ndarray:
        """Get hard regime assignments."""
        if self.gamma_hat is None:
            return None
        return np.argmax(self.gamma_hat, axis=1)

    def get_regime_causal_structures(
        self,
        feature_names: List[str],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Get causal structures for each regime.

        Returns:
            List of dictionaries with causal parents for each regime
        """
        structures = []
        for c, model in enumerate(self.regime_models):
            if model is not None:
                parents = model.get_causal_parents(feature_names, threshold)
                structures.append({
                    'regime': c,
                    'parents': parents
                })
            else:
                structures.append({
                    'regime': c,
                    'parents': None
                })
        return structures

    def _save_checkpoint(self, iteration: int, output_dir: Path):
        """
        Save checkpoint after each EM iteration.

        Args:
            iteration: Current EM iteration number
            output_dir: Directory to save checkpoint
        """
        checkpoint = {
            'iteration': iteration,
            'n_regimes': self.n_regimes,
            'gamma_hat': self.gamma_hat.tolist() if self.gamma_hat is not None else None,
            'regime_model_states': [],
            'prior_network_state': None
        }

        # Save regime model states
        for i, model in enumerate(self.regime_models):
            if model is not None:
                model_path = output_dir / f"checkpoint_iter{iteration}_regime{i}.pt"
                torch.save(model.state_dict(), model_path)
                checkpoint['regime_model_states'].append(str(model_path))
            else:
                checkpoint['regime_model_states'].append(None)

        # Save prior network state
        if self.prior_network is not None:
            prior_path = output_dir / f"checkpoint_iter{iteration}_prior.pt"
            torch.save(self.prior_network.state_dict(), prior_path)
            checkpoint['prior_network_state'] = str(prior_path)

        # Save checkpoint metadata
        with open(output_dir / f"checkpoint_iter{iteration}.json", 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"  Checkpoint saved for iteration {iteration}")

    def _save_partial_results(self, output_dir: Path, regime_idx: int, gamma_hat: np.ndarray):
        """Save partial results when training fails."""
        partial = {
            'failed_regime': regime_idx,
            'gamma_hat': gamma_hat.tolist() if gamma_hat is not None else None,
            'completed_regimes': [i for i, m in enumerate(self.regime_models) if m is not None]
        }
        with open(output_dir / f"partial_results_regime{regime_idx}.json", 'w') as f:
            json.dump(partial, f, indent=2)
        print(f"  Partial results saved for regime {regime_idx}")

    def get_regime_dags(self) -> List[Dict]:
        """
        Get DAG (adjacency matrix) for each regime.

        Returns:
            List of dictionaries with regime DAG info:
            - regime: regime index
            - adjacency_matrix: binary adjacency matrix
            - weighted_adjacency_matrix: weighted adjacency matrix
            - n_samples: number of samples assigned to this regime
        """
        dags = []
        for c, model in enumerate(self.regime_models):
            if model is not None:
                adj = model.get_adj_matrix(samples=1, most_likely_graph=True, squeeze=True)
                weighted_adj = model.get_weighted_adj_matrix(
                    samples=1, most_likely_graph=True, squeeze=True
                ).detach().cpu().numpy()
                dags.append({
                    'regime': c,
                    'adjacency_matrix': adj,
                    'weighted_adjacency_matrix': weighted_adj,
                    'n_samples': int(self.gamma_hat[:, c].sum()) if self.gamma_hat is not None else 0
                })
            else:
                dags.append({
                    'regime': c,
                    'adjacency_matrix': None,
                    'weighted_adjacency_matrix': None,
                    'n_samples': 0
                })
        return dags

    def plot_regime_time_series(
        self,
        dates: np.ndarray = None,
        output_path: str = None,
        title: str = "Regime Assignments Over Time"
    ):
        """
        Plot regime assignments as a time series.

        Args:
            dates: Optional date array for x-axis
            output_path: Path to save figure
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        if self.gamma_hat is None:
            raise ValueError("Model must be fitted first")

        regime_assignments = np.argmax(self.gamma_hat, axis=1)
        n_samples = len(regime_assignments)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

        colors = plt.cm.Set2(np.linspace(0, 1, self.n_regimes))
        x = dates if dates is not None else np.arange(n_samples)

        # Plot regime transitions as colored bands
        current_regime = regime_assignments[0]
        start_idx = 0
        legend_added = set()

        for i in range(1, n_samples + 1):
            if i == n_samples or regime_assignments[i] != current_regime:
                end_idx = i
                label = f'Regime {current_regime}' if current_regime not in legend_added else None
                ax1.axvspan(
                    x[start_idx], x[min(end_idx, n_samples - 1)],
                    alpha=0.3, color=colors[current_regime], label=label
                )
                legend_added.add(current_regime)
                if i < n_samples:
                    current_regime = regime_assignments[i]
                    start_idx = i

        ax1.set_xlim(x[0], x[-1])
        ax1.set_ylabel('Regime')
        ax1.set_title(title)
        ax1.legend(loc='upper right')

        # Bottom: probability heatmap
        im = ax2.imshow(
            self.gamma_hat.T, aspect='auto', cmap='Blues',
            extent=[0, n_samples, -0.5, self.n_regimes - 0.5]
        )
        ax2.set_xlabel('Time' if dates is None else 'Date')
        ax2.set_ylabel('Regime')
        ax2.set_yticks(range(self.n_regimes))
        plt.colorbar(im, ax=ax2, label='Assignment Probability')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved regime plot to: {output_path}")

        return fig

    def plot_regime_with_target(
        self,
        target_values: np.ndarray,
        dates: np.ndarray = None,
        output_path: str = None,
        title: str = "Electricity Price with Regime Detection"
    ):
        """
        Plot TARGET (electricity price) with regime assignments overlaid.

        Args:
            target_values: Array of TARGET values
            dates: Optional date array for x-axis
            output_path: Path to save figure
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        if self.gamma_hat is None:
            raise ValueError("Model must be fitted first")

        regime_assignments = np.argmax(self.gamma_hat, axis=1)
        n_samples = len(regime_assignments)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True
        )

        colors = plt.cm.Set2(np.linspace(0, 1, self.n_regimes))
        x = dates if dates is not None else np.arange(n_samples)

        # Plot regime bands in background
        current_regime = regime_assignments[0]
        start_idx = 0
        legend_entries = set()

        for i in range(1, n_samples + 1):
            if i == n_samples or regime_assignments[i] != current_regime:
                end_idx = i
                label = f'Regime {current_regime}' if current_regime not in legend_entries else None
                ax1.axvspan(
                    x[start_idx], x[min(end_idx, n_samples - 1)],
                    alpha=0.25, color=colors[current_regime], label=label
                )
                legend_entries.add(current_regime)
                if i < n_samples:
                    current_regime = regime_assignments[i]
                    start_idx = i

        # Overlay price time series
        ax1.plot(x, target_values, 'k-', linewidth=1.0, alpha=0.8, label='TARGET')
        ax1.set_ylabel('Electricity Price (TARGET)')
        ax1.set_title(title)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Bottom: regime assignments as step plot
        ax2.step(x, regime_assignments, where='post', color='navy', linewidth=1.5)
        ax2.set_ylabel('Regime')
        ax2.set_xlabel('Time' if dates is None else 'Date')
        ax2.set_yticks(range(self.n_regimes))
        ax2.set_ylim(-0.5, self.n_regimes - 0.5)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved price+regime plot to: {output_path}")

        return fig

    def generate_all_regime_plots(
        self,
        target_values: np.ndarray,
        dates: np.ndarray = None,
        output_dir: str = None,
        country: str = ""
    ) -> Dict:
        """
        Generate both regime visualization plots.

        Args:
            target_values: Array of TARGET values
            dates: Optional date array for x-axis
            output_dir: Directory to save figures
            country: Country code for filenames

        Returns:
            Dictionary with figure objects
        """
        import matplotlib.pyplot as plt

        figs = {}

        # Simple regime plot
        figs['regime_only'] = self.plot_regime_time_series(
            dates=dates,
            output_path=f"{output_dir}/{country}_regime_timeline.png" if output_dir else None,
            title=f"{country} Regime Assignments Over Time"
        )

        # Price + regime overlay
        figs['price_regime'] = self.plot_regime_with_target(
            target_values=target_values,
            dates=dates,
            output_path=f"{output_dir}/{country}_price_regime_overlay.png" if output_dir else None,
            title=f"{country} Electricity Price with Detected Regimes"
        )

        return figs

    def save(self, path: str):
        """Save model state."""
        state = {
            'n_nodes': self.n_nodes,
            'target_idx': self.target_idx,
            'lag': self.lag,
            'n_regimes': self.n_regimes,
            'gamma_hat': self.gamma_hat,
            'model_config': self.model_config,
            'training_params': self.training_params,
        }

        # Save each regime model
        for i, model in enumerate(self.regime_models):
            if model is not None:
                model_path = Path(path).parent / f"regime_model_{i}.pt"
                torch.save(model.state_dict(), model_path)

        # Save prior network
        if self.prior_network is not None:
            prior_path = Path(path).parent / "prior_network.pt"
            torch.save(self.prior_network.state_dict(), prior_path)

        # Save state
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))


def main():
    parser = argparse.ArgumentParser(description="FANTOM with regime detection")
    parser.add_argument('--country', type=str, required=True, choices=['DE', 'FR'],
                        help='Country to model')
    parser.add_argument('--n_regimes', type=int, default=3,
                        help='Initial number of regimes')
    parser.add_argument('--window_size', type=int, default=200,
                        help='Initial window size for regimes')
    parser.add_argument('--min_regime_size', type=int, default=100,
                        help='Minimum samples per regime')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Number of EM iterations')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "regime_results" / f"{args.country}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir

    # Load dataset
    print(f"Loading {args.country} data...")
    dataset = ElectricityDataset(country=args.country, lag=1, imputation='mean')
    print(f"Dataset shape: {dataset.X.shape}")

    # Create regime model
    regime_model = FANTOMRegime(
        n_nodes=dataset.get_num_nodes(),
        target_idx=dataset.get_target_idx(),
        lag=1,
        device=args.device,
        initial_n_regimes=args.n_regimes,
        window_size=args.window_size,
        min_regime_size=args.min_regime_size,
        max_iterations=args.max_iterations,
        model_config=config['model_config'],
        training_params=config['training_params']
    )

    # Fit model with checkpointing
    print("\nFitting regime model...")
    regime_model.fit(dataset.X, verbose=True, checkpoint_dir=checkpoint_dir)

    # Evaluate
    print("\nEvaluating...")
    predictions = regime_model.predict_target(dataset.X)

    from scipy.stats import spearmanr
    spearman, pval = spearmanr(dataset.target, predictions)
    print(f"\nSpearman correlation: {spearman:.4f} (p={pval:.4e})")

    # Get causal structures
    structures = regime_model.get_regime_causal_structures(
        dataset.get_feature_names(),
        threshold=0.5
    )

    print("\nCausal structures by regime:")
    for s in structures:
        print(f"\nRegime {s['regime']}:")
        if s['parents']:
            print(f"  Instantaneous: {len(s['parents']['instantaneous'])} edges")
            for name, _, weight in s['parents']['instantaneous'][:5]:
                print(f"    {name}: {weight:.4f}")

    # Get regime DAGs
    print("\nExtracting regime DAGs...")
    dags = regime_model.get_regime_dags()
    for dag in dags:
        print(f"  Regime {dag['regime']}: {dag['n_samples']} samples")
        if dag['adjacency_matrix'] is not None:
            n_edges = int((dag['adjacency_matrix'] > 0.5).sum())
            print(f"    Total edges: {n_edges}")
            # Save DAG
            np.save(output_dir / f"dag_regime_{dag['regime']}.npy", dag['adjacency_matrix'])
            np.save(output_dir / f"weighted_dag_regime_{dag['regime']}.npy", dag['weighted_adjacency_matrix'])

    # Generate regime plots
    print("\nGenerating regime visualizations...")
    try:
        regime_model.generate_all_regime_plots(
            target_values=dataset.target,
            dates=None,  # Could use dataset.df['DAY_ID'].values[dataset.lag:] if available
            output_dir=str(output_dir),
            country=args.country
        )
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    # Save results
    regime_model.save(str(output_dir / "regime_model.json"))

    # Save summary
    summary = {
        'country': args.country,
        'n_regimes': regime_model.n_regimes,
        'spearman': float(spearman),
        'samples_per_regime': regime_model.gamma_hat.sum(axis=0).tolist() if regime_model.gamma_hat is not None else None,
        'structures': structures,
        'dags': [{'regime': d['regime'], 'n_samples': d['n_samples']} for d in dags]
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
