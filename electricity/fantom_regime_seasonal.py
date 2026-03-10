"""
FANTOM with seasonal regime detection for electricity price prediction.

Improves on window-based initialization by using seasonal patterns
(Winter vs Summer) based on DAY_ID % 365.

Usage:
    python fantom_regime_seasonal.py --country DE --n_regimes 2 --init_mode seasonal
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


class RegimePriorNetwork(nn.Module):
    """Neural network for predicting regime probabilities over time."""

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
    """Train the prior network to predict regime assignments."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        p = model(t)
        loss = criterion(p, gamma)
        loss.backward()
        optimizer.step()

    # Continue training if loss is above threshold
    max_extra_epochs = 5000
    extra = 0
    while loss.item() >= threshold and extra < max_extra_epochs:
        for _ in range(100):
            optimizer.zero_grad()
            p = model(t)
            loss = criterion(p, gamma)
            loss.backward()
            optimizer.step()
        extra += 100

        if loss.item() < threshold:
            break

    model.eval()
    with torch.no_grad():
        p = model(t)

    return p, loss.item(), model


def initialize_seasonal_regimes(day_ids: np.ndarray, n_regimes: int = 2) -> np.ndarray:
    """
    Initialize regime probabilities based on seasonal patterns.

    Uses DAY_ID % 365 to determine season:
    - Winter: Q1 (Jan-Mar, days 0-90) + Q4 (Oct-Dec, days 270-365)
    - Summer: Q2 (Apr-Jun, days 91-181) + Q3 (Jul-Sep, days 182-270)

    Args:
        day_ids: Array of DAY_ID values
        n_regimes: Number of regimes (2 or 4)

    Returns:
        p: Regime probability matrix [N, n_regimes]
    """
    m = len(day_ids)
    seasons = day_ids % 365

    if n_regimes == 2:
        # Winter vs Summer
        winter = (seasons < 90) | (seasons > 270)
        p = np.zeros((m, 2))
        p[winter, 0] = 1.0   # Winter regime
        p[~winter, 1] = 1.0  # Summer regime
        print(f"  Seasonal init (2 regimes): Winter={winter.sum()}, Summer={(~winter).sum()}")

    elif n_regimes == 4:
        # Quarterly
        p = np.zeros((m, 4))
        q1 = (seasons >= 0) & (seasons < 91)
        q2 = (seasons >= 91) & (seasons < 182)
        q3 = (seasons >= 182) & (seasons < 273)
        q4 = (seasons >= 273)

        p[q1, 0] = 1.0  # Q1 (Winter)
        p[q2, 1] = 1.0  # Q2 (Spring)
        p[q3, 2] = 1.0  # Q3 (Summer)
        p[q4, 3] = 1.0  # Q4 (Fall)
        print(f"  Seasonal init (4 regimes): Q1={q1.sum()}, Q2={q2.sum()}, Q3={q3.sum()}, Q4={q4.sum()}")

    else:
        raise ValueError(f"n_regimes must be 2 or 4 for seasonal init, got {n_regimes}")

    return p


def initialize_window_regimes(m: int, n_regimes: int, window_size: int = 200) -> np.ndarray:
    """Original window-based initialization."""
    p = np.zeros((m, n_regimes))
    for c in range(n_regimes):
        if c == n_regimes - 1:
            p[c * window_size:, c] = 1.0
        else:
            p[c * window_size:(c + 1) * window_size, c] = 1.0
    return p


class FANTOMRegimeSeasonal:
    """
    FANTOM with seasonal regime detection.

    Improved version that initializes regimes based on seasonal patterns
    rather than arbitrary time windows.
    """

    def __init__(
        self,
        n_nodes: int,
        target_idx: int,
        lag: int = 1,
        device: str = "cpu",
        initial_n_regimes: int = 2,
        init_mode: str = "seasonal",  # 'seasonal' or 'window'
        window_size: int = 200,
        min_regime_size: int = 50,  # Reduced from 100
        max_iterations: int = 3,
        prior_threshold: float = 0.85,
        model_config: Optional[Dict] = None,
        training_params: Optional[Dict] = None
    ):
        """
        Initialize FANTOM with seasonal regime detection.

        Args:
            n_nodes: Number of variables
            target_idx: Index of TARGET variable
            lag: Temporal lag
            device: PyTorch device
            initial_n_regimes: Starting number of regimes (2 or 4 for seasonal)
            init_mode: 'seasonal' or 'window'
            window_size: Window size for window-based init
            min_regime_size: Minimum samples per regime (zeta)
            max_iterations: Number of EM iterations
            prior_threshold: Loss threshold for prior network
            model_config: FANTOM model configuration
            training_params: Training parameters
        """
        self.n_nodes = n_nodes
        self.target_idx = target_idx
        self.lag = lag
        self.device = torch.device(device)
        self.initial_n_regimes = initial_n_regimes
        self.init_mode = init_mode
        self.window_size = window_size
        self.min_regime_size = min_regime_size
        self.max_iterations = max_iterations
        self.prior_threshold = prior_threshold

        # Default configurations
        self.model_config = model_config or {
            'lambda_dag': 100.0,
            'lambda_sparse': 1.0,
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

    def fit(
        self,
        X: np.ndarray,
        day_ids: np.ndarray = None,
        verbose: bool = True,
        checkpoint_dir: Path = None
    ) -> 'FANTOMRegimeSeasonal':
        """
        Fit FANTOM with regime detection.

        Args:
            X: Data of shape [N, lag+1, n_nodes]
            day_ids: DAY_ID values for seasonal initialization
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints

        Returns:
            self
        """
        m = X.shape[0]
        n_regimes = self.initial_n_regimes

        # Initialize regime probabilities
        if self.init_mode == 'seasonal':
            if day_ids is None:
                raise ValueError("day_ids required for seasonal initialization")
            p = initialize_seasonal_regimes(day_ids, n_regimes)
        else:
            p = initialize_window_regimes(m, n_regimes, self.window_size)
            if verbose:
                print(f"  Window init: {n_regimes} regimes of size ~{self.window_size}")

        # Standardized data for emission computation
        X_std = X.copy()
        for t in range(X_std.shape[1]):
            mean = X_std[:, t, :].mean(axis=0)
            std = X_std[:, t, :].std(axis=0)
            std[std == 0] = 1
            X_std[:, t, :] = (X_std[:, t, :] - mean) / std

        # EM iterations
        gamma_hat = None
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

                # Get regime data based on current assignments
                if it == 0:
                    # Use initial assignments
                    gamma = p[:, c]
                else:
                    gamma = gamma_hat[:, c]

                mask = gamma > 0.5
                regime_data = X[mask]

                if len(regime_data) < 10:
                    if verbose:
                        print(f"  Regime {c} has too few samples ({len(regime_data)}), skipping")
                    self.regime_models.append(None)
                    continue

                if verbose:
                    print(f"  Regime {c}: {len(regime_data)} samples")

                # Standardize regime data
                regime_data_std = regime_data.copy()
                for t in range(regime_data.shape[1]):
                    mean = regime_data[:, t, :].mean(axis=0)
                    std = regime_data[:, t, :].std(axis=0)
                    std[std == 0] = 1
                    regime_data_std[:, t, :] = (regime_data[:, t, :] - mean) / std

                # Train model
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
                    continue

            # E-step: Compute regime assignments
            pall = np.sum(p * log_pdf_emission, axis=1, keepdims=True)
            pall[pall == 0] = 1e-10  # Avoid division by zero

            gamma_hat = (p * log_pdf_emission) / pall
            gamma_hat = gamma_hat / gamma_hat.sum(axis=1, keepdims=True)

            # Hard assignment
            idx = np.argmax(gamma_hat, axis=-1)
            gamma_hat = np.zeros_like(gamma_hat)
            gamma_hat[np.arange(m), idx] = 1

            # Train prior network
            t_idx = torch.tensor(
                np.linspace(0, 20 * n_regimes, m).reshape((m, 1)),
                dtype=torch.float32
            )
            self.prior_network = RegimePriorNetwork(n_regimes)
            p, loss, self.prior_network = train_prior_network(
                self.prior_network,
                t_idx,
                torch.tensor(gamma_hat, dtype=torch.float32),
                threshold=self.prior_threshold
            )
            p = p.detach().numpy()

            # Prune small regimes
            gamma_sum = np.sum(gamma_hat, axis=0)
            valid_regimes = gamma_sum >= self.min_regime_size

            if verbose:
                print(f"\nIteration {it + 1} results:")
                print(f"  Samples per regime: {gamma_sum}")
                print(f"  Valid regimes (>= {self.min_regime_size}): {valid_regimes.sum()}")

            # Update models and gamma_hat for valid regimes only
            if len(self.regime_models) == len(valid_regimes):
                self.regime_models = [m for i, m in enumerate(self.regime_models) if valid_regimes[i]]
            else:
                new_models = []
                for i, valid in enumerate(valid_regimes):
                    if valid and i < len(self.regime_models):
                        new_models.append(self.regime_models[i])
                self.regime_models = new_models

            gamma_hat = gamma_hat[:, valid_regimes]
            p = p[:, valid_regimes]
            n_regimes = int(valid_regimes.sum())

            # Retrain prior network with pruned regimes
            if n_regimes < self.prior_network.n_regimes and n_regimes > 1:
                self.prior_network = RegimePriorNetwork(n_regimes)
                t_idx = torch.tensor(
                    np.linspace(0, 20 * n_regimes, m).reshape((m, 1)),
                    dtype=torch.float32
                )
                p, loss, self.prior_network = train_prior_network(
                    self.prior_network,
                    t_idx,
                    torch.tensor(gamma_hat, dtype=torch.float32),
                    threshold=self.prior_threshold
                )
                p = p.detach().numpy()

            if verbose:
                print(f"  Active regimes after pruning: {n_regimes}")
                print(f"  Prior network loss: {loss:.4f}")

            # Save checkpoint
            self.gamma_hat = gamma_hat
            self.n_regimes = n_regimes

            if checkpoint_dir:
                self._save_checkpoint(it + 1, checkpoint_dir)

            # Check for convergence (single regime)
            if n_regimes <= 1:
                if verbose:
                    print("\nConverged to single regime - stopping early")
                break

        self.gamma_hat = gamma_hat
        self.n_regimes = n_regimes

        return self

    def predict_target(
        self,
        X: np.ndarray,
        use_regime_weights: bool = True
    ) -> np.ndarray:
        """Predict TARGET using regime-specific models."""
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

        # Standardize input
        X_std = X.copy()
        for t in range(X_std.shape[1]):
            mean = X_std[:, t, :].mean(axis=0)
            std = X_std[:, t, :].std(axis=0)
            std[std == 0] = 1
            X_std[:, t, :] = (X_std[:, t, :] - mean) / std

        # Get predictions from each regime model
        for c, model in enumerate(self.regime_models):
            if model is not None:
                model.eval()
                with torch.no_grad():
                    pred = model.predict_target(
                        torch.tensor(X_std, dtype=torch.float32)
                    ).cpu().numpy()
                    predictions[:, c] = pred

        # Weighted combination
        if use_regime_weights:
            final_pred = np.sum(predictions * regime_probs, axis=1)
        else:
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
        """Get causal structures for each regime."""
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
        """Save checkpoint after each EM iteration."""
        checkpoint = {
            'iteration': iteration,
            'n_regimes': self.n_regimes,
            'gamma_hat': self.gamma_hat.tolist() if self.gamma_hat is not None else None,
            'init_mode': self.init_mode,
            'regime_model_states': [],
            'prior_network_state': None
        }

        for i, model in enumerate(self.regime_models):
            if model is not None:
                model_path = output_dir / f"checkpoint_iter{iteration}_regime{i}.pt"
                torch.save(model.state_dict(), model_path)
                checkpoint['regime_model_states'].append(str(model_path))
            else:
                checkpoint['regime_model_states'].append(None)

        if self.prior_network is not None:
            prior_path = output_dir / f"checkpoint_iter{iteration}_prior.pt"
            torch.save(self.prior_network.state_dict(), prior_path)
            checkpoint['prior_network_state'] = str(prior_path)

        with open(output_dir / f"checkpoint_iter{iteration}.json", 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"  Checkpoint saved for iteration {iteration}")

    def save(self, path: str):
        """Save model state."""
        state = {
            'n_nodes': self.n_nodes,
            'target_idx': self.target_idx,
            'lag': self.lag,
            'n_regimes': self.n_regimes,
            'init_mode': self.init_mode,
            'gamma_hat': self.gamma_hat,
            'model_config': self.model_config,
            'training_params': self.training_params,
        }

        for i, model in enumerate(self.regime_models):
            if model is not None:
                model_path = Path(path).parent / f"regime_model_{i}.pt"
                torch.save(model.state_dict(), model_path)

        if self.prior_network is not None:
            prior_path = Path(path).parent / "prior_network.pt"
            torch.save(self.prior_network.state_dict(), prior_path)

        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))


def main():
    parser = argparse.ArgumentParser(description="FANTOM with seasonal regime detection")
    parser.add_argument('--country', type=str, required=True, choices=['DE', 'FR', 'ALL'])
    parser.add_argument('--n_regimes', type=int, default=2, choices=[2, 4])
    parser.add_argument('--init_mode', type=str, default='seasonal', choices=['seasonal', 'window'])
    parser.add_argument('--min_regime_size', type=int, default=50)
    parser.add_argument('--max_iterations', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "seasonal_results" / f"{args.country}_{args.n_regimes}_{args.init_mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    # For 'ALL', pass None to ElectricityDataset to get joint model with all features
    country_arg = None if args.country == 'ALL' else args.country
    print(f"Loading {args.country} data...")
    dataset = ElectricityDataset(country=country_arg, lag=1, imputation='mean')
    print(f"Dataset shape: {dataset.X.shape}")

    # Get DAY_IDs for seasonal initialization
    # DAY_IDs are in the original DataFrame, need to extract after lag adjustment
    day_ids = dataset.df['DAY_ID'].values[dataset.lag:]
    print(f"DAY_ID range: {day_ids.min()} - {day_ids.max()}")

    # Create regime model
    regime_model = FANTOMRegimeSeasonal(
        n_nodes=dataset.get_num_nodes(),
        target_idx=dataset.get_target_idx(),
        lag=1,
        device=args.device,
        initial_n_regimes=args.n_regimes,
        init_mode=args.init_mode,
        min_regime_size=args.min_regime_size,
        max_iterations=args.max_iterations,
        model_config=config['model_config'],
        training_params=config['training_params']
    )

    # Fit model
    print(f"\nFitting regime model (init_mode={args.init_mode}, n_regimes={args.n_regimes})...")
    regime_model.fit(dataset.X, day_ids=day_ids, verbose=True, checkpoint_dir=output_dir)

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
            print(f"  Lagged: {len(s['parents']['lagged'])} edges")
            for name, _, weight in s['parents']['lagged'][:5]:
                print(f"    {name}: {weight:.4f}")

    # Save regime model
    regime_model.save(str(output_dir / "regime_model.json"))

    # Save summary
    summary = {
        'country': args.country,
        'n_regimes': regime_model.n_regimes,
        'init_mode': args.init_mode,
        'min_regime_size': args.min_regime_size,
        'seed': args.seed,
        'spearman': float(spearman),
        'samples_per_regime': regime_model.gamma_hat.sum(axis=0).tolist() if regime_model.gamma_hat is not None else None,
        'structures': structures,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
