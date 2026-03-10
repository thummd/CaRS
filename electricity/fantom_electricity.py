"""
FANTOM adapted for electricity price prediction.

This module adapts the FANTOM causal discovery model to work with
electricity price data and produce predictions for the TARGET variable.
"""

import sys
import os

from paths import FANTOM_CODE_DIR
# Add FANTOM code to path
FANTOM_PATH = str(FANTOM_CODE_DIR)
sys.path.insert(0, FANTOM_PATH)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import spearmanr

# Import FANTOM components
from fantom import FANTOM_stationary
from deci import DECI


class FANTOMElectricity(FANTOM_stationary):
    """
    FANTOM model adapted for electricity price prediction.

    Key modifications:
    1. Supports prediction extraction for TARGET variable
    2. Adds graph constraints to ensure TARGET is an effect (no outgoing edges)
    3. Provides methods for prediction and evaluation with Spearman correlation
    """

    def __init__(
        self,
        num_nodes: int,
        device: torch.device,
        target_idx: int,
        lag: int = 1,
        allow_instantaneous: bool = True,
        constrain_target: bool = True,
        lambda_dag: float = 100.0,
        lambda_sparse: float = 1.0,
        lambda_sparse_l2: float = 0.0,
        l2_group_mode: str = "column",
        lambda_prior: float = 1.0,
        tau_gumbel: float = 1.0,
        base_distribution_type: str = "spline",
        spline_bins: int = 8,
        var_dist_A_mode: str = "temporal_three",
        norm_layers: bool = True,
        res_connection: bool = True,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        heteroscedastic: bool = True,
        **kwargs
    ):
        """
        Initialize FANTOM for electricity prediction.

        Args:
            num_nodes: Number of variables (features + TARGET)
            device: PyTorch device
            target_idx: Index of TARGET variable in the node list
            lag: Temporal lag (use 1 for day-over-day dependencies)
            allow_instantaneous: Whether same-day variables can affect each other
            constrain_target: If True, TARGET has no outgoing edges (it's an effect)
            ... (other parameters same as FANTOM_stationary)
        """
        self.target_idx = target_idx
        self.constrain_target = constrain_target

        # Build graph constraint matrix if constraining target
        graph_constraint = None
        if constrain_target:
            # Create constraint matrix: NaN = learnable, 0 = forbidden, 1 = required
            # We set TARGET -> any to 0 (forbidden) at all lag levels
            graph_constraint = np.full((lag + 1, num_nodes, num_nodes), np.nan)
            # No edges from TARGET to other variables (TARGET is purely an effect)
            graph_constraint[:, target_idx, :] = 0.0
            # Allow edges TO TARGET (these are what we want to learn)
            graph_constraint[:, :, target_idx] = np.nan
            # No self-loops
            for l in range(lag + 1):
                np.fill_diagonal(graph_constraint[l], 0.0)

        super().__init__(
            num_nodes=num_nodes,
            device=device,
            lag=lag,
            allow_instantaneous=allow_instantaneous,
            lambda_dag=lambda_dag,
            lambda_sparse=lambda_sparse,
            lambda_sparse_l2=lambda_sparse_l2,
            l2_group_mode=l2_group_mode,
            lambda_prior=lambda_prior,
            tau_gumbel=tau_gumbel,
            base_distribution_type=base_distribution_type,
            spline_bins=spline_bins,
            var_dist_A_mode=var_dist_A_mode,
            norm_layers=norm_layers,
            res_connection=res_connection,
            encoder_layer_sizes=encoder_layer_sizes,
            decoder_layer_sizes=decoder_layer_sizes,
            graph_constraint_matrix=graph_constraint,
            heteroscedastic=heteroscedastic,
            **kwargs
        )

    def predict_target(
        self,
        X: torch.Tensor,
        most_likely_graph: bool = True
    ) -> torch.Tensor:
        """
        Predict TARGET values given input data.

        Args:
            X: Input data of shape [N, lag+1, num_nodes]
            most_likely_graph: Use the most probable graph (deterministic)

        Returns:
            Predictions for TARGET of shape [N]
        """
        self.eval()
        with torch.no_grad():
            X = X.to(self.device).float()

            # Get adjacency matrix
            A = self.get_adj_matrix_tensor(
                do_round=True,
                samples=1,
                most_likely_graph=most_likely_graph
            ).squeeze(0)

            # Get weighted adjacency
            W_adj = A * self.ICGNN.get_weighted_adjacency()

            # Get predictions from the SEM
            if self.heteroscedastic:
                predict, var_est = self.ICGNN.predict(X, W_adj)
            else:
                predict = self.ICGNN.predict(X, W_adj)

            # Extract TARGET predictions
            target_pred = predict[..., self.target_idx]

            return target_pred

    def predict_with_uncertainty(
        self,
        X: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict TARGET with uncertainty estimation via graph sampling.

        Args:
            X: Input data of shape [N, lag+1, num_nodes]
            n_samples: Number of graph samples for uncertainty estimation

        Returns:
            mean_pred: Mean predictions of shape [N]
            std_pred: Standard deviation of predictions of shape [N]
        """
        self.eval()
        with torch.no_grad():
            X = X.to(self.device).float()

            predictions = []
            for _ in range(n_samples):
                # Sample a graph
                A = self.get_adj_matrix_tensor(
                    do_round=False,
                    samples=1,
                    most_likely_graph=False
                ).squeeze(0)

                W_adj = A * self.ICGNN.get_weighted_adjacency()

                if self.heteroscedastic:
                    predict, _ = self.ICGNN.predict(X, W_adj)
                else:
                    predict = self.ICGNN.predict(X, W_adj)

                predictions.append(predict[..., self.target_idx])

            predictions = torch.stack(predictions, dim=0)
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)

            return mean_pred, std_pred

    def get_causal_parents(
        self,
        feature_names: List[str],
        threshold: float = 0.5
    ) -> Dict[str, List[Tuple[str, int, float]]]:
        """
        Get the causal parents of TARGET variable.

        Args:
            feature_names: List of feature names
            threshold: Minimum edge probability to consider

        Returns:
            Dictionary with 'instantaneous' and 'lagged' parents,
            each containing (feature_name, lag, weight) tuples
        """
        A = self.get_adj_matrix(samples=1, most_likely_graph=True, squeeze=True)
        W = self.get_weighted_adj_matrix(
            samples=1, most_likely_graph=True, squeeze=True
        ).detach().cpu().numpy()

        parents = {'instantaneous': [], 'lagged': []}

        # Instantaneous parents (lag=0)
        for i in range(self.num_nodes):
            if i != self.target_idx and A[0, i, self.target_idx] >= threshold:
                weight = W[0, i, self.target_idx]
                parents['instantaneous'].append((feature_names[i], 0, weight))

        # Lagged parents
        for lag_idx in range(1, self.lag + 1):
            for i in range(self.num_nodes):
                if A[lag_idx, i, self.target_idx] >= threshold:
                    weight = W[lag_idx, i, self.target_idx]
                    parents['lagged'].append((feature_names[i], lag_idx, weight))

        # Sort by absolute weight
        parents['instantaneous'].sort(key=lambda x: abs(x[2]), reverse=True)
        parents['lagged'].sort(key=lambda x: abs(x[2]), reverse=True)

        return parents

    def get_important_features(
        self,
        feature_names: List[str],
        importance_threshold: float = 0.01
    ) -> Dict[str, List[Tuple]]:
        """
        Extract important features based on adjacency weights.

        Uses the sparsity-regularized adjacency matrix to identify
        features that have significant causal influence on TARGET.

        Args:
            feature_names: List of feature names
            importance_threshold: Minimum absolute weight to consider

        Returns:
            Dictionary with 'instantaneous' and 'lagged' important features
            sorted by absolute weight. Each entry contains:
            - instantaneous: (feature_name, weight)
            - lagged: (feature_name, weight, lag)
        """
        W = self.get_weighted_adj_matrix(
            samples=1, most_likely_graph=True, squeeze=True
        ).detach().cpu().numpy()

        important = {'instantaneous': [], 'lagged': []}

        # Instantaneous (lag=0)
        for i in range(self.num_nodes):
            if i != self.target_idx:
                weight = W[0, i, self.target_idx]
                if abs(weight) >= importance_threshold:
                    important['instantaneous'].append((feature_names[i], float(weight)))

        # Lagged
        for lag_idx in range(1, self.lag + 1):
            for i in range(self.num_nodes):
                weight = W[lag_idx, i, self.target_idx]
                if abs(weight) >= importance_threshold:
                    important['lagged'].append((feature_names[i], float(weight), lag_idx))

        # Sort by absolute importance
        important['instantaneous'].sort(key=lambda x: abs(x[1]), reverse=True)
        important['lagged'].sort(key=lambda x: abs(x[1]), reverse=True)

        return important

    def evaluate_predictions(
        self,
        X: torch.Tensor,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate prediction quality.

        Args:
            X: Input data of shape [N, lag+1, num_nodes]
            y_true: True TARGET values of shape [N]

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict_target(X).cpu().numpy()

        # Spearman correlation (challenge metric)
        spearman_corr, spearman_pval = spearmanr(y_true, y_pred)

        # Additional metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'spearman': spearman_corr,
            'spearman_pval': spearman_pval,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }


def create_model(
    num_nodes: int,
    target_idx: int,
    lag: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_config: Optional[Dict] = None
) -> FANTOMElectricity:
    """
    Factory function to create FANTOMElectricity model.

    Args:
        num_nodes: Number of variables
        target_idx: Index of TARGET variable
        lag: Temporal lag
        device: Device to use
        model_config: Optional configuration dictionary

    Returns:
        Configured FANTOMElectricity model
    """
    default_config = {
        'lambda_dag': 100.0,
        'lambda_sparse': 1.0,
        'lambda_sparse_l2': 0.0,
        'l2_group_mode': 'column',
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

    if model_config:
        default_config.update(model_config)

    device = torch.device(device)

    model = FANTOMElectricity(
        num_nodes=num_nodes,
        device=device,
        target_idx=target_idx,
        lag=lag,
        **default_config
    )

    return model.to(device)


def get_default_training_params() -> Dict[str, Any]:
    """Get default training parameters."""
    return {
        'batch_size': 64,
        'learning_rate': 0.001,
        'max_steps_auglag': 20,
        'max_auglag_inner_epochs': 2000,
        'rho': 1.0,
        'alpha': 0.0,
        'progress_rate': 0.9,
        'safety_rho': 1e9,
        'safety_alpha': 1e9,
        'tol_dag': 1e-6,
        'anneal_entropy': 'noanneal',
        'reconstruction_loss_factor': 1.0,
    }


if __name__ == "__main__":
    # Test the model
    print("Testing FANTOMElectricity...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy data
    num_nodes = 10
    target_idx = 9
    lag = 1
    n_samples = 100

    X = torch.randn(n_samples, lag + 1, num_nodes)

    # Create model
    model = create_model(
        num_nodes=num_nodes,
        target_idx=target_idx,
        lag=lag,
        device=str(device)
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test prediction
    with torch.no_grad():
        pred = model.predict_target(X)
        print(f"Prediction shape: {pred.shape}")

    # Test evaluation
    y_true = np.random.randn(n_samples)
    metrics = model.evaluate_predictions(X, y_true)
    print(f"Evaluation metrics: {metrics}")

    print("FANTOMElectricity tests passed!")
