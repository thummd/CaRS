"""
Feature-Conditioned DS3M (Deep Switching State Space Model).

This model extends DS3M with input-dependent regime transitions:
- Original DS3M: pi(d_t | d_{t-1}) - time-homogeneous transition matrix
- Conditioned:   pi(d_t | d_{t-1}, h_t) - transition depends on hidden state (features)

This allows the model to learn which features drive regime shifts.

Usage:
    python ds3m_conditioned.py --country ALL --mode multivariate --d_dim 2 --train --seed 42
"""

import sys
import os
from pathlib import Path

from paths import DS3M_DIR
# Add DS3M code to path
DS3M_PATH = str(DS3M_DIR)
sys.path.insert(0, DS3M_PATH)
sys.path.insert(0, os.path.join(DS3M_PATH, "src"))

import argparse
import json
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import yaml
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Import original DS3M components
from DSSSMCode import DSSSM, train, test, EarlyStopping

# Import local adapter
from ds3m_adapter import (
    load_qrt_data, prepare_multivariate_train_test_split,
    normalize_invert, evaluation, get_feature_columns
)


class ConditionedDSSSM(DSSSM):
    """
    DS3M with feature-conditioned regime transitions.

    Key modification: The transition matrix is modulated by the hidden state,
    which encodes the input features. This allows regime transitions to depend
    on market conditions rather than being purely time-homogeneous.
    """

    def __init__(
        self,
        x_dim,
        y_dim,
        h_dim,
        z_dim,
        d_dim,
        n_layers,
        device,
        bidirection=False,
        bias=False,
        dataname=None,
        transition_hidden_dim=None,
        gate_init=0.5
    ):
        super().__init__(
            x_dim, y_dim, h_dim, z_dim, d_dim, n_layers,
            device, bidirection, bias, dataname
        )

        # Feature-conditioned transition network
        # Maps hidden state to transition matrix modulation
        if transition_hidden_dim is None:
            transition_hidden_dim = h_dim // 2

        self.transition_net = nn.Sequential(
            nn.Linear(h_dim, transition_hidden_dim),
            nn.ReLU(),
            nn.Linear(transition_hidden_dim, d_dim * d_dim)
        )

        # Learnable gate: blends base transition with conditioned transition
        # gate=1 -> use base (original DS3M)
        # gate=0 -> use conditioned (fully feature-dependent)
        self.gate_logit = nn.Parameter(torch.tensor(gate_init).log())

        # Initialize transition_net to output near-zero (small perturbation)
        nn.init.zeros_(self.transition_net[-1].weight)
        nn.init.zeros_(self.transition_net[-1].bias)

    def TransitionMatrixConditioned(self, h_t):
        """
        Compute feature-conditioned transition matrix.

        Args:
            h_t: Hidden state from forward RNN (batch, h_dim)

        Returns:
            Transition matrix per sample (batch, d_dim, d_dim)
        """
        batch_size = h_t.size(0)

        # Base transition (shared, time-homogeneous)
        base = super().TransitionMatrix()  # (d_dim, d_dim)
        base = base.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, d_dim, d_dim)

        # Feature-conditioned modulation
        modulation_logits = self.transition_net(h_t)  # (batch, d_dim*d_dim)
        modulation_logits = modulation_logits.view(batch_size, self.d_dim, self.d_dim)

        # Apply softmax row-wise (each row sums to 1)
        modulation = torch.softmax(modulation_logits, dim=-1)  # (batch, d_dim, d_dim)

        # Blend with learnable gate
        gate = torch.sigmoid(self.gate_logit)
        transition = gate * base + (1 - gate) * modulation

        return transition

    def forward(self, x, y):
        """
        Forward pass with conditioned transitions.

        Mostly follows original DS3M, but uses per-sample, per-timestep
        transition matrices based on hidden state.
        """
        all_d_posterior = []
        all_d_t_sampled_plot = []
        all_d_t_sampled = []

        all_z_posterior_mean, all_z_posterior_std = [], []
        all_z_t_sampled = []

        all_y_emission_mean, all_y_emission_std = [], []

        kld_gaussian_loss = 0
        kld_category_loss = 0
        nll_loss = 0

        batch_size = x.size(1)

        h0 = torch.zeros((self.n_layers, batch_size, self.h_dim), device=self.device)
        if self.bidirection:
            A0 = torch.zeros((self.n_layers * 2, batch_size, int(self.h_dim / 2)), device=self.device)
        else:
            A0 = torch.zeros((self.n_layers, batch_size, self.h_dim), device=self.device)

        # Initialize d0
        samples = torch.distributions.Categorical(
            torch.ones((self.d_dim)) / self.d_dim
        ).sample((batch_size,)).type(torch.LongTensor)
        d0 = self._one_hot_encode(samples, self.d_dim)
        all_d_posterior.append(torch.ones((batch_size, self.d_dim), device=self.device) / self.d_dim)
        all_d_t_sampled_plot.append(samples.reshape(-1, 1).to(self.device))
        all_d_t_sampled.append(d0)

        # Initialize z0
        z0 = torch.zeros((batch_size, self.z_dim), device=self.device)
        all_z_posterior_std.append(z0)
        all_z_posterior_mean.append(z0)
        all_z_t_sampled.append(z0)

        # Forward RNN
        output_forward, h_forward = self.rnn_forward(x, h0)

        # Backward RNN
        yh_concatenate = torch.cat([y, output_forward], 2)
        yh_concatenate_inverse = torch.flip(yh_concatenate, [0])
        output_backward, h_backward = self.rnn_backward(yh_concatenate_inverse, A0)

        for t in range(x.size(0)):
            # ===== KEY MODIFICATION =====
            # Compute per-sample transition matrix based on hidden state
            h_t = output_forward[t]  # (batch, h_dim)
            Transition_t = self.TransitionMatrixConditioned(h_t)  # (batch, d_dim, d_dim)

            # d prior: batch matrix multiplication
            # all_d_t_sampled[t] is (batch, d_dim)
            # Need: d_prior[b] = all_d_t_sampled[t][b] @ Transition_t[b]
            d_prior = torch.bmm(
                all_d_t_sampled[t].unsqueeze(1),  # (batch, 1, d_dim)
                Transition_t  # (batch, d_dim, d_dim)
            ).squeeze(1)  # (batch, d_dim)

            # d posterior
            d_posterior_list = []
            d_posterior = 0
            for i in range(self.d_dim):
                d_posterior_list.append(
                    self.dposterior_list[i](output_backward[x.size(0) - t - 1])
                )
                d_posterior += d_posterior_list[i] * all_d_t_sampled[t][:, i:(i + 1)]
            all_d_posterior.append(d_posterior)

            d_t_samples = torch.distributions.Categorical(d_posterior).sample().type(torch.LongTensor).to(self.device)
            all_d_t_sampled_plot.append(d_t_samples.reshape(-1, 1))
            all_d_t_sampled.append(self._one_hot_encode(d_t_samples, self.d_dim))

            # z prior and posterior (unchanged from original)
            z_prior_list = []
            z_prior_mean_list = []
            z_prior_std_list = []
            z_prior_mean = 0
            z_prior_std = 0

            z_posterior_list = []
            z_posterior_mean_list = []
            z_posterior_std_list = []
            z_posterior_mean = 0
            z_posterior_std = 0

            for i in range(self.d_dim):
                z_prior_list.append(
                    self.ztrainsition_list[i](torch.cat([output_forward[t], all_z_t_sampled[t]], 1))
                )
                z_prior_mean_list.append(self.ztrainsition_mean_list[i](z_prior_list[i]))
                z_prior_std_list.append(self.ztrainsition_std_list[i](z_prior_list[i]))
                z_prior_mean += z_prior_mean_list[i] * all_d_t_sampled[t + 1][:, i:(i + 1)]
                z_prior_std += z_prior_std_list[i] * all_d_t_sampled[t + 1][:, i:(i + 1)]

                z_posterior_list.append(
                    self.zposterior_list[i](torch.cat([output_backward[x.size(0) - t - 1], all_z_t_sampled[t]], 1))
                )
                z_posterior_mean_list.append(self.zposterior_mean_list[i](z_posterior_list[i]))
                z_posterior_std_list.append(self.zposterior_std_list[i](z_posterior_list[i]))
                z_posterior_mean += z_posterior_mean_list[i] * all_d_t_sampled[t + 1][:, i:(i + 1)]
                z_posterior_std += z_posterior_std_list[i] * all_d_t_sampled[t + 1][:, i:(i + 1)]

            all_z_posterior_mean.append(z_posterior_mean)
            all_z_posterior_std.append(z_posterior_std)

            z_t = self._reparameterized_normal_sample(z_posterior_mean, z_posterior_std)
            all_z_t_sampled.append(z_t)

            # y emission (unchanged)
            y_emission_list = []
            y_emission_mean_list = []
            y_emission_std_list = []
            y_emission_mean = 0
            y_emission_std = 0

            for i in range(self.d_dim):
                y_emission_list.append(
                    self.yemission_list[i](torch.cat([output_forward[t], all_z_t_sampled[t + 1]], 1))
                )
                y_emission_mean_list.append(self.yemission_mean_list[i](y_emission_list[i]))
                y_emission_std_list.append(self.yemission_std_list[i](y_emission_list[i]))
                y_emission_mean += y_emission_mean_list[i] * all_d_t_sampled[t + 1][:, i:(i + 1)]
                y_emission_std += y_emission_std_list[i] * all_d_t_sampled[t + 1][:, i:(i + 1)]

            all_y_emission_mean.append(y_emission_mean)
            all_y_emission_std.append(y_emission_std)

            # Losses
            for i in range(self.d_dim):
                kld_gaussian_loss += torch.sum(
                    self._kld_gauss(
                        z_posterior_mean_list[i], z_posterior_std_list[i],
                        z_prior_mean_list[i], z_prior_std_list[i]
                    ) * d_posterior[:, i:(i + 1)]
                )

            # ===== KL for categorical with per-sample transitions =====
            for i in range(self.d_dim):
                # d_prior for this d_t-1=i is Transition_t[:, i, :]
                d_prior_given_i = Transition_t[:, i, :]  # (batch, d_dim)
                kld_category_loss += torch.sum(
                    self._kld_category(d_posterior_list[i], d_prior_given_i) * all_d_posterior[-2][:, i]
                )

            for i in range(self.d_dim):
                nll_loss += torch.sum(
                    self._nll_gauss(y_emission_mean_list[i], y_emission_std_list[i], y[t]) * d_posterior[:, i:(i + 1)]
                )

        return (
            kld_gaussian_loss, kld_category_loss, nll_loss,
            (all_z_posterior_mean, all_z_posterior_std),
            (all_y_emission_mean, all_y_emission_std),
            all_d_t_sampled_plot, all_z_t_sampled,
            all_d_posterior, all_d_t_sampled
        )

    def get_gate_value(self):
        """Return the current gate value (blend between base and conditioned)."""
        return torch.sigmoid(self.gate_logit).item()


def train_conditioned(model, optimizer, trainX, trainY, epoch, batch_size, n_epochs, status="train"):
    """Training loop for conditioned DS3M."""
    model.train()

    if epoch < n_epochs / 2:
        annealing = 0.01
    else:
        annealing = min(1.0, 0.01 + epoch / n_epochs / 2)

    print(f'Annealing coef: {annealing}, Gate: {model.get_gate_value():.3f}')

    for batch in range(0, trainX.size(1), batch_size):
        batchX = trainX[:, batch:(batch + batch_size), :]
        batchY = trainY[:, batch:(batch + batch_size), :]

        kld_gaussian_loss, kld_category_loss, nll_loss, _, _, _, _, _, _ = model(batchX, batchY)
        kld_loss = kld_gaussian_loss + kld_category_loss
        loss = annealing * kld_loss / (batchX.size(1) * batchX.size(0)) + nll_loss / (batchX.size(1) * batchX.size(0))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

    # Evaluate on full training set
    all_d_t_sampled_plot, all_z_t_sampled, loss, all_d_posterior, all_z_posterior_mean = test(
        model, trainX, trainY, epoch, "train"
    )

    return all_d_t_sampled_plot, all_z_t_sampled, loss, all_d_posterior, all_z_posterior_mean


def forecast_multivariate_conditioned(model, testX, testY, y_moments, d_dim, MC_S=200):
    """Forecast function for conditioned model."""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        outputs = model(testX, testY)
        y_emission = outputs[4]
        all_d_posterior = outputs[7]

        all_y_mean, all_y_std = y_emission
        predictions_norm = torch.stack(all_y_mean, dim=0)

        d_posteriors = torch.stack(all_d_posterior, dim=0)
        regime_assignments = d_posteriors[-1].argmax(dim=1).cpu().numpy()

        predictions_last = predictions_norm[-1].cpu().numpy()
        testY_norm = testY[-1].cpu().numpy()

        # Uncertainty estimation
        all_predictions = [predictions_last]
        for _ in range(MC_S - 1):
            outputs_mc = model(testX, testY)
            y_emission_mc = outputs_mc[4]
            all_y_mean_mc, _ = y_emission_mc
            pred_mc = torch.stack(all_y_mean_mc, dim=0)[-1].cpu().numpy()
            all_predictions.append(pred_mc)

        all_predictions = np.array(all_predictions)

        predictions_mean = np.mean(all_predictions, axis=0)
        predictions_upper = np.quantile(all_predictions, 0.95, axis=0)
        predictions_lower = np.quantile(all_predictions, 0.05, axis=0)

        def invert_norm(data, moments):
            return data * moments[1] + moments[0]

        predictions_mean_inv = invert_norm(predictions_mean, y_moments)
        predictions_upper_inv = invert_norm(predictions_upper, y_moments)
        predictions_lower_inv = invert_norm(predictions_lower, y_moments)
        testY_inv = invert_norm(testY_norm, y_moments)

        pred_flat = predictions_mean_inv.reshape(-1)
        actual_flat = testY_inv.reshape(-1)

        res = evaluation(pred_flat.reshape(1, -1), actual_flat.reshape(1, -1))
        spearman_corr, spearman_pval = spearmanr(actual_flat, pred_flat)
        res['spearman'] = spearman_corr
        res['spearman_pval'] = spearman_pval

    return {
        'predictions': predictions_mean_inv,
        'predictions_upper': predictions_upper_inv,
        'predictions_lower': predictions_lower_inv,
        'original': testY_inv.reshape(-1),
        'regime_assignments': regime_assignments,
        'metrics': res,
        'size': len(actual_flat)
    }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluation(pred, actual):
    """Compute evaluation metrics."""
    pred = pred.flatten()
    actual = actual.flatten()
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    mae = np.mean(np.abs(pred - actual))
    mape = np.mean(np.abs((pred - actual) / (actual + 1e-8))) * 100
    return {'rmse': rmse, 'mae': mae, 'mape': mape}


# Multivariate config
QRT_MULTIVARIATE_CONFIG = {
    'ALL': {
        'timestep': 14,
        'x_dim': 32,
        'y_dim': 1,
        'h_dim': 48,
        'z_dim': 12,
        'd_dim': 2,
        'n_layers': 1,
        'n_epochs': 200,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'clip': 10,
        'bidirection': False,
        'dropout': 0.1,
    },
    'FR': {
        'timestep': 14,
        'x_dim': 32,
        'y_dim': 1,
        'h_dim': 32,
        'z_dim': 8,
        'd_dim': 2,
        'n_layers': 1,
        'n_epochs': 200,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'clip': 10,
        'bidirection': False,
        'dropout': 0.1,
    },
    'DE': {
        'timestep': 14,
        'x_dim': 32,
        'y_dim': 1,
        'h_dim': 32,
        'z_dim': 8,
        'd_dim': 2,
        'n_layers': 1,
        'n_epochs': 200,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'clip': 10,
        'bidirection': False,
        'dropout': 0.1,
    }
}


def run_conditioned_experiment(
    country: str,
    config: dict,
    output_dir: Path,
    train_model: bool = True,
    seed: int = 42,
    verbose: bool = True
):
    """Run conditioned DS3M experiment."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n{'=' * 60}")
    print(f"CONDITIONED DS3M EXPERIMENT")
    print(f"{'=' * 60}")

    # Load data
    print(f"\nLoading MULTIVARIATE data for {country}...")
    df = load_qrt_data()

    country_filter = None if country == 'ALL' else country
    data = prepare_multivariate_train_test_split(
        df,
        country=country_filter,
        feature_cols=None,
        target_col='TARGET',
        timestep=config['timestep'],
        test_ratio=0.2
    )

    trainX = data['trainX'].to(device)
    trainY = data['trainY'].to(device)
    testX = data['testX'].to(device)
    testY = data['testY'].to(device)
    y_moments = data['Y_moments']
    feature_cols = data['feature_cols']
    n_features = data['n_features']

    print(f"Train X shape: {trainX.shape}")
    print(f"Test X shape: {testX.shape}")

    # Create conditioned model
    model = ConditionedDSSSM(
        x_dim=n_features,
        y_dim=config['y_dim'],
        h_dim=config['h_dim'],
        z_dim=config['z_dim'],
        d_dim=config['d_dim'],
        n_layers=config['n_layers'],
        device=device,
        bidirection=config.get('bidirection', False),
        gate_init=0.5  # Start with equal blend
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params}")

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if train_model:
        print("\n--- Training ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        early_stopping = EarlyStopping(patience=20, verbose=True)

        best_validation = float('inf')
        loss_history = {'train': [], 'test': [], 'gate': []}
        start_time = time.time()

        for epoch in range(1, config['n_epochs'] + 1):
            _, _, loss_train, _, _ = train_conditioned(
                model, optimizer, trainX, trainY, epoch,
                config['batch_size'], config['n_epochs']
            )

            _, _, loss_test, _, _ = test(model, testX, testY, epoch, "test")

            loss_history['train'].append(loss_train)
            loss_history['test'].append(loss_test)
            loss_history['gate'].append(model.get_gate_value())

            if loss_test < best_validation:
                best_validation = loss_test
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train,
                    'config': config,
                    'y_moments': y_moments,
                    'feature_cols': feature_cols,
                    'gate': model.get_gate_value(),
                }, checkpoint_dir / 'best.tar')

            scheduler.step(loss_test)
            if verbose and epoch % 10 == 0:
                print(f"LR: {optimizer.param_groups[0]['lr']}, Gate: {model.get_gate_value():.3f}")

            early_stopping(loss_test, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f}s")
        print(f"Final gate value: {model.get_gate_value():.3f}")

        # Plot loss and gate evolution
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(loss_history['train'], label='Train Loss')
        ax1.plot(loss_history['test'], label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training History')
        ax1.legend()

        ax2.plot(loss_history['gate'], color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Gate Value')
        ax2.set_title('Gate Evolution (1=base, 0=conditioned)')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(figures_dir / 'training_history.png', dpi=150)
        plt.close()

    # Load best model
    checkpoint = torch.load(checkpoint_dir / 'best.tar', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"Gate value at best: {checkpoint.get('gate', 'N/A')}")

    # Generate forecasts
    print("\n--- Forecasting ---")
    results = forecast_multivariate_conditioned(
        model, testX, testY, y_moments, config['d_dim'], MC_S=200
    )

    print(f"\nResults:")
    print(f"  RMSE: {results['metrics']['rmse']:.4f}")
    print(f"  Spearman: {results['metrics']['spearman']:.4f}")
    print(f"  Gate: {model.get_gate_value():.3f}")

    # Save results
    results_summary = {
        'country': country,
        'mode': 'conditioned',
        'target': 'TARGET',
        'n_features': n_features,
        'feature_cols': feature_cols,
        'config': config,
        'metrics': {k: float(v) for k, v in results['metrics'].items()},
        'n_regimes_detected': len(np.unique(results['regime_assignments'])),
        'regime_distribution': {
            int(k): int(v) for k, v in
            zip(*np.unique(results['regime_assignments'], return_counts=True))
        },
        'gate_value': model.get_gate_value(),
        'training_samples': int(trainX.shape[1]),
        'test_samples': int(testX.shape[1]),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        'original': results['original'].reshape(-1),
        'prediction': results['predictions'].reshape(-1),
        'regime': results['regime_assignments']
    })
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)

    np.save(output_dir / 'regime_assignments.npy', results['regime_assignments'])

    print(f"\nResults saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Conditioned DS3M for electricity price prediction")
    parser.add_argument('--country', type=str, required=True, choices=['FR', 'DE', 'ALL'])
    parser.add_argument('--mode', type=str, default='multivariate', choices=['multivariate'])
    parser.add_argument('--d_dim', type=int, default=None)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    config = QRT_MULTIVARIATE_CONFIG.get(args.country, QRT_MULTIVARIATE_CONFIG['ALL']).copy()

    if args.epochs:
        config['n_epochs'] = args.epochs
    if args.d_dim is not None:
        config['d_dim'] = args.d_dim
    if args.debug:
        config['n_epochs'] = 10

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d_dim = config.get('d_dim', 2)
        output_dir = Path(__file__).parent / "outputs" / "ds3m_conditioned" / f"{args.country}_d{d_dim}_seed{args.seed}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    results = run_conditioned_experiment(
        country=args.country,
        config=config,
        output_dir=output_dir,
        train_model=args.train,
        seed=args.seed,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Conditioned DS3M Experiment Complete")
    print("=" * 60)
    print(f"Country: {args.country}")
    print(f"d_dim: {config.get('d_dim', 2)}")
    print(f"Spearman: {results['metrics']['spearman']:.4f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
