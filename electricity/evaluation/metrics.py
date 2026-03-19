"""
Comprehensive forecasting metrics for CaRS evaluation.

Provides RMSE, MAE, sMAPE, Spearman correlation, directional accuracy,
CRPS, and regime-stratified evaluation for Applied Energy submission.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from typing import Dict, Optional, Union


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Uses sMAPE to avoid division-by-zero with near-zero electricity prices.
    Range: [0, 200%]. Returns percentage value.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid 0/0: where both are zero, error is zero
    mask = denominator > 0
    if mask.sum() == 0:
        return 0.0
    return float(100.0 * np.mean(numerator[mask] / denominator[mask]))


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    corr, _ = spearmanr(y_true.flatten(), y_pred.flatten())
    return float(corr) if not np.isnan(corr) else 0.0


def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: np.ndarray
) -> float:
    """
    Fraction of correctly predicted price movement directions.

    Args:
        y_true: Actual values at time t
        y_pred: Predicted values at time t
        y_prev: Actual values at time t-1
    """
    actual_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    valid = actual_dir.flatten() != 0
    if valid.sum() == 0:
        return 0.5
    return float(np.mean(actual_dir.flatten()[valid] == pred_dir.flatten()[valid]))


def crps_gaussian(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray
) -> float:
    """
    Continuous Ranked Probability Score for Gaussian predictive distribution.

    Lower is better. Measures calibration of probabilistic forecasts.
    Closed-form for Gaussian: CRPS = sigma * [z*Phi(z) + phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma.
    """
    std_safe = np.maximum(y_pred_std, 1e-6)
    z = (y_true - y_pred_mean) / std_safe
    crps_vals = std_safe * (
        z * (2.0 * norm.cdf(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps_vals))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: Optional[np.ndarray] = None,
    y_pred_std: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all forecasting metrics.

    Args:
        y_true: Ground truth values [N] or [N, 1]
        y_pred: Point predictions [N] or [N, 1]
        y_prev: Previous timestep actuals for directional accuracy (optional)
        y_pred_std: Prediction standard deviation for CRPS (optional)

    Returns:
        Dict with keys: rmse, mae, smape, spearman, directional_accuracy, crps
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    metrics = {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'spearman': spearman_correlation(y_true, y_pred),
    }

    if y_prev is not None:
        y_prev = np.asarray(y_prev).flatten()
        metrics['directional_accuracy'] = directional_accuracy(y_true, y_pred, y_prev)

    if y_pred_std is not None:
        y_pred_std = np.asarray(y_pred_std).flatten()
        metrics['crps'] = crps_gaussian(y_true, y_pred, y_pred_std)

    return metrics


def regime_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime_assignments: np.ndarray,
    timestamps: Union[pd.DatetimeIndex, np.ndarray],
    y_prev: Optional[np.ndarray] = None,
    y_pred_std: Optional[np.ndarray] = None,
    crisis_start: str = '2021-09-01',
    crisis_end: str = '2023-06-01'
) -> Dict[str, Dict[str, float]]:
    """
    Report metrics stratified by regime and calendar period.

    Stratifies by:
    1. Model-assigned regime (regime_0, regime_1, ...)
    2. Calendar period (normal, crisis)
    3. Cross-tabulation (regime_0_normal, regime_0_crisis, ...)

    This validates whether CaRS regimes align with known market periods
    and addresses the confound of test-set regime dominance.

    Args:
        y_true: Ground truth [N]
        y_pred: Predictions [N]
        regime_assignments: Integer regime labels [N]
        timestamps: DatetimeIndex or array of timestamps [N]
        y_prev: Previous timestep actuals (optional)
        y_pred_std: Prediction std (optional)
        crisis_start: Start of EU energy crisis
        crisis_end: End of EU energy crisis

    Returns:
        Nested dict: {stratum_name: {metric_name: value}}
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    regime_assignments = np.asarray(regime_assignments).flatten()

    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)

    crisis_mask = (timestamps >= crisis_start) & (timestamps < crisis_end)
    normal_mask = ~crisis_mask

    results = {}

    # Overall
    results['overall'] = compute_all_metrics(y_true, y_pred, y_prev, y_pred_std)

    # By calendar period
    for period_name, mask in [('normal', normal_mask), ('crisis', crisis_mask)]:
        if mask.sum() > 10:
            kw = {}
            if y_prev is not None:
                kw['y_prev'] = y_prev[mask]
            if y_pred_std is not None:
                kw['y_pred_std'] = y_pred_std[mask]
            results[period_name] = compute_all_metrics(
                y_true[mask], y_pred[mask], **kw
            )
            results[period_name]['n_samples'] = int(mask.sum())

    # By model-assigned regime
    unique_regimes = np.unique(regime_assignments)
    for r in unique_regimes:
        r_mask = regime_assignments == r
        if r_mask.sum() > 10:
            kw = {}
            if y_prev is not None:
                kw['y_prev'] = y_prev[r_mask]
            if y_pred_std is not None:
                kw['y_pred_std'] = y_pred_std[r_mask]
            results[f'regime_{r}'] = compute_all_metrics(
                y_true[r_mask], y_pred[r_mask], **kw
            )
            results[f'regime_{r}']['n_samples'] = int(r_mask.sum())

    # Cross-tabulation: regime x period
    for r in unique_regimes:
        for period_name, period_mask in [('normal', normal_mask), ('crisis', crisis_mask)]:
            combined = (regime_assignments == r) & period_mask
            if combined.sum() > 10:
                kw = {}
                if y_prev is not None:
                    kw['y_prev'] = y_prev[combined]
                if y_pred_std is not None:
                    kw['y_pred_std'] = y_pred_std[combined]
                results[f'regime_{r}_{period_name}'] = compute_all_metrics(
                    y_true[combined], y_pred[combined], **kw
                )
                results[f'regime_{r}_{period_name}']['n_samples'] = int(combined.sum())

    # Regime-period alignment statistics
    regime_crisis_frac = {}
    for r in unique_regimes:
        r_mask = regime_assignments == r
        if r_mask.sum() > 0:
            regime_crisis_frac[f'regime_{r}_crisis_fraction'] = float(
                (r_mask & crisis_mask).sum() / r_mask.sum()
            )
    results['alignment'] = regime_crisis_frac

    return results
