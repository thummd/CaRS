"""
Compare regime assignments between DS3M and FANTOM.

This script analyzes and visualizes the correspondence between:
- DS3M's discrete latent variable d_t (regime assignments)
- FANTOM's EM-based gamma_hat (regime assignments)

Usage:
    python compare_regimes.py --ds3m outputs/ds3m/FR_TARGET_*/
                              --fantom regime_results/FR_*/
                              --country FR
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    confusion_matrix, cohen_kappa_score
)
from typing import Dict, List, Optional, Tuple


def load_ds3m_results(ds3m_dir: Path) -> Dict:
    """
    Load DS3M experiment results.

    Args:
        ds3m_dir: Path to DS3M output directory

    Returns:
        Dictionary with regime assignments, predictions, metrics
    """
    results = {}

    # Load regime assignments
    regime_path = ds3m_dir / 'regime_assignments.npy'
    if regime_path.exists():
        results['regime_assignments'] = np.load(regime_path)
    else:
        raise FileNotFoundError(f"Regime assignments not found: {regime_path}")

    # Load predictions
    pred_path = ds3m_dir / 'predictions.csv'
    if pred_path.exists():
        results['predictions'] = pd.read_csv(pred_path)

    # Load metrics
    metrics_path = ds3m_dir / 'results.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            results['summary'] = json.load(f)

    return results


def load_fantom_results(fantom_dir: Path) -> Dict:
    """
    Load FANTOM regime results.

    Args:
        fantom_dir: Path to FANTOM output directory

    Returns:
        Dictionary with regime assignments and metrics
    """
    results = {}

    # Load regime model state
    model_path = fantom_dir / 'regime_model.json'
    if model_path.exists():
        with open(model_path, 'r') as f:
            model_data = json.load(f)
            if 'gamma_hat' in model_data and model_data['gamma_hat'] is not None:
                gamma_hat = np.array(model_data['gamma_hat'])
                results['regime_assignments'] = np.argmax(gamma_hat, axis=1)
                results['regime_probs'] = gamma_hat
                results['n_regimes'] = model_data.get('n_regimes', gamma_hat.shape[1])

    # Load summary
    summary_path = fantom_dir / 'summary.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)

    return results


def align_regime_lengths(
    ds3m_regimes: np.ndarray,
    fantom_regimes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align regime arrays to the same length.

    Due to different windowing/processing, the arrays may have different lengths.
    We align them from the end (most recent samples).

    Args:
        ds3m_regimes: DS3M regime assignments
        fantom_regimes: FANTOM regime assignments

    Returns:
        Aligned (ds3m, fantom) arrays of same length
    """
    min_len = min(len(ds3m_regimes), len(fantom_regimes))

    # Align from end
    ds3m_aligned = ds3m_regimes[-min_len:]
    fantom_aligned = fantom_regimes[-min_len:]

    return ds3m_aligned, fantom_aligned


def compute_regime_metrics(
    ds3m_regimes: np.ndarray,
    fantom_regimes: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics comparing regime assignments.

    Args:
        ds3m_regimes: DS3M regime assignments
        fantom_regimes: FANTOM regime assignments

    Returns:
        Dictionary of comparison metrics
    """
    # Align lengths
    ds3m, fantom = align_regime_lengths(ds3m_regimes, fantom_regimes)

    metrics = {
        'n_samples': len(ds3m),
        'ds3m_n_regimes': len(np.unique(ds3m)),
        'fantom_n_regimes': len(np.unique(fantom)),
    }

    # Adjusted Rand Index (measures similarity, corrected for chance)
    metrics['adjusted_rand_index'] = adjusted_rand_score(fantom, ds3m)

    # Normalized Mutual Information
    metrics['normalized_mutual_info'] = normalized_mutual_info_score(fantom, ds3m)

    # Cohen's Kappa (agreement beyond chance)
    try:
        metrics['cohens_kappa'] = cohen_kappa_score(fantom, ds3m)
    except:
        metrics['cohens_kappa'] = np.nan

    # Agreement rate (simple accuracy)
    metrics['agreement_rate'] = np.mean(ds3m == fantom)

    # Check if we need to flip DS3M regimes (they might be reversed)
    flipped_ds3m = 1 - ds3m if len(np.unique(ds3m)) == 2 else ds3m
    flipped_agreement = np.mean(flipped_ds3m == fantom)
    if flipped_agreement > metrics['agreement_rate']:
        metrics['agreement_rate_flipped'] = flipped_agreement
        metrics['requires_flip'] = True
    else:
        metrics['agreement_rate_flipped'] = metrics['agreement_rate']
        metrics['requires_flip'] = False

    return metrics


def compute_regime_statistics(regimes: np.ndarray, name: str) -> Dict:
    """Compute regime duration and transition statistics."""
    stats = {'name': name}

    # Regime distribution
    unique, counts = np.unique(regimes, return_counts=True)
    stats['regime_distribution'] = {int(u): int(c) for u, c in zip(unique, counts)}

    # Regime durations (consecutive sequences)
    durations = {int(u): [] for u in unique}
    current_regime = regimes[0]
    current_duration = 1

    for i in range(1, len(regimes)):
        if regimes[i] == current_regime:
            current_duration += 1
        else:
            durations[int(current_regime)].append(current_duration)
            current_regime = regimes[i]
            current_duration = 1
    durations[int(current_regime)].append(current_duration)

    stats['avg_duration'] = {
        k: np.mean(v) if v else 0 for k, v in durations.items()
    }
    stats['total_transitions'] = sum(
        1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1]
    )
    stats['transition_rate'] = stats['total_transitions'] / (len(regimes) - 1)

    return stats


def plot_regime_comparison(
    ds3m_regimes: np.ndarray,
    fantom_regimes: np.ndarray,
    output_path: str,
    title: str = "Regime Comparison: DS3M vs FANTOM"
):
    """
    Create comparison visualization of regime assignments.

    Args:
        ds3m_regimes: DS3M regime assignments
        fantom_regimes: FANTOM regime assignments
        output_path: Path to save figure
        title: Plot title
    """
    ds3m, fantom = align_regime_lengths(ds3m_regimes, fantom_regimes)
    n_samples = len(ds3m)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), height_ratios=[1, 1, 1, 2])

    # DS3M regimes
    ax1 = axes[0]
    cmap_ds3m = plt.cm.Blues
    sns.heatmap(
        ds3m.reshape(1, -1), ax=ax1,
        cmap=cmap_ds3m, cbar=False,
        xticklabels=False, yticklabels=False
    )
    ax1.set_ylabel('DS3M')
    ax1.set_title(title)

    # FANTOM regimes
    ax2 = axes[1]
    cmap_fantom = plt.cm.Oranges
    sns.heatmap(
        fantom.reshape(1, -1), ax=ax2,
        cmap=cmap_fantom, cbar=False,
        xticklabels=False, yticklabels=False
    )
    ax2.set_ylabel('FANTOM')

    # Agreement
    ax3 = axes[2]
    agreement = (ds3m == fantom).astype(int)
    cmap_agree = plt.cm.RdYlGn
    sns.heatmap(
        agreement.reshape(1, -1), ax=ax3,
        cmap=cmap_agree, cbar=False,
        xticklabels=False, yticklabels=False,
        vmin=0, vmax=1
    )
    ax3.set_ylabel('Agree')

    # Time series comparison
    ax4 = axes[3]
    x = np.arange(n_samples)
    ax4.step(x, ds3m + 0.1, where='post', label='DS3M', color='blue', alpha=0.7)
    ax4.step(x, fantom - 0.1, where='post', label='FANTOM', color='orange', alpha=0.7)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Regime')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add metrics annotation
    metrics = compute_regime_metrics(ds3m_regimes, fantom_regimes)
    metrics_text = (
        f"ARI: {metrics['adjusted_rand_index']:.3f}  |  "
        f"NMI: {metrics['normalized_mutual_info']:.3f}  |  "
        f"Agreement: {metrics['agreement_rate']:.1%}"
    )
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to: {output_path}")


def plot_confusion_matrix(
    ds3m_regimes: np.ndarray,
    fantom_regimes: np.ndarray,
    output_path: str,
    title: str = "Regime Confusion Matrix"
):
    """Plot confusion matrix between DS3M and FANTOM regimes."""
    ds3m, fantom = align_regime_lengths(ds3m_regimes, fantom_regimes)

    cm = confusion_matrix(fantom, ds3m)
    n_ds3m = len(np.unique(ds3m))
    n_fantom = len(np.unique(fantom))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=[f'DS3M {i}' for i in range(n_ds3m)],
        yticklabels=[f'FANTOM {i}' for i in range(n_fantom)]
    )
    ax.set_xlabel('DS3M Regime')
    ax.set_ylabel('FANTOM Regime')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: {output_path}")


def plot_regime_durations(
    ds3m_regimes: np.ndarray,
    fantom_regimes: np.ndarray,
    output_path: str
):
    """Compare regime duration distributions."""
    ds3m, fantom = align_regime_lengths(ds3m_regimes, fantom_regimes)

    def get_durations(regimes):
        durations = []
        current_len = 1
        for i in range(1, len(regimes)):
            if regimes[i] == regimes[i-1]:
                current_len += 1
            else:
                durations.append(current_len)
                current_len = 1
        durations.append(current_len)
        return durations

    ds3m_durations = get_durations(ds3m)
    fantom_durations = get_durations(fantom)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(ds3m_durations, bins=20, alpha=0.7, label='DS3M', color='blue')
    axes[0].hist(fantom_durations, bins=20, alpha=0.7, label='FANTOM', color='orange')
    axes[0].set_xlabel('Duration (samples)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Regime Duration Distribution')
    axes[0].legend()

    # Box plot
    data = [ds3m_durations, fantom_durations]
    axes[1].boxplot(data, labels=['DS3M', 'FANTOM'])
    axes[1].set_ylabel('Duration (samples)')
    axes[1].set_title('Regime Duration Comparison')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved duration comparison to: {output_path}")


def generate_comparison_report(
    ds3m_results: Dict,
    fantom_results: Dict,
    output_dir: Path,
    country: str
) -> Dict:
    """
    Generate comprehensive comparison report.

    Args:
        ds3m_results: DS3M experiment results
        fantom_results: FANTOM experiment results
        output_dir: Output directory
        country: Country code

    Returns:
        Report dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ds3m_regimes = ds3m_results['regime_assignments']
    fantom_regimes = fantom_results['regime_assignments']

    # Compute metrics
    metrics = compute_regime_metrics(ds3m_regimes, fantom_regimes)

    # Compute statistics
    ds3m_stats = compute_regime_statistics(ds3m_regimes, 'DS3M')
    fantom_stats = compute_regime_statistics(fantom_regimes, 'FANTOM')

    # Generate plots
    plot_regime_comparison(
        ds3m_regimes, fantom_regimes,
        str(output_dir / f'{country}_regime_comparison.png'),
        title=f'{country}: DS3M vs FANTOM Regimes'
    )

    plot_confusion_matrix(
        ds3m_regimes, fantom_regimes,
        str(output_dir / f'{country}_confusion_matrix.png'),
        title=f'{country}: Regime Confusion Matrix'
    )

    plot_regime_durations(
        ds3m_regimes, fantom_regimes,
        str(output_dir / f'{country}_duration_comparison.png')
    )

    # Build report
    report = {
        'country': country,
        'comparison_metrics': metrics,
        'ds3m_statistics': ds3m_stats,
        'fantom_statistics': fantom_stats,
        'ds3m_prediction_metrics': ds3m_results.get('summary', {}).get('metrics', {}),
        'fantom_prediction_metrics': fantom_results.get('summary', {}).get('spearman', None),
    }

    # Save report
    with open(output_dir / f'{country}_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return report


def main():
    parser = argparse.ArgumentParser(description="Compare DS3M and FANTOM regimes")
    parser.add_argument('--ds3m', type=str, required=True,
                        help='Path to DS3M results directory')
    parser.add_argument('--fantom', type=str, required=True,
                        help='Path to FANTOM results directory')
    parser.add_argument('--country', type=str, default='',
                        help='Country code for labeling')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    ds3m_dir = Path(args.ds3m)
    fantom_dir = Path(args.fantom)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent / "regime_comparison"

    print(f"Loading DS3M results from: {ds3m_dir}")
    print(f"Loading FANTOM results from: {fantom_dir}")

    # Load results
    ds3m_results = load_ds3m_results(ds3m_dir)
    fantom_results = load_fantom_results(fantom_dir)

    print(f"\nDS3M regimes: {len(ds3m_results['regime_assignments'])} samples, "
          f"{len(np.unique(ds3m_results['regime_assignments']))} unique regimes")
    print(f"FANTOM regimes: {len(fantom_results['regime_assignments'])} samples, "
          f"{len(np.unique(fantom_results['regime_assignments']))} unique regimes")

    # Generate report
    country = args.country or 'Unknown'
    report = generate_comparison_report(
        ds3m_results, fantom_results, output_dir, country
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"Adjusted Rand Index: {report['comparison_metrics']['adjusted_rand_index']:.4f}")
    print(f"Normalized Mutual Info: {report['comparison_metrics']['normalized_mutual_info']:.4f}")
    print(f"Agreement Rate: {report['comparison_metrics']['agreement_rate']:.1%}")
    if report['comparison_metrics'].get('requires_flip'):
        print(f"  (with flipped labels: {report['comparison_metrics']['agreement_rate_flipped']:.1%})")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
