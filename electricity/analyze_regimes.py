"""
Regime Analysis for DS3M Models.

Analyzes what distinguishes regime 0 (normal market) from regime 1 (anomaly):
1. Per-regime feature statistics
2. Classification of regimes from features
3. Feature importance for regime prediction
4. Temporal patterns in regime switches

Usage:
    python analyze_regimes.py --checkpoint outputs/ds3m/ALL_TARGET_mv_d2_seed42_20260118_163330
    python analyze_regimes.py --checkpoint outputs/ds3m/ALL_TARGET_mv_d2_seed42_20260118_163330 --output figures/
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
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from DSSSMCode import DSSSM
from ds3m_adapter import load_qrt_data, prepare_multivariate_train_test_split


def load_trained_model(checkpoint_dir: Path, device: str = 'cpu'):
    """Load a trained DS3M model and its configuration."""
    checkpoint_path = checkpoint_dir / 'checkpoints' / 'best.tar'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Determine model type from config
    x_dim = config.get('x_dim', config.get('predict_dim', 1))
    y_dim = config.get('y_dim', config.get('predict_dim', 1))

    model = DSSSM(
        x_dim=x_dim,
        y_dim=y_dim,
        h_dim=config['h_dim'],
        z_dim=config['z_dim'],
        d_dim=config['d_dim'],
        n_layers=config['n_layers'],
        device=device,
        bidirection=config.get('bidirection', False)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config, checkpoint


def get_regime_assignments_full(model, data, device='cpu'):
    """Get regime assignments for full dataset (train + test)."""
    model.eval()

    trainX = data['trainX'].to(device)
    trainY = data['trainY'].to(device)
    testX = data['testX'].to(device)
    testY = data['testY'].to(device)

    with torch.no_grad():
        # Training set
        outputs_train = model(trainX, trainY)
        d_posterior_train = outputs_train[7]
        regimes_train = torch.stack(d_posterior_train, dim=0)[-1].argmax(dim=1).cpu().numpy()

        # Test set
        outputs_test = model(testX, testY)
        d_posterior_test = outputs_test[7]
        regimes_test = torch.stack(d_posterior_test, dim=0)[-1].argmax(dim=1).cpu().numpy()

    return regimes_train, regimes_test


def analyze_feature_statistics(features, regimes, feature_names, output_dir=None):
    """Compute per-regime feature statistics and significance tests."""
    results = []
    n_regimes = len(np.unique(regimes))

    for i, name in enumerate(feature_names):
        row = {'feature': name}

        for regime in range(n_regimes):
            mask = regimes == regime
            values = features[mask, i]
            row[f'regime{regime}_mean'] = values.mean()
            row[f'regime{regime}_std'] = values.std()
            row[f'regime{regime}_n'] = mask.sum()

        # Statistical test between regimes (if 2 regimes)
        if n_regimes == 2:
            values_0 = features[regimes == 0, i]
            values_1 = features[regimes == 1, i]

            # Mann-Whitney U test (non-parametric)
            stat, pval = mannwhitneyu(values_0, values_1, alternative='two-sided')
            row['mw_pvalue'] = pval

            # Effect size (difference in means / pooled std)
            pooled_std = np.sqrt((values_0.std()**2 + values_1.std()**2) / 2)
            effect_size = (values_1.mean() - values_0.mean()) / (pooled_std + 1e-8)
            row['effect_size'] = effect_size

        results.append(row)

    df = pd.DataFrame(results)

    # Sort by effect size magnitude
    if 'effect_size' in df.columns:
        df = df.sort_values('effect_size', key=abs, ascending=False)

    if output_dir:
        df.to_csv(output_dir / 'feature_statistics.csv', index=False)

    return df


def train_regime_classifier(features, regimes, feature_names, output_dir=None):
    """Train classifiers to predict regime from features."""
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = regimes

    results = {}

    # Try multiple classifiers
    classifiers = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    for name, clf in classifiers.items():
        # Cross-validation accuracy
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        results[name] = {
            'accuracy_mean': scores.mean(),
            'accuracy_std': scores.std()
        }
        print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")

    # Fit best classifier (random forest) and get feature importance
    best_clf = classifiers['random_forest']
    best_clf.fit(X, y)

    # Feature importance
    importances = best_clf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Permutation importance (more reliable)
    perm_importance = permutation_importance(best_clf, X, y, n_repeats=10, random_state=42)
    importance_df['perm_importance_mean'] = perm_importance.importances_mean
    importance_df['perm_importance_std'] = perm_importance.importances_std

    importance_df = importance_df.sort_values('perm_importance_mean', ascending=False)

    if output_dir:
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

        with open(output_dir / 'classifier_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    return results, importance_df, best_clf


def analyze_temporal_patterns(regimes, day_ids, output_dir=None):
    """Analyze temporal patterns in regime switches."""
    # Create DataFrame
    df = pd.DataFrame({
        'day_id': day_ids,
        'regime': regimes
    })

    # Detect regime switches
    df['switch'] = (df['regime'] != df['regime'].shift(1)).astype(int)
    df.loc[0, 'switch'] = 0  # First point is not a switch

    n_switches = df['switch'].sum()
    switch_rate = n_switches / len(df)

    # Regime durations
    regime_changes = df[df['switch'] == 1].index.tolist()
    regime_changes = [0] + regime_changes + [len(df)]
    durations = np.diff(regime_changes)

    results = {
        'n_switches': int(n_switches),
        'switch_rate': float(switch_rate),
        'mean_duration': float(durations.mean()),
        'std_duration': float(durations.std()),
        'min_duration': int(durations.min()),
        'max_duration': int(durations.max())
    }

    if output_dir:
        with open(output_dir / 'temporal_patterns.json', 'w') as f:
            json.dump(results, f, indent=2)

    return results, df


def plot_regime_analysis(features, regimes, feature_names, importance_df, output_dir):
    """Generate visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Top features by importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(15)
    ax.barh(range(len(top_features)), top_features['perm_importance_mean'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Permutation Importance')
    ax.set_title('Top Features for Regime Prediction')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150)
    plt.close()

    # 2. Feature distribution by regime for top features
    n_regimes = len(np.unique(regimes))
    top_feature_names = importance_df.head(6)['feature'].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, feat_name in enumerate(top_feature_names):
        feat_idx = feature_names.index(feat_name)
        ax = axes[idx]

        for regime in range(n_regimes):
            mask = regimes == regime
            ax.hist(features[mask, feat_idx], bins=30, alpha=0.5,
                   label=f'Regime {regime} (n={mask.sum()})')

        ax.set_xlabel(feat_name)
        ax.set_ylabel('Count')
        ax.legend()
        ax.set_title(f'{feat_name} by Regime')

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=150)
    plt.close()

    # 3. Regime distribution pie chart
    unique, counts = np.unique(regimes, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique)))
    ax.pie(counts, labels=[f'Regime {r}\n({c} samples, {c/len(regimes)*100:.1f}%)'
                           for r, c in zip(unique, counts)],
           colors=colors, autopct='', startangle=90)
    ax.set_title('Regime Distribution')
    plt.savefig(output_dir / 'regime_distribution.png', dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DS3M regimes")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for analysis results')
    parser.add_argument('--country', type=str, default='ALL',
                       choices=['FR', 'DE', 'ALL'],
                       help='Country (should match trained model)')

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = checkpoint_dir / 'regime_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cpu'

    print(f"\n{'=' * 60}")
    print("REGIME ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Output: {output_dir}")

    # Load model
    print("\nLoading trained model...")
    model, config, checkpoint = load_trained_model(checkpoint_dir, device)
    print(f"Config: d_dim={config['d_dim']}, h_dim={config['h_dim']}")

    # Load data
    print("\nLoading data...")
    df = load_qrt_data()
    country_filter = None if args.country == 'ALL' else args.country

    data = prepare_multivariate_train_test_split(
        df,
        country=country_filter,
        feature_cols=None,
        target_col='TARGET',
        timestep=config.get('timestep', 14),
        test_ratio=0.2
    )

    feature_cols = data['feature_cols']
    n_features = len(feature_cols)
    print(f"Features: {n_features}")

    # Get regime assignments
    print("\nGetting regime assignments...")
    regimes_train, regimes_test = get_regime_assignments_full(model, data, device)

    # Combine for full analysis
    regimes_full = np.concatenate([regimes_train, regimes_test])

    # Get features (last timestep)
    features_train = data['trainX'][-1].numpy()  # (batch, n_features)
    features_test = data['testX'][-1].numpy()
    features_full = np.concatenate([features_train, features_test], axis=0)

    print(f"\nRegime distribution:")
    unique, counts = np.unique(regimes_full, return_counts=True)
    for r, c in zip(unique, counts):
        print(f"  Regime {r}: {c} samples ({c/len(regimes_full)*100:.1f}%)")

    # 1. Feature statistics
    print("\n--- Feature Statistics ---")
    stats_df = analyze_feature_statistics(features_full, regimes_full, feature_cols, output_dir)
    print("\nTop features by effect size:")
    print(stats_df[['feature', 'effect_size', 'mw_pvalue']].head(10).to_string(index=False))

    # 2. Regime classifier
    print("\n--- Regime Classification ---")
    clf_results, importance_df, classifier = train_regime_classifier(
        features_full, regimes_full, feature_cols, output_dir
    )
    print("\nTop features by permutation importance:")
    print(importance_df[['feature', 'perm_importance_mean']].head(10).to_string(index=False))

    # 3. Generate plots
    print("\n--- Generating Plots ---")
    plot_regime_analysis(features_full, regimes_full, feature_cols, importance_df, output_dir)

    # 4. Summary
    summary = {
        'checkpoint': str(checkpoint_dir),
        'country': args.country,
        'd_dim': config['d_dim'],
        'n_samples': len(regimes_full),
        'regime_distribution': {int(k): int(v) for k, v in zip(unique, counts)},
        'classifier_accuracy': clf_results['random_forest']['accuracy_mean'],
        'top_features': importance_df.head(10)['feature'].tolist()
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Analysis Complete")
    print(f"{'=' * 60}")
    print(f"Classifier accuracy: {clf_results['random_forest']['accuracy_mean']:.3f}")
    print(f"Top predictive features: {', '.join(summary['top_features'][:5])}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
