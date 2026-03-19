"""
Rolling-Window and Expanding-Window Evaluation for CaRS.

Addresses the confound that the fixed test set is predominantly crisis-regime
data (post-2021). By evaluating across multiple time windows, we verify that
CaRS generalizes across different market conditions.

Uses expanding windows (growing training set) by default to maximize training
data while evaluating on diverse test periods.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from paths import OUTPUT_DIR
from shared_backbone.data_loader import prepare_unified_ds3m_data
from electricity.evaluation.metrics import compute_all_metrics


RESULTS_DIR = OUTPUT_DIR / "rolling_window"


def generate_window_schedule(
    data_start: str = '2015-01-01',
    data_end: str = '2026-03-09',
    min_train_days: int = 365 * 2,
    step_size_days: int = 90,
    test_window_days: int = 90,
) -> List[Dict[str, str]]:
    """
    Generate a schedule of train/test windows.

    Uses expanding windows: training always starts from data_start,
    with the training end date advancing by step_size_days.

    Args:
        data_start: First available data date
        data_end: Last available data date
        min_train_days: Minimum training period before first evaluation
        step_size_days: Days to advance between windows
        test_window_days: Duration of each test window

    Returns:
        List of dicts with train_end, test_end date strings
    """
    start = pd.Timestamp(data_start)
    end = pd.Timestamp(data_end)

    first_train_end = start + timedelta(days=min_train_days)
    windows = []

    current_train_end = first_train_end
    while current_train_end + timedelta(days=test_window_days) <= end:
        test_end = current_train_end + timedelta(days=test_window_days)
        windows.append({
            'window_id': len(windows),
            'train_start': data_start,
            'train_end': current_train_end.strftime('%Y-%m-%d'),
            'test_start': current_train_end.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
        })
        current_train_end += timedelta(days=step_size_days)

    return windows


def train_and_evaluate_window(
    country: str,
    train_end_date: str,
    test_end_date: str,
    feature_groups: Optional[List[str]] = None,
    seed: int = 42,
    max_auglag_steps: int = 30,
    max_inner_epochs: int = 15,
    device: str = 'cuda',
) -> Dict:
    """
    Train CaRS on a single window and evaluate.

    Uses reduced training budget for computational feasibility.

    Args:
        country: Country code
        train_end_date: Training data cutoff
        test_end_date: Test data cutoff
        feature_groups: Feature groups to load
        seed: Random seed
        max_auglag_steps: Reduced outer loop steps
        max_inner_epochs: Reduced inner loop epochs
        device: Torch device

    Returns:
        Dict with metrics, timing, and window info
    """
    from shared_backbone.models.ds3m_causal import DS3MCausal
    from shared_backbone.training.train_e2e import AugmentedLagrangianTrainer

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load data with date-based split
    data = prepare_unified_ds3m_data(
        country=country,
        timestep=14,
        feature_groups=feature_groups,
        train_end_date=train_end_date,
        test_end_date=test_end_date,
    )

    if data['testX'].shape[1] < 10:
        return {'error': f'Too few test samples: {data["testX"].shape[1]}'}

    x_dim = data['trainX'].shape[-1]
    n_features = len(data['feature_cols'])

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model with standard config
    model = DS3MCausal(
        x_dim=x_dim,
        y_dim=1,
        h_dim=32,
        z_dim=8,
        d_dim=2,
        num_nodes=n_features,
        lag=1,
        sharing_mode='shared_backbone',
        lambda_dag=100.0,
        lambda_sparse=0.01,
        lambda_kl=1.0,
        device=device,
    ).to(device)

    # Create trainer with reduced budget
    trainer = AugmentedLagrangianTrainer(
        model=model,
        learning_rate=0.001,
        max_auglag_steps=max_auglag_steps,
        max_inner_epochs=max_inner_epochs,
        early_stopping_metric='spearman',
        verbose=False,
    )

    trainX = data['trainX'].to(device)
    trainY = data['trainY'].to(device)
    testX = data['testX'].to(device)
    testY = data['testY'].to(device)

    t0 = time.time()
    trainer.train(trainX, trainY, testX, testY)
    train_time = time.time() - t0

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_result = model.predict(testX, n_samples=50)

    preds = pred_result['predictions'][-1].cpu().numpy()
    preds_std = pred_result['predictions_std'][-1].cpu().numpy()
    y_true = testY[-1].cpu().numpy()
    y_prev = testY[-2].cpu().numpy()
    regimes = pred_result['regimes'][-1].cpu().numpy()

    metrics = compute_all_metrics(y_true, preds, y_prev=y_prev, y_pred_std=preds_std)

    return {
        'metrics': metrics,
        'train_time': train_time,
        'n_train': data['n_train'],
        'n_test': data['n_test'],
        'train_end': train_end_date,
        'test_end': test_end_date,
        'regime_distribution': {
            int(r): int((regimes == r).sum())
            for r in np.unique(regimes)
        },
    }


def rolling_window_evaluation(
    country: str,
    feature_groups: Optional[List[str]] = None,
    seed: int = 42,
    step_size_days: int = 90,
    test_window_days: int = 90,
    min_train_days: int = 365 * 2,
    max_auglag_steps: int = 30,
    max_inner_epochs: int = 15,
    device: str = 'cuda',
    save: bool = True,
) -> pd.DataFrame:
    """
    Run expanding-window evaluation across multiple time periods.

    Args:
        country: Country code
        feature_groups: Feature groups to load
        seed: Random seed
        step_size_days: Days between window starts
        test_window_days: Test period duration
        min_train_days: Minimum training period
        max_auglag_steps: Reduced outer loop iterations
        max_inner_epochs: Reduced inner loop epochs
        device: Torch device
        save: Save results to disk

    Returns:
        DataFrame with per-window metrics
    """
    windows = generate_window_schedule(
        min_train_days=min_train_days,
        step_size_days=step_size_days,
        test_window_days=test_window_days,
    )

    print(f"\nRolling window evaluation: {country}")
    print(f"Windows: {len(windows)}, step={step_size_days}d, test={test_window_days}d")
    print("=" * 60)

    results = []
    for w in windows:
        print(f"\nWindow {w['window_id']}: train to {w['train_end']}, "
              f"test to {w['test_end']}")

        try:
            result = train_and_evaluate_window(
                country=country,
                train_end_date=w['train_end'],
                test_end_date=w['test_end'],
                feature_groups=feature_groups,
                seed=seed,
                max_auglag_steps=max_auglag_steps,
                max_inner_epochs=max_inner_epochs,
                device=device,
            )

            if 'error' in result:
                print(f"  Skipped: {result['error']}")
                continue

            result.update(w)
            results.append(result)

            m = result['metrics']
            print(f"  RMSE={m['rmse']:.4f}, Spearman={m['spearman']:.4f}, "
                  f"n_train={result['n_train']}, n_test={result['n_test']}, "
                  f"time={result['train_time']:.0f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if not results:
        print("No successful windows.")
        return pd.DataFrame()

    # Build results DataFrame
    rows = []
    for r in results:
        row = {
            'window_id': r['window_id'],
            'train_end': r['train_end'],
            'test_end': r['test_end'],
            'n_train': r['n_train'],
            'n_test': r['n_test'],
            'train_time': r['train_time'],
        }
        row.update(r['metrics'])
        rows.append(row)

    df = pd.DataFrame(rows)

    if save:
        save_dir = RESULTS_DIR / country
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir / f"rolling_window_seed{seed}.csv", index=False)

        # Also save full results as JSON
        for r in results:
            r.pop('metrics', None)  # Already in DataFrame
        with open(save_dir / f"rolling_window_details_seed{seed}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {save_dir}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Summary for {country}:")
    print(f"  Windows completed: {len(df)}")
    print(f"  RMSE:    mean={df['rmse'].mean():.4f}, std={df['rmse'].std():.4f}")
    print(f"  Spearman: mean={df['spearman'].mean():.4f}, std={df['spearman'].std():.4f}")
    if 'directional_accuracy' in df.columns:
        print(f"  DirAcc:  mean={df['directional_accuracy'].mean():.4f}")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Rolling window evaluation')
    parser.add_argument('--country', type=str, default='DE')
    parser.add_argument('--countries', type=str, default=None,
                        help='Comma-separated country codes')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--step-days', type=int, default=90)
    parser.add_argument('--test-days', type=int, default=90)
    parser.add_argument('--min-train-days', type=int, default=730)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-save', action='store_true')

    args = parser.parse_args()
    countries = args.countries.split(',') if args.countries else [args.country]

    for country in countries:
        rolling_window_evaluation(
            country=country,
            seed=args.seed,
            step_size_days=args.step_days,
            test_window_days=args.test_days,
            min_train_days=args.min_train_days,
            device=args.device,
            save=not args.no_save,
        )
