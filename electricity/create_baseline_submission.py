"""
Baseline Linear Regression Submission for QRT Electricity Challenge.

This follows the exact benchmark approach to verify our pipeline.
Expected: ~15.9% Spearman on test (27.9% on train).

Key insight: Each sample is INDEPENDENT (cross-sectional, not time series).
ID and DAY_ID are used as features!
"""

import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR, SUBMISSIONS_DIR
# Paths
DATA_DIR = DATA_DIR / "qrt"
OUTPUT_DIR = SUBMISSIONS_DIR

def main():
    print("=" * 60)
    print("Baseline Linear Regression Submission")
    print("=" * 60)

    # Load data (note: actual filenames have different suffixes)
    X_train = pd.read_csv(DATA_DIR / "X_train_NHkHMNU.csv")
    Y_train = pd.read_csv(DATA_DIR / "y_train_ZAN5mwg.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test_final.csv")

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape}")

    # Prepare features (exactly like benchmark)
    # Drop COUNTRY (categorical), keep ID and DAY_ID as features!
    X_train_clean = X_train.drop(['COUNTRY'], axis=1).fillna(0)
    X_test_clean = X_test.drop(['COUNTRY'], axis=1).fillna(0)

    # Get target
    Y_train_clean = Y_train['TARGET']

    print(f"\nFeature columns: {list(X_train_clean.columns)}")
    print(f"Number of features: {len(X_train_clean.columns)}")

    # Train linear regression
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_clean, Y_train_clean)

    # Evaluate on training data
    train_pred = lr.predict(X_train_clean)
    train_spearman, _ = spearmanr(Y_train_clean, train_pred)
    print(f"\nTraining Spearman: {train_spearman:.4f} ({train_spearman*100:.2f}%)")

    # Generate test predictions
    test_pred = lr.predict(X_test_clean)

    print(f"\nPrediction statistics:")
    print(f"  Train TARGET - Mean: {Y_train_clean.mean():.4f}, Std: {Y_train_clean.std():.4f}")
    print(f"  Train pred   - Mean: {train_pred.mean():.4f}, Std: {train_pred.std():.4f}")
    print(f"  Test pred    - Mean: {test_pred.mean():.4f}, Std: {test_pred.std():.4f}")

    # Check for bimodal distribution (our previous failure)
    print(f"\nPrediction distribution check:")
    print(f"  Min: {test_pred.min():.4f}")
    print(f"  25%: {np.percentile(test_pred, 25):.4f}")
    print(f"  50%: {np.percentile(test_pred, 50):.4f}")
    print(f"  75%: {np.percentile(test_pred, 75):.4f}")
    print(f"  Max: {test_pred.max():.4f}")

    # Create submission
    submission = pd.DataFrame({
        'ID': X_test['ID'],
        'TARGET': test_pred
    })

    # Sort by ID
    submission = submission.sort_values('ID').reset_index(drop=True)

    # Save
    output_path = OUTPUT_DIR / "Baseline_LR_Y_test.csv"
    submission.to_csv(output_path, index=False)

    print(f"\nSubmission saved to: {output_path}")
    print(f"Submission shape: {submission.shape}")
    print(f"\nFirst 5 rows:")
    print(submission.head())

    # Cross-validation estimate
    print("\n" + "=" * 60)
    print("5-Fold Cross-Validation")
    print("=" * 60)

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_clean)):
        X_tr = X_train_clean.iloc[train_idx]
        X_val = X_train_clean.iloc[val_idx]
        y_tr = Y_train_clean.iloc[train_idx]
        y_val = Y_train_clean.iloc[val_idx]

        lr_cv = LinearRegression()
        lr_cv.fit(X_tr, y_tr)
        val_pred = lr_cv.predict(X_val)

        spearman, _ = spearmanr(y_val, val_pred)
        cv_scores.append(spearman)
        print(f"  Fold {fold+1}: Spearman = {spearman:.4f}")

    print(f"\nCV Mean: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
    print(f"Expected test: ~{np.mean(cv_scores)*100:.1f}%")

    return submission


if __name__ == "__main__":
    main()
