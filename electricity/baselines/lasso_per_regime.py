"""
Lasso Regression per Regime baseline.

Step 1: Regime assignment via Hidden Markov Model on rolling price volatility
        (independent of CaRS to avoid circular validation).
Step 2: Fit LassoCV per regime for sparse linear prediction.

Tests whether regime-conditional sparse linear models are sufficient,
or whether the ICGNN causal structure in CaRS adds value.
"""

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from typing import Dict, Optional


class LassoPerRegimeBaseline:
    """
    Regime-conditional Lasso regression.

    Regimes are assigned independently via HMM on rolling price volatility,
    then a separate LassoCV is fitted per regime.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        volatility_window: int = 24,
        lasso_cv_folds: int = 5,
        max_iter: int = 5000,
        random_state: int = 42
    ):
        """
        Args:
            n_regimes: Number of regimes for HMM
            volatility_window: Rolling window size for volatility computation (hours)
            lasso_cv_folds: Number of CV folds for LassoCV
            max_iter: Maximum Lasso iterations
            random_state: Random seed
        """
        self.n_regimes = n_regimes
        self.volatility_window = volatility_window
        self.lasso_cv_folds = lasso_cv_folds
        self.max_iter = max_iter
        self.random_state = random_state

        self.hmm_ = None
        self.lassos_ = {}
        self.scalers_ = {}

    def _compute_volatility_features(self, Y: np.ndarray) -> np.ndarray:
        """Compute rolling volatility and level features for HMM."""
        Y = Y.flatten()
        n = len(Y)
        vol = np.zeros(n)
        level = np.zeros(n)

        for i in range(self.volatility_window, n):
            window = Y[i - self.volatility_window:i]
            vol[i] = np.std(window)
            level[i] = np.mean(window)

        # Fill initial values
        vol[:self.volatility_window] = vol[self.volatility_window]
        level[:self.volatility_window] = level[self.volatility_window]

        return np.column_stack([vol, level])

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray
    ) -> 'LassoPerRegimeBaseline':
        """
        Fit regime-conditional Lasso model.

        Args:
            X_train: Features [N, n_features]
            Y_train: Target [N]

        Returns:
            self
        """
        Y_train = np.asarray(Y_train).flatten()
        X_train = np.asarray(X_train)

        # Step 1: HMM regime assignment on volatility features
        vol_features = self._compute_volatility_features(Y_train)
        self.hmm_ = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=200,
            random_state=self.random_state
        )
        self.hmm_.fit(vol_features)
        regimes = self.hmm_.predict(vol_features)

        # Step 2: Fit LassoCV per regime
        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.sum() < 20:
                # Too few samples: use all data
                mask = np.ones(len(Y_train), dtype=bool)

            self.scalers_[r] = StandardScaler()
            X_r = self.scalers_[r].fit_transform(X_train[mask])

            n_folds = min(self.lasso_cv_folds, max(2, mask.sum() // 50))
            self.lassos_[r] = LassoCV(
                cv=n_folds,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            self.lassos_[r].fit(X_r, Y_train[mask])

        return self

    def predict(
        self,
        X_test: np.ndarray,
        Y_history: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict using regime-conditional Lasso.

        Args:
            X_test: Test features [N_test, n_features]
            Y_history: Historical target for regime assignment (optional).
                       If None, uses most recent training regime.

        Returns:
            Dict with 'predictions', 'regimes'
        """
        X_test = np.asarray(X_test)
        n_test = len(X_test)

        # Assign regimes
        if Y_history is not None:
            vol_features = self._compute_volatility_features(Y_history)
            regimes = self.hmm_.predict(vol_features[-n_test:])
        else:
            # Default to regime 0
            regimes = np.zeros(n_test, dtype=int)

        predictions = np.zeros(n_test)
        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.sum() > 0 and r in self.lassos_:
                X_r = self.scalers_[r].transform(X_test[mask])
                predictions[mask] = self.lassos_[r].predict(X_r)

        return {
            'predictions': predictions,
            'regimes': regimes
        }

    def get_regime_assignments(self, Y: np.ndarray) -> np.ndarray:
        """Get regime assignments for a target series."""
        vol_features = self._compute_volatility_features(Y.flatten())
        return self.hmm_.predict(vol_features)

    def get_selected_features(self) -> Dict[int, np.ndarray]:
        """Get non-zero Lasso coefficients per regime."""
        result = {}
        for r, lasso in self.lassos_.items():
            result[r] = {
                'coef': lasso.coef_,
                'nonzero_idx': np.where(np.abs(lasso.coef_) > 1e-6)[0],
                'alpha': lasso.alpha_
            }
        return result
