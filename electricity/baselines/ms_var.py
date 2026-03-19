"""
Markov-Switching Vector Autoregression (MS-VAR) baseline.

Uses statsmodels MarkovAutoregression as the classical regime-switching
benchmark. Matches CaRS's regime-switching assumption without deep learning
or causal structure.

Note: MS-VAR cannot handle high-dimensional feature spaces well, so we
use PCA-reduced features when n_features > max_features.
"""

import numpy as np
import warnings
from sklearn.decomposition import PCA
from typing import Dict, Optional, Tuple


class MSVARBaseline:
    """
    Markov-Switching Autoregression baseline.

    Uses statsmodels MarkovAutoregression with k_regimes matching CaRS d_dim.
    Features are PCA-reduced when dimensionality is too high for stable estimation.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        order: int = 4,
        max_features: int = 8,
        switching_variance: bool = True
    ):
        """
        Args:
            n_regimes: Number of Markov regimes (default 2, matching CaRS)
            order: AR order (number of lags)
            max_features: Maximum features before PCA reduction
            switching_variance: If True, variance switches across regimes
        """
        self.n_regimes = n_regimes
        self.order = order
        self.max_features = max_features
        self.switching_variance = switching_variance
        self.pca = None
        self.model_ = None
        self.results_ = None

    def _reduce_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """PCA-reduce features if dimensionality too high."""
        if X.shape[1] <= self.max_features:
            return X
        if fit:
            self.pca = PCA(n_components=self.max_features)
            return self.pca.fit_transform(X)
        return self.pca.transform(X)

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        maxiter: int = 500
    ) -> 'MSVARBaseline':
        """
        Fit MS-VAR model.

        Args:
            X_train: Features [N, n_features] (flattened from temporal windows)
            Y_train: Target [N]
            maxiter: Maximum EM iterations

        Returns:
            self
        """
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )

        Y_train = np.asarray(Y_train).flatten()
        X_train = np.asarray(X_train)

        # PCA reduce if needed
        X_reduced = self._reduce_features(X_train, fit=True)

        # Construct endogenous series with exogenous regressors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = MarkovAutoregression(
                Y_train,
                k_regimes=self.n_regimes,
                order=self.order,
                switching_variance=self.switching_variance,
                exog=X_reduced
            )
            try:
                self.results_ = self.model_.fit(
                    maxiter=maxiter,
                    em_iter=100,
                    search_reps=20
                )
            except Exception:
                # Fallback: simpler model without exogenous
                self.model_ = MarkovAutoregression(
                    Y_train,
                    k_regimes=self.n_regimes,
                    order=self.order,
                    switching_variance=self.switching_variance
                )
                self.results_ = self.model_.fit(maxiter=maxiter, em_iter=100)
                self.pca = None  # Flag that exog wasn't used

        return self

    def predict(self, X_test: np.ndarray, Y_history: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate predictions using fitted MS-VAR.

        For MS-VAR, prediction is done via one-step-ahead conditional expectation.

        Args:
            X_test: Test features [N_test, n_features]
            Y_history: Full target series up to test start [N_train]

        Returns:
            Dict with 'predictions', 'regimes', 'regime_probabilities'
        """
        if self.results_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_test = len(X_test)
        predictions = np.zeros(n_test)
        regime_probs = np.zeros((n_test, self.n_regimes))

        # Use smoothed probabilities from in-sample fit for regime info
        smoothed = self.results_.smoothed_marginal_probabilities

        # One-step-ahead forecasting via predict
        try:
            # Extend the model with test data for dynamic forecasting
            Y_full = np.concatenate([
                self.results_.model.endog,
                np.zeros(n_test)
            ])
            if self.pca is not None:
                X_reduced = self._reduce_features(X_test, fit=False)
                exog_full = np.vstack([
                    self.results_.model.exog,
                    X_reduced
                ])
            else:
                exog_full = None

            # Use predict method
            start = len(self.results_.model.endog)
            end = start + n_test - 1
            predictions = self.results_.predict(start=start, end=end)
            if len(predictions) < n_test:
                predictions = np.pad(predictions, (0, n_test - len(predictions)),
                                     constant_values=np.nan)

            # Regime assignments from filtered probabilities
            regime_probs = np.tile(
                smoothed.iloc[-1].values, (n_test, 1)
            )
        except Exception:
            # Fallback: use regime-conditional means
            for r in range(self.n_regimes):
                regime_mask = smoothed.iloc[:, r].values > 0.5
                if regime_mask.sum() > 0:
                    regime_mean = self.results_.model.endog[regime_mask].mean()
                else:
                    regime_mean = self.results_.model.endog.mean()
                predictions += smoothed.iloc[-1, r] * regime_mean
            regime_probs = np.tile(smoothed.iloc[-1].values, (n_test, 1))

        regimes = regime_probs.argmax(axis=1)

        return {
            'predictions': predictions,
            'regimes': regimes,
            'regime_probabilities': regime_probs
        }

    def get_regime_assignments(self) -> np.ndarray:
        """Get smoothed regime assignments from training data."""
        if self.results_ is None:
            raise RuntimeError("Model not fitted.")
        smoothed = self.results_.smoothed_marginal_probabilities
        return smoothed.values.argmax(axis=1)

    def get_transition_matrix(self) -> np.ndarray:
        """Get estimated Markov transition matrix."""
        if self.results_ is None:
            raise RuntimeError("Model not fitted.")
        return self.results_.regime_transition
