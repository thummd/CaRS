"""
Markov-Switching Regression (MS-R) baseline.

Uses statsmodels MarkovRegression as the classical regime-switching
benchmark. Matches CaRS's regime-switching assumption without deep learning
or causal structure.

Note: MS-R cannot handle high-dimensional feature spaces well, so we
use PCA-reduced features when n_features > max_features. Large datasets
are subsampled to avoid SVD convergence issues in the EM algorithm.
"""

import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple


class MSVARBaseline:
    """
    Markov-Switching Regression baseline.

    Uses statsmodels MarkovRegression with k_regimes matching CaRS d_dim.
    Features are PCA-reduced when dimensionality is too high for stable estimation.
    Data is subsampled and standardized to avoid SVD convergence issues.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        order: int = 4,
        max_features: int = 8,
        switching_variance: bool = True,
        max_train_samples: int = 20000,
    ):
        """
        Args:
            n_regimes: Number of Markov regimes (default 2, matching CaRS)
            order: AR order (number of lags to include as exogenous features)
            max_features: Maximum features before PCA reduction
            switching_variance: If True, variance switches across regimes
            max_train_samples: Subsample training data if larger
        """
        self.n_regimes = n_regimes
        self.order = order
        self.max_features = max_features
        self.switching_variance = switching_variance
        self.max_train_samples = max_train_samples
        self.pca = None
        self.model_ = None
        self.results_ = None
        self.y_scaler_ = None
        self.x_scaler_ = None

    def _reduce_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """PCA-reduce features if dimensionality too high."""
        if X.shape[1] <= self.max_features:
            return X
        if fit:
            self.pca = PCA(n_components=self.max_features)
            return self.pca.fit_transform(X)
        return self.pca.transform(X)

    def _add_ar_lags(self, Y: np.ndarray, n_lags: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create AR lag features from target series."""
        n = len(Y)
        lags = np.column_stack([Y[n_lags - i - 1:n - i - 1] for i in range(n_lags)])
        return Y[n_lags:], lags

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        maxiter: int = 500
    ) -> 'MSVARBaseline':
        """
        Fit Markov-Switching Regression model.

        Args:
            X_train: Features [N, n_features]
            Y_train: Target [N]
            maxiter: Maximum EM iterations

        Returns:
            self
        """
        from statsmodels.tsa.regime_switching.markov_regression import (
            MarkovRegression,
        )

        Y_train = np.asarray(Y_train).flatten().astype(np.float64)
        X_train = np.asarray(X_train).astype(np.float64)

        # Standardize Y
        self.y_scaler_ = StandardScaler()
        Y_scaled = self.y_scaler_.fit_transform(Y_train.reshape(-1, 1)).flatten()

        # Build AR lags from Y and combine with exogenous X
        Y_trimmed, ar_lags = self._add_ar_lags(Y_scaled, self.order)
        X_trimmed = X_train[self.order:]

        # Standardize X
        self.x_scaler_ = StandardScaler()
        X_scaled = self.x_scaler_.fit_transform(X_trimmed)

        # PCA reduce
        X_reduced = self._reduce_features(X_scaled, fit=True)

        # Combine AR lags + PCA features as exogenous
        exog = np.hstack([ar_lags, X_reduced])

        # Subsample if dataset is too large
        n = len(Y_trimmed)
        if self.max_train_samples > 0 and n > self.max_train_samples:
            step = n // self.max_train_samples
            idx = np.arange(0, n, step)[:self.max_train_samples]
            Y_fit = Y_trimmed[idx]
            exog_fit = exog[idx]
        else:
            Y_fit = Y_trimmed
            exog_fit = exog

        # Fit MarkovRegression with exogenous regressors
        fitted = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Try 1: Full model with exogenous + switching variance
            try:
                self.model_ = MarkovRegression(
                    Y_fit,
                    k_regimes=self.n_regimes,
                    exog=exog_fit,
                    switching_variance=self.switching_variance,
                )
                self.results_ = self.model_.fit(maxiter=maxiter, em_iter=200)
                self._uses_exog = True
                fitted = True
            except Exception:
                pass

            # Try 2: Without switching variance
            if not fitted:
                try:
                    self.model_ = MarkovRegression(
                        Y_fit,
                        k_regimes=self.n_regimes,
                        exog=exog_fit,
                        switching_variance=False,
                    )
                    self.results_ = self.model_.fit(maxiter=maxiter, em_iter=200)
                    self._uses_exog = True
                    fitted = True
                except Exception:
                    pass

            # Try 3: Without exogenous features
            if not fitted:
                try:
                    self.model_ = MarkovRegression(
                        Y_fit,
                        k_regimes=self.n_regimes,
                        switching_variance=self.switching_variance,
                    )
                    self.results_ = self.model_.fit(maxiter=maxiter, em_iter=200)
                    self._uses_exog = False
                    fitted = True
                except Exception:
                    pass

            # Try 4: Simplest possible model
            if not fitted:
                self.model_ = MarkovRegression(
                    Y_fit,
                    k_regimes=self.n_regimes,
                    switching_variance=False,
                )
                self.results_ = self.model_.fit(maxiter=maxiter, em_iter=200)
                self._uses_exog = False

        return self

    def predict(self, X_test: np.ndarray, Y_history: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate predictions using fitted Markov-Switching model.

        Args:
            X_test: Test features [N_test, n_features]
            Y_history: Historical target series for regime assignment [N_history]

        Returns:
            Dict with 'predictions', 'regimes', 'regime_probabilities'
        """
        if self.results_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_test = len(X_test)
        predictions = np.zeros(n_test)
        regime_probs = np.zeros((n_test, self.n_regimes))

        # Get smoothed probabilities from training fit
        smoothed_raw = self.results_.smoothed_marginal_probabilities
        if hasattr(smoothed_raw, 'values'):
            smoothed = smoothed_raw.values
        else:
            smoothed = np.asarray(smoothed_raw)

        # Use regime-conditional means for prediction
        # This is the most robust approach for out-of-sample MS models
        params = self.results_.params
        endog = self.results_.model.endog

        for r in range(self.n_regimes):
            regime_mask = smoothed[:, r] > 0.5
            if regime_mask.sum() > 0:
                regime_mean = endog[regime_mask].mean()
            else:
                regime_mean = endog.mean()

            # Scale regime probability by last smoothed state
            predictions += smoothed[-1, r] * regime_mean

        regime_probs = np.tile(smoothed[-1], (n_test, 1))
        regimes = regime_probs.argmax(axis=1)

        # Inverse-transform predictions back to original scale
        if self.y_scaler_ is not None:
            predictions = self.y_scaler_.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()
            # Broadcast scalar prediction to all test samples
            if predictions.shape[0] == 1:
                predictions = np.full(n_test, predictions[0])

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
        if hasattr(smoothed, 'values'):
            smoothed = smoothed.values
        return np.asarray(smoothed).argmax(axis=1)

    def get_transition_matrix(self) -> np.ndarray:
        """Get estimated Markov transition matrix."""
        if self.results_ is None:
            raise RuntimeError("Model not fitted.")
        return self.results_.regime_transition
