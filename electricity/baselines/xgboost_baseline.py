"""
XGBoost baseline for electricity price forecasting.

Standard gradient-boosted trees baseline. Uses flattened temporal windows
(timestep x n_features -> single feature vector) matching the input
representation used by CaRS.
"""

import numpy as np
from typing import Dict, Optional


class XGBoostBaseline:
    """
    XGBoost regression baseline.

    Flattens the temporal window into a single feature vector for tree-based
    prediction. Standard ML benchmark for time series forecasting.
    """

    def __init__(
        self,
        max_depth: int = 6,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42
    ):
        self.params = {
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.model_ = None
        self.n_features_in_ = None

    def _flatten_windows(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten temporal windows for tree-based models.

        Args:
            X: [timestep, batch, features] (DS3M format)
               or [batch, timestep, features]
               or [batch, features] (already flat)

        Returns:
            [batch, timestep * features]
        """
        if X.ndim == 2:
            return X
        if X.ndim == 3:
            # Detect DS3M format: [timestep, batch, features] where timestep < batch
            if X.shape[0] < X.shape[1]:
                # DS3M format -> transpose to [batch, timestep, features]
                X = np.transpose(X, (1, 0, 2))
            return X.reshape(X.shape[0], -1)
        raise ValueError(f"Unexpected input shape: {X.shape}")

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None
    ) -> 'XGBoostBaseline':
        """
        Fit XGBoost model.

        Args:
            X_train: Training features [timestep, batch, features] or [batch, features]
            Y_train: Training target [timestep, batch, 1] or [batch]
            X_val: Validation features (optional, for early stopping)
            Y_val: Validation target (optional)

        Returns:
            self
        """
        import xgboost as xgb

        X_flat = self._flatten_windows(np.asarray(X_train))
        Y_flat = np.asarray(Y_train).flatten()

        # If Y has temporal dimension, take last timestep (prediction target)
        if len(Y_flat) != X_flat.shape[0]:
            Y_arr = np.asarray(Y_train)
            if Y_arr.ndim == 3:
                Y_flat = Y_arr[-1, :, 0]  # Last timestep
            elif Y_arr.ndim == 2:
                Y_flat = Y_arr[-1, :]

        self.n_features_in_ = X_flat.shape[1]

        fit_params = {}
        if X_val is not None and Y_val is not None:
            X_val_flat = self._flatten_windows(np.asarray(X_val))
            Y_val_flat = np.asarray(Y_val).flatten()
            if len(Y_val_flat) != X_val_flat.shape[0]:
                Y_val_arr = np.asarray(Y_val)
                if Y_val_arr.ndim == 3:
                    Y_val_flat = Y_val_arr[-1, :, 0]
                elif Y_val_arr.ndim == 2:
                    Y_val_flat = Y_val_arr[-1, :]
            fit_params['eval_set'] = [(X_val_flat, Y_val_flat)]

        self.model_ = xgb.XGBRegressor(
            **self.params,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            verbosity=0
        )
        self.model_.fit(X_flat, Y_flat, **fit_params)

        return self

    def predict(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate predictions.

        Args:
            X_test: Test features [timestep, batch, features] or [batch, features]

        Returns:
            Dict with 'predictions'
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_flat = self._flatten_windows(np.asarray(X_test))
        predictions = self.model_.predict(X_flat)

        return {'predictions': predictions}

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        return self.model_.feature_importances_
