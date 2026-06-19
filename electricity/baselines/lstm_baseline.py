"""LSTM baseline for electricity price forecasting.

A stacked-LSTM regressor that consumes the same Tier-1 input windows as
CaRS but without regime switching or causal structure. Provides a
recurrent-NN comparison matching CaRS's parameter scale.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class _LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, timestep, features]
        out, _ = self.lstm(x)
        # Predict from the last timestep's hidden state
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMBaseline:
    """Two-layer LSTM with hidden size 32, matching CaRS's GRU encoder scale.

    Train: Adam, learning rate 1e-3, early stopping on validation MAE.
    Predicts one-hour-ahead return from a 14-step lookback window.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        max_epochs: int = 80,
        batch_size: int = 512,
        early_stopping_patience: int = 10,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.model_: Optional[_LSTMRegressor] = None

    @staticmethod
    def _to_batched(X: np.ndarray) -> np.ndarray:
        """Accept DS3M [T, B, F] or [B, T, F]; return [B, T, F]."""
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[:, None, :]
        if X.ndim == 3 and X.shape[0] < X.shape[1]:
            # DS3M [T, B, F] -> [B, T, F]
            X = np.transpose(X, (1, 0, 2))
        return X

    @staticmethod
    def _last_target(Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y)
        if Y.ndim == 3:
            return Y[-1, :, 0]
        if Y.ndim == 2:
            return Y[-1, :]
        return Y.flatten()

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> "LSTMBaseline":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        Xtr = self._to_batched(X_train)
        ytr = self._last_target(Y_train)
        Xtr_t = torch.from_numpy(Xtr.astype(np.float32)).to(self.device)
        ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(self.device)

        if X_val is not None and Y_val is not None:
            Xv = self._to_batched(X_val)
            yv = self._last_target(Y_val)
            Xv_t = torch.from_numpy(Xv.astype(np.float32)).to(self.device)
            yv_t = torch.from_numpy(yv.astype(np.float32)).to(self.device)
        else:
            Xv_t = yv_t = None

        input_dim = Xtr.shape[-1]
        self.model_ = _LSTMRegressor(
            input_dim=input_dim, hidden_dim=self.hidden_dim,
            num_layers=self.num_layers, dropout=self.dropout,
        ).to(self.device)

        opt = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.L1Loss()  # train on MAE (more robust to fat tails on returns)

        best_val = float("inf")
        bad_epochs = 0
        best_state = None
        n_train = Xtr_t.shape[0]
        for epoch in range(self.max_epochs):
            self.model_.train()
            # Random mini-batch order
            perm = torch.randperm(n_train, device=self.device)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = Xtr_t[idx]
                yb = ytr_t[idx]
                opt.zero_grad()
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                opt.step()

            if Xv_t is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_pred = self.model_(Xv_t)
                    val_loss = float(loss_fn(val_pred, yv_t).item())
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    bad_epochs = 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model_.state_dict().items()}
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.early_stopping_patience:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        self.model_.eval()
        X = self._to_batched(X_test)
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            preds = self.model_(X_t).cpu().numpy()
        return {"predictions": preds}
