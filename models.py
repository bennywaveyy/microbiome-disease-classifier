"""
Models for multi-class disease prediction from gut microbiome.

Classic ML:
  - Logistic Regression (L2): a strong, interpretable linear baseline.
  - Random Forest: handles nonlinearities and gives feature importances.
  - XGBoost: gradient boosting, often the strongest classic ML on tabular data.

Deep learning:
  - A small MLP (multi-layer perceptron) implemented in PyTorch, wrapped
    in a scikit-learn-compatible interface so it plugs into the same
    cross-validation loop. Microbiome datasets are usually small (hundreds
    to a few thousand samples) — deeper architectures rarely help here,
    but a regularized MLP is a fair point of comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import os as _os
try:
    if _os.environ.get("SKIP_XGBOOST"):
        raise ImportError("XGBoost disabled by SKIP_XGBOOST env var")
    from xgboost import XGBClassifier as _XGBClassifierRaw
    XGBOOST_AVAILABLE = True

    class XGBClassifier(BaseEstimator, ClassifierMixin):
        """Thin wrapper that label-encodes y so string labels work."""

        def __init__(self, **kwargs):
            self._kwargs = kwargs

        def fit(self, X, y):
            self._le = LabelEncoder()
            y_enc = self._le.fit_transform(y)
            self.classes_ = self._le.classes_
            self._model = _XGBClassifierRaw(**self._kwargs)
            self._model.fit(X, y_enc)
            return self

        def predict(self, X):
            return self._le.inverse_transform(self._model.predict(X))

        def predict_proba(self, X):
            return self._model.predict_proba(X)
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False


# --------------------------------------------------------------------------- #
# PyTorch MLP wrapped in sklearn API
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.4):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """
    A small PyTorch MLP wrapped in a scikit-learn-compatible interface.

    Parameters
    ----------
    hidden : tuple of int, hidden layer widths.
    dropout : float, dropout probability between layers.
    lr : float, Adam learning rate.
    weight_decay : float, L2 regularization (helpful for small datasets).
    epochs : int, training epochs.
    batch_size : int.
    device : "cpu", "cuda", or None (auto).
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 32,
        device: str | None = None,
        verbose: bool = False,
        random_state: int = 42,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Feature-wise standardization. The CLR transform centers each *sample*,
        # but features (taxa) still have very different variances — standardizing
        # per-feature lets the network learn at a uniform scale.
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(np.asarray(X, dtype=np.float32)).astype(np.float32)

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_

        device = self._resolve_device()
        self._device = device

        self._model = _MLP(
            in_dim=X_scaled.shape[1],
            n_classes=len(self.classes_),
            hidden=tuple(self.hidden),
            dropout=self.dropout,
        ).to(device)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_enc, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Class-weighted loss helps when classes are imbalanced
        class_counts = np.bincount(y_enc)
        class_weights = torch.tensor(
            class_counts.sum() / (len(class_counts) * class_counts),
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self._model.train()
        for epoch in range(self.epochs):
            running = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * xb.size(0)
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  epoch {epoch+1:>3}/{self.epochs}  loss={running/len(loader.dataset):.4f}")
        return self

    def _forward(self, X) -> np.ndarray:
        self._model.eval()
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            logits = self._model(torch.tensor(X_scaled, dtype=torch.float32, device=self._device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_proba(self, X) -> np.ndarray:
        return self._forward(X)

    def predict(self, X) -> np.ndarray:
        probs = self.predict_proba(X)
        return self._label_encoder.inverse_transform(probs.argmax(axis=1))


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def build_models(random_state: int = 42) -> dict:
    """Return a dict {name: estimator} of all models to evaluate."""
    models = {
        "Logistic Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "MLP (PyTorch)": MLPClassifier(random_state=random_state),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.7,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=1,
            random_state=random_state,
        )
    return models
