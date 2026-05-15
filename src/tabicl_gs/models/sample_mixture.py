from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from tabicl_gs.models.tabicl import build_tabicl_regressor
from tabicl_gs.models.tabpfn import build_tabpfn_regressor
from tabicl_gs.models.xgboost_model import build_xgboost_regressor


def _as_float_vector(values: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _compute_gate_targets(y: np.ndarray, expert_pred: np.ndarray, prior_pred: np.ndarray) -> np.ndarray:
    y = _as_float_vector(y)
    expert_pred = _as_float_vector(expert_pred)
    prior_pred = _as_float_vector(prior_pred)
    if not (y.shape == expert_pred.shape == prior_pred.shape):
        raise ValueError(
            f"Target/prediction shape mismatch: y={y.shape}, expert={expert_pred.shape}, prior={prior_pred.shape}"
        )

    gap = expert_pred - prior_pred
    gate = np.full_like(y, 0.5, dtype=np.float32)
    stable = np.abs(gap) > 1e-6
    gate[stable] = (y[stable] - prior_pred[stable]) / gap[stable]

    if np.any(~stable):
        unstable = ~stable
        expert_err = np.abs(y[unstable] - expert_pred[unstable])
        prior_err = np.abs(y[unstable] - prior_pred[unstable])
        gate[unstable] = (expert_err <= prior_err).astype(np.float32)

    return np.clip(gate, 0.0, 1.0).astype(np.float32)


@dataclass
class SampleWiseMixtureConfig:
    hidden_dim: int = 32
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    batch_size: int | None = None
    device: str = "cpu"
    random_state: int = 42


class SampleWiseMixtureRegressor:
    def __init__(
        self,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        batch_size: int | None = None,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        self.config = SampleWiseMixtureConfig(
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            lr=float(lr),
            weight_decay=float(weight_decay),
            max_epochs=int(max_epochs),
            batch_size=None if batch_size is None else int(batch_size),
            device=str(device),
            random_state=int(random_state),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expert_pred: np.ndarray,
        prior_pred: np.ndarray,
    ) -> "SampleWiseMixtureRegressor":
        X = np.asarray(X, dtype=np.float32)
        y = _as_float_vector(y)
        expert_pred = _as_float_vector(expert_pred)
        prior_pred = _as_float_vector(prior_pred)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X).astype(np.float32)
        gate_targets = _compute_gate_targets(y, expert_pred, prior_pred)

        hidden_layers: tuple[int, ...] = () if self.config.hidden_dim <= 0 else (self.config.hidden_dim,)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            # The mixture gate is trained on per-fold samples only, so a full-batch
            # optimizer is much stabler than Adam in this small-data regime.
            solver="lbfgs",
            alpha=max(self.config.weight_decay, 1e-8),
            learning_rate_init=self.config.lr,
            max_iter=self.config.max_epochs,
            batch_size="auto" if self.config.batch_size is None else self.config.batch_size,
            random_state=self.config.random_state,
        )
        self.model_.fit(X_scaled, gate_targets)
        return self

    def predict_weights(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("SampleWiseMixtureRegressor must be fit before predict.")
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        weights = self.model_.predict(X_scaled).astype(np.float32)
        return np.clip(weights, 0.0, 1.0).astype(np.float32)

    def predict(self, X: np.ndarray, expert_pred: np.ndarray, prior_pred: np.ndarray) -> np.ndarray:
        weights = self.predict_weights(X)
        expert_pred = _as_float_vector(expert_pred)
        prior_pred = _as_float_vector(prior_pred)
        return (weights * expert_pred + (1.0 - weights) * prior_pred).astype(np.float32)


def _build_expert_regressor(backend: str, config: dict[str, Any]):
    backend = backend.lower()
    if backend == "tabicl":
        return build_tabicl_regressor(**config)
    if backend == "tabpfn":
        return build_tabpfn_regressor(**config)
    if backend == "xgboost":
        return build_xgboost_regressor(**config)
    raise ValueError(f"Unsupported sample_mixture expert backend: {backend}")


def _resolve_expert_device(model, backend: str, fallback: str) -> str:
    backend = backend.lower()
    if backend == "tabicl":
        return str(getattr(model, "device_", fallback))
    if backend == "tabpfn":
        devices = getattr(model, "devices_", ())
        return str(devices[0]) if devices else fallback
    return fallback


class SampleWiseMixtureStage2Regressor:
    def __init__(
        self,
        expert_backend: str = "tabicl",
        expert_config: dict[str, Any] | None = None,
        use_prior_prediction: bool = True,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        batch_size: int | None = None,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        if expert_backend.lower() == "sample_mixture":
            raise ValueError("sample_mixture expert_backend cannot recursively be sample_mixture.")
        self.expert_backend = str(expert_backend)
        self.expert_config = dict(expert_config or {})
        self.use_prior_prediction = bool(use_prior_prediction)
        self.random_state = int(random_state)
        self.device_hint = str(device)
        self.mixture_ = SampleWiseMixtureRegressor(
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
        )

    def _split_inputs(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        if not self.use_prior_prediction:
            return X, np.zeros(X.shape[0], dtype=np.float32)
        if X.shape[1] < 2:
            raise ValueError("sample_mixture with use_prior_prediction=True expects at least two input columns.")
        return X[:, :-1].astype(np.float32), X[:, -1].astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SampleWiseMixtureStage2Regressor":
        X_expert, prior_pred = self._split_inputs(X)
        y = _as_float_vector(y)
        expert_runtime_config = dict(self.expert_config)
        expert_runtime_config["random_state"] = self.random_state
        self.expert_model_ = _build_expert_regressor(self.expert_backend, expert_runtime_config)
        self.expert_model_.fit(X_expert, y)
        expert_train_pred = _as_float_vector(self.expert_model_.predict(X_expert))
        self.mixture_.fit(X_expert, y, expert_train_pred, prior_pred)
        self.device_ = _resolve_expert_device(
            self.expert_model_,
            backend=self.expert_backend,
            fallback=self.expert_config.get("device", self.device_hint),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "expert_model_"):
            raise RuntimeError("SampleWiseMixtureStage2Regressor must be fit before predict.")
        X_expert, prior_pred = self._split_inputs(X)
        expert_pred = _as_float_vector(self.expert_model_.predict(X_expert))
        return self.mixture_.predict(X_expert, expert_pred, prior_pred)
