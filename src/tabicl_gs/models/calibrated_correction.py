from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _as_float_vector(values: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _compute_gate_targets(y: np.ndarray, y_tabicl: np.ndarray, y_bayesb: np.ndarray) -> np.ndarray:
    y = _as_float_vector(y)
    y_tabicl = _as_float_vector(y_tabicl)
    y_bayesb = _as_float_vector(y_bayesb)
    gap = y_tabicl - y_bayesb
    gate = np.full_like(y, 0.0, dtype=np.float32)
    stable = np.abs(gap) > 1e-6
    gate[stable] = (y[stable] - y_bayesb[stable]) / gap[stable]
    if np.any(~stable):
        gate[~stable] = 0.0
    return np.clip(gate, 0.0, 1.0).astype(np.float32)


@dataclass
class CalibratedCorrectionConfig:
    hidden_dim: int = 32
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    batch_size: int | None = None
    device: str = "cpu"
    random_state: int = 42
    use_dual_priors: bool = False


class CalibratedCorrectionRegressor:
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
        use_dual_priors: bool = False,
    ) -> None:
        self.config = CalibratedCorrectionConfig(
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            lr=float(lr),
            weight_decay=float(weight_decay),
            max_epochs=int(max_epochs),
            batch_size=None if batch_size is None else int(batch_size),
            device=str(device),
            random_state=int(random_state),
            use_dual_priors=bool(use_dual_priors),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_tabicl: np.ndarray,
        y_bayesb: np.ndarray,
        y_gblup: np.ndarray | None = None,
    ) -> "CalibratedCorrectionRegressor":
        X = np.asarray(X, dtype=np.float32)
        y = _as_float_vector(y)
        y_tabicl = _as_float_vector(y_tabicl)
        y_bayesb = _as_float_vector(y_bayesb)
        y_gblup_vec = None if y_gblup is None else _as_float_vector(y_gblup)

        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X).astype(np.float32)
        if self.config.use_dual_priors:
            if y_gblup_vec is None:
                raise ValueError("use_dual_priors=True requires y_gblup.")
            alpha_targets = _compute_gate_targets(y, y_bayesb, y_gblup_vec)
            self.alpha_model_ = MLPRegressor(
                hidden_layer_sizes=(() if self.config.hidden_dim <= 0 else (self.config.hidden_dim,)),
                activation="relu",
                solver="lbfgs",
                alpha=max(self.config.weight_decay, 1e-8),
                learning_rate_init=self.config.lr,
                max_iter=self.config.max_epochs,
                batch_size="auto" if self.config.batch_size is None else self.config.batch_size,
                random_state=self.config.random_state,
            )
            self.alpha_model_.fit(X_scaled, alpha_targets)
            alpha_train = np.clip(self.alpha_model_.predict(X_scaled).astype(np.float32), 0.0, 1.0)
            y_prior = alpha_train * y_bayesb + (1.0 - alpha_train) * y_gblup_vec
        else:
            y_prior = y_bayesb

        gate_targets = _compute_gate_targets(y, y_tabicl, y_prior)

        hidden_layers: tuple[int, ...] = () if self.config.hidden_dim <= 0 else (self.config.hidden_dim,)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="lbfgs",
            alpha=max(self.config.weight_decay, 1e-8),
            learning_rate_init=self.config.lr,
            max_iter=self.config.max_epochs,
            batch_size="auto" if self.config.batch_size is None else self.config.batch_size,
            random_state=self.config.random_state,
        )
        self.model_.fit(X_scaled, gate_targets)
        self.gate_train_pred_ = np.clip(self.model_.predict(X_scaled).astype(np.float32), 0.0, 1.0)
        if self.config.use_dual_priors:
            self.alpha_train_pred_ = np.clip(self.alpha_model_.predict(X_scaled).astype(np.float32), 0.0, 1.0)
        else:
            self.alpha_train_pred_ = None
        return self

    def predict_gate(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("CalibratedCorrectionRegressor must be fit before predict.")
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        gate = self.model_.predict(X_scaled).astype(np.float32)
        return np.clip(gate, 0.0, 1.0).astype(np.float32)

    def predict_alpha(self, X: np.ndarray) -> np.ndarray:
        if not self.config.use_dual_priors:
            raise RuntimeError("predict_alpha is only available when use_dual_priors=True.")
        if not hasattr(self, "alpha_model_"):
            raise RuntimeError("CalibratedCorrectionRegressor must be fit before predict_alpha.")
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        alpha = self.alpha_model_.predict(X_scaled).astype(np.float32)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    def predict(
        self,
        X: np.ndarray,
        y_tabicl: np.ndarray,
        y_bayesb: np.ndarray,
        y_gblup: np.ndarray | None = None,
    ) -> np.ndarray:
        gate = self.predict_gate(X)
        y_tabicl = _as_float_vector(y_tabicl)
        y_bayesb = _as_float_vector(y_bayesb)
        if self.config.use_dual_priors:
            if y_gblup is None:
                raise ValueError("use_dual_priors=True requires y_gblup.")
            y_gblup_vec = _as_float_vector(y_gblup)
            alpha = self.predict_alpha(X)
            y_prior = alpha * y_bayesb + (1.0 - alpha) * y_gblup_vec
        else:
            y_prior = y_bayesb
        return (y_prior + gate * (y_tabicl - y_prior)).astype(np.float32)

    def get_group_summary(self) -> dict[str, object]:
        if not hasattr(self, "gate_train_pred_"):
            raise RuntimeError("CalibratedCorrectionRegressor must be fit before get_group_summary.")

        gate_train = np.asarray(self.gate_train_pred_, dtype=np.float32).reshape(-1)
        gate_mean = float(np.mean(gate_train))
        gate_std = float(np.std(gate_train))
        gate_min = float(np.min(gate_train))
        gate_max = float(np.max(gate_train))

        if self.config.use_dual_priors:
            alpha_train = np.asarray(self.alpha_train_pred_, dtype=np.float32).reshape(-1)
            alpha_mean = float(np.mean(alpha_train))
            prior_names = ["BayesB", "GBLUP"]
            prior_weight_group = [[alpha_mean, 1.0 - alpha_mean]]
            alpha_group = [alpha_mean]
            bayes_family_group = ["BayesB"]
        else:
            prior_names = ["GBLUP"]
            prior_weight_group = [[1.0]]
            alpha_group = [0.0]
            bayes_family_group = ["GBLUP"]

        return {
            "group_mode": "sample_wise_correction",
            "assignment_mode": "mean_gate",
            "group_counts": [int(gate_train.shape[0])],
            "group_probs_mean": [1.0],
            "prior_names": prior_names,
            "prior_weight_group": prior_weight_group,
            "alpha_group": alpha_group,
            "w_group": [gate_mean],
            "bayes_family_group": bayes_family_group,
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "gate_min": gate_min,
            "gate_max": gate_max,
        }
