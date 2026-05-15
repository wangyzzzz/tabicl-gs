from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VISparsePriorConfig:
    max_epochs: int = 1500
    lr: float = 0.02
    prior_scale: float = 1.0
    noise_scale: float = 1.0
    guide_init_scale: float = 0.1
    early_stopping_patience: int = 60
    tol: float = 1e-5
    scale_prior_by_n_features: bool = True
    use_local_shrinkage: bool = False
    calibration_enabled: bool = True
    min_train_corr_for_prior: float = 0.05
    max_calibration_z: float = 3.0
    calibration_ridge: float = 1e-6
    random_state: int = 42


class VISparsePriorRegressor:
    """JAX + NumPyro SVI sparse linear prior with an sklearn-like API."""

    def __init__(
        self,
        max_epochs: int = 1500,
        lr: float = 0.02,
        prior_scale: float = 1.0,
        noise_scale: float = 1.0,
        guide_init_scale: float = 0.1,
        early_stopping_patience: int = 60,
        tol: float = 1e-5,
        scale_prior_by_n_features: bool = True,
        use_local_shrinkage: bool = False,
        calibration_enabled: bool = True,
        min_train_corr_for_prior: float = 0.05,
        max_calibration_z: float = 3.0,
        calibration_ridge: float = 1e-6,
        device: str = "cpu",
        random_state: int = 42,
        **_: Any,
    ) -> None:
        # `device` is accepted for compatibility with existing model configs; JAX backend chooses device itself.
        self.config = VISparsePriorConfig(
            max_epochs=int(max_epochs),
            lr=float(lr),
            prior_scale=float(prior_scale),
            noise_scale=float(noise_scale),
            guide_init_scale=float(guide_init_scale),
            early_stopping_patience=int(early_stopping_patience),
            tol=float(tol),
            scale_prior_by_n_features=bool(scale_prior_by_n_features),
            use_local_shrinkage=bool(use_local_shrinkage),
            calibration_enabled=bool(calibration_enabled),
            min_train_corr_for_prior=float(min_train_corr_for_prior),
            max_calibration_z=float(max_calibration_z),
            calibration_ridge=float(calibration_ridge),
            random_state=int(random_state),
        )
        self.device_ = "jax"

    def _standardize_X(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if fit:
            self.x_mean_ = np.nanmean(X, axis=0).astype(np.float32)
            self.x_mean_ = np.where(np.isfinite(self.x_mean_), self.x_mean_, 0.0).astype(np.float32)
            X_filled = np.where(np.isfinite(X), X, self.x_mean_.reshape(1, -1)).astype(np.float32)
            self.x_scale_ = X_filled.std(axis=0).astype(np.float32)
            self.x_scale_ = np.where(self.x_scale_ <= 1e-6, 1.0, self.x_scale_).astype(np.float32)
        else:
            X_filled = np.where(np.isfinite(X), X, self.x_mean_.reshape(1, -1)).astype(np.float32)
        return ((X_filled - self.x_mean_.reshape(1, -1)) / self.x_scale_.reshape(1, -1)).astype(np.float32)

    def _effective_prior_scale(self, n_features: int) -> float:
        prior_scale = max(float(self.config.prior_scale), 1e-8)
        if not self.config.scale_prior_by_n_features:
            return prior_scale
        return float(prior_scale / np.sqrt(max(int(n_features), 1)))

    def _fit_prediction_calibration(self, raw_train_pred: np.ndarray, y: np.ndarray) -> None:
        raw = np.asarray(raw_train_pred, dtype=np.float32).reshape(-1)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        mask = np.isfinite(raw) & np.isfinite(y)
        raw = raw[mask]
        y = y[mask]

        self.calibration_train_corr_ = 0.0
        self.calibration_slope_ = 0.0
        self.calibration_intercept_ = float(getattr(self, "y_mean_", np.mean(y) if y.size else 0.0))
        if raw.size < 3 or not self.config.calibration_enabled:
            self.calibration_slope_ = 1.0 if not self.config.calibration_enabled else 0.0
            self.calibration_intercept_ = 0.0 if not self.config.calibration_enabled else self.calibration_intercept_
            return

        raw_mean = float(np.mean(raw))
        y_mean = float(np.mean(y))
        raw_centered = raw - raw_mean
        y_centered = y - y_mean
        raw_var = float(np.mean(raw_centered * raw_centered))
        y_var = float(np.mean(y_centered * y_centered))
        if raw_var <= 1e-12 or y_var <= 1e-12:
            return

        cov = float(np.mean(raw_centered * y_centered))
        corr = cov / float(np.sqrt(raw_var * y_var))
        self.calibration_train_corr_ = float(corr)
        if (not np.isfinite(corr)) or corr < float(self.config.min_train_corr_for_prior):
            return

        ridge = max(float(self.config.calibration_ridge), 0.0)
        slope = cov / (raw_var + ridge)
        if not np.isfinite(slope) or slope <= 0.0:
            return
        self.calibration_slope_ = float(slope)
        self.calibration_intercept_ = float(y_mean - slope * raw_mean)

    def _calibrate_raw_predictions(self, raw_pred: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_pred, dtype=np.float32)
        pred = self.calibration_intercept_ + self.calibration_slope_ * raw
        max_z = float(self.config.max_calibration_z)
        if max_z > 0.0:
            center = float(getattr(self, "y_mean_", 0.0))
            scale = max(float(getattr(self, "y_scale_", 1.0)), 1e-6)
            pred = np.clip(pred, center - max_z * scale, center + max_z * scale)
        return np.asarray(pred, dtype=np.float32)

    @staticmethod
    def _ensure_numpyro_jax_compat() -> None:
        # NumPyro 0.20.1 still references `xla_pmap_p`; JAX 0.10.0 no longer exports it.
        # A minimal alias keeps import working until we pin a fully compatible version set.
        import jax.extend.core.primitives as primitives

        if not hasattr(primitives, "xla_pmap_p") and hasattr(primitives, "jit_p"):
            primitives.xla_pmap_p = primitives.jit_p

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VISparsePriorRegressor":
        import jax
        import jax.numpy as jnp
        self._ensure_numpyro_jax_compat()
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import SVI, TraceMeanField_ELBO
        from numpyro.infer.autoguide import AutoNormal
        from numpyro.optim import Adam

        X_std = self._standardize_X(X, fit=True)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.y_mean_ = float(np.mean(y))
        self.y_scale_ = float(np.std(y))
        if self.y_scale_ <= 1e-6:
            self.y_scale_ = 1.0
        y_std = ((y - self.y_mean_) / self.y_scale_).astype(np.float32)

        X_j = jnp.asarray(X_std)
        y_j = jnp.asarray(y_std)
        n_features = int(X_std.shape[1])
        effective_prior_scale = self._effective_prior_scale(n_features)

        def model(X_data: jnp.ndarray, y_data: jnp.ndarray | None = None) -> None:
            noise = numpyro.sample("noise", dist.LogNormal(jnp.log(self.config.noise_scale), 0.25))
            if self.config.use_local_shrinkage:
                tau = numpyro.sample(
                    "tau",
                    dist.LogNormal(jnp.log(effective_prior_scale), 0.5).expand([n_features]).to_event(1),
                )
            else:
                tau = numpyro.sample("tau", dist.LogNormal(jnp.log(effective_prior_scale), 0.5))
            beta = numpyro.sample("beta", dist.Normal(jnp.zeros(n_features), tau).to_event(1))
            bias = numpyro.sample("bias", dist.Normal(0.0, 1.0))
            mean = jnp.dot(X_data, beta) + bias
            numpyro.sample("obs", dist.Normal(mean, noise), obs=y_data)

        guide = AutoNormal(model, init_scale=self.config.guide_init_scale)
        optimizer = Adam(self.config.lr)
        svi = SVI(model, guide, optimizer, TraceMeanField_ELBO())
        rng_key = jax.random.PRNGKey(self.config.random_state)
        svi_state = svi.init(rng_key, X_j, y_j)

        best_loss = float("inf")
        best_params = None
        stale_epochs = 0
        for _epoch in range(self.config.max_epochs):
            svi_state, loss = svi.update(svi_state, X_j, y_j)
            loss_value = float(loss)
            if loss_value + self.config.tol < best_loss:
                best_loss = loss_value
                best_params = jax.tree.map(lambda x: np.array(x), svi.get_params(svi_state))
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.early_stopping_patience:
                    break

        if best_params is None:
            best_params = jax.tree.map(lambda x: np.array(x), svi.get_params(svi_state))

        posterior = guide.median(best_params)
        beta_std = np.asarray(posterior["beta"], dtype=np.float32).reshape(-1)
        bias_std = float(np.asarray(posterior["bias"], dtype=np.float32))
        # AutoNormal stores unconstrained loc/scale params; we recover posterior scale for beta approximately from guide params.
        scale_key = next((k for k in best_params if k.endswith("beta_auto_scale") or k.endswith("beta_scale")), None)
        if scale_key is None:
            # Fallback: inspect all keys and find one that matches beta shape.
            beta_scale = np.full_like(beta_std, self.config.guide_init_scale, dtype=np.float32)
            for key, value in best_params.items():
                arr = np.asarray(value)
                if arr.shape == beta_std.shape and "scale" in key:
                    beta_scale = np.asarray(arr, dtype=np.float32)
                    break
        else:
            beta_scale = np.asarray(best_params[scale_key], dtype=np.float32).reshape(-1)

        self.effective_prior_scale_ = float(effective_prior_scale)
        self.raw_coef_std_ = beta_std.astype(np.float32)
        self.coef_std_ = self.raw_coef_std_
        self.coef_std_sd_ = np.abs(beta_scale).astype(np.float32)
        self.intercept_std_ = float(bias_std)
        self.raw_coef_ = (self.y_scale_ * self.raw_coef_std_ / self.x_scale_).astype(np.float32)
        self.raw_coef_var_ = ((self.y_scale_ * self.coef_std_sd_ / self.x_scale_) ** 2).astype(np.float32)
        self.raw_intercept_ = float(
            self.y_mean_ + self.y_scale_ * self.intercept_std_ - np.sum(self.raw_coef_ * self.x_mean_)
        )
        raw_train_pred = self._raw_predict(X)
        self._fit_prediction_calibration(raw_train_pred, y)
        self.coef_ = (self.calibration_slope_ * self.raw_coef_).astype(np.float32)
        self.coef_var_ = ((self.calibration_slope_**2) * self.raw_coef_var_).astype(np.float32)
        self.intercept_ = float(self.calibration_intercept_ + self.calibration_slope_ * self.raw_intercept_)
        self.loss_ = float(best_loss)
        self.guide_params_ = best_params
        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "raw_coef_"):
            raise RuntimeError("VISparsePriorRegressor must be fit before raw prediction.")
        X = np.asarray(X, dtype=np.float32)
        X_filled = np.where(np.isfinite(X), X, self.x_mean_.reshape(1, -1)).astype(np.float32)
        return (X_filled @ self.raw_coef_ + self.raw_intercept_).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("VISparsePriorRegressor must be fit before predict.")
        return self._calibrate_raw_predictions(self._raw_predict(X))

    def block_prior_scores(self, block_summaries: list[dict[str, Any]], method: str = "l2") -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("VISparsePriorRegressor must be fit before block_prior_scores.")
        coef_for_scores = getattr(self, "raw_coef_", self.coef_)
        scores = []
        for summary in block_summaries:
            indices = summary.get("snp_indices")
            if indices is None:
                raise ValueError("block_summaries must include snp_indices for VI block prior aggregation.")
            values = coef_for_scores[np.asarray(indices, dtype=int)]
            if values.size == 0:
                scores.append(0.0)
            elif method == "l1":
                scores.append(float(np.sum(np.abs(values))))
            elif method == "mean_abs":
                scores.append(float(np.mean(np.abs(values))))
            elif method == "l2":
                scores.append(float(np.sqrt(np.sum(values * values))))
            else:
                raise ValueError(f"Unsupported block score method: {method}")
        scores_arr = np.asarray(scores, dtype=np.float32)
        if scores_arr.size == 0:
            return scores_arr
        scale = float(scores_arr.std())
        if scale <= 1e-8:
            return np.zeros_like(scores_arr, dtype=np.float32)
        return ((scores_arr - float(scores_arr.mean())) / scale).astype(np.float32)
