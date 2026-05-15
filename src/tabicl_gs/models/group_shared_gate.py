from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _as_float_vector(values: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _clip_gate_targets(y_true: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> np.ndarray:
    y_true = _as_float_vector(y_true)
    y_left = _as_float_vector(y_left)
    y_right = _as_float_vector(y_right)
    gap = y_left - y_right
    out = np.zeros_like(y_true, dtype=np.float32)
    stable = np.abs(gap) > 1e-6
    out[stable] = (y_true[stable] - y_right[stable]) / gap[stable]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _build_group_membership_from_prior(prior_scores: np.ndarray, num_groups: int) -> np.ndarray:
    prior = np.asarray(prior_scores, dtype=np.float32).reshape(-1)
    if prior.size == 0:
        raise ValueError("prior_scores cannot be empty.")
    ranks = np.argsort(np.argsort(prior))
    groups = np.floor(ranks * num_groups / max(len(prior), 1)).astype(int)
    return np.clip(groups, 0, num_groups - 1).astype(np.int64)


def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = _as_float_vector(y_true) - _as_float_vector(y_pred)
    return float(np.mean(diff * diff))


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = exp.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 1e-8, 1.0, denom)
    return (exp / denom).astype(np.float32)


def _project_to_simplex(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError("Cannot project an empty vector onto simplex.")
    sorted_vals = np.sort(vector)[::-1]
    cssv = np.cumsum(sorted_vals) - 1.0
    indices = np.arange(1, vector.size + 1, dtype=np.float64)
    cond = sorted_vals - cssv / indices > 0
    if not np.any(cond):
        return np.full(vector.shape, 1.0 / vector.size, dtype=np.float32)
    rho = int(np.nonzero(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    denom = projected.sum()
    if denom <= 1e-8:
        return np.full(vector.shape, 1.0 / vector.size, dtype=np.float32)
    return (projected / denom).astype(np.float32)


def _fit_simplex_prior_weights(y_true: np.ndarray, prior_matrix: np.ndarray) -> np.ndarray:
    y = _as_float_vector(y_true).astype(np.float64)
    X = np.asarray(prior_matrix, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("prior_matrix must be 2-dimensional.")
    n_priors = X.shape[1]
    if n_priors == 0:
        raise ValueError("prior_matrix must contain at least one prior.")
    if n_priors == 1:
        return np.ones(1, dtype=np.float32)

    mse = np.mean((X - y[:, None]) ** 2, axis=0)
    inv = 1.0 / np.maximum(mse, 1e-8)
    weights = _project_to_simplex(inv)
    spectral_norm = float(np.linalg.svd(X, compute_uv=False)[0]) if X.size > 0 else 0.0
    lipschitz = 2.0 * (spectral_norm**2) / max(X.shape[0], 1)
    step = 1.0 / max(lipschitz, 1e-6)

    for _ in range(200):
        grad = (2.0 / max(X.shape[0], 1)) * (X.T @ (X @ weights - y))
        updated = _project_to_simplex(weights - step * grad)
        if np.max(np.abs(updated - weights)) < 1e-7:
            weights = updated
            break
        weights = updated
    return weights.astype(np.float32)


def _build_prior_prediction_map(
    y_bayesb: np.ndarray,
    y_gblup: np.ndarray,
    y_bayes_candidates: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    prior_map: dict[str, np.ndarray] = {
        "BayesB": _as_float_vector(y_bayesb),
        "GBLUP": _as_float_vector(y_gblup),
    }
    if y_bayes_candidates:
        for name, values in y_bayes_candidates.items():
            key = str(name)
            if key in prior_map:
                continue
            prior_map[key] = _as_float_vector(values)
    return prior_map


def _derive_legacy_gate_fields(
    prior_names: Sequence[str],
    prior_weight_group: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    prior_names = [str(name) for name in prior_names]
    weights = np.asarray(prior_weight_group, dtype=np.float32)
    gblup_idx = prior_names.index("GBLUP") if "GBLUP" in prior_names else -1
    alpha_group = []
    family_group = []
    for row in weights:
        if gblup_idx >= 0:
            alpha_value = float(np.clip(1.0 - row[gblup_idx], 0.0, 1.0))
        else:
            alpha_value = 1.0
        non_gblup = [(name, float(row[idx])) for idx, name in enumerate(prior_names) if name != "GBLUP"]
        if not non_gblup:
            family_group.append("BayesB")
        else:
            family_group.append(max(non_gblup, key=lambda item: item[1])[0])
        alpha_group.append(alpha_value)
    return np.asarray(alpha_group, dtype=np.float32), family_group


def _legacy_summary_to_prior_weights(summary: Mapping[str, object], num_groups: int) -> tuple[list[str], np.ndarray]:
    alpha_group = np.asarray(summary.get("alpha_group", np.zeros(num_groups)), dtype=np.float32).reshape(-1)
    family_by_group = list(summary.get("bayes_family_group", ["BayesB"] * num_groups))  # type: ignore[arg-type]
    prior_names: list[str] = []
    for family in family_by_group:
        family_name = str(family)
        if family_name != "GBLUP" and family_name not in prior_names:
            prior_names.append(family_name)
    if not prior_names:
        prior_names = ["BayesB"]
    if "GBLUP" not in prior_names:
        prior_names.append("GBLUP")

    weight_rows = np.zeros((num_groups, len(prior_names)), dtype=np.float32)
    gblup_idx = prior_names.index("GBLUP")
    for gid in range(num_groups):
        alpha = float(alpha_group[min(gid, alpha_group.shape[0] - 1)]) if alpha_group.size else 0.0
        family = str(family_by_group[min(gid, len(family_by_group) - 1)]) if family_by_group else "BayesB"
        family_idx = prior_names.index(family) if family in prior_names else None
        if family_idx is not None:
            weight_rows[gid, family_idx] = alpha
        weight_rows[gid, gblup_idx] = max(0.0, 1.0 - alpha)
        row_sum = float(weight_rows[gid].sum())
        if row_sum <= 1e-8:
            weight_rows[gid, gblup_idx] = 1.0
        else:
            weight_rows[gid] /= row_sum
    return prior_names, weight_rows.astype(np.float32)


@dataclass
class GroupSharedGateConfig:
    block_input_dims: list[int]
    prior_scores: list[float]
    num_groups: int = 3
    group_mode: str = "embedding_group"
    assignment_mode: str = "nearest_centroid"
    temperature: float = 1.0
    hidden_dim: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    device: str = "cpu"
    random_state: int = 42


class GroupSharedGateRegressor:
    def __init__(
        self,
        block_input_dims: Sequence[int],
        prior_scores: Sequence[float],
        num_groups: int = 3,
        group_mode: str = "embedding_group",
        assignment_mode: str = "nearest_centroid",
        temperature: float = 1.0,
        hidden_dim: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        self.config = GroupSharedGateConfig(
            block_input_dims=[int(v) for v in block_input_dims],
            prior_scores=[float(v) for v in prior_scores],
            num_groups=int(num_groups),
            group_mode=str(group_mode),
            assignment_mode=str(assignment_mode),
            temperature=float(temperature),
            hidden_dim=int(hidden_dim),
            lr=float(lr),
            weight_decay=float(weight_decay),
            max_epochs=int(max_epochs),
            device=str(device),
            random_state=int(random_state),
        )

    def _block_slices(self) -> list[tuple[int, int]]:
        offsets = []
        start = 0
        for dim in self.config.block_input_dims:
            end = start + int(dim)
            offsets.append((start, end))
            start = end
        return offsets

    @classmethod
    def from_summary(
        cls,
        summary: Mapping[str, object],
        block_input_dims: Sequence[int] | None = None,
        prior_scores: Sequence[float] | None = None,
        random_state: int = 42,
    ) -> "GroupSharedGateRegressor":
        group_counts = list(summary.get("group_counts", []))  # type: ignore[arg-type]
        centroids = np.asarray(summary["group_centroids"], dtype=np.float32)  # type: ignore[index]
        num_groups = int(len(group_counts) if group_counts else centroids.shape[0])
        model = cls(
            block_input_dims=list(block_input_dims or [int(centroids.shape[1])]),
            prior_scores=list(prior_scores or []),
            num_groups=num_groups,
            group_mode=str(summary.get("group_mode", "embedding_group")),
            assignment_mode=str(summary.get("assignment_mode", "nearest_centroid")),
            random_state=int(random_state),
        )
        from sklearn.preprocessing import StandardScaler

        model.scaler_ = StandardScaler()
        model.scaler_.mean_ = np.asarray(summary.get("scaler_mean", np.zeros(centroids.shape[1])), dtype=np.float64)
        model.scaler_.scale_ = np.asarray(summary.get("scaler_scale", np.ones(centroids.shape[1])), dtype=np.float64)
        model.scaler_.var_ = model.scaler_.scale_ * model.scaler_.scale_
        model.scaler_.n_features_in_ = int(centroids.shape[1])
        model.group_centroids_ = centroids.astype(np.float32)
        model.group_counts_ = np.asarray(group_counts, dtype=np.int64)
        model.group_probs_mean_ = np.asarray(summary.get("group_probs_mean", np.zeros(num_groups)), dtype=np.float32)
        model.w_group_ = np.asarray(summary["w_group"], dtype=np.float32)  # type: ignore[index]
        if "prior_names" in summary and "prior_weight_group" in summary:
            model.prior_names_ = [str(name) for name in summary.get("prior_names", [])]  # type: ignore[arg-type]
            model.prior_weight_group_ = np.asarray(summary["prior_weight_group"], dtype=np.float32)  # type: ignore[index]
            alpha_group, bayes_family_group = _derive_legacy_gate_fields(model.prior_names_, model.prior_weight_group_)
            model.alpha_group_ = alpha_group
            model.bayes_family_group_ = bayes_family_group
        else:
            prior_names, prior_weight_group = _legacy_summary_to_prior_weights(summary, num_groups)
            model.prior_names_ = prior_names
            model.prior_weight_group_ = prior_weight_group
            model.alpha_group_ = np.asarray(summary["alpha_group"], dtype=np.float32)  # type: ignore[index]
            model.bayes_family_group_ = list(summary.get("bayes_family_group", ["BayesB"] * num_groups))  # type: ignore[arg-type]
        block_prior_counts = summary.get("block_prior_group_counts")
        model.block_prior_group_counts_ = None if block_prior_counts is None else np.asarray(block_prior_counts, dtype=np.int64)
        return model

    def build_group_features(self, X_core: np.ndarray) -> np.ndarray:
        block_vectors = []
        for start, end in self._block_slices():
            block_vectors.append(X_core[:, start:end].mean(axis=1, keepdims=True))
        block_matrix = np.concatenate(block_vectors, axis=1).astype(np.float32)

        if self.config.group_mode == "embedding_group":
            self.block_prior_group_ids_ = None
            return block_matrix.astype(np.float32)

        if self.config.group_mode != "prior_guided_group":
            raise ValueError(f"Unsupported group_mode: {self.config.group_mode}")

        prior_scores = np.asarray(self.config.prior_scores, dtype=np.float32)
        group_ids = _build_group_membership_from_prior(prior_scores, self.config.num_groups)
        self.block_prior_group_ids_ = group_ids.astype(np.int64)
        summaries = []
        for gid in range(self.config.num_groups):
            mask = group_ids == gid
            if not np.any(mask):
                summaries.append(np.zeros((X_core.shape[0], 1), dtype=np.float32))
            else:
                summaries.append(block_matrix[:, mask].mean(axis=1, keepdims=True).astype(np.float32))
        return np.concatenate(summaries, axis=1).astype(np.float32)

    def _build_group_features(self, X_core: np.ndarray) -> np.ndarray:
        return self.build_group_features(X_core)

    def _fit_group_statistics(
        self,
        X_group: np.ndarray,
        y: np.ndarray,
        y_tabicl: np.ndarray,
        y_bayesb: np.ndarray,
        y_gblup: np.ndarray,
        y_bayes_candidates: Mapping[str, np.ndarray] | None = None,
    ) -> "GroupSharedGateRegressor":
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(np.asarray(X_group, dtype=np.float32)).astype(np.float32)

        self.group_model_ = KMeans(
            n_clusters=self.config.num_groups,
            n_init=10,
            random_state=self.config.random_state,
        )
        self.group_assignments_train_ = self.group_model_.fit_predict(X_scaled).astype(np.int64)
        self.group_probs_train_ = np.eye(self.config.num_groups, dtype=np.float32)[self.group_assignments_train_]
        self.group_counts_ = np.bincount(self.group_assignments_train_, minlength=self.config.num_groups).astype(np.int64)
        self.group_probs_mean_ = self.group_probs_train_.mean(axis=0).astype(np.float32)
        self.group_centroids_ = np.asarray(self.group_model_.cluster_centers_, dtype=np.float32)

        if getattr(self, "block_prior_group_ids_", None) is not None:
            self.block_prior_group_counts_ = np.bincount(
                self.block_prior_group_ids_, minlength=self.config.num_groups
            ).astype(np.int64)
        else:
            self.block_prior_group_counts_ = None

        prior_map = _build_prior_prediction_map(y_bayesb, y_gblup, y_bayes_candidates)
        for name, values in prior_map.items():
            if values.shape[0] != y.shape[0]:
                raise ValueError(f"Prior candidate {name} length {values.shape[0]} does not match y length {y.shape[0]}.")

        self.prior_names_ = list(prior_map.keys())
        global_prior_matrix = np.column_stack([prior_map[name] for name in self.prior_names_]).astype(np.float32)
        global_prior_weights = _fit_simplex_prior_weights(y, global_prior_matrix)
        prior_weight_rows = []
        for gid in range(self.config.num_groups):
            mask = self.group_assignments_train_ == gid
            if not np.any(mask):
                prior_weight_rows.append(global_prior_weights)
                continue
            group_prior_matrix = np.column_stack([prior_map[name][mask] for name in self.prior_names_]).astype(np.float32)
            prior_weight_rows.append(_fit_simplex_prior_weights(y[mask], group_prior_matrix))
        self.prior_weight_group_ = np.asarray(prior_weight_rows, dtype=np.float32)
        self.alpha_group_, self.bayes_family_group_ = _derive_legacy_gate_fields(
            self.prior_names_,
            self.prior_weight_group_,
        )

        prior_weights_train = self.prior_weight_group_[self.group_assignments_train_]
        y_prior = np.sum(prior_weights_train * global_prior_matrix, axis=1).astype(np.float32)

        w_targets = _clip_gate_targets(y, y_tabicl, y_prior)
        w_values = []
        global_w = float(np.mean(w_targets))
        for gid in range(self.config.num_groups):
            mask = self.group_assignments_train_ == gid
            w_values.append(float(np.mean(w_targets[mask])) if np.any(mask) else global_w)
        self.w_group_ = np.clip(np.asarray(w_values, dtype=np.float32), 0.0, 1.0)
        return self

    def fit_from_group_features(
        self,
        X_group: np.ndarray,
        y: np.ndarray,
        y_tabicl: np.ndarray,
        y_bayesb: np.ndarray,
        y_gblup: np.ndarray,
        y_bayes_candidates: Mapping[str, np.ndarray] | None = None,
    ) -> "GroupSharedGateRegressor":
        y = _as_float_vector(y)
        y_tabicl = _as_float_vector(y_tabicl)
        y_bayesb = _as_float_vector(y_bayesb)
        y_gblup = _as_float_vector(y_gblup)
        return self._fit_group_statistics(X_group, y, y_tabicl, y_bayesb, y_gblup, y_bayes_candidates)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_tabicl: np.ndarray,
        y_bayesb: np.ndarray,
        y_gblup: np.ndarray,
        y_bayes_candidates: Mapping[str, np.ndarray] | None = None,
    ) -> "GroupSharedGateRegressor":
        X_core = np.asarray(X, dtype=np.float32)
        y = _as_float_vector(y)
        y_tabicl = _as_float_vector(y_tabicl)
        y_bayesb = _as_float_vector(y_bayesb)
        y_gblup = _as_float_vector(y_gblup)

        X_group = self._build_group_features(X_core)
        return self._fit_group_statistics(X_group, y, y_tabicl, y_bayesb, y_gblup, y_bayes_candidates)

    def _predict_group_probs_from_features(self, X_group: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(np.asarray(X_group, dtype=np.float32)).astype(np.float32)
        centroids = np.asarray(self.group_centroids_, dtype=np.float32)
        diff = X_scaled[:, None, :] - centroids[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2).astype(np.float32)
        if self.config.assignment_mode == "nearest_centroid":
            assignments = np.argmin(dist_sq, axis=1).astype(np.int64)
            return np.eye(self.config.num_groups, dtype=np.float32)[assignments]
        if self.config.assignment_mode != "softmax_distance":
            raise ValueError(f"Unsupported assignment_mode: {self.config.assignment_mode}")
        temperature = max(float(self.config.temperature), 1e-6)
        return _softmax(-dist_sq / temperature)

    def _predict_group_probs(self, X_core: np.ndarray) -> np.ndarray:
        X_group = self._build_group_features(X_core)
        return self._predict_group_probs_from_features(X_group)

    def predict_group_assignments_from_group_features(self, X_group: np.ndarray) -> np.ndarray:
        probs = self._predict_group_probs_from_features(X_group)
        return np.argmax(probs, axis=1).astype(np.int64)

    def predict_group_assignments(self, X: np.ndarray) -> np.ndarray:
        X_core = np.asarray(X, dtype=np.float32)
        return self.predict_group_assignments_from_group_features(self._build_group_features(X_core))

    def predict_from_group_features(
        self,
        X_group: np.ndarray,
        y_tabicl: np.ndarray,
        y_bayesb: np.ndarray,
        y_gblup: np.ndarray,
        y_bayes_candidates: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        y_tabicl = _as_float_vector(y_tabicl)
        y_bayesb = _as_float_vector(y_bayesb)
        y_gblup = _as_float_vector(y_gblup)
        group_probs = self._predict_group_probs_from_features(X_group)

        prior_map = _build_prior_prediction_map(y_bayesb, y_gblup, y_bayes_candidates)
        prior_names = list(getattr(self, "prior_names_", list(prior_map.keys())))
        for name in prior_names:
            if name not in prior_map:
                raise ValueError(f"Missing prior prediction for {name} during predict.")
        prior_matrix = np.column_stack([prior_map[name] for name in prior_names]).astype(np.float32)
        prior_weight_group = np.asarray(
            getattr(self, "prior_weight_group_", _legacy_summary_to_prior_weights(self.get_group_summary(), self.config.num_groups)[1]),
            dtype=np.float32,
        )
        group_predictions = []
        for gid in range(self.config.num_groups):
            w = float(self.w_group_[gid])
            weights = prior_weight_group[gid]
            y_prior = np.sum(prior_matrix * weights.reshape(1, -1), axis=1).astype(np.float32)
            group_predictions.append((y_prior + w * (y_tabicl - y_prior)).reshape(-1, 1).astype(np.float32))
        group_prediction_matrix = np.concatenate(group_predictions, axis=1).astype(np.float32)
        return np.sum(group_probs * group_prediction_matrix, axis=1).astype(np.float32)

    def predict(
        self,
        X: np.ndarray,
        y_tabicl: np.ndarray,
        y_bayesb: np.ndarray,
        y_gblup: np.ndarray,
        y_bayes_candidates: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        X_core = np.asarray(X, dtype=np.float32)
        return self.predict_from_group_features(
            self._build_group_features(X_core),
            y_tabicl=y_tabicl,
            y_bayesb=y_bayesb,
            y_gblup=y_gblup,
            y_bayes_candidates=y_bayes_candidates,
        )

    def get_group_summary(self) -> dict[str, list[float] | list[int] | str]:
        if not hasattr(self, "group_counts_"):
            raise RuntimeError("Model must be fit before get_group_summary.")
        summary: dict[str, list[float] | list[int] | str] = {
            "group_mode": self.config.group_mode,
            "assignment_mode": self.config.assignment_mode,
            "group_counts": self.group_counts_.astype(int).tolist(),
            "group_probs_mean": self.group_probs_mean_.astype(float).tolist(),
            "prior_names": list(getattr(self, "prior_names_", ["BayesB", "GBLUP"])),
            "prior_weight_group": np.asarray(
                getattr(self, "prior_weight_group_", _legacy_summary_to_prior_weights({}, self.config.num_groups)[1]),
                dtype=np.float32,
            ).astype(float).tolist(),
            "alpha_group": self.alpha_group_.astype(float).tolist(),
            "w_group": self.w_group_.astype(float).tolist(),
            "bayes_family_group": list(getattr(self, "bayes_family_group_", ["BayesB"] * self.config.num_groups)),
            "group_centroids": np.asarray(self.group_centroids_, dtype=np.float32).astype(float).tolist(),
            "scaler_mean": np.asarray(self.scaler_.mean_, dtype=np.float32).astype(float).tolist(),
            "scaler_scale": np.asarray(self.scaler_.scale_, dtype=np.float32).astype(float).tolist(),
        }
        if self.block_prior_group_counts_ is not None:
            summary["block_prior_group_counts"] = self.block_prior_group_counts_.astype(int).tolist()
        return summary
