from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class TabPFNVectorMetadata:
    raw_embedding_dim: int
    reduced_embedding_dim: int
    device: str


def build_tabpfn_regressor(**kwargs: Any):
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(**kwargs)


class TabPFNVectorRegressor:
    def __init__(
        self,
        n_estimators: int = 1,
        categorical_features_indices: list[int] | None = None,
        softmax_temperature: float = 0.9,
        average_before_softmax: bool = False,
        model_path: str = "auto",
        device: str = "auto",
        ignore_pretraining_limits: bool = False,
        inference_precision: str = "auto",
        fit_mode: str = "fit_preprocessors",
        memory_saving_mode: str | bool | float | int = "auto",
        random_state: int | None = 0,
        n_jobs: int = -1,
        inference_config: dict[str, Any] | None = None,
        differentiable_input: bool = False,
        embedding_reduce_dim: int | None = None,
    ) -> None:
        self.regressor_kwargs = {
            "n_estimators": n_estimators,
            "categorical_features_indices": categorical_features_indices,
            "softmax_temperature": softmax_temperature,
            "average_before_softmax": average_before_softmax,
            "model_path": model_path,
            "device": device,
            "ignore_pretraining_limits": ignore_pretraining_limits,
            "inference_precision": inference_precision,
            "fit_mode": fit_mode,
            "memory_saving_mode": memory_saving_mode,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "inference_config": inference_config,
            "differentiable_input": differentiable_input,
        }
        self.embedding_reduce_dim = embedding_reduce_dim
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNVectorRegressor":
        self.regressor_ = build_tabpfn_regressor(**self.regressor_kwargs)
        self.regressor_.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32))
        train_embeddings = self._get_embeddings(X)
        self.raw_embedding_dim_ = int(train_embeddings.shape[1])
        self.embedding_dim_ = self.raw_embedding_dim_
        return self

    def _get_embeddings(self, X: np.ndarray) -> np.ndarray:
        from tabpfn.utils import get_embeddings

        embeddings = get_embeddings(self.regressor_, np.asarray(X, dtype=np.float32), data_source="test")
        return np.asarray(embeddings, dtype=np.float32).mean(axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._get_embeddings(X)

    def fit_reducer(self, train_embeddings: np.ndarray) -> None:
        train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
        if self.embedding_reduce_dim is None or self.embedding_reduce_dim >= train_embeddings.shape[1]:
            self.reducer_ = None
            self.embedding_dim_ = train_embeddings.shape[1]
            return
        n_components = min(self.embedding_reduce_dim, train_embeddings.shape[0], train_embeddings.shape[1])
        self.reducer_ = PCA(n_components=n_components, random_state=self.random_state)
        self.reducer_.fit(train_embeddings)
        self.embedding_dim_ = int(n_components)

    def reduce_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if getattr(self, "reducer_", None) is None:
            return embeddings
        return self.reducer_.transform(embeddings).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.regressor_.predict(np.asarray(X, dtype=np.float32)), dtype=np.float32)

    def metadata(self) -> TabPFNVectorMetadata:
        devices = getattr(self.regressor_, "devices_", ())
        device = str(devices[0]) if devices else str(self.regressor_kwargs.get("device", "auto"))
        return TabPFNVectorMetadata(
            raw_embedding_dim=int(self.raw_embedding_dim_),
            reduced_embedding_dim=int(self.embedding_dim_),
            device=device,
        )
