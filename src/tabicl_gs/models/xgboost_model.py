from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class XGBoostVectorMetadata:
    raw_embedding_dim: int
    reduced_embedding_dim: int
    device: str


def build_xgboost_regressor(**kwargs: Any):
    from xgboost import XGBRegressor

    return XGBRegressor(**kwargs)


class XGBoostLeafRegressor:
    def __init__(
        self,
        n_estimators: int = 64,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        tree_method: str = "hist",
        objective: str = "reg:squarederror",
        random_state: int | None = 42,
        n_jobs: int = 1,
        device: str = "cpu",
        embedding_reduce_dim: int | None = None,
    ) -> None:
        self.regressor_kwargs = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "tree_method": tree_method,
            "objective": objective,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "device": device,
        }
        self.embedding_reduce_dim = embedding_reduce_dim
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostLeafRegressor":
        self.regressor_ = build_xgboost_regressor(**self.regressor_kwargs)
        self.regressor_.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32))
        train_leaves = self.transform(X)
        self.raw_embedding_dim_ = int(train_leaves.shape[1])
        self.embedding_dim_ = self.raw_embedding_dim_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        leaves = self.regressor_.apply(np.asarray(X, dtype=np.float32))
        return np.asarray(leaves, dtype=np.float32)

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

    def metadata(self) -> XGBoostVectorMetadata:
        return XGBoostVectorMetadata(
            raw_embedding_dim=int(self.raw_embedding_dim_),
            reduced_embedding_dim=int(self.embedding_dim_),
            device=str(self.regressor_kwargs.get("device", "cpu")),
        )
