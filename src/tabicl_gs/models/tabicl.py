from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


def average_embedding_batches(embeddings: np.ndarray) -> np.ndarray:
    return np.asarray(embeddings, dtype=np.float32).mean(axis=0)


def build_tabicl_regressor(**kwargs: Any):
    from tabicl import TabICLRegressor

    return TabICLRegressor(**kwargs)


@dataclass
class TabICLVectorMetadata:
    raw_embedding_dim: int
    reduced_embedding_dim: int
    device: str
    explained_variance_ratio_sum: float
    explained_variance_curve: list[float] | None


class TabICLVectorRegressor:
    def __init__(
        self,
        n_estimators: int = 1,
        norm_methods: list[str] | str | None = None,
        feat_shuffle_method: str = "none",
        batch_size: int | None = 1,
        kv_cache: str | bool = "repr",
        model_path: str | None = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "tabicl-regressor-v2-20260212.ckpt",
        device: str | None = "cuda",
        use_amp: bool | str = "auto",
        use_fa3: bool | str = "auto",
        offload_mode: str | bool = "auto",
        disk_offload_dir: str | None = None,
        random_state: int | None = 42,
        n_jobs: int | None = None,
        verbose: bool = False,
        inference_config: dict[str, Any] | None = None,
        embedding_reduce_dim: int | None = None,
        embedding_explained_variance_target: float | None = None,
        track_full_explained_variance: bool = False,
    ) -> None:
        self.regressor_kwargs = {
            "n_estimators": n_estimators,
            "norm_methods": norm_methods,
            "feat_shuffle_method": feat_shuffle_method,
            "batch_size": batch_size,
            "kv_cache": kv_cache,
            "model_path": model_path,
            "allow_auto_download": allow_auto_download,
            "checkpoint_version": checkpoint_version,
            "device": device,
            "use_amp": use_amp,
            "use_fa3": use_fa3,
            "offload_mode": offload_mode,
            "disk_offload_dir": disk_offload_dir,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": verbose,
            "inference_config": inference_config,
        }
        self.embedding_reduce_dim = embedding_reduce_dim
        self.embedding_explained_variance_target = embedding_explained_variance_target
        self.track_full_explained_variance = track_full_explained_variance
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabICLVectorRegressor":
        self.regressor_ = build_tabicl_regressor(**self.regressor_kwargs)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.regressor_.fit(X, y)
        self.raw_embedding_dim_ = int(self.regressor_.model_.row_num_cls * self.regressor_.model_.embed_dim)
        self.embedding_dim_ = self.raw_embedding_dim_
        self.explained_variance_ratio_sum_ = 1.0
        self.explained_variance_curve_ = None
        return self

    def _extract_embeddings_and_predictions_for_method(self, Xs: np.ndarray, kv_cache) -> tuple[np.ndarray, np.ndarray]:
        if kv_cache is None:
            raise ValueError("TabICL vector extraction requires kv_cache='repr'.")

        import torch

        batch_size = self.regressor_.batch_size or Xs.shape[0]
        n_batches = int(ceil(Xs.shape[0] / batch_size))
        splits = np.array_split(Xs, n_batches)
        embeddings_out = []
        predictions_out = []
        offset = 0
        for X_batch in splits:
            batch_cache = kv_cache.slice_batch(offset, offset + X_batch.shape[0])
            offset += X_batch.shape[0]
            captured: dict[str, np.ndarray] = {}

            def hook(_module, _inputs, output):
                captured["embeddings"] = output.detach().cpu().numpy()

            handle = self.regressor_.model_.row_interactor.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    predictions = self.regressor_.model_.predict_stats_with_cache(
                        X_test=torch.from_numpy(X_batch).float().to(self.regressor_.device_),
                        output_type="mean",
                        cache=batch_cache,
                        inference_config=self.regressor_.inference_config_,
                    )
            finally:
                handle.remove()

            if "embeddings" not in captured:
                raise RuntimeError("Failed to capture TabICL row representations.")
            embeddings_out.append(np.asarray(captured["embeddings"], dtype=np.float32))
            if hasattr(predictions, "detach"):
                predictions_out.append(predictions.detach().cpu().numpy().astype(np.float32))
            else:
                predictions_out.append(np.asarray(predictions, dtype=np.float32))
        return np.concatenate(embeddings_out, axis=0).astype(np.float32), np.concatenate(predictions_out, axis=0).astype(np.float32)

    def _extract_embeddings_only_for_method(self, Xs: np.ndarray, kv_cache) -> np.ndarray:
        if kv_cache is None:
            raise ValueError("TabICL vector extraction requires kv_cache='repr'.")

        import torch

        batch_size = self.regressor_.batch_size or Xs.shape[0]
        n_batches = int(ceil(Xs.shape[0] / batch_size))
        splits = np.array_split(Xs, n_batches)
        embeddings_out = []
        offset = 0
        for X_batch in splits:
            batch_cache = kv_cache.slice_batch(offset, offset + X_batch.shape[0])
            offset += X_batch.shape[0]
            X_batch_t = torch.from_numpy(X_batch).float().to(self.regressor_.device_)
            with torch.no_grad():
                embeddings = self.regressor_.model_.row_interactor(
                    self.regressor_.model_.col_embedder.forward_with_cache(
                        X_batch_t,
                        col_cache=batch_cache.col_cache,
                        y_train=None,
                        use_cache=True,
                        store_cache=False,
                        mgr_config=self.regressor_.inference_config_.COL_CONFIG,
                    ),
                    mgr_config=self.regressor_.inference_config_.ROW_CONFIG,
                )
            embeddings_out.append(embeddings.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(embeddings_out, axis=0).astype(np.float32)

    def _extract_ensemble_outputs(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        X_num = self.regressor_.X_encoder_.transform(X)
        test_data = self.regressor_.ensemble_generator_.transform(X_num, mode="test")
        all_embeddings = []
        all_predictions = []
        for norm_method, (Xs,) in test_data.items():
            embeddings, predictions = self._extract_embeddings_and_predictions_for_method(
                Xs,
                self.regressor_.model_kv_cache_[norm_method],
            )
            all_embeddings.append(embeddings)
            all_predictions.append(predictions)
        return np.concatenate(all_embeddings, axis=0).astype(np.float32), np.concatenate(all_predictions, axis=0).astype(np.float32)

    def transform_with_scalar(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ensemble_embeddings, ensemble_predictions = self._extract_ensemble_outputs(X)
        return ensemble_embeddings.mean(axis=0).astype(np.float32), ensemble_predictions.mean(axis=0).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X_num = self.regressor_.X_encoder_.transform(X)
        test_data = self.regressor_.ensemble_generator_.transform(X_num, mode="test")
        all_embeddings = []
        for norm_method, (Xs,) in test_data.items():
            embeddings = self._extract_embeddings_only_for_method(
                Xs,
                self.regressor_.model_kv_cache_[norm_method],
            )
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0).astype(np.float32).mean(axis=0)

    def fit_reducer(self, train_embeddings: np.ndarray) -> None:
        train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
        max_components = min(train_embeddings.shape[0], train_embeddings.shape[1])
        if self.embedding_explained_variance_target is not None:
            spectrum_pca = PCA(n_components=max_components, random_state=self.random_state)
            spectrum_pca.fit(train_embeddings)
            cumulative = np.cumsum(spectrum_pca.explained_variance_ratio_)
            chosen = int(np.searchsorted(cumulative, self.embedding_explained_variance_target) + 1)
            chosen = min(chosen, max_components)
            self.reducer_ = spectrum_pca
            self.embedding_dim_ = chosen
            self.explained_variance_ratio_sum_ = float(cumulative[chosen - 1])
            self.explained_variance_curve_ = cumulative.astype(float).tolist()
            return

        if self.embedding_reduce_dim is None or self.embedding_reduce_dim >= train_embeddings.shape[1]:
            self.reducer_ = None
            self.embedding_dim_ = train_embeddings.shape[1]
            self.explained_variance_ratio_sum_ = 1.0
            if self.track_full_explained_variance:
                spectrum_pca = PCA(n_components=max_components, random_state=self.random_state)
                spectrum_pca.fit(train_embeddings)
                self.explained_variance_curve_ = np.cumsum(spectrum_pca.explained_variance_ratio_).astype(float).tolist()
            return
        n_components = min(self.embedding_reduce_dim, train_embeddings.shape[0], train_embeddings.shape[1])
        self.reducer_ = PCA(n_components=n_components, random_state=self.random_state)
        self.reducer_.fit(train_embeddings)
        self.embedding_dim_ = int(n_components)
        self.explained_variance_ratio_sum_ = float(self.reducer_.explained_variance_ratio_.sum())
        if self.track_full_explained_variance:
            spectrum_pca = PCA(n_components=max_components, random_state=self.random_state)
            spectrum_pca.fit(train_embeddings)
            self.explained_variance_curve_ = np.cumsum(spectrum_pca.explained_variance_ratio_).astype(float).tolist()

    def reduce_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if getattr(self, "reducer_", None) is None:
            return embeddings
        transformed = self.reducer_.transform(embeddings).astype(np.float32)
        return transformed[:, : self.embedding_dim_]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.regressor_.predict(np.asarray(X, dtype=np.float32)), dtype=np.float32)

    def metadata(self) -> TabICLVectorMetadata:
        return TabICLVectorMetadata(
            raw_embedding_dim=int(self.raw_embedding_dim_),
            reduced_embedding_dim=int(self.embedding_dim_),
            device=str(self.regressor_.device_),
            explained_variance_ratio_sum=float(self.explained_variance_ratio_sum_),
            explained_variance_curve=self.explained_variance_curve_,
        )
