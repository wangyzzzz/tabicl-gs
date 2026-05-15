from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tabicl_gs.models.block_attention import flat_features_to_block_tensor


def _standardize_prior(prior_scores: Sequence[float]) -> np.ndarray:
    prior = np.asarray(prior_scores, dtype=np.float32).reshape(-1)
    if prior.size == 0:
        raise ValueError("prior_scores cannot be empty.")
    scale = float(prior.std())
    if scale <= 1e-8:
        return np.zeros_like(prior, dtype=np.float32)
    return ((prior - float(prior.mean())) / scale).astype(np.float32)


def _assign_groups_from_prior(prior_scores: np.ndarray, num_groups: int) -> np.ndarray:
    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")
    ranks = np.argsort(np.argsort(prior_scores))
    bins = np.floor(ranks * num_groups / max(len(prior_scores), 1)).astype(int)
    return np.clip(bins, 0, num_groups - 1).astype(np.int64)


@dataclass
class _PoolingConfig:
    block_input_dims: list[int]
    prior_scores: np.ndarray
    model_dim: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 150
    batch_size: int | None = None
    device: str = "cuda"
    random_state: int = 42


class _StaticBlockWeightNet:  # pragma: no cover
    def __init__(self, config: _PoolingConfig):
        import torch
        from torch import nn

        self.config = config
        self.max_block_dim = max(config.block_input_dims)
        self.num_blocks = len(config.block_input_dims)
        self.module = nn.Module()
        self.module.input_proj = nn.Linear(self.max_block_dim, config.model_dim)
        init_logits = torch.from_numpy(config.prior_scores.copy()).float()
        self.module.block_logits = nn.Parameter(init_logits)
        self.module.output = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 1),
        )

    def parameters(self):
        return self.module.parameters()

    def to(self, device):
        self.module.to(device)
        return self

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def forward(self, x):
        import torch

        h = self.module.input_proj(x)
        weights = torch.softmax(self.module.block_logits, dim=0)
        pooled = torch.sum(h * weights.view(1, -1, 1), dim=1)
        return self.module.output(pooled).squeeze(-1)

    def get_block_weights(self):
        import torch

        with torch.no_grad():
            return torch.softmax(self.module.block_logits, dim=0).detach().cpu().numpy().astype(np.float32)


class _GroupWeightedPoolingNet:  # pragma: no cover
    def __init__(self, config: _PoolingConfig, group_ids: np.ndarray, num_groups: int):
        import torch
        from torch import nn

        self.config = config
        self.max_block_dim = max(config.block_input_dims)
        self.num_blocks = len(config.block_input_dims)
        self.num_groups = int(num_groups)
        self.group_ids = torch.from_numpy(group_ids.astype(np.int64))
        self.module = nn.Module()
        self.module.input_proj = nn.Linear(self.max_block_dim, config.model_dim)

        group_prior = np.zeros(self.num_groups, dtype=np.float32)
        for gid in range(self.num_groups):
            mask = group_ids == gid
            group_prior[gid] = float(np.mean(config.prior_scores[mask])) if np.any(mask) else 0.0
        self.module.group_logits = nn.Parameter(torch.from_numpy(group_prior).float())
        self.module.output = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 1),
        )

    def parameters(self):
        return self.module.parameters()

    def to(self, device):
        self.module.to(device)
        self.group_ids = self.group_ids.to(device)
        return self

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def forward(self, x):
        import torch

        h = self.module.input_proj(x)
        group_reprs = []
        for gid in range(self.num_groups):
            mask = self.group_ids == gid
            if bool(mask.any()):
                group_reprs.append(h[:, mask, :].mean(dim=1))
            else:
                group_reprs.append(torch.zeros((h.shape[0], h.shape[2]), device=h.device, dtype=h.dtype))
        groups = torch.stack(group_reprs, dim=1)
        weights = torch.softmax(self.module.group_logits, dim=0)
        pooled = torch.sum(groups * weights.view(1, -1, 1), dim=1)
        return self.module.output(pooled).squeeze(-1)

    def get_group_weights(self):
        import torch

        with torch.no_grad():
            return torch.softmax(self.module.group_logits, dim=0).detach().cpu().numpy().astype(np.float32)


class _BaseWeightedPoolingRegressor:
    def __init__(
        self,
        block_input_dims: Sequence[int],
        prior_scores: Sequence[float],
        model_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 150,
        batch_size: int | None = None,
        device: str = "cuda",
        random_state: int = 42,
    ) -> None:
        self.config = _PoolingConfig(
            block_input_dims=[int(dim) for dim in block_input_dims],
            prior_scores=_standardize_prior(prior_scores),
            model_dim=int(model_dim),
            lr=float(lr),
            weight_decay=float(weight_decay),
            max_epochs=int(max_epochs),
            batch_size=None if batch_size is None else int(batch_size),
            device=str(device),
            random_state=int(random_state),
        )

    def _resolve_device(self):
        import torch

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _set_seed(self):
        import torch

        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

    def _prepare_tensor(self, X: np.ndarray):
        import torch

        X = np.asarray(X, dtype=np.float32)
        tensor = flat_features_to_block_tensor(X, self.config.block_input_dims)
        return torch.from_numpy(tensor).float().to(self.device_)

    def _fit_loop(self, model, X_tensor, y_tensor):
        import torch

        optimizer = torch.optim.AdamW(
            list(model.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        criterion = torch.nn.MSELoss()
        batch_size = self.config.batch_size or X_tensor.shape[0]
        for _ in range(self.config.max_epochs):
            model.train()
            if batch_size >= X_tensor.shape[0]:
                batches = [(X_tensor, y_tensor)]
            else:
                permutation = torch.randperm(X_tensor.shape[0], device=self.device_)
                batches = []
                for start in range(0, X_tensor.shape[0], batch_size):
                    idx = permutation[start : start + batch_size]
                    batches.append((X_tensor[idx], y_tensor[idx]))
            for X_batch, y_batch in batches:
                optimizer.zero_grad(set_to_none=True)
                pred = model.forward(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
        model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if not hasattr(self, "model_"):
            raise RuntimeError("Model must be fit before predict.")
        X_tensor = self._prepare_tensor(X)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_.forward(X_tensor).detach().cpu().numpy().astype(np.float32)
        return pred * self.y_std_ + self.y_mean_


class StaticBlockWeightedRegressor(_BaseWeightedPoolingRegressor):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "StaticBlockWeightedRegressor":
        import torch

        self._set_seed()
        self.device_ = self._resolve_device()
        self.model_ = _StaticBlockWeightNet(self.config).to(self.device_)

        X_tensor = self._prepare_tensor(X)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        self.y_mean_ = float(y_np.mean())
        self.y_std_ = float(y_np.std()) if float(y_np.std()) > 1e-6 else 1.0
        y_tensor = torch.from_numpy(((y_np - self.y_mean_) / self.y_std_).astype(np.float32)).to(self.device_)
        self._fit_loop(self.model_, X_tensor, y_tensor)
        return self

    def get_block_weights(self) -> np.ndarray:
        return self.model_.get_block_weights()


class GroupWeightedPoolingRegressor(_BaseWeightedPoolingRegressor):
    def __init__(
        self,
        block_input_dims: Sequence[int],
        prior_scores: Sequence[float],
        num_groups: int = 3,
        model_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 150,
        batch_size: int | None = None,
        device: str = "cuda",
        random_state: int = 42,
    ) -> None:
        super().__init__(
            block_input_dims=block_input_dims,
            prior_scores=prior_scores,
            model_dim=model_dim,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
        )
        self.num_groups = int(num_groups)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GroupWeightedPoolingRegressor":
        import torch

        self._set_seed()
        self.device_ = self._resolve_device()
        group_ids = _assign_groups_from_prior(self.config.prior_scores, self.num_groups)
        self.model_ = _GroupWeightedPoolingNet(self.config, group_ids=group_ids, num_groups=self.num_groups).to(self.device_)

        X_tensor = self._prepare_tensor(X)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        self.y_mean_ = float(y_np.mean())
        self.y_std_ = float(y_np.std()) if float(y_np.std()) > 1e-6 else 1.0
        y_tensor = torch.from_numpy(((y_np - self.y_mean_) / self.y_std_).astype(np.float32)).to(self.device_)
        self._fit_loop(self.model_, X_tensor, y_tensor)
        return self

    def get_group_weights(self) -> np.ndarray:
        return self.model_.get_group_weights()

