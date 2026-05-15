from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def flat_features_to_block_tensor(X: np.ndarray, block_input_dims: Sequence[int]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D flat feature matrix, got shape={X.shape}")
    dims = [int(dim) for dim in block_input_dims]
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"All block_input_dims must be positive, got {dims}")
    if sum(dims) != X.shape[1]:
        raise ValueError(
            f"Sum of block_input_dims ({sum(dims)}) does not match flat feature width ({X.shape[1]})."
        )

    max_dim = max(dims)
    tensor = np.zeros((X.shape[0], len(dims), max_dim), dtype=np.float32)
    start = 0
    for block_id, dim in enumerate(dims):
        end = start + dim
        tensor[:, block_id, :dim] = X[:, start:end]
        start = end
    return tensor


@dataclass
class BlockAttentionConfig:
    block_input_dims: list[int]
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_multiplier: int = 2
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 150
    batch_size: int | None = None
    device: str = "cuda"
    random_state: int = 42
    use_prior_token: bool = False
    use_block_prior: bool = False


class _BlockAttentionNet:  # pragma: no cover - exercised via regressor tests
    def __init__(self, config: BlockAttentionConfig):
        import torch
        from torch import nn

        self._nn = nn
        self.config = config
        self.max_block_dim = max(config.block_input_dims)
        self.num_blocks = len(config.block_input_dims)
        self.use_prior_token = bool(config.use_prior_token)
        self.use_block_prior = bool(config.use_block_prior)
        self.module = nn.Module()
        self.module.input_proj = nn.Linear(self.max_block_dim, config.model_dim)
        self.module.cls_token = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        self.module.pos_embed = nn.Parameter(torch.zeros(1, self.num_blocks + 1, config.model_dim))
        if self.use_block_prior:
            self.module.prior_scale = nn.Linear(1, config.model_dim)
            self.module.prior_shift = nn.Linear(1, config.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.model_dim * config.ff_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.module.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.module.norm = nn.LayerNorm(config.model_dim)
        self.module.head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 1),
        )
        if self.use_prior_token:
            self.module.gate_head = nn.Sequential(
                nn.Linear(config.model_dim, config.model_dim),
                nn.GELU(),
                nn.Linear(config.model_dim, 1),
            )

        nn.init.trunc_normal_(self.module.cls_token, std=0.02)
        nn.init.trunc_normal_(self.module.pos_embed, std=0.02)

    def parameters(self):
        return self.module.parameters()

    def to(self, device):
        self.module.to(device)
        return self

    def eval(self):
        self.module.eval()

    def train(self):
        self.module.train()

    def forward(self, x, block_prior=None):
        import torch

        h = self.module.input_proj(x)
        if self.use_block_prior:
            if block_prior is None:
                raise ValueError("use_block_prior=True requires block_prior input.")
            prior = block_prior.unsqueeze(-1)
            scale = torch.tanh(self.module.prior_scale(prior))
            shift = self.module.prior_shift(prior)
            h = h * (1.0 + scale) + shift
        cls = self.module.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, h], dim=1) + self.module.pos_embed[:, : h.shape[1] + 1, :]
        encoded = self.module.encoder(tokens)
        pooled = self.module.norm(encoded[:, 0, :])
        delta = self.module.head(pooled).squeeze(-1)
        if not self.use_prior_token:
            return delta
        prior = x[:, -1, 0]
        gate = torch.sigmoid(self.module.gate_head(pooled).squeeze(-1))
        return prior + gate * delta


class BlockAttentionRegressor:
    def __init__(
        self,
        block_input_dims: Sequence[int],
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_multiplier: int = 2,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 150,
        batch_size: int | None = None,
        device: str = "cuda",
        random_state: int = 42,
        use_prior_token: bool = False,
        use_block_prior: bool = False,
        verbose: bool = False,
    ) -> None:
        self.config = BlockAttentionConfig(
            block_input_dims=[int(dim) for dim in block_input_dims],
            model_dim=int(model_dim),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            ff_multiplier=int(ff_multiplier),
            dropout=float(dropout),
            lr=float(lr),
            weight_decay=float(weight_decay),
            max_epochs=int(max_epochs),
            batch_size=None if batch_size is None else int(batch_size),
            device=str(device),
            random_state=int(random_state),
            use_prior_token=bool(use_prior_token),
            use_block_prior=bool(use_block_prior),
        )
        self.verbose = bool(verbose)

    def _resolve_device(self):
        import torch

        requested = self.config.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _set_seed(self) -> None:
        import torch

        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

    def _prepare_tensor(self, X: np.ndarray):
        import torch

        X = np.asarray(X, dtype=np.float32)
        token_width = sum(self.config.block_input_dims)
        token_X = X[:, :token_width] if self.config.use_block_prior else X
        tensor = flat_features_to_block_tensor(token_X, self.config.block_input_dims)
        return torch.from_numpy(tensor).float().to(self.device_)

    def _prepare_block_prior(self, X: np.ndarray):
        import torch

        if not self.config.use_block_prior:
            return None
        n_blocks = len(self.config.block_input_dims)
        if X.shape[1] < n_blocks:
            raise ValueError(f"Expected at least {n_blocks} block prior features, got shape={X.shape}")
        prior = np.asarray(X[:, -n_blocks:], dtype=np.float32)
        return torch.from_numpy(prior).float().to(self.device_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlockAttentionRegressor":
        import torch

        self._set_seed()
        self.device_ = self._resolve_device()
        self.model_ = _BlockAttentionNet(self.config).to(self.device_)

        X_tensor = self._prepare_tensor(X)
        block_prior = self._prepare_block_prior(X)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        self.y_mean_ = float(y_np.mean())
        self.y_std_ = float(y_np.std()) if float(y_np.std()) > 1e-6 else 1.0
        y_tensor = torch.from_numpy(((y_np - self.y_mean_) / self.y_std_).astype(np.float32)).to(self.device_)

        optimizer = torch.optim.AdamW(
            list(self.model_.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        criterion = torch.nn.MSELoss()
        batch_size = self.config.batch_size or X_tensor.shape[0]

        for epoch in range(self.config.max_epochs):
            self.model_.train()
            if batch_size >= X_tensor.shape[0]:
                batches = [(X_tensor, block_prior, y_tensor)]
            else:
                permutation = torch.randperm(X_tensor.shape[0], device=self.device_)
                batches = []
                for start in range(0, X_tensor.shape[0], batch_size):
                    idx = permutation[start : start + batch_size]
                    prior_batch = None if block_prior is None else block_prior[idx]
                    batches.append((X_tensor[idx], prior_batch, y_tensor[idx]))

            epoch_loss = 0.0
            for X_batch, prior_batch, y_batch in batches:
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_.forward(X_batch, block_prior=prior_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())

            if self.verbose and (epoch == 0 or (epoch + 1) % 25 == 0):
                print(f"[block_attention] epoch={epoch + 1} loss={epoch_loss / len(batches):.6f}")

        self.model_.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if not hasattr(self, "model_"):
            raise RuntimeError("BlockAttentionRegressor must be fit before predict.")
        X_tensor = self._prepare_tensor(X)
        block_prior = self._prepare_block_prior(X)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_.forward(X_tensor, block_prior=block_prior).detach().cpu().numpy().astype(np.float32)
        return pred * self.y_std_ + self.y_mean_
