from __future__ import annotations

from typing import Any

import numpy as np

from tabicl_gs.models.block_attention import BlockAttentionRegressor
import tabicl_gs.models.baselines as baselines
from tabicl_gs.models.block_weight_pooling import GroupWeightedPoolingRegressor, StaticBlockWeightedRegressor
from tabicl_gs.models.calibrated_correction import CalibratedCorrectionRegressor
from tabicl_gs.models.group_shared_gate import GroupSharedGateRegressor
from tabicl_gs.models.sample_mixture import SampleWiseMixtureStage2Regressor
from tabicl_gs.models.tabicl import TabICLVectorRegressor, build_tabicl_regressor
from tabicl_gs.models.tabpfn import TabPFNVectorRegressor, build_tabpfn_regressor
from tabicl_gs.models.xgboost_model import XGBoostLeafRegressor, build_xgboost_regressor
from tabicl_gs.eval.splits import make_outer_cv_splits


class _RKHSExpertRegressor:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        output_dir: str,
        rscript_path: str,
        seed: int,
        keep_artifacts: bool = True,
        bandwidth_scale: float | None = None,
    ) -> None:
        self.X_train_ = np.asarray(X_train, dtype=np.float32)
        self.y_train_ = np.asarray(y_train, dtype=np.float32).reshape(-1)
        self.output_dir_ = str(output_dir)
        self.rscript_path_ = str(rscript_path)
        self.seed_ = int(seed)
        self.keep_artifacts_ = bool(keep_artifacts)
        self.bandwidth_scale_ = None if bandwidth_scale is None else float(bandwidth_scale)

    def predict(self, X: np.ndarray) -> np.ndarray:
        result = baselines.run_r_baseline(
            model_name="RKHS",
            X_train=self.X_train_,
            y_train=self.y_train_,
            X_test=np.asarray(X, dtype=np.float32),
            output_dir=self.output_dir_,
            rscript_path=self.rscript_path_,
            seed=self.seed_,
            keep_artifacts=self.keep_artifacts_,
            bandwidth_scale=self.bandwidth_scale_,
        )
        return np.asarray(result.predictions, dtype=np.float32)


def prepare_backend_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    result = dict(config)
    result.pop("embedding_reduce_dim", None)
    result.pop("use_raw_block_embeddings", None)
    result.pop("block_prior_method", None)
    result.pop("block_prior_mode", None)
    result.pop("use_block_prior", None)
    result.pop("use_prior_token", None)
    result.pop("use_prior_prediction", None)
    result.pop("use_oof_gate_training", None)
    result.pop("oof_splits", None)
    result.pop("expert_backend", None)
    result.pop("expert_config", None)
    result["random_state"] = seed
    return result


def create_stage1_encoder(backend: str, config: dict[str, Any], seed: int):
    backend = backend.lower()
    runtime_config = dict(config)
    runtime_config["random_state"] = seed
    if backend == "tabicl":
        return TabICLVectorRegressor(**runtime_config)
    if backend == "tabpfn":
        return TabPFNVectorRegressor(**runtime_config)
    if backend == "xgboost":
        return XGBoostLeafRegressor(**runtime_config)
    raise ValueError(f"Unsupported stage1 backend: {backend}")


def _fit_expert_regressor(backend: str, config: dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, seed: int):
    backend = backend.lower()
    runtime = dict(config)
    runtime["random_state"] = seed
    if backend == "tabicl":
        model = build_tabicl_regressor(**runtime)
        model.fit(X_train, y_train)
        device = str(model.device_)
    elif backend == "tabpfn":
        model = build_tabpfn_regressor(**runtime)
        model.fit(X_train, y_train)
        devices = getattr(model, "devices_", ())
        device = str(devices[0]) if devices else str(runtime.get("device", "auto"))
    elif backend == "xgboost":
        model = build_xgboost_regressor(**runtime)
        model.fit(X_train, y_train)
        device = str(runtime.get("device", "cpu"))
    elif backend == "rkhs":
        output_dir = str(runtime.get("output_dir", "outputs/rkhs_expert"))
        rscript_path = str(runtime.get("rscript_path", "Rscript"))
        keep_artifacts = bool(runtime.get("keep_artifacts", True))
        bandwidth_scale = runtime.get("bandwidth_scale")
        model = _RKHSExpertRegressor(
            X_train=X_train,
            y_train=y_train,
            output_dir=output_dir,
            rscript_path=rscript_path,
            seed=seed,
            keep_artifacts=keep_artifacts,
            bandwidth_scale=bandwidth_scale,
        )
        device = "R"
    else:
        raise ValueError(f"Unsupported calibrated_correction expert backend: {backend}")
    pred = np.asarray(model.predict(X_eval), dtype=np.float32)
    return model, pred, device


def _build_oof_predictions_for_expert(
    backend: str,
    expert_config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_splits: int,
) -> np.ndarray:
    splits = make_outer_cv_splits(X_train, n_splits=n_splits, seed=seed)
    oof = np.zeros(X_train.shape[0], dtype=np.float32)
    for split_id, (inner_train_idx, inner_valid_idx) in enumerate(splits, start=1):
        _, valid_pred, _ = _fit_expert_regressor(
            backend,
            expert_config,
            X_train[inner_train_idx],
            y_train[inner_train_idx],
            X_train[inner_valid_idx],
            seed + split_id,
        )
        oof[inner_valid_idx] = valid_pred
    return oof.astype(np.float32)


def fit_stage2_model(backend: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, config: dict[str, Any], seed: int):
    backend = backend.lower()
    runtime_config = prepare_backend_config(config, seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    if backend == "tabicl":
        model = build_tabicl_regressor(**runtime_config)
        model.fit(X_train, y_train)
        return model, np.asarray(model.predict(X_test), dtype=np.float32), str(model.device_)
    if backend == "tabpfn":
        model = build_tabpfn_regressor(**runtime_config)
        model.fit(X_train, y_train)
        device = str(model.devices_[0]) if getattr(model, "devices_", ()) else str(runtime_config.get("device", "auto"))
        return model, np.asarray(model.predict(X_test), dtype=np.float32), device
    if backend == "xgboost":
        model = build_xgboost_regressor(**runtime_config)
        model.fit(X_train, y_train)
        return model, np.asarray(model.predict(X_test), dtype=np.float32), str(runtime_config.get("device", "cpu"))
    if backend == "block_attention":
        model = BlockAttentionRegressor(**runtime_config)
        model.fit(X_train, y_train)
        return model, np.asarray(model.predict(X_test), dtype=np.float32), str(model.device_)
    if backend == "static_block_weight":
        model = StaticBlockWeightedRegressor(**runtime_config)
        model.fit(X_train, y_train)
        return model, np.asarray(model.predict(X_test), dtype=np.float32), str(model.device_)
    if backend == "group_weight_pooling":
        model = GroupWeightedPoolingRegressor(**runtime_config)
        model.fit(X_train, y_train)
        return model, np.asarray(model.predict(X_test), dtype=np.float32), str(model.device_)
    if backend == "group_shared_gate":
        if not bool(config.get("use_prior_prediction", True)):
            raise ValueError("group_shared_gate requires use_prior_prediction=True.")
        if not bool(config.get("use_dual_priors", False)):
            raise ValueError("group_shared_gate requires use_dual_priors=True.")
        runtime_config = dict(runtime_config)
        runtime_config.pop("use_dual_priors", None)
        expert_backend = str(config.get("expert_backend", "tabicl")).lower()
        expert_config = dict(config.get("expert_config", {}))
        prior_width = 2
        if X_train.shape[1] <= prior_width or X_test.shape[1] <= prior_width:
            raise ValueError(
                f"group_shared_gate expected core features plus 2 prior columns, got train shape={X_train.shape}, test shape={X_test.shape}."
            )
        X_train_core = X_train[:, :-prior_width].astype(np.float32)
        X_test_core = X_test[:, :-prior_width].astype(np.float32)
        y_bayesb_train = X_train[:, -prior_width].astype(np.float32)
        y_bayesb_test = X_test[:, -prior_width].astype(np.float32)
        y_gblup_train = X_train[:, -1].astype(np.float32)
        y_gblup_test = X_test[:, -1].astype(np.float32)
        expert_model, y_tabicl_test, device = _fit_expert_regressor(
            expert_backend,
            expert_config,
            X_train_core,
            y_train,
            X_test_core,
            seed=seed,
        )
        y_tabicl_train = np.asarray(expert_model.predict(X_train_core), dtype=np.float32)
        model = GroupSharedGateRegressor(**runtime_config)
        model.fit(X_train_core, y_train, y_tabicl_train, y_bayesb_train, y_gblup_train)

        class _GroupSharedGateStage2Wrapper:
            def __init__(self, expert_model, gate_model):
                self.expert_model = expert_model
                self.gate_model = gate_model

            def predict(self, X):
                X = np.asarray(X, dtype=np.float32)
                X_core = X[:, :-prior_width].astype(np.float32)
                y_bayesb = X[:, -prior_width].astype(np.float32)
                y_gblup = X[:, -1].astype(np.float32)
                y_tabicl = np.asarray(self.expert_model.predict(X_core), dtype=np.float32)
                return self.gate_model.predict(X_core, y_tabicl, y_bayesb, y_gblup)

            def get_group_summary(self):
                return self.gate_model.get_group_summary()

        wrapped = _GroupSharedGateStage2Wrapper(expert_model, model)
        return wrapped, np.asarray(wrapped.predict(X_test), dtype=np.float32), device
    if backend == "sample_mixture":
        model = SampleWiseMixtureStage2Regressor(
            expert_backend=str(config.get("expert_backend", "tabicl")),
            expert_config=dict(config.get("expert_config", {})),
            use_prior_prediction=bool(config.get("use_prior_prediction", True)),
            **runtime_config,
        )
        model.fit(X_train, y_train)
        return model, np.asarray(model.predict(X_test), dtype=np.float32), str(model.device_)
    if backend == "calibrated_correction":
        if not bool(config.get("use_prior_prediction", True)):
            raise ValueError("calibrated_correction requires use_prior_prediction=True.")
        use_dual_priors = bool(config.get("use_dual_priors", False))
        use_oof_gate_training = bool(config.get("use_oof_gate_training", False))
        oof_splits = int(config.get("oof_splits", 3))
        expert_backend = str(config.get("expert_backend", "tabicl")).lower()
        expert_config = dict(config.get("expert_config", {}))
        prior_width = 2 if use_dual_priors else 1
        if X_train.shape[1] <= prior_width or X_test.shape[1] <= prior_width:
            raise ValueError(
                f"calibrated_correction expected feature columns plus {prior_width} prior columns, "
                f"got train shape={X_train.shape}, test shape={X_test.shape}."
            )
        X_train_core = X_train[:, :-prior_width].astype(np.float32)
        X_test_core = X_test[:, :-prior_width].astype(np.float32)
        y_bayesb_train = X_train[:, -prior_width].astype(np.float32)
        y_bayesb_test = X_test[:, -prior_width].astype(np.float32)
        y_gblup_train = X_train[:, -1].astype(np.float32) if use_dual_priors else None
        y_gblup_test = X_test[:, -1].astype(np.float32) if use_dual_priors else None
        if use_oof_gate_training:
            y_tabicl_train_for_gate = _build_oof_predictions_for_expert(
                expert_backend,
                expert_config,
                X_train_core,
                y_train,
                seed=seed,
                n_splits=oof_splits,
            )
        else:
            y_tabicl_train_for_gate = None

        expert_model, y_tabicl_test, device = _fit_expert_regressor(
            expert_backend,
            expert_config,
            X_train_core,
            y_train,
            X_test_core,
            seed=seed,
        )
        y_tabicl_train = np.asarray(expert_model.predict(X_train_core), dtype=np.float32)

        correction_model = CalibratedCorrectionRegressor(**runtime_config)
        correction_model.fit(
            X_train_core,
            y_train,
            y_tabicl_train if y_tabicl_train_for_gate is None else y_tabicl_train_for_gate,
            y_bayesb_train,
            y_gblup_train,
        )

        class _CalibratedCorrectionStage2Wrapper:
            def __init__(self, expert_model, correction_model):
                self.expert_model = expert_model
                self.correction_model = correction_model

            def predict(self, X):
                X = np.asarray(X, dtype=np.float32)
                prior_width = 2 if self.correction_model.config.use_dual_priors else 1
                X_core = X[:, :-prior_width].astype(np.float32)
                y_bayesb = X[:, -prior_width].astype(np.float32)
                y_gblup = X[:, -1].astype(np.float32) if self.correction_model.config.use_dual_priors else None
                y_tabicl = np.asarray(self.expert_model.predict(X_core), dtype=np.float32)
                return self.correction_model.predict(X_core, y_tabicl, y_bayesb, y_gblup)

            def get_group_summary(self):
                return self.correction_model.get_group_summary()

        wrapped = _CalibratedCorrectionStage2Wrapper(expert_model, correction_model)
        return wrapped, np.asarray(wrapped.predict(X_test), dtype=np.float32), device
    raise ValueError(f"Unsupported stage2 backend: {backend}")
