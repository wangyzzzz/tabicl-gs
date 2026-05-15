import numpy as np

from tabicl_gs.models.model_specs import TwoStageModelSpec
from tabicl_gs.pipeline.experiment import (
    _append_stage2_prior_feature,
    _append_stage2_block_prior_feature,
    _prepare_stage2_config,
    _resolve_stage2_feature_mode,
)


def test_resolve_stage2_feature_mode_defaults_to_reduced_for_block_attention():
    spec = TwoStageModelSpec(
        name="TabICLv2-PCA-Attention",
        stage1_backend="tabicl",
        stage2_backend="block_attention",
        stage1_config={},
        stage2_config={"model_dim": 64},
    )
    assert _resolve_stage2_feature_mode(spec) == "reduced"


def test_resolve_stage2_feature_mode_supports_raw_block_embeddings():
    spec = TwoStageModelSpec(
        name="TabICLv2-Attention",
        stage1_backend="tabicl",
        stage2_backend="block_attention",
        stage1_config={},
        stage2_config={"model_dim": 64, "use_raw_block_embeddings": True},
    )
    assert _resolve_stage2_feature_mode(spec) == "raw"


def test_prepare_stage2_config_uses_raw_block_dims_when_requested():
    spec = TwoStageModelSpec(
        name="TabICLv2-Attention",
        stage1_backend="tabicl",
        stage2_backend="block_attention",
        stage1_config={},
        stage2_config={"model_dim": 64, "use_raw_block_embeddings": True},
    )
    config = _prepare_stage2_config(
        model_spec=spec,
        block_summaries=[
            {"raw_embedding_dim": 512, "reduced_embedding_dim": 18},
            {"raw_embedding_dim": 512, "reduced_embedding_dim": 21},
        ],
        include_block_scalar=False,
    )
    assert config["block_input_dims"] == [512, 512]
    assert "use_raw_block_embeddings" not in config


def test_prepare_stage2_config_uses_reduced_dims_by_default():
    spec = TwoStageModelSpec(
        name="TabICLv2-PCA-Attention",
        stage1_backend="tabicl",
        stage2_backend="block_attention",
        stage1_config={},
        stage2_config={"model_dim": 64},
    )
    config = _prepare_stage2_config(
        model_spec=spec,
        block_summaries=[
            {"raw_embedding_dim": 512, "reduced_embedding_dim": 18},
            {"raw_embedding_dim": 512, "reduced_embedding_dim": 21},
        ],
        include_block_scalar=False,
    )
    assert config["block_input_dims"] == [18, 21]


def test_prepare_stage2_config_supports_static_block_weight_backend():
    spec = TwoStageModelSpec(
        name="TabICLv2-StaticBlockWeight",
        stage1_backend="tabicl",
        stage2_backend="static_block_weight",
        stage1_config={},
        stage2_config={"model_dim": 64, "prior_scores": [0.1, 0.2]},
    )
    config = _prepare_stage2_config(
        model_spec=spec,
        block_summaries=[
            {"raw_embedding_dim": 512, "reduced_embedding_dim": 18},
            {"raw_embedding_dim": 512, "reduced_embedding_dim": 21},
        ],
        include_block_scalar=False,
    )
    assert config["block_input_dims"] == [18, 21]
    assert config["prior_scores"] == [0.1, 0.2]


def test_prepare_stage2_config_supports_group_shared_gate_backend():
    spec = TwoStageModelSpec(
        name="TabICLv2-GroupSharedGate",
        stage1_backend="tabicl",
        stage2_backend="group_shared_gate",
        stage1_config={},
        stage2_config={"num_groups": 3, "prior_scores": [0.1, 0.2]},
    )
    config = _prepare_stage2_config(
        model_spec=spec,
        block_summaries=[
            {"raw_embedding_dim": 512, "reduced_embedding_dim": 18},
            {"raw_embedding_dim": 512, "reduced_embedding_dim": 21},
        ],
        include_block_scalar=False,
    )
    assert config["block_input_dims"] == [18, 21]
    assert config["prior_scores"] == [0.1, 0.2]


def test_append_stage2_prior_feature_supports_group_shared_gate_dual_prior():
    spec = TwoStageModelSpec(
        name="TabICLv2-GroupSharedGate",
        stage1_backend="tabicl",
        stage2_backend="group_shared_gate",
        stage1_config={},
        stage2_config={"use_prior_prediction": True, "use_dual_priors": True},
    )
    X_train = np.ones((3, 10), dtype=np.float32)
    X_test = np.ones((2, 10), dtype=np.float32)
    prior_train = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]], dtype=np.float32)
    prior_test = np.array([[0.7, 0.9], [0.8, 1.0]], dtype=np.float32)
    train_out, test_out = _append_stage2_prior_feature(
        model_spec=spec,
        X_train_stage2=X_train,
        X_test_stage2=X_test,
        prior_train=prior_train,
        prior_test=prior_test,
    )
    assert train_out.shape == (3, 12)
    assert test_out.shape == (2, 12)
    assert np.allclose(train_out[:, -2:], prior_train)


def test_append_stage2_prior_feature_adds_single_prior_token_dimension():
    spec = TwoStageModelSpec(
        name="TabICLv2-Attention-ResidualBayesB",
        stage1_backend="tabicl",
        stage2_backend="block_attention",
        stage1_config={},
        stage2_config={"model_dim": 64, "use_raw_block_embeddings": True, "use_prior_token": True},
    )
    X_train = np.ones((3, 1024), dtype=np.float32)
    X_test = np.ones((2, 1024), dtype=np.float32)
    train_out, test_out = _append_stage2_prior_feature(
        model_spec=spec,
        X_train_stage2=X_train,
        X_test_stage2=X_test,
        prior_train=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        prior_test=np.array([0.4, 0.5], dtype=np.float32),
    )
    assert train_out.shape == (3, 1025)
    assert test_out.shape == (2, 1025)
    assert np.allclose(train_out[:, -1], np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_append_stage2_block_prior_feature_adds_per_block_prior_tail():
    spec = TwoStageModelSpec(
        name="TabICLv2-Attention-BlockPrior",
        stage1_backend="tabicl",
        stage2_backend="block_attention",
        stage1_config={},
        stage2_config={"model_dim": 64, "use_raw_block_embeddings": True, "use_block_prior": True},
    )
    X_train = np.ones((2, 1024), dtype=np.float32)
    X_test = np.ones((1, 1024), dtype=np.float32)
    train_out, test_out = _append_stage2_block_prior_feature(
        model_spec=spec,
        X_train_stage2=X_train,
        X_test_stage2=X_test,
        block_prior=np.array([0.1, 0.2], dtype=np.float32),
    )
    assert train_out.shape == (2, 1026)
    assert test_out.shape == (1, 1026)
    assert np.allclose(train_out[:, -2:], np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float32))
