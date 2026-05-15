import numpy as np

from tabicl_gs.models.model_specs import TwoStageModelSpec
from tabicl_gs.pipeline.experiment import _append_stage2_prior_feature, _resolve_stage2_prior


def test_append_stage2_prior_feature_supports_sample_mixture():
    spec = TwoStageModelSpec(
        name="TabICLv2-SampleMixture",
        stage1_backend="tabicl",
        stage2_backend="sample_mixture",
        stage1_config={},
        stage2_config={
            "expert_backend": "tabicl",
            "use_prior_prediction": True,
        },
    )
    X_train = np.ones((3, 8), dtype=np.float32)
    X_test = np.ones((2, 8), dtype=np.float32)

    train_out, test_out = _append_stage2_prior_feature(
        model_spec=spec,
        X_train_stage2=X_train,
        X_test_stage2=X_test,
        prior_train=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        prior_test=np.array([0.4, 0.5], dtype=np.float32),
    )

    assert train_out.shape == (3, 9)
    assert test_out.shape == (2, 9)
    assert np.allclose(train_out[:, -1], np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_resolve_stage2_prior_supports_sample_mixture():
    spec = TwoStageModelSpec(
        name="TabICLv2-SampleMixture",
        stage1_backend="tabicl",
        stage2_backend="sample_mixture",
        stage1_config={},
        stage2_config={
            "expert_backend": "tabicl",
            "use_prior_prediction": True,
        },
    )

    resolved = _resolve_stage2_prior(
        {
            "stage2_prior": {
                "enabled": True,
                "baseline_model": "BayesB",
            }
        },
        spec,
    )

    assert resolved == {"baseline_model": "BayesB"}
