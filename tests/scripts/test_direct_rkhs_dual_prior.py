from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_direct_rkhs_dual_prior.py"
SPEC = importlib.util.spec_from_file_location("run_direct_rkhs_dual_prior", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fit_single_group_gate_returns_scalar_alpha_w():
    y_true = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    y_model = np.array([0.8, 2.2, 0.3], dtype=np.float32)
    y_bayesb = np.array([0.7, 2.4, 0.2], dtype=np.float32)
    y_gblup = np.array([1.1, 1.7, 0.6], dtype=np.float32)

    gate = MODULE.fit_single_group_gate(
        y_true=y_true,
        y_model=y_model,
        y_bayesb=y_bayesb,
        y_gblup=y_gblup,
    )

    assert set(gate) == {"alpha", "w"}
    assert 0.0 <= gate["alpha"] <= 1.0
    assert 0.0 <= gate["w"] <= 1.0


def test_apply_dual_prior_gate_blends_direct_model_and_priors():
    y_model = np.array([0.2, 0.8], dtype=np.float32)
    y_bayesb = np.array([0.0, 1.0], dtype=np.float32)
    y_gblup = np.array([1.0, 0.0], dtype=np.float32)

    pred = MODULE.apply_dual_prior_gate(
        y_model=y_model,
        y_bayesb=y_bayesb,
        y_gblup=y_gblup,
        alpha=0.25,
        w=0.6,
    )

    y_prior = 0.25 * y_bayesb + 0.75 * y_gblup
    expected = y_prior + 0.6 * (y_model - y_prior)
    np.testing.assert_allclose(pred, expected, rtol=1e-6, atol=1e-6)
