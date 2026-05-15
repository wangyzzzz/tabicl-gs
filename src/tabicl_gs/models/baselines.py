from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BaselineRunResult:
    predictions: np.ndarray
    metadata: dict
    command: list[str]
    beta: np.ndarray | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_r_baseline(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    output_dir: str | Path,
    rscript_path: str,
    seed: int,
    sommer_method: str | None = None,
    keep_artifacts: bool = True,
    return_beta: bool = False,
    bandwidth_scale: float | None = None,
) -> BaselineRunResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _run_with_paths(work_dir: Path) -> BaselineRunResult:
        train_x_path = work_dir / "train_x.csv"
        train_y_path = work_dir / "train_y.csv"
        test_x_path = work_dir / "test_x.csv"
        pred_out = work_dir / "predictions.csv"
        meta_out = work_dir / "metadata.json"
        beta_out = work_dir / "beta.csv"

        np.savetxt(train_x_path, X_train, delimiter=",", fmt="%.6f")
        np.savetxt(train_y_path, y_train.reshape(-1, 1), delimiter=",", fmt="%.6f")
        np.savetxt(test_x_path, X_test, delimiter=",", fmt="%.6f")

        command = [
            rscript_path,
            str(_project_root() / "r" / "run_baseline.R"),
            "--model",
            model_name,
            "--train-x",
            str(train_x_path),
            "--train-y",
            str(train_y_path),
            "--test-x",
            str(test_x_path),
            "--pred-out",
            str(pred_out),
            "--meta-out",
            str(meta_out),
            "--seed",
            str(seed),
        ]
        if sommer_method:
            command.extend(["--sommer-method", sommer_method])
        if bandwidth_scale is not None:
            command.extend(["--bandwidth-scale", str(float(bandwidth_scale))])
        if return_beta:
            command.extend(["--beta-out", str(beta_out)])
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        if keep_artifacts:
            if completed.stdout:
                (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            if completed.stderr:
                (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        predictions = np.loadtxt(pred_out, delimiter=",", dtype=np.float32)
        if predictions.ndim == 0:
            predictions = predictions.reshape(1)
        metadata = json.loads(meta_out.read_text(encoding="utf-8"))
        beta = None
        if return_beta and beta_out.exists():
            beta = np.loadtxt(beta_out, delimiter=",", dtype=np.float32)
            beta = np.asarray(beta, dtype=np.float32).reshape(-1)
        return BaselineRunResult(predictions=predictions, metadata=metadata, command=command, beta=beta)

    if keep_artifacts:
        return _run_with_paths(output_dir)

    with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
        return _run_with_paths(Path(temp_dir))
