from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TwoStageModelSpec:
    name: str
    stage1_backend: str
    stage2_backend: str
    stage1_config: dict[str, Any]
    stage2_config: dict[str, Any]


def _resolve_backend_config(config: dict[str, Any], section: str, backend: str) -> dict[str, Any]:
    section_config = config.get(section, {})
    if not isinstance(section_config, dict):
        raise ValueError(f"{section} config must be a mapping.")
    if backend in section_config and isinstance(section_config[backend], dict):
        return dict(section_config[backend])
    return dict(section_config)


def resolve_two_stage_model_specs(config: dict[str, Any]) -> list[TwoStageModelSpec]:
    raw_specs = config.get("main_models")
    if raw_specs is None:
        return [
            TwoStageModelSpec(
                name="TabICLv2-2stage",
                stage1_backend="tabicl",
                stage2_backend="tabicl",
                stage1_config=_resolve_backend_config(config, "stage1", "tabicl"),
                stage2_config=_resolve_backend_config(config, "stage2", "tabicl"),
            )
        ]

    specs: list[TwoStageModelSpec] = []
    for raw_spec in raw_specs:
        stage1_backend = raw_spec["stage1_backend"]
        stage2_backend = raw_spec["stage2_backend"]
        specs.append(
            TwoStageModelSpec(
                name=raw_spec["name"],
                stage1_backend=stage1_backend,
                stage2_backend=stage2_backend,
                stage1_config=_resolve_backend_config(config, "stage1", stage1_backend),
                stage2_config=_resolve_backend_config(config, "stage2", stage2_backend),
            )
        )
    return specs
