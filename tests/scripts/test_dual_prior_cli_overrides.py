from pathlib import Path

import scripts.run_dual_prior_fixed_block_folds as cli


def test_run_dual_prior_cli_uses_frozen_gate_for_non_fold1(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(
        cli,
        "load_experiment_config",
        lambda path: {"seed": 2026, "trait_col": "Trait"},
    )
    monkeypatch.setattr(
        cli,
        "deep_update",
        lambda base, override: {**base, **override},
    )

    def fake_regular(*, base_config, fold_id, group_size, output_dir):
        calls.append(("regular", int(fold_id)))
        return {"fold": int(fold_id), "pearson": 0.1}

    def fake_frozen(*, base_config, fold_id, group_size, gate_summary_path, output_dir):
        calls.append(("frozen", int(fold_id)))
        return {"fold": int(fold_id), "pearson": 0.2}

    monkeypatch.setattr(cli, "run_dual_prior_fixed_block_on_fold", fake_regular)
    monkeypatch.setattr(cli, "run_dual_prior_fixed_block_with_frozen_gate_on_fold", fake_frozen)
    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "config": "dummy.yaml",
                "output_root": str(tmp_path / "out"),
                "trait_col": "Heading_date",
                "group_size": 315,
                "fold_ids": [1, 2, 3, 4, 5],
                "frozen_gate_summary_path": str(tmp_path / "gate.json"),
            },
        )(),
    )

    cli.main()

    assert calls == [
        ("regular", 1),
        ("frozen", 2),
        ("frozen", 3),
        ("frozen", 4),
        ("frozen", 5),
    ]


def test_run_dual_prior_cli_can_override_num_groups(monkeypatch, tmp_path: Path):
    captured_configs = []

    monkeypatch.setattr(
        cli,
        "load_experiment_config",
        lambda path: {
            "seed": 2026,
            "trait_col": "Trait",
            "stage2": {"group_shared_gate": {"num_groups": 3}},
        },
    )

    def fake_regular(*, base_config, fold_id, group_size, output_dir):
        captured_configs.append(base_config)
        return {"fold": int(fold_id), "pearson": 0.1}

    monkeypatch.setattr(cli, "run_dual_prior_fixed_block_on_fold", fake_regular)
    monkeypatch.setattr(cli, "run_dual_prior_fixed_block_with_frozen_gate_on_fold", fake_regular)
    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "config": "dummy.yaml",
                "output_root": str(tmp_path / "out"),
                "trait_col": "Yield",
                "group_size": 737,
                "num_groups": 7,
                "fold_ids": [1],
                "frozen_gate_summary_path": None,
            },
        )(),
    )

    cli.main()

    assert captured_configs
    assert captured_configs[0]["stage2"]["group_shared_gate"]["num_groups"] == 7
