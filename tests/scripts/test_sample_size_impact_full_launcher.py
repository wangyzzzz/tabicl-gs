from __future__ import annotations

import re
from pathlib import Path


def test_full_sample_size_launcher_is_fold1_only_screening() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_sample_size_impact_full_server_gpu1.sh"
    text = script.read_text(encoding="utf-8")

    assert re.search(r"^FOLD_IDS=\(1\)$", text, flags=re.MULTILINE)
    assert '--fold-ids "${FOLD_IDS[@]}"' in text
    assert "--fold-ids 1 2 3 4 5" not in text


def test_full_sample_size_launcher_skips_completed_rice529_screening_traits() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_sample_size_impact_full_server_gpu1.sh"
    text = script.read_text(encoding="utf-8")

    assert 'DATASETS=("soybean951" "pig3534" "wheat406")' in text
    assert 'DATASETS=("rice529"' not in text
    assert 'DATASETS=("cotton1245"' not in text
    assert 'order = "${DATASETS[*]}".split()' in text
    assert 'order = ["rice529"' not in text


def test_full_sample_size_launcher_only_runs_20_and_100_percent() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_sample_size_impact_full_server_gpu1.sh"
    text = script.read_text(encoding="utf-8")

    assert "PROPORTIONS=(0.2 1.0)" in text
    assert "BLOCK_SEARCH_PROPORTIONS=(0.2 1.0)" in text
    assert "0.4 0.6 0.8" not in text
