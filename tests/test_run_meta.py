"""run_meta.py and seed.py: run IDs, metadata lifecycle, and repeatable random streams."""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from vocdet.utils.run_meta import (
    ENV_GIT_SHA,
    ENV_RUN_ID,
    finish_run_meta,
    make_run_id,
    resolve_run_id,
    start_run_meta,
)
from vocdet.utils.seed import seed_everything


def test_make_run_id() -> None:
    assert make_run_id("E1", "0123456789", datetime(2026, 1, 2, 3, 4)) == "E1_20260102-0304_0123456"


def test_kaggle_environment_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_RUN_ID, "E1_20261028-0900_abc1234")
    assert resolve_run_id("E1") == "E1_20261028-0900_abc1234"


def test_meta_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_GIT_SHA, "f" * 40)
    meta = start_run_meta(tmp_path, "E1_x", "E1", seed=42)
    on_disk = json.loads((tmp_path / "run_meta.json").read_text())
    assert on_disk["status"] == "running" and on_disk["git_sha"] == "f" * 40
    assert on_disk["packages"]["torch"] is not None
    final = finish_run_meta(tmp_path, meta)
    assert final["status"] == "complete" and final["wall_time_s"] >= 0
    assert json.loads((tmp_path / "run_meta.json").read_text()) == final


def test_seed_everything_repeats_all_streams() -> None:
    def draw() -> tuple:
        return random.random(), float(np.random.rand()), float(torch.rand(1))

    seed_everything(42)
    first = draw()
    seed_everything(42)
    assert draw() == first
