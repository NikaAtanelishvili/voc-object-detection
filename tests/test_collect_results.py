"""collect_results.py on fake run folders (PLAN.md P2.5)."""

import csv
import json
from pathlib import Path

import pytest

from collect_results import IncompleteRunError, collect


def fake_run(runs: Path, run_id: str, map50: float, status: str = "complete") -> Path:
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "config.yaml").write_text("exp_id: E1\n")
    meta = {"exp_id": run_id.split("_")[0], "git_sha": "abc", "seed": 42, "status": status}
    meta |= {"gpu_name": "Tesla T4", "wall_time_s": 12.5}
    (d / "run_meta.json").write_text(json.dumps(meta))
    (d / "metrics_val.json").write_text(json.dumps({"map50": map50, "map50_95": 0.3}))
    (d / "train_log.csv").write_text("epoch,loss\n1,0.5\n")
    (d / "predictions_val.json").write_text("[]")  # must NOT be copied
    return d


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def test_collect_copies_small_files_and_writes_row(tmp_path: Path) -> None:
    fake_run(tmp_path / "runs", "E1_20261028-0900_abc1234", 0.61)
    rows = collect(tmp_path / "runs", tmp_path / "results")
    dest = tmp_path / "results" / "runs" / "E1_20261028-0900_abc1234"
    assert sorted(p.name for p in dest.iterdir()) == [
        "config.yaml",
        "metrics.json",
        "run_meta.json",
        "train_log.csv",
    ]
    assert rows[0]["map50"] == 0.61
    (row,) = read_csv(tmp_path / "results" / "experiments.csv")
    assert row["exp_id"] == "E1" and row["map50"] == "0.61" and row["map_small"] == ""


def test_rerun_is_idempotent_and_rows_are_sorted(tmp_path: Path) -> None:
    runs, results = tmp_path / "runs", tmp_path / "results"
    fake_run(runs, "E2_20261028-1000_abc1234", 0.7)
    collect(runs, results)
    collect(runs, results)
    fake_run(runs, "E1_20261028-0900_abc1234", 0.6)
    collect(runs, results)
    assert [r["run_id"] for r in read_csv(results / "experiments.csv")] == [
        "E1_20261028-0900_abc1234",
        "E2_20261028-1000_abc1234",
    ]


def test_incomplete_runs_are_refused(tmp_path: Path) -> None:
    d = fake_run(tmp_path / "runs", "E1_x", 0.6)
    (d / "config.yaml").unlink()
    with pytest.raises(IncompleteRunError, match="config.yaml"):
        collect(tmp_path / "runs", tmp_path / "results")
    fake_run(tmp_path / "runs2", "E1_y", 0.6, status="running")
    with pytest.raises(IncompleteRunError, match="running"):
        collect(tmp_path / "runs2", tmp_path / "results")
