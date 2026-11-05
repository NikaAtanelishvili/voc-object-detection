"""collect_results.py on fake run folders (PLAN.md P2.5)."""

import csv
import json
from pathlib import Path

import pytest

from collect_results import IncompleteRunError, ManifestError, collect, load_manifest


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
    rows = collect(tmp_path / "runs", tmp_path / "results", {"E1": "E1_20261028-0900_abc1234"})
    dest = tmp_path / "results" / "runs" / "E1_20261028-0900_abc1234"
    assert sorted(p.name for p in dest.iterdir()) == [
        "config.yaml",
        "metrics.json",
        "run_meta.json",
        "train_log.csv",
    ]
    assert rows[0]["map50"] == 0.61
    assert b"\r" not in (tmp_path / "results" / "experiments.csv").read_bytes()
    (row,) = read_csv(tmp_path / "results" / "experiments.csv")
    assert row["exp_id"] == "E1" and row["map50"] == "0.61" and row["map_small"] == ""
    assert row["median_iter_s"] == ""  # this log has no timing column


def test_median_iter_s_comes_from_the_train_log(tmp_path: Path) -> None:
    d = fake_run(tmp_path / "runs", "E1_20261028-0900_abc1234", 0.61)
    # A slow first iteration (CUDA start-up) does not move the median.
    (d / "train_log.csv").write_text("epoch,step,iter_s\n0,0,9.0\n0,1,0.5\n0,2,0.4\n0,3,0.6\n")
    (row,) = collect(tmp_path / "runs", tmp_path / "results", {"E1": d.name})
    assert row["median_iter_s"] == pytest.approx(0.55)


def test_rerun_is_idempotent_and_rows_are_sorted(tmp_path: Path) -> None:
    runs, results = tmp_path / "runs", tmp_path / "results"
    fake_run(runs, "E2_20261028-1000_abc1234", 0.7)
    collect(runs, results, {"E2": "E2_20261028-1000_abc1234"})
    collect(runs, results, {"E2": "E2_20261028-1000_abc1234"})
    fake_run(runs, "E1_20261028-0900_abc1234", 0.6)
    collect(runs, results, {"E2": "E2_20261028-1000_abc1234", "E1": "E1_20261028-0900_abc1234"})
    assert [r["run_id"] for r in read_csv(results / "experiments.csv")] == [
        "E1_20261028-0900_abc1234",
        "E2_20261028-1000_abc1234",
    ]


def test_incomplete_runs_are_refused(tmp_path: Path) -> None:
    d = fake_run(tmp_path / "runs", "E1_x", 0.6)
    (d / "config.yaml").unlink()
    with pytest.raises(IncompleteRunError, match="config.yaml"):
        collect(tmp_path / "runs", tmp_path / "results", {"E1": "E1_x"})
    fake_run(tmp_path / "runs2", "E1_y", 0.6, status="running")
    with pytest.raises(IncompleteRunError, match="running"):
        collect(tmp_path / "runs2", tmp_path / "results", {"E1": "E1_y"})


def test_only_manifest_runs_are_collected(tmp_path: Path) -> None:
    # A superseded run sits next to the canonical one in runs/, as after a rerun.
    runs, results = tmp_path / "runs", tmp_path / "results"
    fake_run(runs, "E1_20261028-0800_old0000", 0.5)
    fake_run(runs, "E1_20261028-0900_abc1234", 0.6)
    (row,) = collect(runs, results, {"E1": "E1_20261028-0900_abc1234"})
    assert row["run_id"] == "E1_20261028-0900_abc1234"
    assert not (results / "runs" / "E1_20261028-0800_old0000").exists()


def test_unlisted_results_folder_stays_out_of_the_csv(tmp_path: Path) -> None:
    runs, results = tmp_path / "runs", tmp_path / "results"
    fake_run(runs, "E1_20261028-0800_old0000", 0.5)
    fake_run(runs, "E1_20261028-0900_abc1234", 0.6)
    collect(runs, results, {"E1": "E1_20261028-0800_old0000"})
    (row,) = collect(runs, results, {"E1": "E1_20261028-0900_abc1234"})  # the run was replaced
    assert row["run_id"] == "E1_20261028-0900_abc1234"


def test_fresh_clone_uses_the_committed_results(tmp_path: Path) -> None:
    # runs/ is gitignored, so a clone has only results/runs/.
    runs, results = tmp_path / "runs", tmp_path / "results"
    fake_run(runs, "E1_20261028-0900_abc1234", 0.6)
    collect(runs, results, {"E1": "E1_20261028-0900_abc1234"})
    (results / "experiments.csv").unlink()
    (row,) = collect(tmp_path / "empty_runs", results, {"E1": "E1_20261028-0900_abc1234"})
    assert row["map50"] == 0.6


def test_manifest_errors_are_specific(tmp_path: Path) -> None:
    runs, results = tmp_path / "runs", tmp_path / "results"
    fake_run(runs, "E2_20261028-1000_abc1234", 0.7)
    with pytest.raises(ManifestError, match="is a run of 'E2'"):
        collect(runs, results, {"E1": "E2_20261028-1000_abc1234"})
    with pytest.raises(ManifestError, match="neither runs/ nor results/runs/"):
        collect(runs, results, {"E3": "E3_missing"})


def test_load_manifest_rejects_non_string_values(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text("E1: E1_20261028-0900_abc1234\n")
    assert load_manifest(good) == {"E1": "E1_20261028-0900_abc1234"}
    bad = tmp_path / "bad.yaml"
    bad.write_text("E1: [a, b]\n")
    with pytest.raises(ValueError):
        load_manifest(bad)
