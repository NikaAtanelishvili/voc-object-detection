"""final_test_eval.py: test results are recorded once per model; a rerun needs a logged reason."""

import json
from pathlib import Path

import pytest

import final_test_eval as fte


def fake_predict_output(root: Path, exp_id: str, run_id: str, map50: float) -> None:
    out = root / exp_id
    out.mkdir(parents=True)
    info = {"exp_id": exp_id, "run_id": run_id, "weights": "w.pt", "weights_sha256": "abc"}
    (out / "model.json").write_text(json.dumps(info))
    metrics = dict.fromkeys(fte.METRIC_COLUMNS, 0.5) | {"map50": map50}
    (out / "metrics_test.json").write_text(json.dumps(metrics))
    (out / "predictions_test.json").write_text("[]")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the script's results/ and runs/ at a temporary repo."""
    monkeypatch.setattr(fte, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fte, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(fte, "TEST_CSV", tmp_path / "results" / "test_metrics.csv")
    monkeypatch.setattr(fte, "LOG_CSV", tmp_path / "results" / "test_eval_log.csv")
    (tmp_path / "results" / "runs" / "E6s_run").mkdir(parents=True)
    return tmp_path


def record(source: Path, reason: str | None = None) -> int:
    args = type("Args", (), {"source": source, "force_rerun_with_reason": reason})()
    return fte.run_record(args)


def test_record_writes_one_row_and_keeps_predictions_out_of_results(repo: Path) -> None:
    fake_predict_output(repo / "pulled", "E6s", "E6s_run", 0.8)
    record(repo / "pulled")
    rows = (repo / "results" / "test_metrics.csv").read_text().splitlines()
    assert rows[0].startswith("exp_id,run_id,weights,weights_sha256,map50")
    assert rows[1].startswith("E6s,E6s_run,w.pt,abc,0.8")
    assert (repo / "results" / "runs" / "E6s_run" / "metrics_test.json").exists()
    assert (repo / "runs" / "E6s_run" / "predictions_test.json").exists()
    assert not (repo / "results" / "runs" / "E6s_run" / "predictions_test.json").exists()


def test_second_record_is_refused_without_a_reason(repo: Path) -> None:
    fake_predict_output(repo / "first", "E6s", "E6s_run", 0.8)
    record(repo / "first")
    fake_predict_output(repo / "second", "E6s", "E6s_run", 0.9)
    with pytest.raises(fte.AlreadyEvaluatedError, match="once per model"):
        record(repo / "second")
    assert "0.8" in (repo / "results" / "test_metrics.csv").read_text()


def test_forced_rerun_is_logged_with_its_reason(repo: Path) -> None:
    fake_predict_output(repo / "first", "E6s", "E6s_run", 0.8)
    record(repo / "first")
    fake_predict_output(repo / "second", "E6s", "E6s_run", 0.9)
    record(repo / "second", reason="kernel image changed mid-run")
    log = (repo / "results" / "test_eval_log.csv").read_text().splitlines()
    assert log[0] == "recorded_utc,exp_id,source,reason"
    assert log[1].endswith("E6s,second,kernel image changed mid-run")
    assert "0.9" in (repo / "results" / "test_metrics.csv").read_text()


def test_predict_refuses_recorded_models_before_doing_any_work(repo: Path) -> None:
    fake_predict_output(repo / "first", "E6s", "E6s_run", 0.8)
    record(repo / "first")
    with pytest.raises(fte.AlreadyEvaluatedError):
        fte.check_once(["E1", "E6s"], reason=None, csv_path=repo / "results" / "test_metrics.csv")
    fte.check_once(["E1"], reason=None, csv_path=repo / "results" / "test_metrics.csv")
