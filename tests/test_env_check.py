"""env_check.py must work without torch or a GPU (it does locally and in CI)."""

import json
from pathlib import Path

import env_check


def test_env_check_writes_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["env_check.py", "--out", str(tmp_path / "out")])
    assert env_check.main() == 0
    env = json.loads((tmp_path / "out" / "env.json").read_text())
    assert env["python"]
    assert "torch" in env["packages"]
    assert isinstance(env["kaggle_input"], list)
