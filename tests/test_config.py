"""Config validation: bad configs must fail locally, not after a Kaggle session starts."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from vocdet.config import load_job_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(text)
    return path


def test_env_check_config_is_valid() -> None:
    cfg = load_job_config(REPO_ROOT / "configs" / "kaggle" / "env_check.yaml")
    assert cfg.exp_id == "env_check"
    assert "{run_dir}" in cfg.kaggle.command
    assert cfg.kaggle.gpu is True


def test_defaults(tmp_path: Path) -> None:
    cfg = load_job_config(write(tmp_path, "exp_id: E1\nkaggle:\n  command: echo hi\n"))
    assert cfg.kaggle.gpu is True
    assert cfg.kaggle.pip == []
    assert cfg.kaggle.sources == []


def test_typo_in_kaggle_section_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="gpus"):
        load_job_config(write(tmp_path, "exp_id: E1\nkaggle:\n  command: x\n  gpus: true\n"))


def test_exp_id_must_be_slug_safe(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="exp_id"):
        load_job_config(write(tmp_path, "exp_id: E 1/x\nkaggle:\n  command: x\n"))


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping"):
        load_job_config(write(tmp_path, "- just\n- a list\n"))
