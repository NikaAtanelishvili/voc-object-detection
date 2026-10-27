"""kaggle_run.py: git-state guard, run IDs, kernel metadata, and the rendered runner script."""

import ast
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from kaggle_run import (
    GitStateError,
    build_metadata,
    ensure_clean_and_pushed,
    https_repo_url,
    kernel_slug,
    make_run_id,
    render_runner,
)
from vocdet.config import JobConfig

SHA = "0123456789abcdef0123456789abcdef01234567"


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo_with_origin(tmp_path: Path) -> Path:
    """A clone of a local bare repo, with one pushed commit."""
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "--quiet", "--bare", "-b", "main", str(origin))
    repo = tmp_path / "work"
    git(tmp_path, "clone", "--quiet", str(origin), str(repo))
    (repo / "a.txt").write_text("a")
    git(repo, "add", "a.txt")
    git(repo, "commit", "--quiet", "-m", "first")
    git(repo, "push", "--quiet", "origin", "HEAD:main")
    return repo


def test_clean_pushed_repo_returns_sha(repo_with_origin: Path) -> None:
    sha = ensure_clean_and_pushed(repo_with_origin)
    assert len(sha) == 40


def test_dirty_tree_is_refused(repo_with_origin: Path) -> None:
    (repo_with_origin / "untracked.txt").write_text("x")
    with pytest.raises(GitStateError, match="dirty"):
        ensure_clean_and_pushed(repo_with_origin)


def test_unpushed_commit_is_refused(repo_with_origin: Path) -> None:
    (repo_with_origin / "a.txt").write_text("changed")
    git(repo_with_origin, "commit", "--quiet", "-am", "local only")
    with pytest.raises(GitStateError, match="git push"):
        ensure_clean_and_pushed(repo_with_origin)


@pytest.mark.parametrize(
    "remote",
    [
        "git@github.com:user/repo.git",
        "git@github.com:user/repo",
        "https://github.com/user/repo.git",
    ],
)
def test_https_repo_url(remote: str) -> None:
    assert https_repo_url(remote) == "https://github.com/user/repo.git"


def test_https_repo_url_rejects_unknown_hosts() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        https_repo_url("ssh://example.com/repo.git")


def test_run_id_follows_contract() -> None:
    assert make_run_id("E1", SHA, datetime(2026, 10, 28, 9, 5)) == "E1_20261028-0905_0123456"


def test_kernel_slug_is_kaggle_safe() -> None:
    assert kernel_slug("env_check") == "vocdet-env-check"
    assert kernel_slug("E0a") == "vocdet-e0a"


def test_metadata_gpu_and_sources() -> None:
    meta = build_metadata("me", "E1", gpu=True, sources=["vocdet-prepare-data", "other/kernel"])
    assert meta["id"] == "me/vocdet-e1"
    assert meta["title"] == "vocdet-e1"
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    assert meta["is_private"] is True
    assert meta["kernel_type"] == "script"
    assert meta["kernel_sources"] == ["me/vocdet-prepare-data", "other/kernel"]


def test_rendered_runner_is_valid_python_with_values() -> None:
    cfg = JobConfig.model_validate(
        {"exp_id": "E1", "kaggle": {"command": "python x.py --out {run_dir}", "pip": ["a==1"]}}
    )
    source = render_runner("https://github.com/u/r.git", SHA, "E1_run", cfg, gpu=False)
    tree = ast.parse(source)
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.isupper()
        and isinstance(node.value, (ast.Constant, ast.List))
    }
    assert constants["COMMIT_SHA"] == SHA
    assert constants["RUN_ID"] == "E1_run"
    assert constants["COMMAND"] == "python x.py --out /kaggle/working/E1_run"
    assert constants["PIP_PACKAGES"] == ["a==1"]
    assert constants["GPU_REQUESTED"] is False
