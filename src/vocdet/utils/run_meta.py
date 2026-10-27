"""Run identity and metadata: the run artifact contract from PLAN.md §2.

Every training or evaluation run writes runs/<run_id>/run_meta.json, so every number can be traced
to the code (git SHA), the environment, and the seed that produced it.
"""

import importlib.metadata
import json
import os
import platform
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# Set by the Kaggle runner (kaggle/templates/runner.py.j2) for the command it runs.
ENV_RUN_ID = "VOCDET_RUN_ID"
ENV_GIT_SHA = "VOCDET_GIT_SHA"
RECORDED_PACKAGES = (
    "torch",
    "torchvision",
    "torchmetrics",
    "pycocotools",
    "ultralytics",
    "numpy",
    "vocdet",
)


def make_run_id(exp_id: str, commit_sha: str, now: datetime) -> str:
    """run_id = <exp_id>_<yyyymmdd-hhmm>_<sha7>, the contract from PLAN.md §2."""
    return f"{exp_id}_{now:%Y%m%d-%H%M}_{commit_sha[:7]}"


def git_sha() -> str:
    """Commit being run: set by the Kaggle runner, else read from the local checkout.

    A dirty local tree is marked with a "-dirty" suffix, so results from uncommitted code are
    visibly not reproducible.
    """
    if os.environ.get(ENV_GIT_SHA):
        return os.environ[ENV_GIT_SHA]
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def resolve_run_id(exp_id: str) -> str:
    """The Kaggle runner's run ID if set (so outputs land in the pulled folder), else a new one."""
    return os.environ.get(ENV_RUN_ID) or make_run_id(exp_id, git_sha(), datetime.now())


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_info() -> dict:
    """Interpreter, key package versions, and the accelerator, recorded with every run."""
    import torch  # local import: keeps `vocdet --help` fast

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "gpu_name": gpu,
        "packages": {name: package_version(name) for name in RECORDED_PACKAGES},
    }


def start_run_meta(run_dir: Path, run_id: str, exp_id: str, seed: int) -> dict:
    """Write run_meta.json with status "running" at the start: a crashed run still has metadata."""
    meta = {
        "run_id": run_id,
        "exp_id": exp_id,
        "git_sha": git_sha(),
        "seed": seed,
        "status": "running",
        "start_utc": datetime.now(UTC).isoformat(),
        **environment_info(),
    }
    _write(run_dir, meta)
    return meta


def finish_run_meta(run_dir: Path, meta: dict) -> dict:
    """Add end time and wall time, mark the run complete, and rewrite run_meta.json."""
    end = datetime.now(UTC)
    start = datetime.fromisoformat(meta["start_utc"])
    meta = {
        **meta,
        "status": "complete",
        "end_utc": end.isoformat(),
        "wall_time_s": round((end - start).total_seconds(), 1),
    }
    _write(run_dir, meta)
    return meta


def _write(run_dir: Path, meta: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
