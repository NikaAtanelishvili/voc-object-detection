"""Push one config as a Kaggle kernel that runs the repo at the current, pushed commit.

Usage: uv run python scripts/kaggle_run.py --config configs/kaggle/env_check.yaml [--gpu|--cpu]
       [--sources slug1,slug2] [--dry-run]

Refuses a dirty tree or an unpushed HEAD (CLAUDE.md hard rule 6): the kernel clones the public repo
at HEAD's SHA, so local-only changes would silently not run. The kernel folder is written to
kaggle/kernels/<run_id>/ (gitignored); scripts/kaggle_pull.py reads it back to find the kernel.
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from vocdet.config import JobConfig, load_job_config
from vocdet.utils.run_meta import make_run_id

logger = logging.getLogger("kaggle_run")

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "kaggle" / "templates"
KERNELS_DIR = REPO_ROOT / "kaggle" / "kernels"
# T4 x2. Never P100 (sm_60): recent PyTorch wheels may not support it (PLAN.md §9).
GPU_ACCELERATOR = "NvidiaTeslaT4"


class GitStateError(RuntimeError):
    """The local repo is not in a state Kaggle can reproduce."""


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def ensure_clean_and_pushed(repo: Path) -> str:
    """Return HEAD's SHA, or raise if the tree is dirty or HEAD is on no remote branch."""
    status = git(repo, "status", "--porcelain")
    if status:
        raise GitStateError(f"Working tree is dirty; commit or stash first:\n{status}")
    git(repo, "fetch", "--quiet", "origin")
    if not git(repo, "branch", "--remotes", "--contains", "HEAD"):
        raise GitStateError("HEAD is not on any remote branch; run `git push` first.")
    return git(repo, "rev-parse", "HEAD")


def https_repo_url(remote_url: str) -> str:
    """Kaggle clones anonymously over HTTPS, so convert an SSH remote to its HTTPS form."""
    match = re.fullmatch(r"git@github\.com:(.+?)(\.git)?", remote_url)
    if match:
        return f"https://github.com/{match.group(1)}.git"
    if remote_url.startswith("https://"):
        return remote_url
    raise ValueError(f"Unsupported remote URL (expected GitHub SSH or HTTPS): {remote_url}")


def kernel_slug(exp_id: str) -> str:
    """One Kaggle kernel per experiment; each launch is a new version of it."""
    return "vocdet-" + exp_id.lower().replace("_", "-")


def build_metadata(username: str, exp_id: str, gpu: bool, sources: list[str]) -> dict:
    """kernel-metadata.json fields as documented by the installed Kaggle CLI (2.2.5)."""
    slug = kernel_slug(exp_id)
    return {
        "id": f"{username}/{slug}",
        # The CLI warns unless the title slugifies to the id's slug, so use the slug itself.
        "title": slug,
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [s if "/" in s else f"{username}/{s}" for s in sources],
        "model_sources": [],
    }


def render_runner(repo_url: str, commit_sha: str, run_id: str, cfg: JobConfig, gpu: bool) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), undefined=StrictUndefined)
    return env.get_template("runner.py.j2").render(
        repo_url=repo_url,
        commit_sha=commit_sha,
        run_id=run_id,
        command=cfg.kaggle.command.format(run_dir=f"/kaggle/working/{run_id}"),
        pip=cfg.kaggle.pip,
        gpu=gpu,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    accel = parser.add_mutually_exclusive_group()
    accel.add_argument("--gpu", dest="gpu", action="store_true", default=None)
    accel.add_argument("--cpu", dest="gpu", action="store_false")
    parser.add_argument("--sources", help="Comma-separated kernel slugs; overrides the config.")
    parser.add_argument("--dry-run", action="store_true", help="Write the kernel folder, no push.")
    args = parser.parse_args()

    cfg = load_job_config(args.config)
    gpu = cfg.kaggle.gpu if args.gpu is None else args.gpu
    sources = args.sources.split(",") if args.sources else cfg.kaggle.sources
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        raise SystemExit("KAGGLE_USERNAME is not set; put it in .env (see .env.example).")

    commit_sha = ensure_clean_and_pushed(REPO_ROOT)
    repo_url = https_repo_url(git(REPO_ROOT, "remote", "get-url", "origin"))
    run_id = make_run_id(cfg.exp_id, commit_sha, datetime.now())

    kernel_dir = KERNELS_DIR / run_id
    kernel_dir.mkdir(parents=True, exist_ok=False)
    (kernel_dir / "runner.py").write_text(
        render_runner(repo_url, commit_sha, run_id, cfg, gpu), encoding="utf-8"
    )
    # The config travels with the kernel folder: provenance, and kaggle_pull reads pull_pattern.
    shutil.copyfile(args.config, kernel_dir / "config.yaml")
    metadata = build_metadata(username, cfg.exp_id, gpu, sources)
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Kernel folder: %s (commit %s, gpu=%s)", kernel_dir, commit_sha[:7], gpu)

    if args.dry_run:
        logger.info("Dry run: not pushing.")
        return 0
    push = ["kaggle", "kernels", "push", "--path", str(kernel_dir)]
    if gpu:
        push += ["--accelerator", GPU_ACCELERATOR]
    subprocess.run(push, check=True)
    print(f"Pushed {metadata['id']} as run {run_id}. Pull with: make kaggle-pull RUN={run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
