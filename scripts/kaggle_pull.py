"""Wait for a Kaggle kernel launched by kaggle_run.py to finish, then download its outputs.

Usage: uv run python scripts/kaggle_pull.py <run_id> [--poll-s 60]

Outputs land in runs/<run_id>/ (the runner writes everything under /kaggle/working/<run_id>/, and
`kaggle kernels output` keeps that relative path). The kernel's own log is moved in as kernel.log.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from vocdet.config import load_job_config

logger = logging.getLogger("kaggle_pull")

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNELS_DIR = REPO_ROOT / "kaggle" / "kernels"
RUNS_DIR = REPO_ROOT / "runs"
RUNNING_STATES = {"QUEUED", "RUNNING", "NEW_SCRIPT"}
MAX_PAGE_SIZE = 200  # the CLI's documented maximum for `kernels output --page-size`
RETRY_PAUSE_S = 120


def parse_status(cli_output: str) -> str:
    """Extract e.g. COMPLETE from `user/slug has status "KernelWorkerStatus.COMPLETE"`."""
    match = re.search(r'has status "([^"]+)"', cli_output)
    if not match:
        raise ValueError(f"Unexpected `kaggle kernels status` output: {cli_output!r}")
    return match.group(1).rsplit(".", 1)[-1].upper()


def wait_for_kernel(kernel_id: str, poll_s: int) -> str:
    """Poll until the kernel leaves the queued/running states; return the final state."""
    while True:
        out = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id], check=True, capture_output=True, text=True
        ).stdout
        status = parse_status(out)
        if status not in RUNNING_STATES:
            return status
        logger.info("%s is %s; checking again in %ds", kernel_id, status, poll_s)
        time.sleep(poll_s)


def download_outputs(kernel_id: str, pull_pattern: str | None) -> None:
    """Download kernel outputs into runs/, retrying once after a pause.

    The CLI lists every output file page by page and filters by pattern on our side, so a kernel
    with ~23k outputs (the prep kernel's images and labels) can hit Kaggle's rate limit (HTTP 429).
    The largest page size cuts the number of requests; the retry skips files already downloaded.
    """
    command = ["kaggle", "kernels", "output", kernel_id, "--path", str(RUNS_DIR)]
    command += ["--page-size", str(MAX_PAGE_SIZE)]
    if pull_pattern:
        command += ["--file-pattern", pull_pattern]
    if subprocess.run(command).returncode == 0:
        return
    logger.warning("Download failed (often HTTP 429 rate limit); retrying in %ds", RETRY_PAUSE_S)
    time.sleep(RETRY_PAUSE_S)
    subprocess.run(command, check=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument("--poll-s", type=int, default=60)
    args = parser.parse_args()

    metadata_path = KERNELS_DIR / args.run_id / "kernel-metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"No kernel folder for run {args.run_id} at {metadata_path.parent}")
    kernel_id = json.loads(metadata_path.read_text())["id"]
    pull_pattern = load_job_config(metadata_path.parent / "config.yaml").kaggle.pull_pattern

    status = wait_for_kernel(kernel_id, args.poll_s)
    logger.info("%s finished with status %s; downloading outputs", kernel_id, status)
    RUNS_DIR.mkdir(exist_ok=True)
    download_outputs(kernel_id, pull_pattern)

    run_dir = RUNS_DIR / args.run_id
    # Outputs belong to the kernel's latest version; if a newer run was pushed to the same kernel,
    # its folder name will not match ours, and we must not mistake it for this run.
    if not (run_dir / "kaggle_run.json").exists():
        raise SystemExit(
            f"{run_dir}/kaggle_run.json missing: the kernel failed before finishing, or a newer "
            f"version of {kernel_id} replaced this run. Check runs/ and the Kaggle UI."
        )
    kernel_log = RUNS_DIR / f"{kernel_id.split('/')[1]}.log"
    if kernel_log.exists():
        kernel_log.replace(run_dir / "kernel.log")
    print(f"Run {args.run_id}: status {status}, outputs in {run_dir}")
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
