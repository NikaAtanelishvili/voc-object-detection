"""Collect finished runs into results/: small artifacts per run, one experiments.csv row each.

Usage: uv run python scripts/collect_results.py [run_id ...]   (default: every run in runs/)

For each run, copies config.yaml, run_meta.json, metrics_val.json (as metrics.json) and, if
under 1 MB, train_log.csv into results/runs/<run_id>/. Predictions and weights stay in runs/ (hard
rule 2). experiments.csv is then rebuilt from all of results/runs/, so rerunning is idempotent: one
row per run, sorted by run_id.
"""

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("collect_results")

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("config.yaml", "run_meta.json", "metrics_val.json")
MAX_LOG_BYTES = 1_000_000
METRIC_COLUMNS = ("map50", "map50_95", "map_small", "map_medium", "map_large", "mar_100")
COLUMNS = ("run_id", "exp_id", "git_sha", "seed", "gpu_name", "wall_time_s", *METRIC_COLUMNS)


class IncompleteRunError(RuntimeError):
    """A run folder lacks a file of the run artifact contract (PLAN.md §2)."""


def collect_run(run_dir: Path, results_runs: Path) -> Path:
    """Copy one run's small artifacts; return its results folder."""
    missing = [name for name in REQUIRED if not (run_dir / name).exists()]
    if missing:
        raise IncompleteRunError(f"{run_dir} is missing {missing}")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    if meta.get("status") != "complete":
        raise IncompleteRunError(f"{run_dir}: run_meta status is {meta.get('status')!r}")

    dest = results_runs / run_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "config.yaml", dest / "config.yaml")
    shutil.copyfile(run_dir / "run_meta.json", dest / "run_meta.json")
    shutil.copyfile(run_dir / "metrics_val.json", dest / "metrics.json")
    log = run_dir / "train_log.csv"
    if log.exists():
        if log.stat().st_size <= MAX_LOG_BYTES:
            shutil.copyfile(log, dest / "train_log.csv")
        else:
            logger.warning("%s is over 1 MB; left in runs/ only", log)
    return dest


def experiment_row(result_dir: Path) -> dict:
    meta = json.loads((result_dir / "run_meta.json").read_text())
    metrics = json.loads((result_dir / "metrics.json").read_text())
    row = {
        "run_id": result_dir.name,
        "exp_id": meta["exp_id"],
        "git_sha": meta["git_sha"],
        "seed": meta["seed"],
        "gpu_name": meta.get("gpu_name"),
        "wall_time_s": meta.get("wall_time_s"),
    }
    row.update({name: metrics.get(name) for name in METRIC_COLUMNS})
    return row


def rebuild_experiments_csv(results_runs: Path, csv_path: Path) -> list[dict]:
    rows = [experiment_row(d) for d in sorted(results_runs.iterdir()) if d.is_dir()]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        # lineterminator: csv defaults to "\r\n", which shows up as noise in git diffs.
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def collect(runs_dir: Path, results_dir: Path, run_ids: list[str] | None = None) -> list[dict]:
    """Collect the given runs (default: every run with metrics_val.json); rebuild the CSV."""
    if run_ids is None:
        run_ids = sorted(d.name for d in runs_dir.iterdir() if (d / "metrics_val.json").exists())
    for run_id in run_ids:
        dest = collect_run(runs_dir / run_id, results_dir / "runs")
        logger.info("Collected %s -> %s", run_id, dest)
    return rebuild_experiments_csv(results_dir / "runs", results_dir / "experiments.csv")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_ids", nargs="*", help="Run IDs under runs/ (default: all with metrics)"
    )
    args = parser.parse_args()
    rows = collect(REPO_ROOT / "runs", REPO_ROOT / "results", args.run_ids or None)
    print(f"results/experiments.csv: {len(rows)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
