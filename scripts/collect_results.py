"""Collect the canonical runs into results/: small artifacts per run, one experiments.csv row each.

Usage: uv run python scripts/collect_results.py   (reads results/canonical_runs.yaml)

The manifest maps each experiment ID to its one canonical run_id, so superseded runs and path checks
that are still in runs/ never reach the results. For each listed run, copies config.yaml,
run_meta.json, metrics_val.json (as metrics.json) and, if under 1 MB, train_log.csv into
results/runs/<run_id>/. A run already in results/ but not in runs/ (a fresh clone) is used as
committed. Predictions and weights stay in runs/ (hard rule 2). experiments.csv is then rebuilt from
the listed runs only, so rerunning is idempotent: one row per experiment, sorted by run_id.
"""

import argparse
import csv
import json
import logging
import shutil
import statistics
import sys
from pathlib import Path

import yaml
from pydantic import TypeAdapter

logger = logging.getLogger("collect_results")

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "results" / "canonical_runs.yaml"
REQUIRED = ("config.yaml", "run_meta.json", "metrics_val.json")
MAX_LOG_BYTES = 1_000_000
METRIC_COLUMNS = ("map50", "map50_95", "map_small", "map_medium", "map_large", "mar_100")
COLUMNS = (
    "run_id", "exp_id", "git_sha", "seed", "gpu_name", "wall_time_s", "median_iter_s",
    *METRIC_COLUMNS,
)  # fmt: skip


class IncompleteRunError(RuntimeError):
    """A run folder lacks a file of the run artifact contract (PLAN.md §2)."""


class ManifestError(ValueError):
    """canonical_runs.yaml names a run that does not exist or belongs to another experiment."""


def load_manifest(path: Path) -> dict[str, str]:
    """Read the exp_id -> run_id mapping; pydantic rejects anything that is not str -> str."""
    with path.open() as f:
        return TypeAdapter(dict[str, str]).validate_python(yaml.safe_load(f) or {})


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


def median_iter_s(train_log: Path) -> float | None:
    """Median seconds per training iteration, the speed used for GPU estimates (§5).

    The median ignores the slow first iterations (CUDA start-up, the loaders' first batches)
    without choosing how many to drop. None for runs without a training log or timing column.
    """
    if not train_log.exists():
        return None
    with train_log.open(newline="") as f:
        times = [float(row["iter_s"]) for row in csv.DictReader(f) if row.get("iter_s")]
    return round(statistics.median(times), 4) if times else None


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
        "median_iter_s": median_iter_s(result_dir / "train_log.csv"),
    }
    row.update({name: metrics.get(name) for name in METRIC_COLUMNS})
    return row


def rebuild_experiments_csv(result_dirs: list[Path], csv_path: Path) -> list[dict]:
    rows = [experiment_row(d) for d in sorted(result_dirs)]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        # lineterminator: csv defaults to "\r\n", which shows up as noise in git diffs.
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def collect(runs_dir: Path, results_dir: Path, manifest: dict[str, str]) -> list[dict]:
    """Collect every run named in the manifest; rebuild the CSV from exactly those runs."""
    result_dirs = []
    for exp_id, run_id in manifest.items():
        if (runs_dir / run_id).is_dir():
            dest = collect_run(runs_dir / run_id, results_dir / "runs")
            logger.info("Collected %s -> %s", run_id, dest)
        else:
            dest = results_dir / "runs" / run_id
            if not dest.is_dir():
                raise ManifestError(f"{exp_id}: {run_id} is in neither runs/ nor results/runs/")
        meta_exp_id = json.loads((dest / "run_meta.json").read_text())["exp_id"]
        if meta_exp_id != exp_id:
            raise ManifestError(f"{exp_id}: {run_id} is a run of {meta_exp_id!r}")
        result_dirs.append(dest)

    listed = {d.name for d in result_dirs}
    for d in sorted((results_dir / "runs").iterdir()):
        if d.is_dir() and d.name not in listed:
            logger.warning("%s is not in the manifest; left out of experiments.csv", d)
    return rebuild_experiments_csv(result_dirs, results_dir / "experiments.csv")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="exp_id -> run_id YAML")
    args = parser.parse_args()
    rows = collect(REPO_ROOT / "runs", REPO_ROOT / "results", load_manifest(args.manifest))
    print(f"results/experiments.csv: {len(rows)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
