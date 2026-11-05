"""Paired bootstrap: a 95% CI per run and for the difference between runs (P5.1, ADR-014).

Usage: uv run python scripts/bootstrap_ci.py [--split val|test] [--iterations 1000] [--seed 0]
       [--force]

Val has only 572 images, and two models scored on the same images share most of their noise: an easy
resample lifts both. So overlap of two separate CIs says little about their difference. Instead,
every iteration resamples *images* with replacement, the draw depends only on the seed, and every
run is re-scored on the same draws. The difference of two runs per iteration then gives the CI of
the delta (2.5th-97.5th percentile). Scoring uses vocdet.eval.resample, which runs the evaluator's
pycocotools matching once per run and only the accumulation per draw (equal to evaluate() on the
resampled set; ~100x faster), so 1000 draws are affordable on test too.

--split val: every canonical run, PAIRS, results/runs/<run_id>/bootstrap_ci.json and
results/paired_deltas.csv. --split test: the runs recorded in results/test_metrics.csv by
final_test_eval.py (their saved predictions; no new test evaluation), TEST_PAIRS,
bootstrap_ci_test.json and results/paired_deltas_test.csv. Runs on part of the split are skipped.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from collect_results import MANIFEST, load_manifest
from vocdet.data.splits import load_splits
from vocdet.eval.evaluator import evaluate
from vocdet.eval.resample import MatchedImages
from vocdet.eval.schema import (
    GroundTruth,
    Prediction,
    ground_truth_from_annotations,
    read_predictions,
)

logger = logging.getLogger("bootstrap_ci")

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("map50", "map50_95")
# (a, b) by experiment ID: the table reports a - b. Pairs whose runs do not exist yet are left out.
PAIRS = (
    ("E2", "E0b"),
    ("E3", "E0b"),
    ("E2", "E1"),
    ("E3", "E2"),
    ("E4", "E2"),
    ("E4", "E0b"),
    ("E5", "E2"),
    ("E5", "E0b"),
    ("E6s", "E3"),
    ("E6s", "E0b"),
    ("E6s", "E2"),
    ("E6n", "E2"),
    ("E6n", "E0b"),
    ("E6s", "E6n"),
    ("E0a_yolo_timing_s", "E3"),
    ("E0a_yolo_timing_s", "E0b"),
    ("E0a_yolo_timing_n", "E2"),
)
# Test pairs (owner, 2026-11-06): the deployment candidate against the alternatives.
TEST_PAIRS = (("E6s", "E0b"), ("E6s", "E3"), ("E6s", "E6n"), ("E6s", "E1"))


def resample_indices(num_images: int, iterations: int, seed: int) -> np.ndarray:
    """(iterations, num_images) image indices. Same arguments, same draws: this is the pairing."""
    return np.random.default_rng(seed).integers(0, num_images, size=(iterations, num_images))


def bootstrap_scores(
    predictions: list[Prediction], ground_truth: list[GroundTruth], indices: np.ndarray
) -> dict[str, list[float]]:
    """Per-iteration values of each metric, in draw order; images indexed in ground_truth order."""
    by_id = {p.image_id: p for p in predictions}
    matched = MatchedImages([by_id[g.image_id] for g in ground_truth], ground_truth)
    scores = [matched.score(picks) for picks in indices]
    return {name: [s[name] for s in scores] for name in METRICS}


def confidence_interval(values: list[float]) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def delta_row(a: dict, b: dict, metric: str) -> dict:
    """a - b on one metric: full-val point estimate and the paired 95% CI."""
    if a["settings"] != b["settings"]:
        raise ValueError(f"Not paired: {a['run_id']} and {b['run_id']} used different settings")
    deltas = np.array(a["values"][metric]) - np.array(b["values"][metric])
    low, high = confidence_interval(deltas.tolist())
    return {
        "a": a["run_id"],
        "b": b["run_id"],
        "metric": metric,
        "delta": a["point"][metric] - b["point"][metric],
        "ci95_low": low,
        "ci95_high": high,
        "excludes_zero": low > 0 or high < 0,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Recompute existing scores.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "voc_prepared")
    args = parser.parse_args()

    ids = load_splits(REPO_ROOT / "splits")[args.split]
    annotations = pd.read_parquet(args.data_dir / "annotations.parquet")
    gt = ground_truth_from_annotations(annotations, ids)
    settings = {"iterations": args.iterations, "seed": args.seed, "num_images": len(gt)}
    indices = resample_indices(len(gt), args.iterations, args.seed)
    if args.split == "val":
        runs, pairs, suffix = load_manifest(MANIFEST), PAIRS, ""
    else:
        with (REPO_ROOT / "results" / "test_metrics.csv").open(newline="") as f:
            runs = {row["exp_id"]: row["run_id"] for row in csv.DictReader(f)}
        pairs, suffix = TEST_PAIRS, "_test"

    results = {}
    for exp_id, run_id in runs.items():
        out = REPO_ROOT / "results" / "runs" / run_id / f"bootstrap_ci{suffix}.json"
        saved = json.loads(out.read_text()) if out.exists() else {}
        if not args.force and saved.get("settings") == settings and "values" in saved:
            logger.info("%s: up to date", run_id)
            results[exp_id] = saved
            continue
        source = REPO_ROOT / "runs" / run_id / f"predictions_{args.split}.json"
        predictions = read_predictions(source)
        if len(predictions) != len(gt):
            logger.info("%s: covers %d of %d images; skipped", run_id, len(predictions), len(gt))
            continue
        values = bootstrap_scores(predictions, gt, indices)
        full = evaluate(predictions, gt)
        result = {
            "run_id": run_id,
            "split": args.split,
            "settings": settings,
            "point": {name: full[name] for name in METRICS},
            "ci95": {name: confidence_interval(values[name]) for name in METRICS},
            "values": values,
        }
        out.write_text(json.dumps(result, indent=2) + "\n")
        logger.info("%s: map50 CI95 %s", run_id, result["ci95"]["map50"])
        results[exp_id] = result

    rows = [
        delta_row(results[a], results[b], metric)
        for a, b in pairs
        if a in results and b in results
        for metric in METRICS
    ]
    deltas_csv = REPO_ROOT / "results" / f"paired_deltas{suffix}.csv"
    with deltas_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("%s: %d rows", deltas_csv, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
