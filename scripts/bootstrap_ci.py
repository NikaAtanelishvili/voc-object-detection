"""Paired bootstrap on val: a 95% CI per run and for the difference between runs (P5.1, ADR-014).

Usage: uv run python scripts/bootstrap_ci.py [--iterations 200] [--seed 0] [--workers 4] [--force]

Val has only 572 images, and two models scored on the same images share most of their noise: an easy
resample lifts both. So overlap of two separate CIs says little about their difference. Instead,
every iteration resamples val *images* with replacement, the draw depends only on the seed, and
every run is re-scored on the same draws with the one evaluator (ADR-008). The difference of two
runs per iteration then gives the CI of the delta (2.5th-97.5th percentile).

Per-iteration scores are saved in results/runs/<run_id>/bootstrap_ci.json, so a new run costs only
its own scoring. Pairs go to results/paired_deltas.csv. An image drawn twice is entered twice under
distinct IDs, because the evaluator rejects duplicate IDs. Runs on a subset of val are skipped.
"""

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from collect_results import MANIFEST, load_manifest
from vocdet.data.splits import load_splits
from vocdet.eval.evaluator import evaluate
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
    ("E5", "E2"),
    ("E0a_yolo_timing_s", "E3"),
    ("E0a_yolo_timing_s", "E0b"),
    ("E0a_yolo_timing_n", "E2"),
)


def resample_indices(num_images: int, iterations: int, seed: int) -> np.ndarray:
    """(iterations, num_images) image indices. Same arguments, same draws: this is the pairing."""
    return np.random.default_rng(seed).integers(0, num_images, size=(iterations, num_images))


def score_resample(
    predictions: list[Prediction], ground_truth: list[GroundTruth], picks: np.ndarray
) -> dict[str, float]:
    """The evaluator's METRICS on one resample. `predictions` follows the order of ground_truth."""
    sample_gt, sample_pred = [], []
    for k, index in enumerate(picks):
        g, p = ground_truth[index], predictions[index]
        sample_gt.append(GroundTruth(f"{g.image_id}#{k}", g.boxes, g.labels, g.difficult))
        sample_pred.append(Prediction(f"{g.image_id}#{k}", p.boxes, p.scores, p.labels))
    metrics = evaluate(sample_pred, sample_gt)
    return {name: metrics[name] for name in METRICS}


def bootstrap_scores(
    predictions: list[Prediction],
    ground_truth: list[GroundTruth],
    indices: np.ndarray,
    workers: int = 1,
) -> dict[str, list[float]]:
    """Per-iteration values of each metric, in draw order."""
    by_id = {p.image_id: p for p in predictions}
    ordered = [by_id[g.image_id] for g in ground_truth]
    arguments = [(ordered, ground_truth, picks) for picks in indices]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(score_resample, *zip(*arguments, strict=True)))
    else:
        scores = [score_resample(*a) for a in arguments]
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
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--force", action="store_true", help="Recompute existing scores.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "voc_prepared")
    args = parser.parse_args()

    val_ids = load_splits(REPO_ROOT / "splits")["val"]
    annotations = pd.read_parquet(args.data_dir / "annotations.parquet")
    val_gt = ground_truth_from_annotations(annotations, val_ids)
    settings = {"iterations": args.iterations, "seed": args.seed, "num_images": len(val_gt)}
    indices = resample_indices(len(val_gt), args.iterations, args.seed)

    results = {}
    for exp_id, run_id in load_manifest(MANIFEST).items():
        out = REPO_ROOT / "results" / "runs" / run_id / "bootstrap_ci.json"
        saved = json.loads(out.read_text()) if out.exists() else {}
        if not args.force and saved.get("settings") == settings and "values" in saved:
            logger.info("%s: up to date", run_id)
            results[exp_id] = saved
            continue
        predictions = read_predictions(REPO_ROOT / "runs" / run_id / "predictions_val.json")
        if len(predictions) != len(val_gt):
            logger.info(
                "%s: covers %d of %d val images; skipped", run_id, len(predictions), len(val_gt)
            )
            continue
        values = bootstrap_scores(predictions, val_gt, indices, args.workers)
        full = evaluate(predictions, val_gt)
        result = {
            "run_id": run_id,
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
        for a, b in PAIRS
        if a in results and b in results
        for metric in METRICS
    ]
    deltas_csv = REPO_ROOT / "results" / "paired_deltas.csv"
    with deltas_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("%s: %d rows", deltas_csv, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
