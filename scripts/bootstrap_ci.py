"""Bootstrap 95% confidence interval on val map50 for each canonical run (PLAN.md P5.1).

Usage: uv run python scripts/bootstrap_ci.py [--iterations 200] [--seed 0] [--force]

Val has only 572 images, so a difference of a point or two between runs may be noise. Each
iteration resamples val *images* with replacement and re-scores the saved predictions_val.json with
the one evaluator (ADR-008); the interval is the 2.5th-97.5th percentile of those map50 values. An
image drawn twice is entered twice under distinct IDs, because the evaluator rejects duplicate IDs.
Results go to results/runs/<run_id>/bootstrap_ci.json (predictions stay in runs/, hard rule 2), and
a run that already has one with the same settings is skipped. ~3 s per iteration on CPU.
"""

import argparse
import json
import logging
import sys
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


def bootstrap_map50(
    predictions: list[Prediction], ground_truth: list[GroundTruth], iterations: int, seed: int
) -> list[float]:
    """map50 of `iterations` image resamples with replacement, in order. Same seed, same values."""
    by_id = {p.image_id: p for p in predictions}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        picks = rng.integers(0, len(ground_truth), size=len(ground_truth))
        sample_gt, sample_pred = [], []
        for k, index in enumerate(picks):
            g = ground_truth[index]
            p = by_id[g.image_id]
            sample_gt.append(GroundTruth(f"{g.image_id}#{k}", g.boxes, g.labels, g.difficult))
            sample_pred.append(Prediction(f"{g.image_id}#{k}", p.boxes, p.scores, p.labels))
        values.append(evaluate(sample_pred, sample_gt)["map50"])
    return values


def confidence_interval(values: list[float]) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Recompute existing intervals.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "voc_prepared")
    args = parser.parse_args()

    val_ids = load_splits(REPO_ROOT / "splits")["val"]
    annotations = pd.read_parquet(args.data_dir / "annotations.parquet")
    val_gt = ground_truth_from_annotations(annotations, val_ids)
    settings = {"iterations": args.iterations, "seed": args.seed}

    for run_id in load_manifest(MANIFEST).values():
        out = REPO_ROOT / "results" / "runs" / run_id / "bootstrap_ci.json"
        if not args.force and out.exists() and json.loads(out.read_text())["settings"] == settings:
            logger.info("%s: up to date", run_id)
            continue
        predictions = read_predictions(REPO_ROOT / "runs" / run_id / "predictions_val.json")
        if len(predictions) != len(val_gt):
            logger.info("%s: predictions cover %d of %d val images; skipped (smoke run)",
                        run_id, len(predictions), len(val_gt))  # fmt: skip
            continue
        values = bootstrap_map50(predictions, val_gt, args.iterations, args.seed)
        low, high = confidence_interval(values)
        result = {
            "run_id": run_id,
            "settings": settings,
            "map50": evaluate(predictions, val_gt)["map50"],
            "ci95_low": low,
            "ci95_high": high,
        }
        out.write_text(json.dumps(result, indent=2) + "\n")
        logger.info("%s: map50 CI95 [%.4f, %.4f]", run_id, low, high)
    return 0


if __name__ == "__main__":
    sys.exit(main())
