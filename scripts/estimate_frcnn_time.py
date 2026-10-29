"""Estimate a Faster R-CNN run's training time from a timing run, by replaying batch shapes.

Usage: uv run python scripts/estimate_frcnn_time.py --timing-run results/runs/<run_id>
       --config configs/experiments/E1.yaml --annotations data/voc_prepared/annotations.parquet
       --out results/estimates/E1.json

Step time on Kaggle's T4 is bimodal: a step whose padded batch shape has not been seen before costs
seconds more than a repeat (DEVIATIONS 2026-10-29, P3.4). The order of batches is fixed by the
seeded shuffle, so this script replays the training loader on CPU, predicts every batch's padded
shape, checks the prediction against the timing run's train_log.csv, and then prices the target run
as new-shape steps plus repeat steps. Validation and start-up are not included.
"""

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from vocdet.config import FrcnnConfig, load_job_config
from vocdet.data.splits import load_splits

logger = logging.getLogger("estimate_frcnn_time")

SIZE_DIVISIBLE = (
    32  # torchvision GeneralizedRCNNTransform default: batches pad to a multiple of this
)


def resized_hw(height: int, width: int, min_size: int, max_size: int) -> tuple[int, int]:
    """Image size after torchvision's detection resize: short side to min_size, long side capped."""
    scale = min(min_size / min(height, width), max_size / max(height, width))
    return math.floor(height * scale), math.floor(width * scale)


def batch_shapes(
    sizes: list[tuple[int, int]], cfg: FrcnnConfig, epochs: int | None = None
) -> list[tuple[int, int]]:
    """Padded (H, W) of every training batch, in the order train_frcnn sees them.

    Uses a real DataLoader over indices with the same seeded generator as train_frcnn, so the
    shuffle is torch's own, not a re-implementation (worker count does not change the order). Each
    batch pads to the largest resized height and width, rounded up to SIZE_DIVISIBLE.
    """
    t = cfg.train
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        list(range(len(sizes))), batch_size=t.batch_size, shuffle=True, generator=generator
    )
    shapes = []
    for _ in range(epochs or t.epochs):
        for batch in loader:
            hw = [resized_hw(*sizes[i], cfg.model.min_size, cfg.model.max_size) for i in batch]
            shapes.append(
                (
                    math.ceil(max(h for h, _ in hw) / SIZE_DIVISIBLE) * SIZE_DIVISIBLE,
                    math.ceil(max(w for _, w in hw) / SIZE_DIVISIBLE) * SIZE_DIVISIBLE,
                )
            )
    return shapes


def first_occurrences(shapes: list[tuple[int, int]]) -> list[bool]:
    seen: set[tuple[int, int]] = set()
    flags = []
    for shape in shapes:
        flags.append(shape not in seen)
        seen.add(shape)
    return flags


def fit_step_times(iter_s: list[float], is_new: list[bool]) -> dict:
    """Median step time for new and repeat shapes, and how cleanly the two groups separate.

    separation = fastest new-shape step / slowest repeat step. Above 1 means every new-shape step
    was slower than every repeat step, i.e. the shape replay explains the timing completely.
    """
    new = [t for t, n in zip(iter_s, is_new, strict=True) if n]
    repeat = [t for t, n in zip(iter_s, is_new, strict=True) if not n]
    if not new or not repeat:
        raise ValueError("The timing run needs both new-shape and repeat-shape steps to fit.")
    return {
        "new_shape_s": statistics.median(new),
        "repeat_s": statistics.median(repeat),
        "separation": min(new) / max(repeat),
        "num_new": len(new),
        "num_repeat": len(repeat),
    }


def train_sizes(annotations: Path, cfg: FrcnnConfig, splits_dir: Path) -> list[tuple[int, int]]:
    """(height, width) of the run's training images, in split-file order like the dataset."""
    ann = pd.read_parquet(annotations, columns=["image_id", "width", "height"])
    size = ann.groupby("image_id")[["height", "width"]].first()
    ids = load_splits(splits_dir)["train"][: cfg.train.limit_train_images]
    return [(int(size.at[i, "height"]), int(size.at[i, "width"])) for i in ids]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timing-run", type=Path, required=True, help="results/runs/<run_id>")
    parser.add_argument("--config", type=Path, required=True, help="Config of the run to price.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    timing_cfg = load_job_config(args.timing_run / "config.yaml", FrcnnConfig)
    with (args.timing_run / "train_log.csv").open(newline="") as f:
        iter_s = [float(row["iter_s"]) for row in csv.DictReader(f)]
    timing_shapes = batch_shapes(
        train_sizes(args.annotations, timing_cfg, args.splits_dir), timing_cfg
    )
    if len(timing_shapes) != len(iter_s):
        raise ValueError(f"Replayed {len(timing_shapes)} steps but the log has {len(iter_s)}.")
    fit = fit_step_times(iter_s, first_occurrences(timing_shapes))
    if fit["separation"] <= 1:
        raise ValueError(f"Shape replay does not explain the step times: {fit}")

    target_cfg = load_job_config(args.config, FrcnnConfig)
    target_shapes = batch_shapes(
        train_sizes(args.annotations, target_cfg, args.splits_dir), target_cfg
    )
    num_new = sum(first_occurrences(target_shapes))
    train_s = num_new * fit["new_shape_s"] + (len(target_shapes) - num_new) * fit["repeat_s"]
    estimate = {
        "timing_run": args.timing_run.name,
        "config": str(args.config),
        "fit": fit,
        "steps": len(target_shapes),
        "distinct_shapes": num_new,
        "train_hours": round(train_s / 3600, 3),
        "excludes": "validation, final prediction, kernel start-up",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(estimate, indent=2) + "\n")
    logger.info("Estimate written to %s: %s", args.out, estimate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
