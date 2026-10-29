"""Estimate a YOLO run's GPU time from a 3-epoch timing run (P4.3, DEVIATIONS 2026-10-29).

Usage: uv run python scripts/estimate_yolo_time.py --timing-run runs/<run_id>
       --config configs/experiments/E2.yaml --out results/estimates/E2.json

The timing run trains 3 full-data epochs with close_mosaic=1, so its train_log.csv (Ultralytics
results.csv, "time" = cumulative seconds since training started) gives:
  epoch 1: start-up + first mosaic epoch (cold caches, CUDA warm-up)
  epoch 2: one warm mosaic epoch
  epoch 3: one warm no-mosaic epoch (mosaic loads 4 images per sample, so the two differ)
Each epoch includes its val pass. The target run is priced as epoch 1, then its remaining mosaic
epochs, then its close_mosaic plain epochs, plus everything in the kernel outside training (clone,
pip, dataset root, final prediction and val), taken from the timing kernel's wall time. Early
stopping (patience) can only shorten the run, so the result is an upper bound.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from vocdet.config import YoloConfig, load_job_config

ULTRALYTICS_CLOSE_MOSAIC = 10  # default in Ultralytics 8.4.191 (cfg/default.yaml)


def epoch_times(train_log: Path) -> tuple[float, float, float]:
    """(first epoch, warm mosaic epoch, warm no-mosaic epoch) in seconds from cumulative times."""
    with train_log.open(newline="") as f:
        cumulative = [float(row["time"]) for row in csv.DictReader(f)]
    if len(cumulative) != 3:
        raise ValueError(f"{train_log}: expected 3 epochs of a timing run, got {len(cumulative)}")
    return cumulative[0], cumulative[1] - cumulative[0], cumulative[2] - cumulative[1]


def estimate_seconds(
    first_s: float,
    mosaic_s: float,
    plain_s: float,
    outside_training_s: float,
    epochs: int,
    close_mosaic: int,
) -> float:
    """Seconds for a run of `epochs` whose last `close_mosaic` epochs have no mosaic."""
    plain_epochs = min(close_mosaic, epochs)
    mosaic_epochs = epochs - plain_epochs
    if mosaic_epochs >= 1:
        return (
            outside_training_s + first_s + (mosaic_epochs - 1) * mosaic_s + plain_epochs * plain_s
        )
    # Mosaic never used: the first epoch is a plain one plus the same start-up cost.
    return outside_training_s + first_s - mosaic_s + epochs * plain_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timing-run", type=Path, required=True, help="Pulled run folder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_job_config(args.config, YoloConfig)
    first_s, mosaic_s, plain_s = epoch_times(args.timing_run / "train_log.csv")
    with (args.timing_run / "train_log.csv").open(newline="") as f:
        training_s = float(list(csv.DictReader(f))[-1]["time"])
    kernel_s = json.loads((args.timing_run / "kaggle_run.json").read_text())["wall_time_s"]
    close_mosaic = cfg.train.close_mosaic
    if close_mosaic is None:
        close_mosaic = ULTRALYTICS_CLOSE_MOSAIC
    seconds = estimate_seconds(
        first_s, mosaic_s, plain_s, kernel_s - training_s, cfg.train.epochs, close_mosaic
    )
    result = {
        "timing_run": args.timing_run.name,
        "config": str(args.config),
        "first_epoch_s": round(first_s, 1),
        "mosaic_epoch_s": round(mosaic_s, 1),
        "plain_epoch_s": round(plain_s, 1),
        "outside_training_s": round(kernel_s - training_s, 1),
        "epochs": cfg.train.epochs,
        "close_mosaic": close_mosaic,
        "gpu_hours_upper_bound": round(seconds / 3600, 3),
        "note": "upper bound: early stopping (patience) can only shorten the run",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))  # noqa: T201  (script output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
