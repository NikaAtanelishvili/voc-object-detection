"""E0b: evaluate a COCO-pretrained YOLO model on VOC val with no training (PLAN.md P2.6).

It measures how much of the final accuracy comes from COCO pretraining alone. It is also the first
end-to-end run of predict -> evaluate -> collect, before any training code exists.
"""

import logging
import shutil
from pathlib import Path

import pandas as pd

from vocdet.config import ZeroShotConfig
from vocdet.data.splits import load_splits
from vocdet.eval.evaluator import evaluate, write_metrics
from vocdet.eval.predict import predict_yolo
from vocdet.eval.schema import ground_truth_from_annotations, write_predictions
from vocdet.utils.run_meta import finish_run_meta, start_run_meta
from vocdet.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def run_zero_shot(
    cfg: ZeroShotConfig,
    config_path: Path,
    data_dir: Path,
    out_dir: Path,
    splits_dir: Path = Path("splits"),
    limit: int | None = None,
    device: str | None = None,
    model=None,
) -> dict:
    """Predict, evaluate, and write the run artifacts (PLAN.md §2) into out_dir; return the metrics.

    `limit` evaluates only the first N images (local smoke tests). `model` lets tests inject a fake
    instead of loading Ultralytics weights.
    """
    seed_everything(cfg.seed)
    meta = start_run_meta(out_dir, out_dir.name, cfg.exp_id, cfg.seed)
    shutil.copyfile(config_path, out_dir / "config.yaml")

    split = cfg.predict.split
    image_ids = load_splits(splits_dir)[split][:limit]
    image_paths = [data_dir / "yolo" / "images" / split / f"{i}.jpg" for i in image_ids]
    if model is None:
        from ultralytics import YOLO  # heavy import, only when actually predicting

        model = YOLO(cfg.predict.weights)
    predictions = predict_yolo(model, image_paths, cfg.predict.imgsz, cfg.predict.batch, device)
    write_predictions(predictions, out_dir / f"predictions_{split}.json")

    annotations = pd.read_parquet(data_dir / "annotations.parquet")
    metrics = evaluate(predictions, ground_truth_from_annotations(annotations, image_ids))
    write_metrics(metrics, out_dir / f"metrics_{split}.json")

    meta["data_dir"] = str(data_dir)
    meta["num_images"] = len(image_ids)
    finish_run_meta(out_dir, meta)
    return metrics
