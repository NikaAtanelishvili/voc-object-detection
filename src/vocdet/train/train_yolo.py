"""Fine-tune an Ultralytics YOLO26 model on VOC (P4.1, P4.2): YAML config in, run artifacts out.

Unlike Faster R-CNN (ADR-012), the training loop is Ultralytics' own; this module only decides
what goes in and where things land. Ultralytics reads data from a dataset YAML and writes a label
`.cache` next to the label files, but the prepared data is mounted read-only under /kaggle/input.
So each run builds a writable dataset root: images are symlinked (no 2 GB copy), label files are
copied, and a fresh voc_yolo.yaml points at that root. Ultralytics finds a label by replacing
`/images/` with `/labels/` in the (unresolved) image path, so symlinked images still find the
copied labels.
"""

import logging
import shutil
from pathlib import Path

import pandas as pd
import yaml

from vocdet.config import YoloConfig
from vocdet.constants import VOC_CLASSES
from vocdet.data.splits import load_splits
from vocdet.eval.evaluator import evaluate, write_metrics
from vocdet.eval.predict import predict_yolo
from vocdet.eval.schema import (
    GroundTruth,
    Prediction,
    ground_truth_from_annotations,
    write_predictions,
)
from vocdet.utils.run_meta import finish_run_meta, start_run_meta
from vocdet.utils.seed import seed_everything

logger = logging.getLogger(__name__)

# A larger map50 gap between Ultralytics and our difficult-removed score than the remaining
# differences explain (ADR-013) points at an adapter bug: class map, box scaling, or image order.
CROSSCHECK_TOLERANCE = 0.02


def build_dataset_root(data_dir: Path, root: Path, split_ids: dict[str, list[str]]) -> Path:
    """Create root/{images,labels}/<split>/ for the given IDs and return root/voc_yolo.yaml.

    Only the splits passed in are written, so the test split never reaches Ultralytics
    (hard rule 4).
    """
    source = data_dir / "yolo"
    for split, ids in split_ids.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            image = source / "images" / split / f"{image_id}.jpg"
            link = root / "images" / split / f"{image_id}.jpg"
            if not link.is_symlink():
                link.symlink_to(image.resolve())
            label = f"{image_id}.txt"
            shutil.copyfile(source / "labels" / split / label, root / "labels" / split / label)
        logger.info("Dataset root %s: %d %s images", root, len(ids), split)

    dataset_yaml = root / "voc_yolo.yaml"
    spec = {"path": str(root.resolve()), **{s: f"images/{s}" for s in split_ids}}
    spec["names"] = dict(enumerate(VOC_CLASSES))
    dataset_yaml.write_text(yaml.safe_dump(spec, sort_keys=False))
    return dataset_yaml


def ultralytics_train_args(cfg: YoloConfig, dataset_yaml: Path, out_dir: Path) -> dict:
    """Map our config onto Ultralytics `train()` keyword arguments (names checked against 8.4.191).

    `optimizer` is not passed, so the model default ("auto") applies. Augmentation keys left as None
    in the config are not passed either, so Ultralytics keeps its defaults for them.
    """
    t = cfg.train
    args = {
        "data": str(dataset_yaml),
        "epochs": t.epochs,
        "imgsz": t.imgsz,
        "batch": t.batch,
        "seed": cfg.seed,
        "deterministic": True,
        "patience": t.patience,
        # Absolute, so Ultralytics writes inside the run folder instead of its own runs/ directory.
        "project": str(out_dir.resolve()),
        "name": "ultralytics",
        "exist_ok": True,
    }
    if t.freeze is not None:
        args["freeze"] = t.freeze
    if t.close_mosaic is not None:
        args["close_mosaic"] = t.close_mosaic
    args.update(t.augment.model_dump(exclude_none=True))
    return args


def ultralytics_val_crosscheck(
    model,
    dataset_yaml: Path,
    cfg: YoloConfig,
    out_dir: Path,
    predictions: list[Prediction],
    val_gt: list[GroundTruth],
) -> dict:
    """Ultralytics' own val mAP next to ours (P4.2, ADR-013). Our evaluator stays authoritative.

    The YOLO labels leave difficult objects out, so Ultralytics scores a detection on one as a
    false positive, while our evaluator ignores it. Scoring our predictions once more with
    difficult objects removed from the ground truth separates that effect from the rest of the gap.
    """
    results = model.val(
        data=str(dataset_yaml), split="val", imgsz=cfg.train.imgsz, batch=cfg.train.batch,
        project=str(out_dir.resolve()), name="ultralytics_val", exist_ok=True, plots=False,
    )  # fmt: skip
    no_difficult = [
        GroundTruth(
            g.image_id, g.boxes[~g.difficult], g.labels[~g.difficult], g.difficult[~g.difficult]
        )
        for g in val_gt
    ]
    ours_without_difficult = evaluate(predictions, no_difficult)
    check = {
        "ultralytics_map50": float(results.box.map50),
        "ultralytics_map50_95": float(results.box.map),
        "ours_map50_difficult_removed": ours_without_difficult["map50"],
        "ours_map50_95_difficult_removed": ours_without_difficult["map50_95"],
    }
    gap = check["ultralytics_map50"] - check["ours_map50_difficult_removed"]
    if abs(gap) > CROSSCHECK_TOLERANCE:
        logger.warning("Ultralytics map50 differs from ours (difficult removed) by %.4f", gap)
    return check


def train_yolo(
    cfg: YoloConfig,
    config_path: Path,
    data_dir: Path,
    out_dir: Path,
    splits_dir: Path = Path("splits"),
    load_model=None,
) -> dict:
    """Train, predict val with best.pt, evaluate, write the run artifacts; return the metrics.

    data_dir is a voc_prepared/ folder with annotations.parquet and yolo/{images,labels}/<split>/.
    `load_model` (weights -> model) lets tests inject a fake instead of ultralytics.YOLO.
    """
    t = cfg.train
    seed_everything(cfg.seed)
    meta = start_run_meta(out_dir, out_dir.name, cfg.exp_id, cfg.seed)
    shutil.copyfile(config_path, out_dir / "config.yaml")

    splits = load_splits(splits_dir)
    split_ids = {
        "train": splits["train"][: t.limit_train_images],
        "val": splits["val"][: t.limit_val_images],
    }
    dataset_root = out_dir / "dataset"
    dataset_yaml = build_dataset_root(data_dir, dataset_root, split_ids)

    if load_model is None:
        from ultralytics import YOLO  # heavy import, only when actually training

        load_model = YOLO
    model = load_model(t.weights)
    model.train(**ultralytics_train_args(cfg, dataset_yaml, out_dir))

    # Same layout as the Faster R-CNN runs: train_log.csv and weights/ at the top of the run folder.
    save_dir = Path(model.trainer.save_dir)
    shutil.copyfile(save_dir / "results.csv", out_dir / "train_log.csv")
    shutil.move(save_dir / "weights", out_dir / "weights")

    # Final artifacts come from best.pt through the same adapter and evaluator as every model.
    best = load_model(str(out_dir / "weights" / "best.pt"))
    val_images = [data_dir / "yolo" / "images" / "val" / f"{i}.jpg" for i in split_ids["val"]]
    predictions = predict_yolo(best, val_images, t.imgsz, t.batch)
    write_predictions(predictions, out_dir / "predictions_val.json")
    annotations = pd.read_parquet(data_dir / "annotations.parquet")
    val_gt = ground_truth_from_annotations(annotations, split_ids["val"])
    metrics = evaluate(predictions, val_gt)
    write_metrics(metrics, out_dir / "metrics_val.json")
    meta["ultralytics_val_crosscheck"] = ultralytics_val_crosscheck(
        best, dataset_yaml, cfg, out_dir, predictions, val_gt
    )

    # ~11k symlinks and labels: rebuilt in seconds, and not worth saving as kernel output.
    shutil.rmtree(dataset_root)

    meta.update({"data_dir": str(data_dir), "weights": t.weights})
    meta.update(
        {"num_train_images": len(split_ids["train"]), "num_val_images": len(split_ids["val"])}
    )
    finish_run_meta(out_dir, meta)
    return metrics
