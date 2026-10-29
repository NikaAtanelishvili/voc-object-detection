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
import time
from concurrent.futures import ThreadPoolExecutor
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

# Ultralytics' val pipeline scores ~2 points above its own predict() (measured, ADR-013); an adapter
# bug (class map, box scaling, image IDs) costs tens of points. A gap above this is checked by hand.
CROSSCHECK_TOLERANCE = 0.05
IO_THREADS = 32  # dataset root build: threads waiting on the mount, not computing


def link_image_copy_label(
    image_id: str, images: Path, labels: Path, out_images: Path, out_labels: Path
) -> None:
    """Symlink one image into out_images and copy its label file into out_labels."""
    link = out_images / f"{image_id}.jpg"
    if not link.is_symlink():
        link.symlink_to(images / f"{image_id}.jpg")
    shutil.copyfile(labels / f"{image_id}.txt", out_labels / f"{image_id}.txt")


def build_dataset_root(data_dir: Path, root: Path, split_ids: dict[str, list[str]]) -> Path:
    """Create root/{images,labels}/<split>/ for the given IDs and return root/voc_yolo.yaml.

    Only the splits passed in are written, so the test split never reaches Ultralytics
    (hard rule 4). Every file operation on the /kaggle/input mount waits ~0.1 s, so done one by one
    the 5717 train+val images took ~11 GPU minutes (DEVIATIONS 2026-10-29). Source folders are
    resolved once, and the per-image work runs in a thread pool, since it waits on I/O, not CPU.
    """
    for split, ids in split_ids.items():
        images = (data_dir / "yolo" / "images" / split).resolve()
        labels = data_dir / "yolo" / "labels" / split
        out_images, out_labels = root / "images" / split, root / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=IO_THREADS) as pool:
            futures = [
                pool.submit(link_image_copy_label, i, images, labels, out_images, out_labels)
                for i in ids
            ]
            for future in futures:
                future.result()  # re-raises a worker's exception here
        logger.info(
            "Dataset root %s: %d %s images in %.1f s",
            root,
            len(ids),
            split,
            time.monotonic() - start,
        )

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
