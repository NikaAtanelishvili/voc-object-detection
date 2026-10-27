"""The single evaluator for every model: a thin wrapper over torchmetrics MeanAveragePrecision.

Difficult objects are passed as `iscrowd=1`, which pycocotools treats as ignore regions: a detection
matched to one is neither a TP nor an FP, and an unmatched one is not a miss (VOC protocol).

Two metric instances are used on purpose (ADR-008):
- the default COCO thresholds (IoU 0.50:0.95) give map50_95, map50, the size buckets, and mar_100;
- a second instance with iou_thresholds=[0.5] and class_metrics=True gives per-class AP at IoU 0.5.
  With the default thresholds, torchmetrics' per-class numbers would be averaged over 0.50:0.95.

AP uses COCO 101-point interpolation, so numbers are not directly comparable to the VOC all-point AP
in published tables (PLAN.md §4).
"""

import json
import logging
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision

from vocdet.constants import VOC_CLASSES
from vocdet.eval.schema import GroundTruth, Prediction

logger = logging.getLogger(__name__)


def _to_torchmetrics(
    predictions: list[Prediction], ground_truth: list[GroundTruth]
) -> tuple[list[dict], list[dict]]:
    """Pair predictions with ground truth by image_id, in ground-truth order.

    An image with no prediction entry is an error, not an empty prediction: a silently dropped image
    would inflate precision.
    """
    by_id = {p.image_id: p for p in predictions}
    extra = set(by_id) - {g.image_id for g in ground_truth}
    if extra:
        raise ValueError(
            f"{len(extra)} predictions for images not in the split, e.g. {sorted(extra)[:3]}"
        )
    preds, targets = [], []
    for gt in ground_truth:
        if gt.image_id not in by_id:
            raise ValueError(f"No prediction entry for image {gt.image_id}")
        p = by_id[gt.image_id]
        preds.append(
            {
                "boxes": torch.from_numpy(p.boxes),
                "scores": torch.from_numpy(p.scores),
                "labels": torch.from_numpy(p.labels),
            }
        )
        targets.append(
            {
                "boxes": torch.from_numpy(gt.boxes),
                "labels": torch.from_numpy(gt.labels),
                "iscrowd": torch.from_numpy(gt.difficult.astype("int64")),
            }
        )
    return preds, targets


def _value(t: torch.Tensor) -> float | None:
    """torchmetrics reports -1 when a metric is undefined (e.g. no small objects); use None."""
    v = float(t)
    return None if v == -1.0 else v


def evaluate(predictions: list[Prediction], ground_truth: list[GroundTruth]) -> dict:
    """Return the flat metrics dict used everywhere in the project."""
    preds, targets = _to_torchmetrics(predictions, ground_truth)

    coco = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")
    coco.update(preds, targets)
    c = coco.compute()

    at50 = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.5],
        class_metrics=True,
        backend="pycocotools",
    )
    at50.update(preds, targets)
    r = at50.compute()

    ap50_per_class: dict[str, float | None] = {name: None for name in VOC_CLASSES}
    # `classes` lists the labels present in predictions or targets; map_per_class follows its order.
    for label, ap in zip(
        r["classes"].reshape(-1).tolist(), r["map_per_class"].reshape(-1), strict=True
    ):
        ap50_per_class[VOC_CLASSES[label]] = _value(ap)

    return {
        "map50": _value(c["map_50"]),
        "map50_95": _value(c["map"]),
        "map_small": _value(c["map_small"]),
        "map_medium": _value(c["map_medium"]),
        "map_large": _value(c["map_large"]),
        "mar_100": _value(c["mar_100"]),
        "ap50_per_class": ap50_per_class,
        "num_images": len(ground_truth),
    }


def write_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    logger.info("map50=%.4f map50_95=%.4f -> %s", metrics["map50"], metrics["map50_95"], path)
