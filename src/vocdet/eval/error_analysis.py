"""TIDE-lite: classify every detection and every missed object (PLAN.md §4), with our own code.

Per image, detections with score >= score_threshold are processed from highest to lowest score:

1. Find the same-class ground-truth box (difficult included) with the highest IoU, as in VOC.
   If IoU >= 0.5: a difficult box -> Ignored; an unmatched box -> TP (now matched); an already
   matched box -> Dupe.
2. Otherwise it is a false positive, classified in this order (non-difficult ground truth only):
   - Cls:  IoU >= 0.5 with an object of another class (right box, wrong class)
   - Loc:  0.1 <= IoU < 0.5 with an object of the same class (right class, poor box)
   - Both: 0.1 <= IoU < 0.5 with an object of another class
   - Bkg:  IoU < 0.1 with every object
3. Every non-difficult object never matched as TP is a Miss.

Why a score threshold: mAP inference keeps detections down to 0.001 confidence, and counting those
would bury real errors under thousands of near-zero-score Bkg hits. Reports use each model's
operating threshold (ADR-009).

Confusion matrix (21 x 21, index 20 = background), rows = true class, columns = predicted class:
TP -> [c, c]; Cls -> [true, predicted]; Loc/Both/Dupe/Bkg -> [background, predicted];
Miss -> [true, background]. Rows count events, not objects: an object with a Cls error is usually
also a Miss.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vocdet.constants import VOC_CLASSES
from vocdet.eval.schema import GroundTruth, Prediction

IOU_FG = 0.5  # foreground: a correct box
IOU_BG = 0.1  # below this the detection does not touch the object at all
BACKGROUND = len(VOC_CLASSES)  # row/column 20 of the confusion matrix
FP_TYPES = ("Cls", "Loc", "Both", "Dupe", "Bkg")
OUTCOMES = ("TP", *FP_TYPES, "Ignored")


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (len(a), len(b)) for xyxy boxes, written out by hand.

    Intersection = overlap of the x ranges times overlap of the y ranges (zero if they do not
    overlap); union = area(a) + area(b) - intersection.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


@dataclass
class DetectionRecord:
    image_id: str
    box: list[float]
    label: int  # predicted class
    score: float
    outcome: str  # one of OUTCOMES
    true_label: int | None  # class of the object it was compared with (None for Bkg)
    iou: float  # IoU with that object


@dataclass
class MissRecord:
    image_id: str
    box: list[float]
    label: int
    area: float


@dataclass
class ErrorAnalysis:
    detections: list[DetectionRecord]
    misses: list[MissRecord]
    confusion: np.ndarray  # (21, 21) int64


def _best(iou_row: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    """Index and IoU of the best ground-truth box among those selected by mask; (-1, 0) if none."""
    if not mask.any():
        return -1, 0.0
    candidates = np.where(mask, iou_row, -1.0)
    j = int(candidates.argmax())
    return j, float(candidates[j])


def analyze_image(
    pred: Prediction, gt: GroundTruth, score_threshold: float
) -> tuple[list[DetectionRecord], list[MissRecord]]:
    keep = pred.scores >= score_threshold
    boxes, scores, labels = pred.boxes[keep], pred.scores[keep], pred.labels[keep]
    order = np.argsort(-scores, kind="stable")
    iou = box_iou(boxes, gt.boxes)
    matched = np.zeros(len(gt.boxes), dtype=bool)
    records = []

    for d in order:
        label = int(labels[d])
        same = gt.labels == label
        j, best_same = _best(iou[d], same)
        if best_same >= IOU_FG:
            if gt.difficult[j]:
                outcome = "Ignored"
            elif not matched[j]:
                outcome = "TP"
                matched[j] = True
            else:
                outcome = "Dupe"
            true_label, best_iou = label, best_same
        else:
            jo, best_other = _best(iou[d], ~same & ~gt.difficult)
            js, best_same_nd = _best(iou[d], same & ~gt.difficult)
            if best_other >= IOU_FG:
                outcome, true_label, best_iou = "Cls", int(gt.labels[jo]), best_other
            elif best_same_nd >= IOU_BG:
                outcome, true_label, best_iou = "Loc", label, best_same_nd
            elif best_other >= IOU_BG:
                outcome, true_label, best_iou = "Both", int(gt.labels[jo]), best_other
            else:
                outcome, true_label, best_iou = "Bkg", None, max(best_other, best_same_nd)
        records.append(
            DetectionRecord(
                pred.image_id,
                boxes[d].tolist(),
                label,
                float(scores[d]),
                outcome,
                true_label,
                best_iou,
            )
        )

    misses = []
    for j in np.where(~matched & ~gt.difficult)[0]:
        x1, y1, x2, y2 = gt.boxes[j].tolist()
        misses.append(
            MissRecord(gt.image_id, [x1, y1, x2, y2], int(gt.labels[j]), (x2 - x1) * (y2 - y1))
        )
    return records, misses


def analyze(
    predictions: list[Prediction], ground_truth: list[GroundTruth], score_threshold: float = 0.5
) -> ErrorAnalysis:
    """Run analyze_image over a split and build the confusion matrix."""
    by_id = {p.image_id: p for p in predictions}
    detections: list[DetectionRecord] = []
    misses: list[MissRecord] = []
    for gt in ground_truth:
        pred = by_id.get(gt.image_id)
        if pred is None:
            raise ValueError(f"No prediction entry for image {gt.image_id}")
        d, m = analyze_image(pred, gt, score_threshold)
        detections.extend(d)
        misses.extend(m)

    confusion = np.zeros((BACKGROUND + 1, BACKGROUND + 1), dtype=np.int64)
    for r in detections:
        if r.outcome in ("TP", "Cls"):
            confusion[r.true_label, r.label] += 1
        elif r.outcome != "Ignored":
            confusion[BACKGROUND, r.label] += 1
    for m in misses:
        confusion[m.label, BACKGROUND] += 1
    return ErrorAnalysis(detections, misses, confusion)


def summarize(result: ErrorAnalysis) -> dict:
    """Counts overall and per class: detection outcomes by predicted class, misses by true class."""
    overall = {name: 0 for name in (*OUTCOMES, "Miss")}
    per_class = {c: {name: 0 for name in (*OUTCOMES, "Miss")} for c in VOC_CLASSES}
    for r in result.detections:
        overall[r.outcome] += 1
        per_class[VOC_CLASSES[r.label]][r.outcome] += 1
    for m in result.misses:
        overall["Miss"] += 1
        per_class[VOC_CLASSES[m.label]]["Miss"] += 1
    return {
        "overall": overall,
        "per_class": per_class,
        "confusion_labels": [*VOC_CLASSES, "background"],
        "confusion": result.confusion.tolist(),
    }


def write_error_analysis(result: ErrorAnalysis, out_dir: Path) -> None:
    """errors.json (summary) and error_records.json (every detection and miss, for galleries)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "errors.json").write_text(json.dumps(summarize(result), indent=2) + "\n")
    records = {
        "detections": [asdict(r) for r in result.detections],
        "misses": [asdict(m) for m in result.misses],
    }
    (out_dir / "error_records.json").write_text(json.dumps(records) + "\n")  # as the hook expects
