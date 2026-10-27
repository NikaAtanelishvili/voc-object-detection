"""The one prediction schema every model produces, and the matching ground truth.

Faster R-CNN, YOLO, and every exported ONNX/OpenVINO variant are converted to `Prediction`, and
all metrics are computed from it by the same evaluator. That is what makes comparisons fair
(PLAN.md §2). Boxes are absolute, 0-based xyxy floats in original image pixels; labels are VOC
indices 0-19. JSON files store the float32 values unrounded, so evaluating a saved file gives
exactly the metrics computed in memory during the run.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 20


def _as_boxes(boxes: np.ndarray | list) -> np.ndarray:
    """(N, 4) float32 array; an empty input becomes shape (0, 4), not (0,)."""
    arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if (arr[:, 2] < arr[:, 0]).any() or (arr[:, 3] < arr[:, 1]).any():
        raise ValueError("boxes must be xyxy with x2 >= x1 and y2 >= y1")
    return arr


def _as_labels(labels: np.ndarray | list, n: int) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(arr) != n:
        raise ValueError(f"expected {n} labels, got {len(arr)}")
    if ((arr < 0) | (arr >= NUM_CLASSES)).any():
        raise ValueError(f"labels must be VOC indices in [0, {NUM_CLASSES - 1}]")
    return arr


@dataclass
class Prediction:
    """Detections for one image. Arrays are validated and normalized in __post_init__."""

    image_id: str
    boxes: np.ndarray  # (N, 4) float32, absolute 0-based xyxy
    scores: np.ndarray  # (N,) float32 in [0, 1]
    labels: np.ndarray  # (N,) int64 in [0, 19]

    def __post_init__(self) -> None:
        self.boxes = _as_boxes(self.boxes)
        self.labels = _as_labels(self.labels, len(self.boxes))
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        if len(self.scores) != len(self.boxes):
            raise ValueError(f"expected {len(self.boxes)} scores, got {len(self.scores)}")
        if ((self.scores < 0) | (self.scores > 1)).any():
            raise ValueError("scores must be in [0, 1]")


@dataclass
class GroundTruth:
    """All annotated objects of one image; difficult ones are ignore regions in evaluation."""

    image_id: str
    boxes: np.ndarray  # (M, 4) float32, absolute 0-based xyxy
    labels: np.ndarray  # (M,) int64 in [0, 19]
    difficult: np.ndarray  # (M,) bool

    def __post_init__(self) -> None:
        self.boxes = _as_boxes(self.boxes)
        self.labels = _as_labels(self.labels, len(self.boxes))
        self.difficult = np.asarray(self.difficult, dtype=bool).reshape(-1)
        if len(self.difficult) != len(self.boxes):
            raise ValueError(
                f"expected {len(self.boxes)} difficult flags, got {len(self.difficult)}"
            )


def write_predictions(predictions: list[Prediction], path: Path) -> None:
    """One JSON list, one object per image.

    No rounding: float32 -> Python float -> JSON -> float32 is exact. Rounding boxes to 0.01 px once
    shifted E0b's map50_95 by 2e-4 between the in-run and the file-based evaluation.
    """
    records = [
        {
            "image_id": p.image_id,
            "boxes": p.boxes.tolist(),
            "scores": p.scores.tolist(),
            "labels": p.labels.tolist(),
        }
        for p in predictions
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records))


def read_predictions(path: Path) -> list[Prediction]:
    return [Prediction(**record) for record in json.loads(path.read_text())]


def write_ground_truth(ground_truth: list[GroundTruth], path: Path) -> None:
    records = [
        {
            "image_id": g.image_id,
            "boxes": g.boxes.tolist(),
            "labels": g.labels.tolist(),
            "difficult": g.difficult.tolist(),
        }
        for g in ground_truth
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records))


def read_ground_truth(path: Path) -> list[GroundTruth]:
    return [GroundTruth(**record) for record in json.loads(path.read_text())]


def ground_truth_from_annotations(
    annotations: pd.DataFrame, image_ids: list[str]
) -> list[GroundTruth]:
    """Ground truth for a split from the canonical table, in image_ids order.

    Every listed image gets an entry, even one without objects, so it still counts for false
    positives.
    """
    groups = {image_id: g for image_id, g in annotations.groupby("image_id")}
    result = []
    for image_id in image_ids:
        g = groups.get(image_id)
        if g is None:
            result.append(GroundTruth(image_id, np.zeros((0, 4)), [], []))
        else:
            boxes = g[["x1", "y1", "x2", "y2"]].to_numpy()
            result.append(GroundTruth(image_id, boxes, g["class_id"].to_numpy(), g["difficult"]))
    return result
