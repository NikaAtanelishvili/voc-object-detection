"""Model adapters: run a model over images and return `Prediction`s (eval/schema.py).

predict_yolo serves both the zero-shot COCO model (E0b) and our fine-tuned VOC models (P4.2): the
class map translates the model's own class indices to VOC indices, and classes without a VOC
counterpart are dropped. predict_frcnn serves our Faster R-CNN (P3.3).
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torchvision.io import ImageReadMode, decode_image

from vocdet.constants import COCO_TO_VOC, VOC_CLASSES
from vocdet.data.dataset import LABEL_OFFSET
from vocdet.eval.schema import Prediction

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 0.001  # standard for mAP (PLAN.md §4)
MAX_DETECTIONS = 100  # kept per image after mapping to VOC classes
MODEL_MAX_DET = 300  # asked from the model before mapping, so non-VOC classes cannot use up the 100


def build_class_map(model_names: dict[int, str]) -> dict[int, int]:
    """Model class index -> VOC index, by name.

    VOC names map to themselves (fine-tuned models); COCO names go through COCO_TO_VOC (zero-shot);
    anything else is dropped. Raises unless all 20 VOC classes are covered.
    """
    class_map = {}
    for index, name in model_names.items():
        voc_name = name if name in VOC_CLASSES else COCO_TO_VOC.get(name)
        if voc_name is not None:
            class_map[int(index)] = VOC_CLASSES.index(voc_name)
    if sorted(class_map.values()) != list(range(len(VOC_CLASSES))):
        raise ValueError(
            f"Model classes do not cover all 20 VOC classes: {sorted(class_map.values())}"
        )
    return class_map


def to_voc_prediction(
    image_id: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    model_labels: np.ndarray,
    class_map: dict[int, int],
    max_detections: int = MAX_DETECTIONS,
) -> Prediction:
    """Map labels to VOC, drop unmapped classes, keep the top max_detections by score."""
    keep = np.array([int(label) in class_map for label in model_labels], dtype=bool)
    boxes, scores, model_labels = boxes[keep], scores[keep], model_labels[keep]
    order = np.argsort(-scores, kind="stable")[:max_detections]
    labels = np.array([class_map[int(label)] for label in model_labels[order]], dtype=np.int64)
    return Prediction(image_id, boxes[order], scores[order], labels)


def _batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def predict_yolo(
    model,
    image_paths: list[Path],
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
) -> list[Prediction]:
    """Run an Ultralytics model and convert every result. image_id = file stem.

    Ultralytics returns boxes already scaled back to original image pixels, in xyxy.
    """
    class_map = build_class_map(model.names)
    predictions = []
    for chunk in _batches([str(p) for p in image_paths], batch):
        results = model.predict(
            chunk,
            conf=SCORE_THRESHOLD,
            max_det=MODEL_MAX_DET,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        for path, r in zip(chunk, results, strict=True):
            predictions.append(
                to_voc_prediction(
                    Path(path).stem,
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy().astype(np.int64),
                    class_map,
                )
            )
        logger.info("Predicted %d / %d images", len(predictions), len(image_paths))
    return predictions


def frcnn_output_to_prediction(image_id: str, output: dict) -> Prediction:
    """One torchvision detection output -> Prediction.

    torchvision labels are VOC class_id + LABEL_OFFSET (0 is background and never returned), so
    the offset is subtracted here: the exact inverse of the dataset's `+ LABEL_OFFSET`. The model
    already applies the 0.001 score threshold and the top-100 cap (models/frcnn.py).
    """
    labels = output["labels"].cpu().numpy() - LABEL_OFFSET
    return Prediction(
        image_id,
        output["boxes"].float().cpu().numpy(),
        output["scores"].float().cpu().numpy(),
        labels,
    )


@torch.no_grad()
def predict_frcnn(
    model,
    image_ids: list[str],
    image_dir: Path,
    device: torch.device,
    batch_size: int = 8,
) -> list[Prediction]:
    """Run Faster R-CNN over images (read like the training dataset: RGB floats in [0, 1])."""
    model.eval()
    predictions = []
    for chunk in _batches(image_ids, batch_size):
        images = [
            decode_image(str(image_dir / f"{i}.jpg"), mode=ImageReadMode.RGB)
            .float()
            .div(255)
            .to(device)
            for i in chunk
        ]
        outputs = model(images)
        predictions.extend(
            frcnn_output_to_prediction(i, out) for i, out in zip(chunk, outputs, strict=True)
        )
    logger.info("Faster R-CNN predicted %d images", len(predictions))
    return predictions
