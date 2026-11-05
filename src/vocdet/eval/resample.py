"""Score many image resamples of one prediction set fast, with the evaluator's own code (ADR-014).

Re-running evaluate() per bootstrap draw repeats the expensive part every time: pycocotools matches
detections to ground truth image by image (COCOeval.evaluate), and that matching does not depend on
which other images are in the sample. Only the accumulation over images (COCOeval.accumulate) does.
So the matching runs once, on COCO objects built by torchmetrics exactly as evaluate() builds them,
and each draw re-runs pycocotools' accumulate over the matched images in draw order, duplicates
included. tests/test_resample.py checks that this equals evaluate() on the resampled set (to float32
precision: torchmetrics returns float32).

Only mAP50 and mAP50-95 over all object sizes are computed, so matching is restricted to the "all"
area range and 100 detections per image, the values evaluate() reports for those two metrics.
Uses torchmetrics' private `_get_coco_datasets` (pinned torchmetrics==1.9.0).
"""

import contextlib
import copy
import io

import numpy as np
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection import MeanAveragePrecision

from vocdet.eval.evaluator import _to_torchmetrics
from vocdet.eval.schema import GroundTruth, Prediction


class MatchedImages:
    """Per-image COCO matching of one prediction set, ready to score any resample of its images."""

    def __init__(self, predictions: list[Prediction], ground_truth: list[GroundTruth]) -> None:
        preds, targets = _to_torchmetrics(predictions, ground_truth)
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")
        metric.update(preds, targets)
        coco_preds, coco_target = metric._coco_backend._get_coco_datasets(
            metric.groundtruth_labels,
            metric.groundtruth_box,
            metric.groundtruth_mask,
            metric.groundtruth_crowds,
            metric.groundtruth_area,
            metric.detection_labels,
            metric.detection_box,
            metric.detection_mask,
            metric.detection_scores,
            metric.iou_type,
            average=metric.average,
        )
        self.coco_eval = COCOeval(coco_target, coco_preds, iouType="bbox")
        params = self.coco_eval.params
        params.iouThrs = np.array(metric.iou_thresholds, dtype=np.float64)
        params.recThrs = np.array(metric.rec_thresholds, dtype=np.float64)
        params.maxDets = [100]
        params.areaRng, params.areaRngLbl = [[0, 1e10]], ["all"]
        with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints progress
            self.coco_eval.evaluate()
        self.num_images = len(params.imgIds)
        self.num_categories = len(params.catIds)

    def score(self, picks: np.ndarray) -> dict[str, float]:
        """mAP50 and mAP50-95 of the resample made of images picks[0], picks[1], ... (by index)."""
        resample = copy.copy(self.coco_eval)
        resample.params = copy.copy(self.coco_eval.params)
        resample._paramsEval = copy.copy(self.coco_eval._paramsEval)
        # evalImgs is ordered category-major, then area range (one here), then image.
        resample.evalImgs = [
            self.coco_eval.evalImgs[k * self.num_images + int(i)]
            for k in range(self.num_categories)
            for i in picks
        ]
        positions = list(range(len(picks)))  # each drawn image, duplicates included, in draw order
        resample.params.imgIds = positions
        resample._paramsEval.imgIds = positions
        with contextlib.redirect_stdout(io.StringIO()):
            resample.accumulate()
        # The same reduction as COCOeval.summarize: mean over defined (> -1) precision values.
        precision = resample.eval["precision"][:, :, :, 0, 0]  # [IoU, recall, category]
        at50 = precision[np.isclose(resample.params.iouThrs, 0.5)]
        return {"map50": _mean_defined(at50), "map50_95": _mean_defined(precision)}


def _mean_defined(values: np.ndarray) -> float:
    defined = values[values > -1]
    return float(defined.mean()) if defined.size else -1.0
