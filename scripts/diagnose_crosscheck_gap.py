"""Split the gap between Ultralytics val map50 and ours into its causes (ADR-013). CPU, val only.

Usage: uv run python scripts/diagnose_crosscheck_gap.py --weights runs/<run_id>/weights/best.pt
       --out results/diagnostics/crosscheck_gap_<run_id>.json

All scores use ground truth without difficult objects, as the YOLO labels do. Steps:
  1. our predict path (predict_yolo settings) scored by our evaluator      -> ours_map50
  2. the same predictions scored by Ultralytics' matching + ap_per_class    -> metric effect
  3. the same, keeping 300 detections per image instead of 100              -> cap effect
  4. model.val() with batch 16 and batch 1 (rect batches share padding)     -> inference path effect
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, box_iou

from vocdet.data.splits import load_splits
from vocdet.eval.evaluator import evaluate
from vocdet.eval.predict import MODEL_MAX_DET, SCORE_THRESHOLD, build_class_map, to_voc_prediction
from vocdet.eval.schema import GroundTruth, Prediction, ground_truth_from_annotations
from vocdet.train.train_yolo import build_dataset_root

REPO_ROOT = Path(__file__).resolve().parents[1]


def ultralytics_map50(predictions: list[Prediction], gt: list[GroundTruth]) -> float:
    """map50 of our predictions as Ultralytics' validator would compute it."""
    validator = BaseValidator.__new__(BaseValidator)  # only match_predictions is used
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    by_id = {g.image_id: g for g in gt}
    tp, conf, pred_cls, target_cls = [], [], [], []
    for p in predictions:
        g = by_id[p.image_id]
        target_cls.append(g.labels)
        if len(p.scores) == 0:
            continue
        labels = torch.tensor(p.labels)
        if len(g.labels) == 0:
            correct = np.zeros((len(labels), 10), dtype=bool)
        else:
            iou = box_iou(torch.tensor(g.boxes), torch.tensor(p.boxes))
            correct = validator.match_predictions(labels, torch.tensor(g.labels), iou).numpy()
        tp.append(correct)
        conf.append(p.scores)
        pred_cls.append(p.labels)
    ap = ap_per_class(
        np.concatenate(tp),
        np.concatenate(conf),
        np.concatenate(pred_cls),
        np.concatenate(target_cls),
    )[5]  # (classes, 10 IoU thresholds)
    return float(ap[:, 0].mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "voc_prepared")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    ids = load_splits(REPO_ROOT / "splits")["val"]
    annotations = pd.read_parquet(args.data_dir / "annotations.parquet")
    gt = [
        GroundTruth(
            g.image_id, g.boxes[~g.difficult], g.labels[~g.difficult], g.difficult[~g.difficult]
        )
        for g in ground_truth_from_annotations(annotations, ids)
    ]
    model = YOLO(str(args.weights))
    class_map = build_class_map(model.names)

    raw = {}  # image_id -> model output before our top-100 cut
    paths = [str(args.data_dir / "yolo" / "images" / "val" / f"{i}.jpg") for i in ids]
    for start in range(0, len(paths), 16):
        chunk = paths[start : start + 16]
        results = model.predict(
            chunk, conf=SCORE_THRESHOLD, max_det=MODEL_MAX_DET, imgsz=args.imgsz, device="cpu",
            verbose=False,
        )  # fmt: skip
        for path, r in zip(chunk, results, strict=True):
            raw[Path(path).stem] = (
                r.boxes.xyxy.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy().astype(np.int64)
            )  # fmt: skip

    top100 = [to_voc_prediction(i, *raw[i], class_map, max_detections=100) for i in ids]
    top300 = [to_voc_prediction(i, *raw[i], class_map, max_detections=300) for i in ids]
    result = {
        "weights": str(args.weights),
        "ours_map50": evaluate(top100, gt)["map50"],
        "ultralytics_metric_on_our_predictions": ultralytics_map50(top100, gt),
        "ultralytics_metric_on_our_predictions_top300": ultralytics_map50(top300, gt),
    }
    with tempfile.TemporaryDirectory() as tmp:
        dataset_yaml = build_dataset_root(args.data_dir, Path(tmp) / "root", {"val": ids})
        # Ultralytics requires a train key; it is never read by val.
        dataset_yaml.write_text(dataset_yaml.read_text() + "train: images/val\n")
        for batch in (16, 1):
            r = model.val(
                data=str(dataset_yaml), split="val", imgsz=args.imgsz, batch=batch, device="cpu",
                plots=False, project=tmp, name="val", exist_ok=True, verbose=False,
            )  # fmt: skip
            result[f"model_val_map50_batch{batch}"] = float(r.box.map50)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))  # noqa: T201  (script output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
