"""The one and only use of the test split: final models scored once each (P5.2, hard rule 4).

Usage:
  predict (Kaggle CPU kernel, configs/kaggle/final_test_eval.yaml):
      python scripts/final_test_eval.py predict --data-dir kaggle --out-dir <run_dir>
  record (locally, after `make kaggle-pull`):
      uv run python scripts/final_test_eval.py record --from runs/<run_id>
      [--force-rerun-with-reason "<text>"]

`predict` runs each model in FINAL_MODELS (the canonical run of each experiment) on every test image
through the same adapters and evaluator as val, and writes <out>/<exp_id>/{predictions_test.json,
metrics_test.json, model.json}. model.json holds the sha256 of the weights actually used, so the
result can be tied to the Hub backup. On Kaggle the weights come from the training kernels' outputs
attached as sources. `record` turns that into results/test_metrics.csv (one row per experiment) and
results/runs/<run_id>/metrics_test.json, and keeps predictions in runs/<run_id>/ for the paired
bootstrap. Both refuse an experiment that already has a row, unless --force-rerun-with-reason is
given; the reason is appended to results/test_eval_log.csv. Test results never feed back into any
choice: model selection is done (ADR-003).
"""

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch

from collect_results import MANIFEST, load_manifest
from vocdet.config import FrcnnConfig, YoloConfig, ZeroShotConfig, load_job_config
from vocdet.data.splits import load_splits
from vocdet.eval.evaluator import evaluate, write_metrics
from vocdet.eval.predict import predict_frcnn, predict_yolo
from vocdet.eval.schema import ground_truth_from_annotations, write_predictions

logger = logging.getLogger("final_test_eval")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
TEST_CSV = RESULTS / "test_metrics.csv"
LOG_CSV = RESULTS / "test_eval_log.csv"
# PLAN.md P5.2: E1-E3 and the selected model; the owner added E0b and the fallback E6n (2026-11-06).
FINAL_MODELS = ("E0b", "E1", "E2", "E3", "E6s", "E6n")
METRIC_COLUMNS = ("map50", "map50_95", "map_small", "map_medium", "map_large", "mar_100")
COLUMNS = ("exp_id", "run_id", "weights", "weights_sha256", *METRIC_COLUMNS, "recorded_utc")


class AlreadyEvaluatedError(RuntimeError):
    """The experiment already has a test result; test evaluation happens once per model."""


def recorded_experiments(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as f:
        return {row["exp_id"] for row in csv.DictReader(f)}


def check_once(exp_ids: list[str], reason: str | None, csv_path: Path | None = None) -> None:
    """Refuse experiments with a recorded test result unless a reason is given."""
    csv_path = csv_path or TEST_CSV  # read at call time, not bound when the function is defined
    done = sorted(set(exp_ids) & recorded_experiments(csv_path))
    if done and not reason:
        raise AlreadyEvaluatedError(
            f"{done} already have test results in {csv_path}. Test evaluation happens once per "
            'model; rerun only with --force-rerun-with-reason "<why>".'
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_weights(run_id: str, on_kaggle: bool) -> Path:
    if on_kaggle:
        from vocdet.utils.kaggle_input import find_run_file

        return find_run_file(run_id, "weights/best.pt")
    return REPO_ROOT / "runs" / run_id / "weights" / "best.pt"


def predict_one(exp_id: str, run_id: str, data_dir: Path, test_ids: list[str], on_kaggle: bool):
    """(predictions, weights label, sha256 or None) for one experiment on the test images."""
    config = RESULTS / "runs" / run_id / "config.yaml"
    images = data_dir / "yolo" / "images" / "test"
    if exp_id == "E0b":
        from ultralytics import YOLO

        cfg = load_job_config(config, ZeroShotConfig)
        model = YOLO(cfg.predict.weights)  # COCO weights, downloaded into the working folder
        paths = [images / f"{i}.jpg" for i in test_ids]
        predictions = predict_yolo(model, paths, cfg.predict.imgsz, cfg.predict.batch, "cpu")
        weights = Path(cfg.predict.weights)
        return predictions, str(weights), sha256(weights) if weights.exists() else None
    weights = find_weights(run_id, on_kaggle)
    if exp_id == "E1":
        from vocdet.models.frcnn import build_frcnn

        cfg = load_job_config(config, FrcnnConfig)
        model = build_frcnn(
            pretrained=False, min_size=cfg.model.min_size, max_size=cfg.model.max_size
        )
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"])
        predictions = predict_frcnn(model, test_ids, images, torch.device("cpu"))
    else:
        from ultralytics import YOLO

        cfg = load_job_config(config, YoloConfig)
        paths = [images / f"{i}.jpg" for i in test_ids]
        predictions = predict_yolo(
            YOLO(str(weights)), paths, cfg.train.imgsz, cfg.train.batch, "cpu"
        )
    return predictions, str(weights), sha256(weights)


def run_predict(args: argparse.Namespace) -> int:
    exp_ids = args.models or list(FINAL_MODELS)
    check_once(exp_ids, args.force_rerun_with_reason)
    on_kaggle = args.data_dir == "kaggle"
    if on_kaggle:
        from vocdet.config import DataConfig
        from vocdet.data.prepare import find_prepared_dir

        data_dir = find_prepared_dir(
            load_job_config(REPO_ROOT / "configs/data/voc2012.yaml", DataConfig).prep_run_id
        )
    else:
        data_dir = Path(args.data_dir)
    test_ids = load_splits(REPO_ROOT / "splits")["test"]
    gt = ground_truth_from_annotations(pd.read_parquet(data_dir / "annotations.parquet"), test_ids)
    manifest = load_manifest(MANIFEST)
    for exp_id in exp_ids:
        run_id = manifest[exp_id]
        out = args.out_dir / exp_id
        out.mkdir(parents=True, exist_ok=True)
        predictions, weights, digest = predict_one(exp_id, run_id, data_dir, test_ids, on_kaggle)
        write_predictions(predictions, out / "predictions_test.json")
        metrics = evaluate(predictions, gt)
        write_metrics(metrics, out / "metrics_test.json")
        info = {"exp_id": exp_id, "run_id": run_id, "weights": weights, "weights_sha256": digest}
        (out / "model.json").write_text(json.dumps(info, indent=2) + "\n")
        logger.info("%s (%s): test map50 %.4f", exp_id, run_id, metrics["map50"])
    return 0


def run_record(args: argparse.Namespace) -> int:
    folders = sorted(d for d in args.source.iterdir() if (d / "model.json").exists())
    infos = [json.loads((d / "model.json").read_text()) for d in folders]
    check_once([i["exp_id"] for i in infos], args.force_rerun_with_reason)

    rows = {}
    if TEST_CSV.exists():
        with TEST_CSV.open(newline="") as f:
            rows = {row["exp_id"]: row for row in csv.DictReader(f)}
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for folder, info in zip(folders, infos, strict=True):
        metrics = json.loads((folder / "metrics_test.json").read_text())
        rows[info["exp_id"]] = {
            **info,
            **{m: metrics[m] for m in METRIC_COLUMNS},
            "recorded_utc": now,
        }
        result_dir = RESULTS / "runs" / info["run_id"]
        shutil.copyfile(folder / "metrics_test.json", result_dir / "metrics_test.json")
        run_dir = REPO_ROOT / "runs" / info["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "predictions_test.json", run_dir / "predictions_test.json")
        if args.force_rerun_with_reason:
            new_log = not LOG_CSV.exists()
            with LOG_CSV.open("a", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                if new_log:
                    writer.writerow(["recorded_utc", "exp_id", "source", "reason"])
                writer.writerow(
                    [now, info["exp_id"], args.source.name, args.force_rerun_with_reason]
                )

    with TEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[e] for e in FINAL_MODELS if e in rows)
    logger.info("%s: %d rows", TEST_CSV, len(rows))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pred = sub.add_parser("predict", help="Score the final models on test (Kaggle CPU kernel).")
    pred.add_argument("--data-dir", default="kaggle", help="voc_prepared/ path, or 'kaggle'.")
    pred.add_argument("--out-dir", type=Path, required=True)
    pred.add_argument("--models", nargs="*", choices=FINAL_MODELS, help="Default: all.")
    rec = sub.add_parser("record", help="Write a pulled predict run into results/.")
    rec.add_argument("--from", dest="source", type=Path, required=True)
    for p in (pred, rec):
        p.add_argument("--force-rerun-with-reason", help="Allow an experiment that has a result.")
    args = parser.parse_args()
    return run_predict(args) if args.command == "predict" else run_record(args)


if __name__ == "__main__":
    sys.exit(main())
