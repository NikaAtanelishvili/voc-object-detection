"""train_frcnn.py: LR schedule, configs, and a tiny end-to-end CPU run with resume."""

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from vocdet.config import FrcnnConfig, FrcnnTrainSettings, load_job_config
from vocdet.data.parse_voc import build_annotations
from vocdet.train.train_frcnn import TRAIN_LOG_COLUMNS, lr_at, start_log, train_frcnn
from vocdet.utils.kaggle_input import find_run_file

REPO_ROOT = Path(__file__).resolve().parents[1]
VOC_FIXTURE = Path(__file__).parent / "fixtures" / "voc"


def test_lr_schedule() -> None:
    t = FrcnnTrainSettings(lr=0.01, warmup_iters=500, warmup_factor=0.001, milestones=[8, 11])
    assert lr_at(0, 0, t) == pytest.approx(0.01 * 0.001)  # warm-up starts low
    assert lr_at(250, 0, t) == pytest.approx(0.01 * (0.001 + 0.999 * 0.5))  # linear in between
    assert lr_at(500, 0, t) == pytest.approx(0.01)  # warm-up done
    assert lr_at(5000, 7, t) == pytest.approx(0.01)
    assert lr_at(6000, 8, t) == pytest.approx(0.001)  # first milestone
    assert lr_at(8000, 11, t) == pytest.approx(0.0001)  # second milestone


@pytest.mark.parametrize("name", ["E1", "E0a_frcnn", "E0a_frcnn_timing"])
def test_experiment_configs_are_valid(name: str) -> None:
    cfg = load_job_config(REPO_ROOT / "configs" / "experiments" / f"{name}.yaml", FrcnnConfig)
    assert cfg.model.min_size == 600 and cfg.model.max_size == 1000
    assert "vocdet-prepare-data" in cfg.kaggle.sources


def test_milestones_must_lie_inside_training() -> None:
    with pytest.raises(ValueError, match="milestones"):
        FrcnnTrainSettings(epochs=12, milestones=[8, 12])


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Path, Path]:
    """voc_prepared/ with one train and one val image, and the matching splits/ folder."""
    data = tmp_path / "voc_prepared"
    build_annotations(VOC_FIXTURE, data / "annotations.parquet")
    for split, image_id, size in (
        ("train", "2008_000001", (500, 375)),
        ("val", "2008_000002", (400, 300)),
    ):
        (data / "yolo" / "images" / split).mkdir(parents=True)
        Image.new("RGB", size, (120, 90, 60)).save(
            data / "yolo" / "images" / split / f"{image_id}.jpg"
        )
    splits = tmp_path / "splits"
    splits.mkdir()
    for name, ids in (("train", "2008_000001"), ("val", "2008_000002"), ("test", "")):
        (splits / f"{name}.txt").write_text(ids + "\n")
    return data, splits


def tiny_config(tmp_path: Path, epochs: int, milestones: list[int]) -> tuple[FrcnnConfig, Path]:
    raw = {
        "exp_id": "E_test",
        "model": {"pretrained": False, "min_size": 128, "max_size": 160},
        "train": {
            "epochs": epochs, "batch_size": 1, "warmup_iters": 1, "milestones": milestones,
            "amp": False, "num_workers": 0, "eval_every": 1,
        },
        "kaggle": {"command": "unused"},
    }  # fmt: skip
    path = tmp_path / f"cfg_{epochs}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return FrcnnConfig.model_validate(raw), path


def rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def test_end_to_end_training_and_resume(prepared: tuple[Path, Path], tmp_path: Path) -> None:
    data, splits = prepared
    cpu = torch.device("cpu")

    cfg1, path1 = tiny_config(tmp_path, epochs=1, milestones=[])
    first = tmp_path / "runs" / "first"
    metrics = train_frcnn(cfg1, path1, data, first, splits_dir=splits, device=cpu)
    for name in (
        "config.yaml",
        "run_meta.json",
        "train_log.csv",
        "val_log.csv",
        "metrics_val.json",
        "predictions_val.json",
        "weights/best.pt",
        "weights/last.pt",
    ):
        assert (first / name).exists(), name  # fmt: skip
    log = rows(first / "train_log.csv")
    assert len(log) == 1  # one image, batch 1, one epoch
    assert 0 <= float(log[0]["data_s"]) <= float(log[0]["iter_s"])  # data wait is part of the step
    assert rows(first / "val_log.csv")[0]["is_best"] == "True"
    assert metrics["num_images"] == 1
    assert json.loads((first / "metrics_val.json").read_text()) == metrics
    meta = json.loads((first / "run_meta.json").read_text())
    assert meta["status"] == "complete" and meta["best_epoch"] == 0

    # A session that died during epoch 1 leaves log rows from it, after the epoch-0 last.pt.
    with (first / "train_log.csv").open("a") as f:
        f.write("1,1,0.001,9.9,1,1,1,1,1.0,0.1,0.5\n")

    # Resume as a 2-epoch run in a new, empty folder (a new Kaggle session): only epoch 1 is
    # trained, and the logs are carried over from the source run without the unfinished epoch.
    cfg2, path2 = tiny_config(tmp_path, epochs=2, milestones=[1])
    second = tmp_path / "runs" / "second"
    second.mkdir(parents=True)
    train_frcnn(cfg2, path2, data, second, splits_dir=splits, resume=first / "weights" / "last.pt",
                device=cpu)  # fmt: skip
    epochs = [int(r["epoch"]) for r in rows(second / "train_log.csv")]
    assert epochs == [0, 1]  # one row per epoch: the planted epoch-1 row is gone
    assert float(rows(second / "train_log.csv")[1]["loss"]) != pytest.approx(9.9)
    assert [int(r["epoch"]) for r in rows(second / "val_log.csv")] == [0, 1]
    steps = [int(r["step"]) for r in rows(second / "train_log.csv")]
    assert steps == [0, 1]  # the global step continues across the resume
    assert float(rows(second / "train_log.csv")[1]["lr"]) == pytest.approx(0.001)  # milestone 1
    assert (second / "weights" / "best.pt").exists()
    meta2 = json.loads((second / "run_meta.json").read_text())
    assert meta2["resumed_from"] == str(first / "weights" / "last.pt")


def test_start_log_without_a_source_log_writes_the_header(tmp_path: Path) -> None:
    # Fresh run, or a source run that died before writing its logs: header only, no crash.
    for source in (None, tmp_path / "missing" / "train_log.csv"):
        start_log(tmp_path / "train_log.csv", TRAIN_LOG_COLUMNS, source, last_epoch=0)
        assert (tmp_path / "train_log.csv").read_text() == ",".join(TRAIN_LOG_COLUMNS) + "\n"


def test_find_run_file_finds_last_pt_at_the_deep_mount(tmp_path: Path) -> None:
    run = "E1_20261029-1430_abc1234"
    last = tmp_path / "input" / "notebooks" / "owner" / "vocdet-e1" / run / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"x")
    assert find_run_file(run, "weights/last.pt", tmp_path / "input") == last
    with pytest.raises(FileNotFoundError, match="found 0"):
        find_run_file("E1_20991231-2359_0000000", "weights/last.pt", tmp_path / "input")
