"""train_yolo.py: writable dataset root, Ultralytics argument mapping, and the run layout."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from pydantic import ValidationError

from vocdet.config import YoloAugmentSettings, YoloConfig, load_job_config
from vocdet.constants import VOC_CLASSES
from vocdet.data.parse_voc import build_annotations
from vocdet.eval.evaluator import evaluate
from vocdet.eval.schema import GroundTruth, Prediction, read_predictions
from vocdet.train.train_yolo import (
    build_dataset_root,
    train_yolo,
    ultralytics_train_args,
    ultralytics_val_crosscheck,
)

VOC_FIXTURE = Path(__file__).parent / "fixtures" / "voc"

IDS = {"train": ["2008_000001", "2008_000003"], "val": ["2008_000002"], "test": ["2008_000004"]}
LABELS = {
    "2008_000001": "12 0.5 0.5 0.4 0.6\n",
    "2008_000003": "0 0.3 0.3 0.2 0.2\n14 0.6 0.6 0.2 0.4\n",
    "2008_000002": "7 0.5 0.5 0.5 0.5\n",
    "2008_000004": "1 0.5 0.5 0.1 0.1\n",
}


@pytest.fixture
def prepared(tmp_path: Path):
    """A read-only voc_prepared/yolo (like /kaggle/input) and the matching splits/ folder."""
    data = tmp_path / "voc_prepared"
    for split, ids in IDS.items():
        for kind in ("images", "labels"):
            (data / "yolo" / kind / split).mkdir(parents=True)
        for image_id in ids:
            Image.new("RGB", (320, 240), (120, 90, 60)).save(
                data / "yolo" / "images" / split / f"{image_id}.jpg"
            )
            (data / "yolo" / "labels" / split / f"{image_id}.txt").write_text(LABELS[image_id])
    splits = tmp_path / "splits"
    splits.mkdir()
    for split, ids in IDS.items():
        (splits / f"{split}.txt").write_text("\n".join(ids) + "\n")
    read_only = [d for d in (data / "yolo").rglob("*") if d.is_dir()]
    for d in read_only:
        d.chmod(0o555)
    yield data, splits
    for d in read_only:
        d.chmod(0o755)  # so pytest can clean up tmp_path


def make_config(**train) -> YoloConfig:
    return YoloConfig.model_validate(
        {
            "exp_id": "E_test",
            "train": {"weights": "yolo26n.pt", **train},
            "kaggle": {"command": "unused"},
        }  # fmt: skip
    )


def test_dataset_root_symlinks_images_copies_labels_and_leaves_out_test(
    prepared, tmp_path: Path
) -> None:
    data, _ = prepared
    root = tmp_path / "root"
    dataset_yaml = build_dataset_root(data, root, {s: IDS[s] for s in ("train", "val")})

    image = root / "images" / "train" / "2008_000001.jpg"
    assert image.is_symlink()
    assert image.resolve() == (data / "yolo" / "images" / "train" / "2008_000001.jpg").resolve()
    label = root / "labels" / "train" / "2008_000003.txt"
    assert not label.is_symlink() and label.read_text() == LABELS["2008_000003"]

    spec = yaml.safe_load(dataset_yaml.read_text())
    assert spec["path"] == str(root.resolve()) and Path(spec["path"]).is_absolute()
    assert spec["train"] == "images/train" and spec["val"] == "images/val"
    assert "test" not in spec  # hard rule 4: Ultralytics never sees the test split
    assert spec["names"] == dict(enumerate(VOC_CLASSES))
    assert not (root / "images" / "test").exists()


def test_ultralytics_finds_labels_through_symlinked_images(prepared, tmp_path: Path) -> None:
    from ultralytics.data import YOLODataset

    data, _ = prepared
    root = tmp_path / "root"
    build_dataset_root(data, root, {s: IDS[s] for s in ("train", "val")})
    dataset = YOLODataset(
        img_path=str(root / "images" / "train"),
        data={"names": dict(enumerate(VOC_CLASSES)), "nc": 20, "channels": 3},
        augment=False,
    )
    by_image = {
        Path(lb["im_file"]).stem: sorted(lb["cls"].ravel().tolist()) for lb in dataset.labels
    }
    assert by_image == {"2008_000001": [12.0], "2008_000003": [0.0, 14.0]}
    # The label cache lands in the writable copy; the read-only source is untouched.
    assert (root / "labels" / "train.cache").exists()
    assert not list((data / "yolo").rglob("*.cache"))


def test_train_args_pass_only_what_the_config_sets(tmp_path: Path) -> None:
    args = ultralytics_train_args(make_config(), tmp_path / "data.yaml", Path("runs/x"))
    assert args["seed"] == 42 and args["deterministic"] is True
    assert args["epochs"] == 50 and args["imgsz"] == 640 and args["batch"] == 16
    assert args["patience"] == 15
    assert Path(args["project"]).is_absolute() and args["name"] == "ultralytics"
    assert "optimizer" not in args and "freeze" not in args and "mosaic" not in args

    e4 = make_config(augment={"mosaic": 0, "mixup": 0, "hsv_h": 0, "hsv_s": 0, "hsv_v": 0,
                              "scale": 0, "translate": 0})  # fmt: skip
    args = ultralytics_train_args(e4, tmp_path / "data.yaml", tmp_path)
    assert args["mosaic"] == 0 and args["translate"] == 0 and "fliplr" not in args
    assert ultralytics_train_args(make_config(freeze=10), tmp_path, tmp_path)["freeze"] == 10
    assert "close_mosaic" not in args
    assert (
        ultralytics_train_args(make_config(close_mosaic=1), tmp_path, tmp_path)["close_mosaic"] == 1
    )


def test_unknown_augmentation_key_is_an_error() -> None:
    with pytest.raises(ValidationError):
        YoloAugmentSettings.model_validate({"mosiac": 0})


class FakeYOLO:
    """Stands in for ultralytics.YOLO: writes what Ultralytics writes and predicts one fixed box."""

    names = dict(enumerate(VOC_CLASSES))

    def __init__(self, weights: str) -> None:
        self.weights = weights

    def train(self, **kwargs) -> None:
        self.kwargs = kwargs
        spec = yaml.safe_load(Path(kwargs["data"]).read_text())
        self.num_train_images = len(os.listdir(Path(spec["path"]) / spec["train"]))
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        (save_dir / "weights").mkdir(parents=True)
        (save_dir / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.1\n")
        for name in ("best.pt", "last.pt"):
            (save_dir / "weights" / name).write_bytes(b"fake")
        self.trainer = SimpleNamespace(save_dir=save_dir)

    def predict(self, sources: list[str], **kwargs) -> list:
        boxes = SimpleNamespace(
            xyxy=torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
            conf=torch.tensor([0.9]),
            cls=torch.tensor([11.0]),
        )
        return [SimpleNamespace(boxes=boxes) for _ in sources]

    def val(self, **kwargs) -> SimpleNamespace:
        self.val_kwargs = kwargs
        assert Path(kwargs["data"]).exists()  # the dataset root still exists during val
        return SimpleNamespace(box=SimpleNamespace(map50=0.25, map=0.125))


def test_train_yolo_writes_the_run_artifacts(prepared, tmp_path: Path) -> None:
    data, splits = prepared
    build_annotations(VOC_FIXTURE, data / "annotations.parquet")
    cfg = make_config(epochs=1, limit_train_images=1)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump()))
    out_dir = tmp_path / "runs" / "E_test_20261029-0000_abc1234"

    models = []
    metrics = train_yolo(
        cfg, config_path, data, out_dir, splits_dir=splits,
        load_model=lambda w: models.append(FakeYOLO(w)) or models[-1],
    )  # fmt: skip

    trained, best = models
    assert trained.weights == "yolo26n.pt" and trained.num_train_images == 1
    assert best.weights == str(out_dir / "weights" / "best.pt")  # val uses best.pt
    assert best.val_kwargs["split"] == "val" and best.val_kwargs["plots"] is False
    assert (out_dir / "weights" / "last.pt").exists()
    assert (out_dir / "train_log.csv").read_text().startswith("epoch,")
    assert not (out_dir / "dataset").exists()
    preds = read_predictions(out_dir / "predictions_val.json")
    assert [p.image_id for p in preds] == ["2008_000002"]
    assert json.loads((out_dir / "metrics_val.json").read_text())["map50"] == metrics["map50"]
    meta = json.loads((out_dir / "run_meta.json").read_text())
    assert meta["status"] == "complete" and meta["num_train_images"] == 1
    check = meta["ultralytics_val_crosscheck"]
    assert check["ultralytics_map50"] == 0.25 and check["ultralytics_map50_95"] == 0.125
    assert check["ours_map50_difficult_removed"] == metrics["map50"]  # val image has no difficult


def test_crosscheck_counts_a_hit_on_a_difficult_object_as_false_positive(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # One dog is difficult. The higher-scored detection lies on it: ignored by our evaluator,
    # but a false positive once difficult objects are removed (as in the YOLO labels).
    gt = [
        GroundTruth(
            "img",
            np.array([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=np.float32),
            np.array([11, 11]),
            np.array([False, True]),
        )
    ]
    preds = [
        Prediction(
            "img",
            np.array([[200, 200, 300, 300], [0, 0, 100, 100]], dtype=np.float32),
            np.array([0.9, 0.6], dtype=np.float32),
            np.array([11, 11]),
        )
    ]
    dataset_yaml = tmp_path / "voc_yolo.yaml"
    dataset_yaml.write_text("unused by the fake\n")
    check = ultralytics_val_crosscheck(
        FakeYOLO("best.pt"), dataset_yaml, make_config(), tmp_path, preds, gt
    )
    assert evaluate(preds, gt)["map50"] == pytest.approx(1.0)
    assert check["ours_map50_difficult_removed"] < 0.6
    # The fake's 0.25 is far from our difficult-removed score, so the gap is flagged.
    assert "differs from ours" in caplog.text


@pytest.mark.parametrize("name", ["E0a_yolo", "E0a_yolo_timing_n", "E0a_yolo_timing_s", "E2", "E3"])
def test_experiment_configs_are_valid(name: str) -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / f"{name}.yaml"
    cfg = load_job_config(path, YoloConfig)
    assert cfg.kaggle.pip == ["ultralytics==8.4.191"]
    assert "vocdet-prepare-data" in cfg.kaggle.sources
