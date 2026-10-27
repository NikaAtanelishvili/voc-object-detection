"""zero_shot.py end to end with a fake model, and find_prepared_dir."""

import json
import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

from test_predict import FakeYOLO
from vocdet.config import ZeroShotConfig, load_job_config
from vocdet.data.parse_voc import build_annotations
from vocdet.data.prepare import find_prepared_dir
from vocdet.eval.zero_shot import run_zero_shot

REPO_ROOT = Path(__file__).resolve().parents[1]
VOC_FIXTURE = Path(__file__).parent / "fixtures" / "voc"


def test_e0b_config_is_valid() -> None:
    cfg = load_job_config(REPO_ROOT / "configs" / "experiments" / "E0b.yaml", ZeroShotConfig)
    assert cfg.predict.split == "val"
    assert "vocdet-prepare-data" in cfg.kaggle.sources
    assert any(p.startswith("ultralytics==") for p in cfg.kaggle.pip)


def test_zero_shot_config_rejects_test_split(tmp_path: Path) -> None:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "experiments" / "E0b.yaml").read_text())
    raw["predict"]["split"] = "test"
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="split"):
        load_job_config(tmp_path / "c.yaml", ZeroShotConfig)


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal voc_prepared/ (parquet + val images) and a splits/ folder."""
    data = tmp_path / "input" / "prep-kernel" / "voc_prepared"
    build_annotations(VOC_FIXTURE, data / "annotations.parquet")
    (data / "yolo" / "images" / "val").mkdir(parents=True)
    Image.new("RGB", (400, 300)).save(data / "yolo" / "images" / "val" / "2008_000002.jpg")
    splits = tmp_path / "splits"
    splits.mkdir()
    for name, ids in (("train", "2008_000001"), ("val", "2008_000002"), ("test", "")):
        (splits / f"{name}.txt").write_text(ids + "\n")
    return data, splits


def test_find_prepared_dir(prepared: tuple[Path, Path], tmp_path: Path) -> None:
    data, _ = prepared
    assert find_prepared_dir(tmp_path / "input") == data
    shutil.copytree(data, tmp_path / "input" / "other" / "voc_prepared")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        find_prepared_dir(tmp_path / "input")


def test_run_zero_shot_writes_run_artifacts(prepared: tuple[Path, Path], tmp_path: Path) -> None:
    data, splits = prepared
    config_path = REPO_ROOT / "configs" / "experiments" / "E0b.yaml"
    cfg = load_job_config(config_path, ZeroShotConfig)
    out = tmp_path / "runs" / "E0b_20261028-0900_abc1234"
    metrics = run_zero_shot(cfg, config_path, data, out, splits_dir=splits, model=FakeYOLO())
    for name in ("config.yaml", "run_meta.json", "predictions_val.json", "metrics_val.json"):
        assert (out / name).exists(), name
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["status"] == "complete" and meta["data_dir"] == str(data)
    assert metrics["num_images"] == 1
    assert metrics["map50"] == pytest.approx(0.0)  # the fake's single "cat" box hits nothing
