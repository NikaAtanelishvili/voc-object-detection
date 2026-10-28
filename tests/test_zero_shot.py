"""zero_shot.py end to end with a fake model; data config, find_prepared_dir, resolve-only."""

import json
import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

from test_predict import FakeYOLO
from vocdet.cli import main
from vocdet.config import DataConfig, ZeroShotConfig, load_job_config
from vocdet.data.parse_voc import build_annotations
from vocdet.data.prepare import find_prepared_dir
from vocdet.eval.zero_shot import run_zero_shot

REPO_ROOT = Path(__file__).resolve().parents[1]
VOC_FIXTURE = Path(__file__).parent / "fixtures" / "voc"
RUN = "prepare_data_20261028-0208_7b7605e"


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
    # The deeper of the two mount layouts seen on Kaggle (ADR-011).
    data = tmp_path / "input" / "notebooks" / "owner" / "vocdet-prepare-data" / RUN / "voc_prepared"
    build_annotations(VOC_FIXTURE, data / "annotations.parquet")
    (data / "yolo" / "images" / "val").mkdir(parents=True)
    Image.new("RGB", (400, 300)).save(data / "yolo" / "images" / "val" / "2008_000002.jpg")
    splits = tmp_path / "splits"
    splits.mkdir()
    for name, ids in (("train", "2008_000001"), ("val", "2008_000002"), ("test", "")):
        (splits / f"{name}.txt").write_text(ids + "\n")
    return data, splits


def test_find_prepared_dir_at_any_depth(prepared: tuple[Path, Path], tmp_path: Path) -> None:
    data, _ = prepared
    assert find_prepared_dir(RUN, tmp_path / "input") == data
    shallow_root = tmp_path / "input2"
    shutil.copytree(data, shallow_root / "vocdet-prepare-data" / RUN / "voc_prepared")
    assert find_prepared_dir(RUN, shallow_root).parts[-3:] == (
        "vocdet-prepare-data",
        RUN,
        "voc_prepared",
    )


def test_find_prepared_dir_fails_fast_on_zero_or_many(
    prepared: tuple[Path, Path], tmp_path: Path
) -> None:
    data, _ = prepared
    other_run = "prepare_data_20991231-2359_0000000"
    with pytest.raises(FileNotFoundError, match="found 0") as excinfo:
        find_prepared_dir(other_run, tmp_path / "input")
    assert "notebooks" in str(excinfo.value)  # the listing shows what is mounted
    shutil.copytree(data, tmp_path / "input" / "vocdet-prepare-data" / RUN / "voc_prepared")
    with pytest.raises(FileNotFoundError, match="found 2"):
        find_prepared_dir(RUN, tmp_path / "input")


def test_data_config(tmp_path: Path) -> None:
    cfg = load_job_config(REPO_ROOT / "configs" / "data" / "voc2012.yaml", DataConfig)
    assert cfg.prep_run_id == RUN
    (tmp_path / "d.yaml").write_text("prep_run_id: voc_prepared\n")
    with pytest.raises(ValueError, match="prep_run_id"):
        load_job_config(tmp_path / "d.yaml", DataConfig)


def test_resolve_only_prints_path_and_skips_prediction(
    prepared: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    data, _ = prepared
    config = str(REPO_ROOT / "configs" / "experiments" / "E0b.yaml")
    assert main(["zero-shot", "--config", config, "--data-dir", str(data), "--resolve-only"]) == 0
    assert f"Prepared data: {data}" in capsys.readouterr().out


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
