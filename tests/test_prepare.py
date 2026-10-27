"""prepare.py: the full pipeline on the fixture folder, including the frozen-splits check."""

import shutil
from pathlib import Path

import pytest
from PIL import Image
from pycocotools.coco import COCO

from vocdet.data.prepare import prepare_all
from vocdet.data.splits import SplitsFrozenError

VOC_FIXTURE = Path(__file__).parent / "fixtures" / "voc"
COUNTS = {"train": 1, "val": 1}


@pytest.fixture
def voc_dir(tmp_path: Path) -> Path:
    d = tmp_path / "VOC2012"
    shutil.copytree(VOC_FIXTURE, d)
    (d / "JPEGImages").mkdir()
    for image_id, size in (("2008_000001", (500, 375)), ("2008_000002", (400, 300))):
        Image.new("RGB", size).save(d / "JPEGImages" / f"{image_id}.jpg")
    return d


def splits_for_fixture(split_dir: Path) -> Path:
    """The splits the fixture's official lists produce: 1 train image, 10% of 1 rounds to 0 val."""
    split_dir.mkdir()
    (split_dir / "train.txt").write_text("2008_000001\n")
    (split_dir / "val.txt").write_text("\n")
    (split_dir / "test.txt").write_text("2008_000002\n")
    return split_dir


def test_prepare_all_writes_every_artifact(voc_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "voc_prepared"
    prepare_all(voc_dir, splits_for_fixture(tmp_path / "splits"), out, expected_counts=COUNTS)
    for rel in [
        "versions.json",
        "annotations.parquet",
        "yolo/voc_yolo.yaml",
        "yolo/labels/train/2008_000001.txt",
        "yolo/images/test/2008_000002.jpg",
        "coco/test.json",
        "eda/summary.json",
        "EDA.md",
        "figures/eda_samples.jpg",
    ]:
        assert (out / rel).exists(), rel
    assert not (out / "yolo/images/train/2008_000001.jpg").is_symlink()  # copy mode by default
    assert COCO(str(out / "coco/test.json")).getImgIds() == [2008000002]


def test_prepare_all_refuses_different_splits(voc_dir: Path, tmp_path: Path) -> None:
    split_dir = splits_for_fixture(tmp_path / "splits")
    (split_dir / "test.txt").write_text("2008_000001\n")
    with pytest.raises(SplitsFrozenError):
        prepare_all(voc_dir, split_dir, tmp_path / "out", expected_counts=COUNTS)
