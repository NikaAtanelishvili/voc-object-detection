"""integrity.py: passes on consistent fixture data and names each kind of failure."""

import shutil
from pathlib import Path

import pytest
from PIL import Image

from vocdet.data.integrity import IntegrityError, check_integrity
from vocdet.data.parse_voc import parse_dataset

VOC_FIXTURE = Path(__file__).parent / "fixtures" / "voc"
IDS = ["2008_000001", "2008_000002"]
SPLITS = {"train": ["2008_000001"], "val": [], "test": ["2008_000002"]}
COUNTS = {"train": 1, "val": 1}


@pytest.fixture
def voc_dir(tmp_path: Path) -> Path:
    """Copy of the fixture folder plus tiny JPEGs, so tests can delete files."""
    d = tmp_path / "VOC2012"
    shutil.copytree(VOC_FIXTURE, d)
    (d / "JPEGImages").mkdir()
    for image_id in IDS:
        Image.new("RGB", (8, 8)).save(d / "JPEGImages" / f"{image_id}.jpg")
    return d


@pytest.fixture
def annotations():
    return parse_dataset(VOC_FIXTURE, IDS)


def test_consistent_data_passes(annotations, voc_dir: Path) -> None:
    dist = check_integrity(annotations, SPLITS, voc_dir, expected_counts=COUNTS)
    assert dist.at["dog", "train"] == 1
    assert dist.at["bottle", "train"] == 0  # difficult objects are not counted
    assert dist.at["person", "test"] == 1
    assert len(dist) == 20


def test_wrong_official_count_fails(annotations, voc_dir: Path) -> None:
    with pytest.raises(IntegrityError, match="expected 5717"):
        check_integrity(annotations, SPLITS, voc_dir, expected_counts={"train": 5717, "val": 1})


def test_missing_image_fails(annotations, voc_dir: Path) -> None:
    (voc_dir / "JPEGImages" / "2008_000002.jpg").unlink()
    with pytest.raises(IntegrityError, match="1 image files missing"):
        check_integrity(annotations, SPLITS, voc_dir, expected_counts=COUNTS)


def test_split_mismatch_fails(annotations, voc_dir: Path) -> None:
    bad = {"train": [], "val": [], "test": IDS}
    with pytest.raises(IntegrityError, match="do not equal the official train"):
        check_integrity(annotations, bad, voc_dir, expected_counts=COUNTS)


def test_zero_area_box_fails(annotations, voc_dir: Path) -> None:
    annotations.loc[0, "x2"] = annotations.loc[0, "x1"]
    with pytest.raises(IntegrityError, match="zero-area"):
        check_integrity(annotations, SPLITS, voc_dir, expected_counts=COUNTS)


def test_only_difficult_image_is_logged_not_fatal(annotations, voc_dir: Path, caplog) -> None:
    annotations.loc[annotations["image_id"] == "2008_000001", "difficult"] = True
    check_integrity(annotations, SPLITS, voc_dir, expected_counts=COUNTS)
    assert "only difficult objects" in caplog.text
