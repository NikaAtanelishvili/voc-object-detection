"""eda.py: size buckets and an end-to-end run on the fixture (tables, figures, markdown)."""

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from vocdet.data.eda import run_eda, size_bucket
from vocdet.data.parse_voc import parse_dataset

VOC_FIXTURE = Path(__file__).parent / "fixtures" / "voc"
IDS = ["2008_000001", "2008_000002"]


@pytest.mark.parametrize(
    ("area", "bucket"),
    [(0.5, "small"), (32 * 32 - 1, "small"), (32 * 32, "medium"), (96 * 96 - 1, "medium"),
     (96 * 96, "large")],
)  # fmt: skip
def test_size_bucket_boundaries(area: float, bucket: str) -> None:
    assert size_bucket(area) == bucket


def test_run_eda_on_fixture(tmp_path: Path) -> None:
    voc_dir = tmp_path / "VOC2012"
    shutil.copytree(VOC_FIXTURE, voc_dir)
    (voc_dir / "JPEGImages").mkdir()
    for image_id, size in zip(IDS, [(500, 375), (400, 300)], strict=True):
        Image.new("RGB", size).save(voc_dir / "JPEGImages" / f"{image_id}.jpg")
    splits = {"train": ["2008_000001"], "val": ["2008_000002"], "test": ["2009_000001"]}
    report = tmp_path / "reports" / "EDA.md"

    summary = run_eda(parse_dataset(VOC_FIXTURE, IDS), splits, voc_dir, tmp_path / "eda", report)

    assert summary["images"] == {"train": 1, "val": 1, "test": 1}
    assert summary["objects_non_difficult"] == {"train": 1, "val": 2}  # difficult bottle excluded
    assert summary["top_class"] == "dog"
    assert json.loads((tmp_path / "eda" / "summary.json").read_text()) == summary
    text = report.read_text()
    assert "{" not in text.split("## Tables")[0]  # every template field was filled
    figures = sorted((tmp_path / "reports" / "figures").glob("eda_*"))
    assert len(figures) == 7
    assert all(f.stat().st_size < 500_000 for f in figures)
