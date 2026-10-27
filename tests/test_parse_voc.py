"""parse_voc.py: the 1-based -> 0-based conversion, flags, parts, and invalid boxes."""

from pathlib import Path

import pandas as pd
import pytest

from vocdet.constants import VOC_CLASSES
from vocdet.data.parse_voc import (
    COLUMNS,
    AnnotationError,
    build_annotations,
    parse_annotation,
)

VOC_FIXTURE = Path(__file__).parent / "fixtures" / "voc"
ANN = VOC_FIXTURE / "Annotations"


def test_coordinates_are_converted_to_zero_based_xyxy() -> None:
    dog, bottle = parse_annotation(ANN / "2008_000001.xml")
    # xmin=1, xmax=500 in a 500-wide image covers every pixel column -> [0, 500].
    assert (dog["x1"], dog["y1"], dog["x2"], dog["y2"]) == (0.0, 9.0, 500.0, 375.0)
    assert (bottle["x1"], bottle["y1"], bottle["x2"], bottle["y2"]) == (47.0, 239.0, 60.0, 271.0)
    # Width in canonical coordinates equals the number of covered pixels (60 - 48 + 1).
    assert bottle["x2"] - bottle["x1"] == 13


def test_flags_class_ids_and_image_fields() -> None:
    dog, bottle = parse_annotation(ANN / "2008_000001.xml")
    assert dog["image_id"] == "2008_000001"
    assert (dog["width"], dog["height"]) == (500, 375)
    assert dog["truncated"] is True and dog["difficult"] is False
    assert bottle["difficult"] is True and bottle["truncated"] is False
    assert VOC_CLASSES[dog["class_id"]] == "dog"
    assert VOC_CLASSES[bottle["class_id"]] == "bottle"


def test_part_boxes_are_ignored_and_missing_flags_default_to_false() -> None:
    person, cat = parse_annotation(ANN / "2008_000002.xml")
    assert (person["x1"], person["y1"], person["x2"], person["y2"]) == (99.0, 49.0, 200.5, 290.0)
    assert cat["class_name"] == "cat"
    assert cat["difficult"] is False and cat["truncated"] is False


def write_xml(tmp_path: Path, name: str, box: tuple[str, str, str, str], w: int = 100) -> Path:
    xmin, ymin, xmax, ymax = box
    path = tmp_path / "bad.xml"
    path.write_text(
        f"<annotation><size><width>{w}</width><height>100</height></size><object><name>{name}"
        f"</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax>"
        f"<ymax>{ymax}</ymax></bndbox></object></annotation>"
    )
    return path


@pytest.mark.parametrize(
    "box",
    [
        ("10", "10", "101", "50"),  # xmax beyond width
        ("0", "10", "50", "50"),  # xmin=0 is impossible for 1-based indices -> x1 = -1
        ("50", "10", "49", "50"),  # zero width after conversion (x1 = 49 = x2)
    ],
)
def test_invalid_boxes_raise(tmp_path: Path, box: tuple[str, str, str, str]) -> None:
    with pytest.raises(AnnotationError, match="outside"):
        parse_annotation(write_xml(tmp_path, "dog", box))


def test_single_pixel_box_is_valid(tmp_path: Path) -> None:
    (row,) = parse_annotation(write_xml(tmp_path, "dog", ("50", "10", "50", "10")))
    assert (row["x2"] - row["x1"], row["y2"] - row["y1"]) == (1.0, 1.0)


def test_unknown_class_raises(tmp_path: Path) -> None:
    with pytest.raises(AnnotationError, match="unknown class"):
        parse_annotation(write_xml(tmp_path, "unicorn", ("10", "10", "20", "20")))


def test_build_annotations_writes_parquet(tmp_path: Path) -> None:
    out = tmp_path / "annotations.parquet"
    df = build_annotations(VOC_FIXTURE, out)
    assert list(df.columns) == COLUMNS
    assert len(df) == 4
    roundtrip = pd.read_parquet(out)
    pd.testing.assert_frame_equal(roundtrip, df)
    assert roundtrip["difficult"].dtype == bool
