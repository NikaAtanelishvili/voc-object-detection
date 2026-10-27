"""export_yolo.py and export_coco.py on the fixture VOC folder."""

import json
from pathlib import Path

import pytest
import yaml
from pycocotools.coco import COCO

from vocdet.constants import VOC_CLASSES
from vocdet.data.export_coco import build_coco, coco_image_id, export_coco
from vocdet.data.export_yolo import export_yolo, from_yolo_line, to_yolo_line
from vocdet.data.parse_voc import parse_dataset

VOC_FIXTURE = Path(__file__).parent / "fixtures" / "voc"
SPLITS = {"train": ["2008_000001"], "val": ["2008_000002"], "test": []}


@pytest.fixture
def annotations():
    return parse_dataset(VOC_FIXTURE, ["2008_000001", "2008_000002"])


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """Placeholder JPEG files; the exporter only links or copies them."""
    d = tmp_path / "JPEGImages"
    d.mkdir()
    for image_id in ("2008_000001", "2008_000002"):
        (d / f"{image_id}.jpg").write_bytes(b"\xff\xd8 fake jpeg")
    return d


def test_yolo_round_trip_within_1e4(annotations) -> None:
    for row in annotations.itertuples():
        line = to_yolo_line(row.class_id, row.x1, row.y1, row.x2, row.y2, row.width, row.height)
        cls, x1, y1, x2, y2 = from_yolo_line(line, row.width, row.height)
        assert cls == row.class_id
        for got, want in ((x1, row.x1), (y1, row.y1), (x2, row.x2), (y2, row.y2)):
            assert abs(got - want) < 1e-4


def test_yolo_line_is_normalized_center_format() -> None:
    # 0-based box [0, 0, 500, 375] in a 500x375 image is the whole image.
    assert (
        to_yolo_line(11, 0, 0, 500, 375, 500, 375)
        == "11 0.50000000 0.50000000 1.00000000 1.00000000"
    )


@pytest.mark.parametrize("mode", ["symlink", "copy"])
def test_export_yolo_layout(annotations, image_dir: Path, tmp_path: Path, mode: str) -> None:
    out = tmp_path / "yolo"
    yaml_path = export_yolo(annotations, SPLITS, image_dir, out, mode=mode)

    img = out / "images" / "train" / "2008_000001.jpg"
    assert img.is_symlink() == (mode == "symlink")
    assert img.read_bytes() == (image_dir / "2008_000001.jpg").read_bytes()

    # 2008_000001 has a dog and a difficult bottle: only the dog is a training label.
    lines = (out / "labels" / "train" / "2008_000001.txt").read_text().splitlines()
    assert [line.split()[0] for line in lines] == [str(VOC_CLASSES.index("dog"))]
    assert len((out / "labels" / "val" / "2008_000002.txt").read_text().splitlines()) == 2

    config = yaml.safe_load(yaml_path.read_text())
    assert config["train"] == "images/train" and config["test"] == "images/test"
    assert Path(config["path"]).is_absolute()
    assert config["names"] == dict(enumerate(VOC_CLASSES))


def test_coco_image_id() -> None:
    assert coco_image_id("2008_000123") == 2008000123
    with pytest.raises(ValueError, match="YYYY_NNNNNN"):
        coco_image_id("img_1")


def test_coco_json_loads_in_pycocotools(annotations, tmp_path: Path) -> None:
    export_coco(annotations, SPLITS, tmp_path)
    coco = COCO(str(tmp_path / "train.json"))
    assert coco.getImgIds() == [2008000001]
    assert sorted(coco.getCatIds()) == list(range(1, 21))
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[2008000001]))
    by_class = {coco.loadCats(a["category_id"])[0]["name"]: a for a in anns}
    assert by_class["bottle"]["iscrowd"] == 1  # difficult -> ignore region
    assert by_class["dog"]["iscrowd"] == 0
    # 0-based xyxy (0, 9, 500, 375) -> xywh (0, 9, 500, 366); area = w * h.
    assert by_class["dog"]["bbox"] == [0.0, 9.0, 500.0, 366.0]
    assert by_class["dog"]["area"] == 500.0 * 366.0
    assert json.loads((tmp_path / "test.json").read_text())["images"] == []


def test_coco_rejects_split_image_without_annotations(annotations) -> None:
    with pytest.raises(ValueError, match="no annotations"):
        build_coco(annotations, ["2008_000001", "2008_009999"])
