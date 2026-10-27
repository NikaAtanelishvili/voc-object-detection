"""Parse VOC XML annotations into one canonical table (one row per object), saved as parquet.

Coordinates: VOC XML boxes are 1-based inclusive pixel indices. We convert to 0-based float xyxy
with (xmin - 1, ymin - 1, xmax, ymax): a box covering pixels 1..W becomes [0, W], so the canonical
width x2 - x1 equals the number of covered pixels. Every later stage (exporters, dataset, evaluator)
reads this table, so the conversion happens exactly once, here.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from vocdet.constants import VOC_CLASSES

logger = logging.getLogger(__name__)

COLUMNS = [
    "image_id",
    "width",
    "height",
    "class_name",
    "class_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "difficult",
    "truncated",
]
CLASS_TO_ID = {name: i for i, name in enumerate(VOC_CLASSES)}


class AnnotationError(ValueError):
    """An XML annotation is malformed or has an impossible box."""


def _int_flag(obj: ET.Element, tag: str) -> bool:
    """`difficult`/`truncated` are "0"/"1"; some VOC2012 objects omit them, which means 0."""
    text = obj.findtext(tag)
    return text is not None and int(text.strip()) == 1


def parse_annotation(xml_path: Path) -> list[dict]:
    """Return one dict per object in a VOC XML file, with 0-based xyxy coordinates.

    Uses obj.find("bndbox") (a direct child) on purpose: person objects can contain <part> elements
    (head, hand, foot) with their own <bndbox>, which must not be mistaken for the person's box.
    """
    root = ET.parse(xml_path).getroot()
    image_id = xml_path.stem
    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))
    if width <= 0 or height <= 0:
        raise AnnotationError(f"{xml_path}: invalid image size {width}x{height}")

    rows = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name").strip()
        if class_name not in CLASS_TO_ID:
            raise AnnotationError(f"{xml_path}: unknown class {class_name!r}")
        box = obj.find("bndbox")
        # float() first: a few VOC2012 files store coordinates like "273.5".
        xmin, ymin, xmax, ymax = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, y1, x2, y2 = xmin - 1, ymin - 1, xmax, ymax
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise AnnotationError(
                f"{xml_path}: {class_name} box (x1={x1}, y1={y1}, x2={x2}, y2={y2}) is outside "
                f"the {width}x{height} image or has zero area"
            )
        rows.append(
            {
                "image_id": image_id,
                "width": width,
                "height": height,
                "class_name": class_name,
                "class_id": CLASS_TO_ID[class_name],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "difficult": _int_flag(obj, "difficult"),
                "truncated": _int_flag(obj, "truncated"),
            }
        )
    return rows


def read_image_ids(list_file: Path) -> list[str]:
    """Read an ImageSets/Main/*.txt list (one image ID per line)."""
    return [line.strip() for line in list_file.read_text().splitlines() if line.strip()]


def parse_dataset(voc_dir: Path, image_ids: list[str]) -> pd.DataFrame:
    """Parse the given images' annotations into the canonical DataFrame (fixed column order)."""
    rows: list[dict] = []
    for image_id in image_ids:
        rows.extend(parse_annotation(voc_dir / "Annotations" / f"{image_id}.xml"))
    df = pd.DataFrame(rows, columns=COLUMNS)
    logger.info("Parsed %d objects from %d images", len(df), len(image_ids))
    return df


def build_annotations(voc_dir: Path, out_path: Path) -> pd.DataFrame:
    """Parse all detection images (ImageSets/Main/trainval.txt) and write the parquet file.

    Only the detection trainval list is parsed: Annotations/ also holds action- and layout-only
    images that are not part of the detection task (ADR-007).
    """
    image_ids = read_image_ids(voc_dir / "ImageSets" / "Main" / "trainval.txt")
    df = parse_dataset(voc_dir, image_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)
    return df
