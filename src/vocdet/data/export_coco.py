"""Export the canonical table to one COCO-format JSON per split, the evaluator's ground truth.

- category_id = VOC class_id + 1 (COCO ids start at 1)
- bbox = [x, y, w, h] in 0-based absolute pixels; area = w * h (box area drives the size buckets)
- iscrowd = difficult: COCO eval treats crowd boxes as ignore regions, so a detection on a difficult
  object is neither a true nor a false positive, matching the VOC protocol (PLAN.md §3)
- image id = int(YYYY_NNNNNN without "_"), deterministic and reversible
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

from vocdet.constants import VOC_CLASSES

logger = logging.getLogger(__name__)

VOC_ID_PATTERN = re.compile(r"^(\d{4})_(\d{6})$")


def coco_image_id(voc_id: str) -> int:
    """'2008_000123' -> 2008000123. COCO needs integer image ids."""
    match = VOC_ID_PATTERN.match(voc_id)
    if not match:
        raise ValueError(f"Unexpected VOC image id {voc_id!r} (expected YYYY_NNNNNN)")
    return int(match.group(1) + match.group(2))


def build_coco(annotations: pd.DataFrame, image_ids: list[str]) -> dict:
    """COCO dict for the given images.

    Image sizes come from the annotation rows, so every listed image needs at least one row
    (difficult objects count). VOC detection images always have one; anything else is a data error.
    """
    split_df = annotations[annotations["image_id"].isin(set(image_ids))]
    sizes = split_df.groupby("image_id")[["width", "height"]].first()
    missing = set(image_ids) - set(sizes.index)
    if missing:
        raise ValueError(
            f"{len(missing)} split images have no annotations, e.g. {sorted(missing)[:3]}"
        )

    images = [
        {
            "id": coco_image_id(image_id),
            "file_name": f"{image_id}.jpg",
            "width": int(sizes.at[image_id, "width"]),
            "height": int(sizes.at[image_id, "height"]),
        }
        for image_id in sorted(image_ids)
    ]
    coco_annotations = []
    for ann_id, row in enumerate(split_df.itertuples(), start=1):
        w, h = row.x2 - row.x1, row.y2 - row.y1
        coco_annotations.append(
            {
                "id": ann_id,
                "image_id": coco_image_id(row.image_id),
                "category_id": int(row.class_id) + 1,
                "bbox": [float(row.x1), float(row.y1), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": int(row.difficult),
            }
        )
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(VOC_CLASSES)]
    return {"images": images, "annotations": coco_annotations, "categories": categories}


def export_coco(annotations: pd.DataFrame, splits: dict[str, list[str]], out_dir: Path) -> None:
    """Write out_dir/<split>.json for every split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, image_ids in splits.items():
        coco = build_coco(annotations, image_ids)
        (out_dir / f"{split}.json").write_text(json.dumps(coco))
        logger.info(
            "COCO %s: %d images, %d objects", split, len(coco["images"]), len(coco["annotations"])
        )
