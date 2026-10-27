"""Export the canonical table to the Ultralytics YOLO dataset layout.

Layout under out_dir:
    images/{train,val,test}/<id>.jpg   symlinks to JPEGImages (or copies, see `mode`)
    labels/{train,val,test}/<id>.txt   one "cls cx cy w h" line per non-difficult object, normalized
    voc_yolo.yaml                      path, train, val, test, names

Difficult objects are excluded from labels, so YOLO never trains on them. Evaluation does not use
these labels: it uses the COCO JSON, where difficult objects become ignore regions (export_coco.py).
"""

import logging
import shutil
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from vocdet.constants import VOC_CLASSES

logger = logging.getLogger(__name__)


def to_yolo_line(
    class_id: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> str:
    """0-based absolute xyxy -> "cls cx cy w h" normalized to [0, 1] by the image size.

    8 decimals (not the usual 6) keep the round-trip error far below 1e-4 px for VOC-sized images.
    """
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"{class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}"


def from_yolo_line(line: str, width: int, height: int) -> tuple[int, float, float, float, float]:
    """Inverse of to_yolo_line, used by the round-trip test."""
    cls, cx, cy, w, h = line.split()
    cx_abs, cy_abs = float(cx) * width, float(cy) * height
    w_abs, h_abs = float(w) * width, float(h) * height
    return (
        int(cls),
        cx_abs - w_abs / 2,
        cy_abs - h_abs / 2,
        cx_abs + w_abs / 2,
        cy_abs + h_abs / 2,
    )


def place_image(src: Path, dst: Path, mode: Literal["symlink", "copy"]) -> None:
    """Symlink is fast and free locally. On Kaggle, kernel outputs do not keep symlinks into
    /kaggle/input, so the prep kernel uses copies."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def export_yolo(
    annotations: pd.DataFrame,
    splits: dict[str, list[str]],
    image_dir: Path,
    out_dir: Path,
    mode: Literal["symlink", "copy"] = "symlink",
) -> Path:
    """Write images, labels, and voc_yolo.yaml; return the yaml path."""
    kept = annotations[~annotations["difficult"]]
    by_image = {image_id: group for image_id, group in kept.groupby("image_id")}

    for split, image_ids in splits.items():
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        n_empty = 0
        for image_id in image_ids:
            place_image(image_dir / f"{image_id}.jpg", img_out / f"{image_id}.jpg", mode)
            lines = []
            if image_id in by_image:
                for row in by_image[image_id].itertuples():
                    lines.append(
                        to_yolo_line(
                            row.class_id, row.x1, row.y1, row.x2, row.y2, row.width, row.height
                        )
                    )
            else:
                n_empty += 1  # only difficult objects: an empty label file = a background image
            (lbl_out / f"{image_id}.txt").write_text("".join(f"{line}\n" for line in lines))
        logger.info("YOLO %s: %d images (%d with no labels)", split, len(image_ids), n_empty)

    yaml_path = out_dir / "voc_yolo.yaml"
    config = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(VOC_CLASSES)),
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return yaml_path
