"""Integrity checks on the prepared data; any mismatch stops the pipeline (CLAUDE.md hard rule 8).

Checks: official list sizes match PLAN.md §3, splits partition them exactly, every image file
exists, no zero-area boxes, and which images have no non-difficult object (allowed, but logged:
they act as background-only images in training). Also logs the class distribution per split.
"""

import logging
from pathlib import Path

import pandas as pd

from vocdet.constants import VOC_CLASSES
from vocdet.data.parse_voc import read_image_ids

logger = logging.getLogger(__name__)

# PLAN.md §3: detection ImageSets/Main list sizes for VOC2012.
EXPECTED_OFFICIAL_COUNTS = {"train": 5717, "val": 5823}


class IntegrityError(RuntimeError):
    """The data on disk disagrees with PLAN.md or with itself."""


def check_integrity(
    annotations: pd.DataFrame,
    splits: dict[str, list[str]],
    voc_dir: Path,
    expected_counts: dict[str, int] = EXPECTED_OFFICIAL_COUNTS,
) -> pd.DataFrame:
    """Run every check; return the per-split class distribution (objects, difficult excluded).

    expected_counts is a parameter only so tests can use a tiny fixture; real runs use the default.
    """
    main = voc_dir / "ImageSets" / "Main"
    official = {name: read_image_ids(main / f"{name}.txt") for name in expected_counts}
    errors = []
    for name, expected in expected_counts.items():
        if len(official[name]) != expected:
            errors.append(
                f"official {name}.txt has {len(official[name])} images, expected {expected}"
            )

    if set(splits["train"]) | set(splits["val"]) != set(official["train"]):
        errors.append("splits train + val do not equal the official train list")
    if set(splits["test"]) != set(official["val"]):
        errors.append("split test does not equal the official val list")
    if set(splits["train"]) & set(splits["val"]):
        errors.append("splits train and val overlap")

    all_ids = [image_id for ids in splits.values() for image_id in ids]
    missing_images = [i for i in all_ids if not (voc_dir / "JPEGImages" / f"{i}.jpg").exists()]
    if missing_images:
        errors.append(f"{len(missing_images)} image files missing, e.g. {missing_images[:3]}")

    zero_area = annotations[
        (annotations["x2"] <= annotations["x1"]) | (annotations["y2"] <= annotations["y1"])
    ]
    if len(zero_area):
        errors.append(
            f"{len(zero_area)} zero-area boxes, e.g. {zero_area['image_id'].head(3).tolist()}"
        )

    annotated = set(annotations["image_id"])
    unannotated = [i for i in all_ids if i not in annotated]
    if unannotated:
        errors.append(
            f"{len(unannotated)} split images have no annotation rows, e.g. {unannotated[:3]}"
        )

    if errors:
        raise IntegrityError("Data integrity check failed:\n- " + "\n- ".join(errors))

    kept = annotations[~annotations["difficult"]]
    with_objects = set(kept["image_id"])
    for split, ids in splits.items():
        only_difficult = [i for i in ids if i not in with_objects]
        if only_difficult:
            logger.warning(
                "%s: %d images have only difficult objects (kept as background-only): %s",
                split,
                len(only_difficult),
                only_difficult[:10],
            )

    distribution = pd.DataFrame(
        {
            split: kept[kept["image_id"].isin(set(ids))]["class_name"]
            .value_counts()
            .reindex(VOC_CLASSES, fill_value=0)
            for split, ids in splits.items()
        }
    )
    logger.info("Objects per class and split (difficult excluded):\n%s", distribution.to_string())
    logger.info("All integrity checks passed")
    return distribution
