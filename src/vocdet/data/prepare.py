"""The whole data pipeline in one call, run by the `vocdet-prepare-data` Kaggle CPU kernel (P1.9).

download -> parse -> verify splits -> integrity -> export (YOLO + COCO) -> EDA. The output folder is
what training kernels mount via kernel_sources, so it holds everything they need and nothing else.
"""

import importlib.metadata
import json
import logging
from pathlib import Path

from vocdet.data.eda import run_eda
from vocdet.data.export_coco import export_coco
from vocdet.data.export_yolo import export_yolo
from vocdet.data.integrity import EXPECTED_OFFICIAL_COUNTS, check_integrity
from vocdet.data.parse_voc import build_annotations
from vocdet.data.splits import build_splits

logger = logging.getLogger(__name__)

RECORDED_PACKAGES = ("pandas", "numpy", "pyarrow", "matplotlib", "tabulate", "pycocotools", "torch")


def package_versions() -> dict[str, str | None]:
    """Versions actually used, so a Kaggle image change is visible in the outputs."""
    versions: dict[str, str | None] = {}
    for name in RECORDED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def prepare_all(
    voc_dir: Path,
    splits_dir: Path,
    out_dir: Path,
    image_mode: str = "copy",
    expected_counts: dict[str, int] = EXPECTED_OFFICIAL_COUNTS,
) -> None:
    """Build every prepared artifact from an extracted VOC2012 folder into out_dir.

    build_splits recomputes the splits from this machine's official lists and compares them with the
    committed files: identical content is a no-op, any difference raises SplitsFrozenError. That
    proves Kaggle's copy of the data yields exactly our frozen splits.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = package_versions()
    logger.info("Package versions: %s", versions)
    (out_dir / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")

    annotations = build_annotations(voc_dir, out_dir / "annotations.parquet")
    splits = build_splits(voc_dir, splits_dir)
    check_integrity(annotations, splits, voc_dir, expected_counts=expected_counts)
    export_yolo(annotations, splits, voc_dir / "JPEGImages", out_dir / "yolo", mode=image_mode)
    export_coco(annotations, splits, out_dir / "coco")
    run_eda(annotations, splits, voc_dir, out_dir / "eda", out_dir / "EDA.md")
    logger.info("Prepared data written to %s", out_dir)


def find_prepared_dir(root: Path = Path("/kaggle/input")) -> Path:
    """Locate voc_prepared/ under a Kaggle input mount and log where it is.

    The mount layout of kernel_sources is not documented precisely, so we search the two plausible
    depths instead of hard-coding a path (PLAN.md P1.9: the first GPU kernel logs the real path).
    """
    patterns = ("*/voc_prepared/annotations.parquet", "*/*/voc_prepared/annotations.parquet")
    found = sorted({p.parent for pattern in patterns for p in root.glob(pattern)})
    if len(found) != 1:
        listing = sorted(str(p) for p in root.glob("*/*/*"))[:40]
        raise FileNotFoundError(
            f"Expected exactly one voc_prepared/ under {root}, found {found}. "
            f"First entries 3 levels down: {listing}"
        )
    logger.info("Prepared data found at %s", found[0])
    return found[0]
