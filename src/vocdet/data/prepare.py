"""The whole data pipeline in one call, run by the `vocdet-prepare-data` Kaggle CPU kernel (P1.9).

download -> parse -> verify splits -> integrity -> export (YOLO + COCO) -> EDA. The output folder is
what training kernels mount via kernel_sources, so it holds everything they need and nothing else.
"""

import importlib.metadata
import json
import logging
import os
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


def find_prepared_dir(prep_run_id: str, root: Path = Path("/kaggle/input")) -> Path:
    """Return the one <prep_run_id>/voc_prepared/ folder under root that has annotations.parquet.

    Searched recursively because Kaggle mounts the same kernel_sources at different depths in
    different kernels (/kaggle/input/<slug>/... vs /kaggle/input/notebooks/<owner>/<slug>/...,
    ADR-011). Matching on the prep run ID keeps the data version pinned. Zero or several matches
    fail immediately, listing what is there.
    """
    matches = []
    seen = set()  # real paths already walked, so symlink loops cannot recurse forever
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        here = Path(dirpath)
        if here.name == prep_run_id and (here / "voc_prepared" / "annotations.parquet").is_file():
            matches.append(here / "voc_prepared")
        # Never descend into prepared data: ~23k images and labels, and nothing to find there.
        dirnames[:] = [d for d in dirnames if d != "voc_prepared"]

    if len(matches) != 1:
        listing = sorted(str(p) for depth in ("*", "*/*", "*/*/*") for p in root.glob(depth))[:60]
        raise FileNotFoundError(
            f"Expected exactly one {prep_run_id}/voc_prepared/annotations.parquet under {root}, "
            f"found {len(matches)}: {[str(m) for m in matches]}. Entries under {root}: {listing}"
        )
    logger.info("Prepared data found at %s", matches[0])
    return matches[0]
