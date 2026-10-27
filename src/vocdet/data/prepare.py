"""The whole data pipeline in one call, run by the `vocdet-prepare-data` Kaggle CPU kernel (P1.9).

download -> parse -> verify splits -> integrity -> export (YOLO + COCO) -> EDA. The output folder is
what training kernels mount via kernel_sources, so it holds everything they need and nothing else.
"""

import importlib.metadata
import json
import logging
import time
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


def wait_for_prepared_dir(path: Path, timeout_s: float = 120, poll_s: float = 10) -> Path:
    """Return path once path/annotations.parquet exists; fail with a listing of what is there.

    The path comes from configs/data/voc2012.yaml (observed on real kernels). The wait covers an
    input mount that is still attaching when the command starts: E0b's first run found nothing under
    /kaggle/input although the same metadata mounts correctly on CPU and GPU kernels (ADR-011).
    """
    deadline = time.monotonic() + timeout_s
    while not (path / "annotations.parquet").exists():
        if time.monotonic() >= deadline:
            root = next(parent for parent in path.parents if parent.exists())
            listing = sorted(str(p) for p in root.glob("*/*"))[:40]
            raise FileNotFoundError(
                f"No prepared data at {path} after {timeout_s:.0f} s. Under {root}: {listing}"
            )
        logger.warning("Prepared data not at %s yet; waiting %.0f s", path, poll_s)
        time.sleep(poll_s)
    logger.info("Prepared data found at %s", path)
    return path
