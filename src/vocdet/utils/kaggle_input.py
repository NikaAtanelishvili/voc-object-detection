"""Find one file of an earlier run's output inside a Kaggle kernel's /kaggle/input."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

KAGGLE_INPUT = Path("/kaggle/input")


def find_run_file(run_id: str, relative: str, root: Path = KAGGLE_INPUT) -> Path:
    """Return the one <run_id>/<relative> file under root.

    Searched recursively because Kaggle mounts the same kernel_sources at different depths in
    different kernels (/kaggle/input/<slug>/... vs /kaggle/input/notebooks/<owner>/<slug>/...,
    ADR-011). Matching on the run ID pins the exact run. Zero or several matches fail immediately,
    listing what is there.
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
        if here.name == run_id and (here / relative).is_file():
            matches.append(here / relative)
        # Never descend into prepared data: ~23k images and labels, and nothing to find there.
        dirnames[:] = [d for d in dirnames if d != "voc_prepared"]

    if len(matches) != 1:
        listing = sorted(str(p) for depth in ("*", "*/*", "*/*/*") for p in root.glob(depth))[:60]
        raise FileNotFoundError(
            f"Expected exactly one {run_id}/{relative} under {root}, "
            f"found {len(matches)}: {[str(m) for m in matches]}. Entries under {root}: {listing}"
        )
    logger.info("Found %s", matches[0])
    return matches[0]
