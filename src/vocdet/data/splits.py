"""Build the frozen train/val/test image-ID lists (PLAN.md §3, ADR-002).

- train = official VOC2012 `train` minus a random 10% (seed 42)
- val   = that 10%, used for all model selection and tuning
- test  = official VOC2012 `val`, used only once per final model in Phase 5

VOC2012 test labels are not public, so the official val becomes our untouched test set.
"""

import logging
import random
from pathlib import Path

from vocdet.data.parse_voc import read_image_ids

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


class SplitsFrozenError(RuntimeError):
    """Existing split files differ from what would be written (hard rule 5)."""


def make_splits(
    official_train: list[str], official_val: list[str], val_fraction: float = 0.1, seed: int = 42
) -> dict[str, list[str]]:
    """Return sorted ID lists for train/val/test.

    Input IDs are sorted before sampling, so the result depends only on the set of IDs and the seed,
    not on the order of lines in the input files.
    """
    for ids in (official_train, official_val):
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate image IDs in the official lists")
    if set(official_train) & set(official_val):
        raise ValueError("Official train and val lists overlap")

    pool = sorted(official_train)
    n_val = round(len(pool) * val_fraction)
    val = sorted(random.Random(seed).sample(pool, n_val))
    val_set = set(val)
    train = [image_id for image_id in pool if image_id not in val_set]
    return {"train": train, "val": val, "test": sorted(official_val)}


def write_splits(splits: dict[str, list[str]], out_dir: Path) -> None:
    """Write splits/<name>.txt. Rewriting identical content is a no-op; different content raises.

    This makes `vocdet splits` safe to rerun anywhere (e.g. on Kaggle) without ever silently
    changing the frozen splits. Changing them on purpose means deleting the files under a new ADR.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        path = out_dir / f"{name}.txt"
        content = "\n".join(splits[name]) + "\n"
        if path.exists() and path.read_text() != content:
            raise SplitsFrozenError(
                f"{path} exists with different content. Splits are frozen (CLAUDE.md hard rule 5); "
                "changing them needs a new ADR and rerunning every experiment."
            )
        path.write_text(content)
        logger.info("%s: %d images -> %s", name, len(splits[name]), path)


def build_splits(voc_dir: Path, out_dir: Path) -> dict[str, list[str]]:
    """Read the official Main lists from an extracted VOC2012 folder and write the split files."""
    main = voc_dir / "ImageSets" / "Main"
    splits = make_splits(read_image_ids(main / "train.txt"), read_image_ids(main / "val.txt"))
    write_splits(splits, out_dir)
    return splits


def load_splits(split_dir: Path) -> dict[str, list[str]]:
    """Read the committed splits/<name>.txt files; later stages use these and never recompute."""
    return {name: read_image_ids(split_dir / f"{name}.txt") for name in SPLIT_NAMES}
