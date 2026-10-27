"""splits.py: disjoint, deterministic, sizes add up, and frozen files are protected."""

import random
from pathlib import Path

import pytest

from vocdet.data.splits import SplitsFrozenError, load_splits, make_splits, write_splits

OFFICIAL_TRAIN = [f"2008_{i:06d}" for i in range(1000)]
OFFICIAL_VAL = [f"2009_{i:06d}" for i in range(300)]


def test_sets_are_disjoint_and_sizes_add_up() -> None:
    s = make_splits(OFFICIAL_TRAIN, OFFICIAL_VAL)
    train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(val) == 100
    assert train | val == set(OFFICIAL_TRAIN)
    assert test == set(OFFICIAL_VAL)


def test_deterministic_across_calls_and_input_order() -> None:
    shuffled = OFFICIAL_TRAIN.copy()
    random.Random(0).shuffle(shuffled)
    assert make_splits(OFFICIAL_TRAIN, OFFICIAL_VAL) == make_splits(shuffled, OFFICIAL_VAL)


def test_seed_changes_val() -> None:
    assert (
        make_splits(OFFICIAL_TRAIN, OFFICIAL_VAL, seed=1)["val"]
        != make_splits(OFFICIAL_TRAIN, OFFICIAL_VAL)["val"]
    )


def test_outputs_are_sorted() -> None:
    s = make_splits(OFFICIAL_TRAIN, OFFICIAL_VAL)
    for ids in s.values():
        assert ids == sorted(ids)


def test_overlapping_or_duplicate_inputs_raise() -> None:
    with pytest.raises(ValueError, match="overlap"):
        make_splits(OFFICIAL_TRAIN, OFFICIAL_TRAIN[:5])
    with pytest.raises(ValueError, match="Duplicate"):
        make_splits(OFFICIAL_TRAIN + OFFICIAL_TRAIN[:1], OFFICIAL_VAL)


def test_rewriting_identical_splits_is_allowed(tmp_path: Path) -> None:
    s = make_splits(OFFICIAL_TRAIN, OFFICIAL_VAL)
    write_splits(s, tmp_path)
    write_splits(s, tmp_path)
    assert (tmp_path / "val.txt").read_text().splitlines() == s["val"]


def test_changing_frozen_splits_raises(tmp_path: Path) -> None:
    write_splits(make_splits(OFFICIAL_TRAIN, OFFICIAL_VAL), tmp_path)
    with pytest.raises(SplitsFrozenError, match="frozen"):
        write_splits(make_splits(OFFICIAL_TRAIN, OFFICIAL_VAL, seed=7), tmp_path)


def test_load_splits_round_trip(tmp_path: Path) -> None:
    s = make_splits(OFFICIAL_TRAIN, OFFICIAL_VAL)
    write_splits(s, tmp_path)
    assert load_splits(tmp_path) == s
