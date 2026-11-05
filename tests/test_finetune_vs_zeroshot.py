"""finetune_vs_zeroshot.py: which classes get a fresh head row, and the group means."""

import pytest

from finetune_vs_zeroshot import fresh_head_classes, group_means, per_class_deltas
from vocdet.constants import VOC_CLASSES


def test_fresh_head_classes_are_the_six_renamed_ones() -> None:
    assert fresh_head_classes() == [
        "aeroplane", "diningtable", "motorbike", "pottedplant", "sofa", "tvmonitor",
    ]  # fmt: skip


def test_group_means_split_fresh_from_copied() -> None:
    run = {name: 0.5 for name in VOC_CLASSES} | {"sofa": 0.1, "dog": 0.9}
    base = {name: 0.5 for name in VOC_CLASSES}
    means = group_means(per_class_deltas(run, base), ["sofa", "tvmonitor"])
    assert means["fresh_head_row"] == pytest.approx(-0.2)
    assert means["copied_head_row"] == pytest.approx(0.4 / 18)
