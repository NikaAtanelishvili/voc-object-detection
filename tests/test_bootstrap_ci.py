"""bootstrap_ci.py: zero width for a resample-invariant score; a fixed seed reproduces it."""

import numpy as np

from bootstrap_ci import bootstrap_map50, confidence_interval
from vocdet.eval.schema import GroundTruth, Prediction


def image(i: int, label: int, hit: bool) -> tuple[GroundTruth, Prediction]:
    box = np.array([[10, 10, 60, 60]], dtype=np.float32)
    gt = GroundTruth(f"img{i}", box, np.array([label]), np.array([False]))
    pred_box = box if hit else box + 200  # a miss lands far away: one FP and one FN
    return gt, Prediction(f"img{i}", pred_box, np.array([0.9], dtype=np.float32), np.array([label]))


def test_perfect_predictions_give_a_zero_width_interval() -> None:
    pairs = [image(i, label=i % 2, hit=True) for i in range(6)]
    gt, preds = [g for g, _ in pairs], [p for _, p in pairs]
    low, high = confidence_interval(bootstrap_map50(preds, gt, iterations=5, seed=0))
    assert low == high == 1.0


def test_fixed_seed_gives_identical_intervals_and_misses_widen_it() -> None:
    pairs = [image(i, label=0, hit=i % 3 != 0) for i in range(9)]
    gt, preds = [g for g, _ in pairs], [p for _, p in pairs]
    first = bootstrap_map50(preds, gt, iterations=8, seed=7)
    assert first == bootstrap_map50(preds, gt, iterations=8, seed=7)
    low, high = confidence_interval(first)
    assert low < high
