"""bootstrap_ci.py: pairing, zero-width deltas, reproducibility, and why pairing matters."""

import numpy as np
import pytest

from bootstrap_ci import (
    bootstrap_scores,
    confidence_interval,
    delta_row,
    resample_indices,
)
from vocdet.eval.schema import GroundTruth, Prediction


def image(i: int, label: int, hit: bool) -> tuple[GroundTruth, Prediction]:
    box = np.array([[10, 10, 60, 60]], dtype=np.float32)
    gt = GroundTruth(f"img{i}", box, np.array([label]), np.array([False]))
    pred_box = box if hit else box + 200  # a miss lands far away: one FP and one FN
    return gt, Prediction(f"img{i}", pred_box, np.array([0.9], dtype=np.float32), np.array([label]))


def run(run_id: str, gt: list, preds: list, indices: np.ndarray) -> dict:
    values = bootstrap_scores(preds, gt, indices)
    point = {name: float(np.mean(v)) for name, v in values.items()}
    return {"run_id": run_id, "settings": {"seed": 0}, "point": point, "values": values}


def test_perfect_predictions_give_a_zero_width_interval() -> None:
    pairs = [image(i, label=i % 2, hit=True) for i in range(6)]
    gt, preds = [g for g, _ in pairs], [p for _, p in pairs]
    values = bootstrap_scores(preds, gt, resample_indices(6, 5, seed=0))
    assert confidence_interval(values["map50"]) == (1.0, 1.0)


def test_fixed_seed_gives_identical_draws_and_values() -> None:
    assert np.array_equal(resample_indices(9, 8, seed=7), resample_indices(9, 8, seed=7))
    assert not np.array_equal(resample_indices(9, 8, seed=7), resample_indices(9, 8, seed=8))
    pairs = [image(i, label=0, hit=i % 3 != 0) for i in range(9)]
    gt, preds = [g for g, _ in pairs], [p for _, p in pairs]
    indices = resample_indices(9, 8, seed=7)
    assert bootstrap_scores(preds, gt, indices) == bootstrap_scores(preds, gt, indices)


def test_identical_models_have_a_zero_delta_though_their_own_ci_is_wide() -> None:
    pairs = [image(i, label=0, hit=i % 3 != 0) for i in range(9)]
    gt, preds = [g for g, _ in pairs], [p for _, p in pairs]
    indices = resample_indices(9, 8, seed=7)
    a, b = run("a", gt, preds, indices), run("b", gt, preds, indices)
    low, high = confidence_interval(a["values"]["map50"])
    assert low < high
    row = delta_row(a, b, "map50")
    assert (row["ci95_low"], row["ci95_high"]) == (0.0, 0.0)
    assert row["excludes_zero"] is False


def test_pairing_is_tighter_than_separate_draws() -> None:
    # b equals a except on one image, so nearly all resample noise is shared.
    pairs_a = [image(i, label=0, hit=i % 3 != 0) for i in range(12)]
    pairs_b = [image(i, label=0, hit=i % 3 != 0 or i == 0) for i in range(12)]
    gt = [g for g, _ in pairs_a]
    preds_a, preds_b = [p for _, p in pairs_a], [p for _, p in pairs_b]
    indices = resample_indices(12, 30, seed=1)
    paired = delta_row(run("a", gt, preds_b, indices), run("b", gt, preds_a, indices), "map50")
    other = resample_indices(12, 30, seed=2)
    a, b = run("a", gt, preds_b, indices), run("b", gt, preds_a, other)
    unpaired = delta_row(a, b, "map50")
    assert paired["ci95_high"] - paired["ci95_low"] < unpaired["ci95_high"] - unpaired["ci95_low"]


def test_delta_row_refuses_runs_with_different_draws() -> None:
    a = {"run_id": "a", "settings": {"seed": 0}, "point": {}, "values": {}}
    b = {"run_id": "b", "settings": {"seed": 1}, "point": {}, "values": {}}
    with pytest.raises(ValueError, match="Not paired"):
        delta_row(a, b, "map50")
