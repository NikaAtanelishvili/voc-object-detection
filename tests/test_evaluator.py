"""evaluator.py on synthetic cases with known answers (PLAN.md P2.2)."""

import numpy as np
import pytest

from vocdet.constants import VOC_CLASSES
from vocdet.eval.evaluator import evaluate
from vocdet.eval.schema import GroundTruth, Prediction

CAT, DOG = VOC_CLASSES.index("cat"), VOC_CLASSES.index("dog")


def gt_set() -> list[GroundTruth]:
    """Two images: a cat and a dog, then two cats. All boxes are 'medium' sized."""
    return [
        GroundTruth("a", [[10, 10, 60, 60], [100, 100, 180, 160]], [CAT, DOG], [False, False]),
        GroundTruth("b", [[0, 0, 50, 40], [60, 60, 120, 130]], [CAT, CAT], [False, False]),
    ]


def perfect(gts: list[GroundTruth]) -> list[Prediction]:
    return [Prediction(g.image_id, g.boxes, np.full(len(g.boxes), 0.9), g.labels) for g in gts]


def test_perfect_predictions_score_one() -> None:
    m = evaluate(perfect(gt_set()), gt_set())
    assert m["map50"] == pytest.approx(1.0)
    assert m["map50_95"] == pytest.approx(1.0)
    assert m["ap50_per_class"]["cat"] == pytest.approx(1.0)
    assert m["ap50_per_class"]["dog"] == pytest.approx(1.0)
    assert m["ap50_per_class"]["person"] is None  # no ground truth, no predictions
    assert m["num_images"] == 2


def test_no_predictions_score_zero() -> None:
    empty = [Prediction(g.image_id, [], [], []) for g in gt_set()]
    m = evaluate(empty, gt_set())
    assert m["map50"] == pytest.approx(0.0)


def test_detection_on_difficult_object_is_ignored() -> None:
    """A higher-scoring detection on a difficult cat ranks first; as an FP it would lower AP."""
    gts = [GroundTruth("a", [[10, 10, 60, 60], [100, 100, 180, 160]], [CAT, CAT], [False, True])]
    base = [Prediction("a", [[10, 10, 60, 60]], [0.9], [CAT])]
    with_difficult_hit = [
        Prediction("a", [[10, 10, 60, 60], [100, 100, 180, 160]], [0.9, 0.95], [CAT, CAT])
    ]
    assert evaluate(base, gts)["map50"] == pytest.approx(1.0)  # the difficult cat is not a miss
    assert evaluate(with_difficult_hit, gts)["map50"] == pytest.approx(1.0)  # nor an FP


def test_same_detection_on_a_normal_object_would_count() -> None:
    """Control for the test above: the same boxes without the difficult flag change the score."""
    gts = [GroundTruth("a", [[10, 10, 60, 60], [100, 100, 180, 160]], [CAT, CAT], [False, False])]
    base = [Prediction("a", [[10, 10, 60, 60]], [0.9], [CAT])]
    assert evaluate(base, gts)["map50"] < 1.0  # second cat is now a miss


def test_swapping_a_label_lowers_that_class_ap() -> None:
    preds = perfect(gt_set())
    # Second cat in image b predicted as dog, with the top score so the dog FP ranks first (with
    # tied scores the true dog would rank first and dog AP would stay 1.0).
    preds[1].labels = np.array([CAT, DOG])
    preds[1].scores = np.array([0.9, 0.95], dtype=np.float32)
    m = evaluate(preds, gt_set())
    assert m["ap50_per_class"]["cat"] < 1.0  # one cat is now missed
    assert m["ap50_per_class"]["dog"] < 1.0  # a high-scoring dog FP ranks above the true dog
    assert m["map50"] < 1.0


def test_map50_equals_mean_of_per_class_ap50() -> None:
    preds = perfect(gt_set())
    preds[1].labels = np.array([CAT, DOG])
    m = evaluate(preds, gt_set())
    per_class = [v for v in m["ap50_per_class"].values() if v is not None]
    assert m["map50"] == pytest.approx(np.mean(per_class))


def test_missing_or_extra_prediction_entries_raise() -> None:
    with pytest.raises(ValueError, match="No prediction entry"):
        evaluate(perfect(gt_set())[:1], gt_set())
    extra = [*perfect(gt_set()), Prediction("zzz", [], [], [])]
    with pytest.raises(ValueError, match="not in the split"):
        evaluate(extra, gt_set())
