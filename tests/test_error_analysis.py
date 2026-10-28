"""error_analysis.py: one hand-built detection for every outcome, IoU, and the confusion matrix."""

from pathlib import Path

import numpy as np
import pytest

from vocdet.constants import VOC_CLASSES
from vocdet.eval.error_analysis import BACKGROUND, analyze, box_iou, summarize, write_error_analysis
from vocdet.eval.schema import GroundTruth, Prediction

CAT, DOG, PERSON = (VOC_CLASSES.index(c) for c in ("cat", "dog", "person"))

# Ground truth: a cat, a dog, and a difficult cat.
GT = GroundTruth(
    "img",
    [[0, 0, 100, 100], [200, 0, 300, 100], [0, 200, 100, 300]],
    [CAT, DOG, CAT],
    [False, False, True],
)
# One detection per outcome. Comments give the IoU that decides it.
DETECTIONS = [
    ([0, 0, 100, 100], 0.99, CAT),  # TP: IoU 1.0 with the cat
    ([2, 2, 100, 100], 0.90, CAT),  # Dupe: IoU 0.96 with the already matched cat
    ([0, 0, 98, 100], 0.80, DOG),  # Cls: IoU 0.98 with the cat, no dog nearby
    ([200, 0, 240, 100], 0.70, DOG),  # Loc: IoU 0.4 with the dog
    ([200, 0, 240, 100], 0.65, CAT),  # Both: IoU 0.4 with the dog, wrong class
    ([400, 400, 450, 450], 0.60, PERSON),  # Bkg: touches nothing
    ([0, 200, 100, 300], 0.55, CAT),  # Ignored: on the difficult cat
    ([0, 0, 100, 100], 0.10, CAT),  # below the 0.5 threshold: not analysed at all
]
PRED = Prediction(
    "img",
    [d[0] for d in DETECTIONS],
    [d[1] for d in DETECTIONS],
    [d[2] for d in DETECTIONS],
)


def test_box_iou_known_values() -> None:
    a = [[0, 0, 10, 10]]
    b = [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 1 / 3, 0.0]])
    assert box_iou(np.zeros((0, 4)), b).shape == (0, 3)
    assert box_iou(a, np.zeros((0, 4))).shape == (1, 0)


def test_every_outcome_is_classified() -> None:
    result = analyze([PRED], [GT], score_threshold=0.5)
    outcomes = [r.outcome for r in sorted(result.detections, key=lambda r: -r.score)]
    assert outcomes == ["TP", "Dupe", "Cls", "Loc", "Both", "Bkg", "Ignored"]
    cls = next(r for r in result.detections if r.outcome == "Cls")
    assert cls.true_label == CAT and cls.label == DOG
    # The dog was never detected correctly -> one miss; the difficult cat is never a miss.
    assert [(m.label, m.area) for m in result.misses] == [(DOG, 100 * 100)]


def test_summary_counts() -> None:
    summary = summarize(analyze([PRED], [GT], score_threshold=0.5))
    assert summary["overall"] == {
        "TP": 1, "Cls": 1, "Loc": 1, "Both": 1, "Dupe": 1, "Bkg": 1, "Ignored": 1, "Miss": 1,
    }  # fmt: skip
    assert summary["per_class"]["dog"]["Cls"] == 1  # counted under the predicted class
    assert summary["per_class"]["dog"]["Miss"] == 1  # counted under the true class


def test_confusion_matrix_entries() -> None:
    c = analyze([PRED], [GT], score_threshold=0.5).confusion
    assert c.shape == (21, 21)
    assert c[CAT, CAT] == 1  # TP
    assert c[CAT, DOG] == 1  # Cls: true cat predicted as dog
    assert c[BACKGROUND, CAT] == 2  # Dupe + Both
    assert c[BACKGROUND, DOG] == 1  # Loc
    assert c[BACKGROUND, PERSON] == 1  # Bkg
    assert c[DOG, BACKGROUND] == 1  # Miss
    assert c.sum() == 7  # the ignored detection is not counted anywhere


def test_lower_threshold_includes_low_score_detection() -> None:
    result = analyze([PRED], [GT], score_threshold=0.05)
    low = next(r for r in result.detections if r.score == pytest.approx(0.10))
    assert low.outcome == "Dupe"  # the cat was already matched by the 0.99 detection


def test_missing_prediction_entry_raises() -> None:
    with pytest.raises(ValueError, match="No prediction entry"):
        analyze([], [GT])


def test_write_error_analysis(tmp_path: Path) -> None:
    write_error_analysis(analyze([PRED], [GT]), tmp_path)
    assert (tmp_path / "errors.json").exists() and (tmp_path / "error_records.json").exists()
