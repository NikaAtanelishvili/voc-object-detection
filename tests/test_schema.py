"""schema.py: validation, empty predictions, JSON round trips, ground truth from the table."""

from pathlib import Path

import numpy as np
import pytest

from vocdet.data.parse_voc import parse_dataset
from vocdet.eval.schema import (
    GroundTruth,
    Prediction,
    ground_truth_from_annotations,
    read_ground_truth,
    read_predictions,
    write_ground_truth,
    write_predictions,
)

VOC_FIXTURE = Path(__file__).parent / "fixtures" / "voc"


def test_empty_prediction_has_well_shaped_arrays() -> None:
    p = Prediction("a", [], [], [])
    assert p.boxes.shape == (0, 4) and p.scores.shape == (0,) and p.labels.shape == (0,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"boxes": [[10, 0, 5, 5]], "scores": [0.5], "labels": [0]}, "x2 >= x1"),
        ({"boxes": [[0, 0, 5, 5]], "scores": [0.5], "labels": [20]}, "VOC indices"),
        ({"boxes": [[0, 0, 5, 5]], "scores": [1.5], "labels": [0]}, r"\[0, 1\]"),
        ({"boxes": [[0, 0, 5, 5]], "scores": [0.5, 0.4], "labels": [0]}, "scores"),
    ],
)
def test_invalid_predictions_raise(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Prediction("a", **kwargs)


def test_prediction_json_round_trip(tmp_path: Path) -> None:
    preds = [
        Prediction("a", [[1.234, 2.0, 30.5, 40.25]], [0.87654], [14]),
        Prediction("b", [], [], []),
    ]
    write_predictions(preds, tmp_path / "p.json")
    back = read_predictions(tmp_path / "p.json")
    assert [p.image_id for p in back] == ["a", "b"]
    np.testing.assert_allclose(back[0].boxes, [[1.23, 2.0, 30.5, 40.25]], atol=1e-5)
    assert back[0].labels.tolist() == [14]
    assert back[1].boxes.shape == (0, 4)


def test_ground_truth_json_round_trip(tmp_path: Path) -> None:
    gts = [GroundTruth("a", [[0, 0, 10, 10], [5, 5, 6, 6]], [3, 4], [False, True])]
    write_ground_truth(gts, tmp_path / "g.json")
    (back,) = read_ground_truth(tmp_path / "g.json")
    assert back.difficult.tolist() == [False, True]
    np.testing.assert_array_equal(back.boxes, gts[0].boxes)


def test_ground_truth_from_annotations_keeps_order_and_difficult() -> None:
    df = parse_dataset(VOC_FIXTURE, ["2008_000001", "2008_000002"])
    gts = ground_truth_from_annotations(df, ["2008_000002", "2008_000001", "2008_009999"])
    assert [g.image_id for g in gts] == ["2008_000002", "2008_000001", "2008_009999"]
    assert gts[1].difficult.tolist() == [False, True]  # dog, difficult bottle
    assert gts[2].boxes.shape == (0, 4)  # unknown image -> empty ground truth
