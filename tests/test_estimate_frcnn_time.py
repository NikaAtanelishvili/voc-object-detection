"""estimate_frcnn_time.py: resize rule, batch padding, and the new-shape / repeat fit."""

import pytest

from estimate_frcnn_time import batch_shapes, first_occurrences, fit_step_times, resized_hw
from vocdet.config import FrcnnConfig


def test_resized_hw_matches_torchvision_rule() -> None:
    assert resized_hw(375, 500, 600, 1000) == (600, 800)  # short side to 600
    assert resized_hw(200, 500, 600, 1000) == (400, 1000)  # long side capped at 1000


def test_batch_shapes_pad_to_the_largest_image_and_a_multiple_of_32() -> None:
    cfg = FrcnnConfig.model_validate(
        {
            "exp_id": "t",
            "model": {"min_size": 600, "max_size": 1000},
            "train": {"epochs": 1, "batch_size": 2, "milestones": []},
            "kaggle": {"command": "unused"},
        }
    )
    # A landscape and a portrait image in one batch pad to the union of both: 800 x 800.
    assert batch_shapes([(375, 500), (500, 375)], cfg) == [(800, 800)]


def test_fit_separates_new_and_repeat_steps() -> None:
    is_new = first_occurrences([(1, 1), (1, 1), (2, 2), (1, 1)])
    assert is_new == [True, False, True, False]
    fit = fit_step_times([8.0, 0.3, 7.0, 0.2], is_new)
    assert fit["new_shape_s"] == pytest.approx(7.5) and fit["repeat_s"] == pytest.approx(0.25)
    assert fit["separation"] == pytest.approx(7.0 / 0.3)
    with pytest.raises(ValueError, match="both"):
        fit_step_times([1.0], [True])
