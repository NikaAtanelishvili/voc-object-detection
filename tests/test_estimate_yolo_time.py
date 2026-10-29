"""estimate_yolo_time.py: epoch times from cumulative seconds, and the pricing formula."""

from pathlib import Path

import pytest

from estimate_yolo_time import epoch_times, estimate_seconds


def test_epoch_times_come_from_cumulative_seconds(tmp_path: Path) -> None:
    log = tmp_path / "train_log.csv"
    log.write_text("epoch,time\n1,100\n2,160\n3,210\n")
    assert epoch_times(log) == (100, 60, 50)
    log.write_text("epoch,time\n1,100\n")
    with pytest.raises(ValueError, match="expected 3 epochs"):
        epoch_times(log)


def test_estimate_prices_mosaic_and_plain_epochs() -> None:
    # 50 epochs, last 10 plain: start-up epoch + 39 warm mosaic + 10 plain + outside training.
    assert estimate_seconds(100, 60, 50, 300, epochs=50, close_mosaic=10) == pytest.approx(
        300 + 100 + 39 * 60 + 10 * 50
    )
    # The timing run itself is reproduced exactly (3 epochs, close_mosaic=1).
    assert estimate_seconds(100, 60, 50, 0, epochs=3, close_mosaic=1) == pytest.approx(210)
    # No mosaic epoch at all: the first epoch's start-up stays, its mosaic cost becomes plain.
    assert estimate_seconds(100, 60, 50, 0, epochs=2, close_mosaic=10) == pytest.approx(140)
