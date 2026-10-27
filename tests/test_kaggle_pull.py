"""kaggle_pull.py: parsing the CLI's status line."""

import pytest

from kaggle_pull import parse_status


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('me/vocdet-e1 has status "KernelWorkerStatus.COMPLETE"', "COMPLETE"),
        ('me/vocdet-e1 has status "KernelWorkerStatus.RUNNING"\n', "RUNNING"),
        ('me/vocdet-e1 has status "error"', "ERROR"),
    ],
)
def test_parse_status(line: str, expected: str) -> None:
    assert parse_status(line) == expected


def test_parse_status_rejects_unexpected_output() -> None:
    with pytest.raises(ValueError, match="Unexpected"):
        parse_status("404 Not found")
