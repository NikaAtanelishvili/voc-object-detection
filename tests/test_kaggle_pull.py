"""kaggle_pull.py: parsing the CLI's status line, and polling through transient API failures."""

import subprocess

import pytest

import kaggle_pull
from kaggle_pull import parse_status, wait_for_kernel


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


def fake_status_calls(monkeypatch: pytest.MonkeyPatch, outcomes: list[tuple[int, str]]) -> None:
    """Make each `kaggle kernels status` call return the next (returncode, stdout)."""
    calls = iter(outcomes)

    def run(args, **kwargs):
        code, out = next(calls)
        return subprocess.CompletedProcess(args, code, stdout=out, stderr="429 Too Many Requests")

    monkeypatch.setattr(kaggle_pull.subprocess, "run", run)


def status(state: str) -> tuple[int, str]:
    return 0, f'me/vocdet-e2 has status "KernelWorkerStatus.{state}"'


def test_wait_survives_a_failed_status_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_status_calls(
        monkeypatch, [status("RUNNING"), (1, ""), status("RUNNING"), status("COMPLETE")]
    )
    assert wait_for_kernel("me/vocdet-e2", poll_s=0, sleep=lambda s: None) == "COMPLETE"


def test_wait_gives_up_after_consecutive_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_status_calls(monkeypatch, [(1, "")] * kaggle_pull.MAX_STATUS_FAILURES)
    with pytest.raises(RuntimeError, match="failed 5 times in a row: 429"):
        wait_for_kernel("me/vocdet-e2", poll_s=0, sleep=lambda s: None)
