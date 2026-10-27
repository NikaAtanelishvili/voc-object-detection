"""Smoke tests for the package scaffold: importable, versioned, CLI wired."""

import pytest

import vocdet
from vocdet.cli import main


def test_version_is_set() -> None:
    assert isinstance(vocdet.__version__, str)
    assert vocdet.__version__.count(".") == 2


def test_cli_version_flag_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert vocdet.__version__ in capsys.readouterr().out
