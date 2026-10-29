"""build_report_tables.py: the experiment table and marker replacement."""

from pathlib import Path

import pytest

from build_report_tables import experiments_table, replace_block


def test_replace_block_keeps_hand_written_text_and_is_idempotent() -> None:
    text = "intro\n<!-- BEGIN:t -->\nold\n<!-- END:t -->\nouter\n"
    once = replace_block(text, "t", "new")
    assert once == "intro\n<!-- BEGIN:t -->\nnew\n<!-- END:t -->\nouter\n"
    assert replace_block(once, "t", "new") == once
    empty = "<!-- BEGIN:t -->\n<!-- END:t -->\n"  # a fresh report has nothing in between
    assert replace_block(empty, "t", "new") == "<!-- BEGIN:t -->\nnew\n<!-- END:t -->\n"


def test_replace_block_requires_one_marker_pair() -> None:
    with pytest.raises(ValueError, match="marker"):
        replace_block("no markers here", "t", "x")


def test_experiments_table_formats_metrics(tmp_path: Path) -> None:
    csv = tmp_path / "experiments.csv"
    csv.write_text(
        "run_id,exp_id,git_sha,seed,gpu_name,wall_time_s,median_iter_s,map50,map50_95,"
        "map_small,map_medium,map_large,mar_100\n"
        "E1_x,E1,abc,42,Tesla T4,100.0,0.25,0.71234,0.4,,0.3,0.5,0.6\n"
    )
    table = experiments_table(csv)
    assert "| E1_x " in table and "0.712" in table and "mAP50-95" in table
