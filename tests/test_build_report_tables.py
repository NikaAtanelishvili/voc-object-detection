"""build_report_tables.py: the experiment table and marker replacement."""

from pathlib import Path

import pytest

from build_report_tables import deltas_table, experiments_table, final_test_table, replace_block


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


def test_deltas_table_names_experiments_and_signs_values(tmp_path: Path) -> None:
    path = tmp_path / "paired_deltas.csv"
    path.write_text(
        "a,b,metric,delta,ci95_low,ci95_high,excludes_zero\n"
        "E2_20261029-1808_ae67b2f,E0b_20261028-0321_fa7911f,map50,-0.0154,-0.03,0.002,False\n"
    )
    table = deltas_table(path)
    assert "E2 − E0b" in table and "-0.015" in table and "+0.002" in table and "| no" in table


def test_test_table_keeps_the_csv_order(tmp_path: Path) -> None:
    path = tmp_path / "test_metrics.csv"
    header = "exp_id,run_id,weights,weights_sha256,map50,map50_95,map_small,map_medium,map_large,"
    path.write_text(
        header + "mar_100,recorded_utc\n"
        "E0b,E0b_r,w,h,0.805,0.626,0.2,0.4,0.7,0.7,t\nE6s,E6s_r,w,h,0.856,0.689,0.3,0.5,0.8,0.8,t\n"
    )
    table = final_test_table(path)
    assert table.index("E0b") < table.index("E6s") and "0.856" in table
