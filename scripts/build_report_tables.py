"""Regenerate the tables in reports/*.md from results/, so no metric in a report is typed by hand.

Usage: uv run python scripts/build_report_tables.py   (make report)

Each generated table lives between `<!-- BEGIN:<name> -->` and `<!-- END:<name> -->` markers in its
report. Everything outside the markers is hand-written and left untouched, so rerunning only
refreshes numbers. P3.6 adds the experiment table; P5.1 adds the rest (PLAN.md).
"""

import logging
import re
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("build_report_tables")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_COLUMNS = {
    "run_id": "Run",
    "exp_id": "Exp",
    "map50": "mAP50",
    "map50_95": "mAP50-95",
    "map_small": "AP small",
    "map_medium": "AP medium",
    "map_large": "AP large",
    "mar_100": "AR@100",
    "wall_time_s": "Wall time (s)",
}


def experiments_table(csv_path: Path) -> str:
    """One row per collected run, in run_id order, with the val metrics from experiments.csv."""
    df = pd.read_csv(csv_path)[list(EXPERIMENT_COLUMNS)].rename(columns=EXPERIMENT_COLUMNS)
    return df.to_markdown(index=False, floatfmt=".3f")


def deltas_table(csv_path: Path) -> str:
    """Paired bootstrap differences (ADR-014), the only way two runs are compared in reports."""
    df = pd.read_csv(csv_path)
    experiment = df[["a", "b"]].replace(r"_\d{8}-\d{4}_[0-9a-f]{7}$", "", regex=True)
    table = pd.DataFrame(
        {
            "Comparison": experiment["a"] + " − " + experiment["b"],
            "Metric": df["metric"],
            "Δ": df["delta"],
            "CI95 low": df["ci95_low"],
            "CI95 high": df["ci95_high"],
            "CI excludes 0": df["excludes_zero"].map({True: "yes", False: "no"}),
        }
    )
    return table.to_markdown(index=False, floatfmt="+.3f")


def replace_block(text: str, name: str, content: str) -> str:
    """Put content between the name's markers; fail if the markers are missing or repeated."""
    # The block may be empty (a fresh report), so the END marker is matched without a newline.
    pattern = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(<!-- END:{name} -->)", re.DOTALL)
    if len(pattern.findall(text)) != 1:
        raise ValueError(f"Expected exactly one BEGIN/END:{name} marker pair")
    return pattern.sub(lambda m: m.group(1) + content + "\n" + m.group(2), text)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = REPO_ROOT / "reports" / "EXPERIMENTS.md"
    text = report.read_text()
    table = experiments_table(REPO_ROOT / "results" / "experiments.csv")
    text = replace_block(text, "experiments", table)
    text = replace_block(text, "deltas", deltas_table(REPO_ROOT / "results" / "paired_deltas.csv"))
    report.write_text(text)
    logger.info("Updated %s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
