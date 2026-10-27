"""Run identity and metadata: the run artifact contract from PLAN.md §2.

Every training or evaluation run writes runs/<run_id>/run_meta.json, so every number can be traced
to the code (git SHA), the environment, and the seed that produced it.
"""

from datetime import datetime


def make_run_id(exp_id: str, commit_sha: str, now: datetime) -> str:
    """run_id = <exp_id>_<yyyymmdd-hhmm>_<sha7>, the contract from PLAN.md §2."""
    return f"{exp_id}_{now:%Y%m%d-%H%M}_{commit_sha[:7]}"
