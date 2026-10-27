"""Pydantic models for the YAML configs, so a typo in a config fails before a GPU session starts."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class KaggleSettings(BaseModel):
    """How a config is executed as a Kaggle kernel (read by `scripts/kaggle_run.py`)."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(
        description="Shell command run from the repo root on Kaggle. `{run_dir}` is replaced by "
        "/kaggle/working/<run_id>, the only folder whose contents are pulled back."
    )
    gpu: bool = Field(default=True, description="Request a T4 GPU (False = CPU-only kernel).")
    pip: list[str] = Field(
        default_factory=list,
        description="Packages missing from the Kaggle image. The repo itself is installed with "
        "--no-deps so Kaggle's preinstalled torch is never replaced.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Kernel slugs whose outputs are mounted under /kaggle/input (prepared data).",
    )
    pull_pattern: str | None = Field(
        default=None,
        description="Regex (re.search on each output path) selecting which outputs to download; "
        "None downloads everything. Lets the prep kernel keep 2 GB of images on Kaggle.",
    )


class JobConfig(BaseModel):
    """Top-level fields every runnable config shares.

    Experiment-specific sections (model, training, ...) are validated by their own models when those
    phases are implemented; until then they are allowed through untouched.
    """

    model_config = ConfigDict(extra="allow")

    exp_id: str = Field(
        pattern=r"^[A-Za-z0-9_]+$", description="Experiment ID, e.g. E1 or env_check."
    )
    kaggle: KaggleSettings


def load_job_config(path: Path) -> JobConfig:
    """Read and validate a YAML config. Raises pydantic.ValidationError with the offending field."""
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: expected a YAML mapping at the top level, got {type(raw).__name__}"
        )
    return JobConfig.model_validate(raw)
