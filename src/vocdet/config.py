"""Pydantic models for the YAML configs, so a typo in a config fails before a GPU session starts."""

from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class PredictSettings(BaseModel):
    """Inference settings for an evaluation-only run (E0b zero-shot)."""

    model_config = ConfigDict(extra="forbid")

    weights: str = Field(description="Ultralytics weights name or path, e.g. yolo26n.pt.")
    imgsz: int = 640
    batch: int = 16
    # Only val: the test split is reserved for scripts/final_test_eval.py (hard rule 4).
    split: Literal["val"] = "val"


class ZeroShotConfig(JobConfig):
    """E0b: a pretrained COCO model on VOC val, no training. Unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    predict: PredictSettings


class DataConfig(BaseModel):
    """configs/data/voc2012.yaml: where kernels find the prepared data."""

    model_config = ConfigDict(extra="forbid")

    prep_run_id: str
    kaggle_prepared_dir: str

    @model_validator(mode="after")
    def _path_matches_run(self) -> "DataConfig":
        if self.prep_run_id not in self.kaggle_prepared_dir:
            raise ValueError("kaggle_prepared_dir must contain prep_run_id (update both together)")
        return self


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_job_config(path: Path, model: type[ConfigT] = JobConfig) -> ConfigT:
    """Read a YAML config and validate it against `model`.

    Raises pydantic.ValidationError naming the offending field.
    """
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: expected a YAML mapping at the top level, got {type(raw).__name__}"
        )
    return model.model_validate(raw)
