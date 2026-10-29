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


class FrcnnModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pretrained: bool = True
    min_size: int = 600
    max_size: int = 1000


class FrcnnTrainSettings(BaseModel):
    """Optimisation settings for the Faster R-CNN loop (ADR-012)."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=12, ge=1)
    batch_size: int = Field(default=8, ge=1)
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: int = Field(default=500, ge=0)
    warmup_factor: float = Field(default=0.001, gt=0, le=1, description="LR multiplier at step 0.")
    milestones: list[int] = Field(default=[8, 11], description="Epochs where LR is times gamma.")
    gamma: float = 0.1
    amp: bool = True
    num_workers: int = 2
    eval_every: int = Field(default=2, ge=1, description="Validate every N epochs and at the end.")
    limit_train_images: int | None = Field(default=None, description="Smoke runs only (E0a).")
    limit_val_images: int | None = Field(default=None, description="Smoke runs only (E0a).")

    @model_validator(mode="after")
    def _milestones_inside_training(self) -> "FrcnnTrainSettings":
        if self.milestones != sorted(self.milestones) or any(
            m <= 0 or m >= self.epochs for m in self.milestones
        ):
            raise ValueError("milestones must be sorted epochs strictly between 0 and epochs")
        return self


class FrcnnConfig(JobConfig):
    """E1 / E0a-frcnn: Faster R-CNN fine-tuning. Unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    model: FrcnnModelSettings
    train: FrcnnTrainSettings


class YoloAugmentSettings(BaseModel):
    """Ultralytics augmentation hyperparameters. None keeps the Ultralytics default (E2, E3, E5);
    E4 sets them to 0. Only these keys are allowed, so a typo cannot silently do nothing."""

    model_config = ConfigDict(extra="forbid")

    mosaic: float | None = None
    mixup: float | None = None
    hsv_h: float | None = None
    hsv_s: float | None = None
    hsv_v: float | None = None
    scale: float | None = None
    translate: float | None = None
    fliplr: float | None = None


class YoloTrainSettings(BaseModel):
    """Arguments passed to Ultralytics `train()` (P4.1). The optimizer stays the model default."""

    model_config = ConfigDict(extra="forbid")

    weights: str = Field(description="Ultralytics COCO weights to fine-tune, e.g. yolo26n.pt.")
    epochs: int = Field(default=50, ge=1)
    imgsz: int = 640
    batch: int = Field(default=16, ge=1)
    patience: int = Field(default=15, ge=0, description="Early-stopping epochs without val gain.")
    freeze: int | None = Field(default=None, description="Freeze the first N layers (E5: 10).")
    augment: YoloAugmentSettings = Field(default_factory=YoloAugmentSettings)
    limit_train_images: int | None = Field(default=None, description="Smoke runs only (E0a).")
    limit_val_images: int | None = Field(default=None, description="Smoke runs only (E0a).")


class YoloConfig(JobConfig):
    """E0a-yolo, E2-E5: YOLO26 fine-tuning. Unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    train: YoloTrainSettings


class DataConfig(BaseModel):
    """configs/data/voc2012.yaml: where kernels find the prepared data."""

    model_config = ConfigDict(extra="forbid")

    prep_run_id: str = Field(
        pattern=r"^prepare_data_\d{8}-\d{4}_[0-9a-f]{7}$",
        description="Run ID of the prep kernel whose output is mounted (pins the data version).",
    )


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
