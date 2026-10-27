# Architecture Decision Records

One entry per design decision. Newest at the bottom. Never edit an accepted ADR's decision; supersede it
with a new ADR and link both ways.

## Template

```
## ADR-NNN: <title>
- **Date:** YYYY-MM-DD
- **Status:** proposed | accepted | superseded by ADR-MMM
- **Context:** the problem and the constraints
- **Decision:** what we chose
- **Alternatives considered:** what else, and why not
- **Consequences:** what this makes easier or harder
- **Pinned versions:** dependencies added under this decision (exact versions)
```

---

## ADR-001: Stack choice

- **Date:** 2026-10-28
- **Status:** accepted
- **Context:** 7-day deadline; training only on free Kaggle GPUs; deployment target is a 2-vCPU machine
  with no GPU; the owner must explain every line in an interview without AI tools.
- **Decision:**
  - Python ≥3.10 (Kaggle compatibility), `uv` + `pyproject.toml` with src layout, lockfile committed.
  - PyTorch + torchvision. Two-stage baseline: `fasterrcnn_mobilenet_v3_large_fpn` (COCO weights) with an
    own, explicit training loop. One-stage: Ultralytics YOLO26 n/s (NMS-free end-to-end inference, clean
    ONNX/OpenVINO export).
  - Metrics: `torchmetrics` `MeanAveragePrecision` (pycocotools backend), one evaluator for all models.
    Error analysis: own "TIDE-lite" implementation.
  - Tracking: run folders + `results/experiments.csv` committed to git; W&B optional.
  - Optimization: ONNX Runtime; OpenVINO FP32 and INT8 (NNCF via Ultralytics export).
  - Serving: FastAPI + Uvicorn with Gradio mounted at `/ui`; hosted on a Hugging Face Docker Space (free
    CPU), weights on the HF Hub at a pinned revision.
  - Compute: Kaggle kernels driven by the `kaggle` CLI.
  - Quality: ruff, pytest, pre-commit, GitHub Actions (CPU-only torch wheels).
  - Config: YAML validated by pydantic.
- **Alternatives considered:**
  - Detectron2 / MMDetection for the two-stage model: heavier, less explainable, install friction on
    Kaggle. torchvision's reference model is enough for a baseline.
  - pip/poetry: slower, and uv's lockfile plus index mapping (CPU torch wheels) is simpler.
  - MLflow/W&B as the primary tracker: needs a server or account; git-committed results are reviewable
    and survive without external services.
  - The `tidecv` package: unmaintained; our own version is short and defensible.
- **Consequences:** Ultralytics is AGPL-3.0 (noted in README). torchmetrics/pycocotools use COCO 101-point
  interpolation, so numbers are not directly comparable to published VOC all-point AP.
- **Pinned versions (P0.2, dev group):** `ruff==0.17.3`, `pytest==9.1.1`,
  `pre-commit==4.7.0 ; python_version >= '3.11'` (pre-commit 4.7 requires 3.11; CI uses 3.11).
  Pre-commit hooks: `ruff-pre-commit v0.17.3`, `pre-commit-hooks v6.0.0`. CI actions:
  `actions/checkout@v7.0.1`, `astral-sh/setup-uv@v10.3.2`. P0.5 (dev group, local tooling):
  `kaggle==2.2.5 ; python_version >= '3.11'` (requires 3.11), `huggingface_hub==2.3.0`. Runtime dependencies are pinned as they are
  added and listed in the ADR of the phase that adds them.
