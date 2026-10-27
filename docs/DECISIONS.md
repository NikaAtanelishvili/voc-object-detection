# Architecture Decision Records

One entry per design decision. Newest at the bottom. Never edit an accepted ADR's decision; supersede it
with a new ADR and link both ways.

Numbering: PLAN.md reserves **ADR-002** (split protocol, P1.4), **ADR-003** (model selection, P4.6), and
**ADR-004** (deployment variant, P6.6) by name. Other ADRs take the next free number from ADR-005 on, in the
order they are written, so the plan's references stay valid.

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
- **Status:** accepted; the Python floor is superseded by [ADR-005](#adr-005-python-311-floor)
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
  `kaggle==2.2.5 ; python_version >= '3.11'` (requires 3.11), `huggingface_hub==2.3.0`. P0.6:
  runtime `pydantic==2.14.1`, `pyyaml==6.0.3`; dev `jinja2==3.1.6` (renders the Kaggle runner). Runtime dependencies are pinned as they are
  added and listed in the ADR of the phase that adds them.

---

## ADR-005: Python 3.11 floor

- **Date:** 2026-10-28
- **Status:** accepted (supersedes the Python line of ADR-001)
- **Context:** ADR-001 chose Python ≥3.10 "to match Kaggle". The `env_check` kernel showed Kaggle actually
  runs Python 3.13.15. Several tools need ≥3.11: kaggle 2.2.5, pre-commit 4.7.0, and onnxruntime 1.31.1,
  which the deployed app will need.
- **Decision:** `requires-python = ">=3.11"`, ruff `target-version = "py311"`. CI tests 3.11 and 3.13. The
  app Docker image uses 3.13. Owner decision on 2026-10-28.
- **Alternatives considered:** keep 3.10 and put `python_version >= '3.11'` markers on every package that
  needs it. That supports a version that no environment in this project uses, and it adds markers to explain.
- **Consequences:** no environment-marker workarounds in `pyproject.toml`. Stdlib features from 3.11 are
  allowed (e.g. `datetime.UTC`, `tarfile` extraction filters from 3.11.4).
- **Pinned versions:** unchanged; the markers on `kaggle==2.2.5` and `pre-commit==4.7.0` were removed.
