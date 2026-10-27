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

## ADR-002: Split protocol

- **Date:** 2026-10-28
- **Status:** accepted. **Splits are frozen**; changing them requires a new ADR and rerunning every
  experiment (CLAUDE.md hard rule 5).
- **Context:** VOC2012 test labels are not public. We need an untouched test set for the final numbers, a
  validation set for every model-selection and tuning decision, and identical splits for every model.
- **Decision:**
  - `test` = official VOC2012 `ImageSets/Main/val.txt`. It is used only in Phase 5 by
    `scripts/final_test_eval.py`, once per final model.
  - `val` = a random 10% of official `ImageSets/Main/train.txt`: `round(0.1 × n)` images sampled with
    `random.Random(42).sample` from the **sorted** ID list, so the result depends only on the set of IDs and
    the seed.
  - `train` = the rest of official `train`.
  - Files: `splits/{train,val,test}.txt`, sorted, one ID per line, committed with commit type `data`.
    `vocdet splits` regenerates them. `write_splits` refuses to overwrite a split file with different
    content (`SplitsFrozenError`), so rerunning it anywhere is safe.
  - Later stages read the committed files (`load_splits`) and never recompute them.
  - EDA inspects only train and val labels; test labels are not looked at before Phase 5.
- **Verified on 2026-10-28:** the official lists have the sizes PLAN.md §3 expects (5,717 train, 5,823
  val, no overlap). The resulting split sizes are given by `wc -l splits/*.txt`.
- **Alternatives considered:**
  - k-fold CV: too expensive on the GPU budget.
  - Stratifying val by class: VOC images are multi-label, so stratification is ill-defined. With ~570
    images, every class still appears in val (checked by P1.7).
  - Using official val as val and holding out part of train as test: that would make the test set smaller
    and non-standard.
- **Consequences:** test numbers are on the full official VOC2012 val (comparable in protocol, but not in
  interpolation, to published "VOC2012 val" results). Val is small, so val mAP is noisier; single-seed
  differences of under about one point should not be over-interpreted.
- **Pinned versions:** none added.

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

---

## ADR-006: VOC2012 download source and integrity check

- **Date:** 2026-10-28
- **Status:** accepted
- **Context:** PLAN.md §3 named the Ultralytics assets mirror as the fallback, verified with torchvision's
  MD5. Checked on 2026-10-28: that mirror serves `VOCtrainval_11-May-2012.zip` (1,950,180,009 bytes), a
  different file from the official `.tar`, so torchvision's MD5 cannot verify it.
- **Decision:**
  - Primary URL, filename, MD5 (`6cd6e144f989b92b3379bac3b3de84fd`), and base dir are read **at runtime** from
    `torchvision.datasets.voc.DATASET_YEAR_DICT["2012"]`. The primary is the Oxford host, which redirects
    (301) to `https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar`
    (1,999,639,040 bytes).
  - Fallback: `https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar`. It serves a tar of exactly the
    same size, and it is verified with **the same torchvision MD5**.
  - **pjreddie.com is a personal website** (Joseph Redmon, author of YOLO), not an official or institutional
    mirror. It may disappear or change at any time. We do not trust the host. **The MD5 check is the
    integrity guarantee**: any file whose MD5 differs is deleted and never extracted.
  - Downloads stream to `<file>.part` and are renamed only when complete. A cached archive is reused only if
    its MD5 matches. Extraction uses `tarfile` `filter="data"` and writes a marker file when it finishes.
  - Once the Kaggle prep kernel (P1.9) has run, its output is the working cache, so later runs do not
    download at all.
- **Alternatives considered:**
  - Ultralytics zip as the fallback: needs its own checksum (computed by a 1.95 GB download we would then
    have to trust), zip handling, and possibly a different folder layout.
  - No fallback: a single point of failure on the Oxford host during the 7-day window.
- **Consequences:** MD5 protects against corruption and accidental substitution. It is not a defence against
  a deliberate collision attack. That is acceptable for a public research dataset whose expected hash comes
  from torchvision, not from the mirror.
- **Pinned versions:** runtime `torch==2.11.0`, `torchvision==0.26.0` (CPU wheels locally and in CI; the
  same versions Kaggle preinstalls, see `docs/ENVIRONMENT.md`).

---

## ADR-007: Canonical annotation table and coordinate convention

- **Date:** 2026-10-28
- **Status:** accepted
- **Context:** VOC XML boxes are 1-based, inclusive pixel indices. torchvision, Ultralytics, and pycocotools
  all expect continuous 0-based coordinates. An off-by-one here silently shifts every box in every model.
- **Decision:**
  - `src/vocdet/data/parse_voc.py` is the **only** place that reads XML. It converts each box to 0-based
    float xyxy as `(xmin - 1, ymin - 1, xmax, ymax)`. A box covering pixel columns 1..W becomes `[0, W]`, so
    `x2 - x1` equals the number of covered pixels and a 1-pixel box has width 1.
  - Validity is asserted per box: `0 ≤ x1 < x2 ≤ width` and `0 ≤ y1 < y2 ≤ height`. Any violation raises
    `AnnotationError` with the file name; nothing is clipped silently.
  - The output is one parquet table, one row per object, with columns `image_id, width, height, class_name,
    class_id, x1, y1, x2, y2, difficult, truncated`. `class_id` indexes `VOC_CLASSES` (0–19). The two
    flags are booleans; a missing tag means 0.
  - Only `<object>/<bndbox>` (a direct child) is read. Person-layout `<part>` boxes are ignored.
  - Only the images in `ImageSets/Main/trainval.txt` are parsed. `Annotations/` also holds action- and
    layout-only images that are not part of the detection benchmark.
- **Alternatives considered:** keeping 1-based integers and converting in each exporter. That means three
  places to get the same off-by-one right instead of one.
- **Consequences:** exporters, the dataset, and the evaluator all trust the table without re-checking.
  Tests cover border boxes, single-pixel boxes, fractional coordinates, parts, and invalid boxes.
- **Pinned versions:** runtime `pandas==2.3.3`, `numpy==2.1.3` (the same as Kaggle), `pyarrow==26.0.0`
  (Kaggle's pyarrow version is not recorded yet; P1.9 checks it). Added later in Phase 1: `pycocotools==2.0.11` (P1.5, the same as
  Kaggle), `matplotlib==3.11.2` and `tabulate==0.10.0` (P1.8, EDA figures and markdown tables; Kaggle
  versions are checked in P1.9).
