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

---

## ADR-008: Evaluator design

- **Date:** 2026-10-28
- **Status:** accepted
- **Context:** every model must be scored by one evaluator (PLAN.md §2, §4). torchmetrics 1.9.0 (the
  Kaggle version) is the backend chosen in ADR-001. Its `class_metrics=True` reports per-class AP
  averaged over IoU 0.50:0.95, but we want per-class AP at IoU 0.5.
- **Decision:**
  - `src/vocdet/eval/evaluator.py::evaluate(predictions, ground_truth)` is the only metric code.
  - Instance 1, `MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")` with the
    default COCO thresholds, provides `map50`, `map50_95`, `map_small/medium/large` (over 0.50:0.95, COCO
    convention), and `mar_100`.
  - Instance 2, the same but with `iou_thresholds=[0.5]` and `class_metrics=True`, provides
    `ap50_per_class`. A test checks that `map50` equals the mean of the per-class AP50 values.
  - Difficult objects are passed as `iscrowd = 1` (ignore regions). Undefined metrics (torchmetrics `-1`,
    e.g. a class with no ground truth) become `None` in JSON.
  - Every ground-truth image must have a prediction entry, possibly empty. A missing or extra entry raises,
    so an image dropped by a bug cannot silently inflate precision.
- **Alternatives considered:**
  - One instance with `extended_summary=True`, slicing the precision array at IoU 0.5 by hand. That avoids
    computing twice but is harder to read and explain.
  - Our own AP code: more to defend, and no gain over the reference implementation.
- **Consequences:**
  - Evaluation runs pycocotools twice (about 8 s for val, including imports).
  - COCO 101-point interpolation, so numbers are not comparable to VOC all-point AP tables. Every report
    states this.
  - COCO "crowd" IoU is intersection over **detection** area, not union. Any detection lying mostly inside
    a difficult box is ignored. This is slightly more lenient than VOC's standard-IoU rule for difficult
    objects, and it applies identically to every model.
- **Pinned versions:** runtime `torchmetrics==1.9.0` (the same as Kaggle).

---

## ADR-009: TIDE-lite error analysis rules

- **Date:** 2026-10-28
- **Status:** accepted
- **Context:** PLAN.md §4 asks for our own error breakdown (Cls, Loc, Both, Dupe, Bkg, Miss) plus a
  confusion matrix. The `tidecv` package is unmaintained (ADR-001), and the exact rules must be explainable.
- **Decision:** `src/vocdet/eval/error_analysis.py`:
  - **Score threshold.** Only detections with score ≥ a threshold are analysed. The CLI default is 0.5;
    reports use each model's F1-optimal operating threshold on val. With the mAP setting (conf 0.001),
    near-zero-score background hits would swamp the counts.
  - **Matching.** Per image, detections in descending score order. Each is compared with the same-class
    ground-truth box of highest IoU (difficult boxes included), following the VOC devkit. At IoU ≥ 0.5: a
    difficult box makes it Ignored, an unmatched box makes it TP, and an already matched box makes it Dupe.
  - **False-positive order:** Cls (IoU ≥ 0.5 with another class) → Loc (0.1 ≤ IoU < 0.5, same class) →
    Both (0.1 ≤ IoU < 0.5, other class) → Bkg. Only non-difficult objects count here. This mirrors the
    TIDE priority.
  - **Miss** = every non-difficult object never matched as TP.
  - **Confusion matrix:** 21×21, index 20 = background, rows = true class, columns = predicted class. Cls
    goes to `[true, predicted]`; Loc, Both, Dupe, and Bkg go to `[background, predicted]`; Miss goes to
    `[true, background]`. Cells count events, so row sums are not object counts.
  - The IoU is computed by hand (`box_iou`, vectorised numpy).
- **Alternatives considered:**
  - Weighting errors by their mAP impact, as TIDE does: more faithful, but requires re-running the evaluator
    per error type. Counts are enough to rank error sources and are easy to defend.
  - Matching each detection to the best *unmatched* object instead of the VOC argmax rule: this differs only
    in crowded scenes. We keep the devkit behaviour so our TP counts match VOC conventions.
- **Consequences:** counts depend on the threshold, so every report must state it. Ignored detections
  appear in the records but in no count or matrix cell.
- **Pinned versions:** none added.

---

## ADR-010: Linux-only lock resolution and the ultralytics pin

- **Date:** 2026-10-28
- **Status:** accepted
- **Context:** adding `ultralytics==8.4.191` failed to lock. ultralytics requires
  `numpy<2.0 or >2.3.4` **on macOS only**, which conflicts with our `numpy==2.1.3` pin (the Kaggle
  version). uv resolves for every platform by default.
- **Decision:**
  - `[tool.uv] environments = ["sys_platform == 'linux'"]`. The lockfile covers Linux only, which is every
    environment of this project: the dev machine, GitHub Actions, Kaggle, and the Docker Space.
  - Pin `ultralytics==8.4.191`, the latest release on 2026-10-28, which includes the YOLO26 weights. On
    Kaggle it is installed from the config's `kaggle.pip`, under a constraints file that pins the
    preinstalled torch/torchvision. The runner records their versions before and after and fails on any
    change (PLAN.md P4.1 criterion, implemented early because E0b is the first kernel to install
    ultralytics).
- **Alternatives considered:** bumping numpy past 2.3.4 to satisfy macOS. That would diverge from Kaggle's
  numpy 2.1.3, which we keep on purpose (`--no-deps` installs mean Kaggle's versions run).
- **Consequences:** `uv sync` on macOS is unsupported. That is acceptable for this project and stated in
  the README (P8.2).
- **Observation to follow up:** in 8.4.191, a loaded `yolo26n.pt` reports `Detect.end2end = False`, so
  `predict()` runs NMS (iou 0.7). PLAN.md §1 describes YOLO26 as NMS-free. Phase 4 (training) and Phase 6
  (export) must check how end-to-end mode is enabled before claiming NMS-free inference.
- **Pinned versions:** runtime `ultralytics==8.4.191`, plus its resolved dependencies in `uv.lock`.

---

## ADR-011: Prepared-data location on Kaggle

- **Date:** 2026-10-28 (revised the same day after the root cause was found)
- **Status:** accepted
- **Context:** training and evaluation kernels mount the prep kernel's output via `kernel_sources`. Two
  E0b runs failed before predicting because they could not find it:
  1. `E0b_20261028-0257_5978b2b` searched `/kaggle/input` with
     `glob("*/*/voc_prepared/annotations.parquet")`, i.e. only two levels below the root, and found
     nothing.
  2. `E0b_20261028-0305_3fc1694` used the path seen by two `env_check` kernels (CPU and T4):
     `/kaggle/input/vocdet-prepare-data/<prep_run_id>/voc_prepared/`. It waited 120 s, then listed
     `/kaggle/input`: the only entry was `/kaggle/input/notebooks/nikaatanelishvili`.
- **Root cause:** Kaggle mounts the same `kernel_sources` entry at **different depths in different
  kernels**. The `vocdet-env-check` kernel sees `/kaggle/input/vocdet-prepare-data/...`. The
  `vocdet-e0b` kernel sees `/kaggle/input/notebooks/<owner>/vocdet-prepare-data/...`, one level deeper.
  The first glob was too shallow for the second layout, and the pinned path came from the first. Nothing
  showed a *delayed* mount: the second run's listing showed the data mounted elsewhere, not missing. Why
  the two kernels differ is not documented; we treat the depth as unknowable in advance.
- **Decision:**
  - `configs/data/voc2012.yaml` holds only `prep_run_id`, which pins the data version (validated format
    `prepare_data_<yyyymmdd-hhmm>_<sha7>`).
  - `data/prepare.py::find_prepared_dir(prep_run_id)` walks `/kaggle/input` **recursively** (`os.walk`,
    following symlinks, with a loop guard). It never descends into `voc_prepared/` (~23k files). It looks
    for `<prep_run_id>/voc_prepared/annotations.parquet`.
  - It requires **exactly one** match. Zero or several matches fail immediately with a listing of the top
    three levels of `/kaggle/input`. There is **no waiting**: no evidence of a delayed mount, so no retry
    for it.
  - Before any GPU launch of a kernel that reads prepared data, the same kernel runs once on **CPU** with
    `--resolve-only` (via `kaggle_run.py --cpu --append-args "--resolve-only"`). That prints the resolved
    path and exits, so a path problem costs no GPU time.
- **Alternatives considered:**
  - Listing both observed layouts as candidates: this breaks again if a third layout appears.
  - A recursive search for any `voc_prepared/`: robust, but no longer tied to a data version.
  - The fixed path plus a 120 s wait (the first revision of this ADR): it addressed a cause (a slow mount)
    that the evidence did not support.
- **Consequences:** startup cost is one directory walk that skips the image folders (seconds). A second
  prep run with a different run ID can be attached side by side without ambiguity. Two copies of the same
  run ID fail loudly instead of being picked arbitrarily.
- **Pinned versions:** none added.

## ADR-012: Faster R-CNN training loop and resume

- **Date:** 2026-10-29
- **Status:** accepted
- **Context:** P3.2 asks for our own explicit training loop (CLAUDE.md: no framework hides it) with
  warm-up plus step decay, AMP, gradient-norm logging, periodic validation, best/last checkpoints, and
  resume, because a Kaggle session can die. Each Kaggle launch is a new session with a new run folder
  under `/kaggle/working`. Nothing from an earlier session is on disk unless it is attached as an input.
- **Decision:**
  - **Loop** (`train/train_frcnn.py::train_one_epoch`): per iteration, set the LR, forward (the model
    returns its four losses in train mode), sum them, then `GradScaler.scale(loss).backward()` and
    `unscale_`. Measure the gradient norm with `clip_grad_norm_(max_norm=inf)` (it measures and never
    clips), then `scaler.step` and `scaler.update`. One CSV row per iteration holds the LR, total loss, the
    four component losses, and the gradient norm. A non-finite loss raises `FloatingPointError`
    immediately.
  - **LR** is one pure function, `lr_at(step, epoch)`: a linear warm-up from `lr * warmup_factor` over the
    first `warmup_iters` steps, times `gamma` for every milestone epoch reached. This is the schedule of
    `MultiStepLR` plus a warm-up, without two chained scheduler objects (DEVIATIONS 2026-10-29). Nothing
    needs saving for resume.
  - **Validation** runs every `eval_every` epochs and after the last one, through the shared evaluator
    (ADR-008). `best.pt` holds the weights with the highest val map50. The final `predictions_val.json`
    and `metrics_val.json` are re-predicted from `best.pt`, so they are right after a resume too.
  - **`last.pt`** is written after every epoch. It holds the model, optimizer, AMP scaler, shuffle
    generator, the Python/NumPy/torch/CUDA RNG states, the epoch, the global step, and the best score. A
    session that dies mid-epoch loses at most that epoch.
  - **Resume into any folder.** The source run is the folder above `last.pt`. Its `best.pt` is copied
    over. Both logs are rebuilt from the source logs, keeping only rows up to the checkpoint's epoch
    (`start_log`). Rows of the unfinished epoch are dropped because that epoch is trained again. A missing
    source log becomes a header-only file. `run_meta.json` records `resumed_from`.

### Resuming a dead Kaggle run

**How the new session gets `last.pt`.** It attaches the dead run's kernel output as a kernel source.
This is the same mechanism the prep data uses (ADR-011), and it is found the same way:
`utils/kaggle_input.py::find_run_file(<run_id>, "weights/last.pt")` searches `/kaggle/input` at any
depth. The resume **must be a different kernel**. A kernel source always mounts the source kernel's
*latest* version, and a new version of `vocdet-e1` would be the resume itself. So
`kaggle_run.py --resume-from <run_id>`:
- pushes to `vocdet-<exp>-from-<yyyymmdd-hhmm>`, named after the dead run's timestamp;
- adds the dead run's kernel, read from `kaggle/kernels/<run_id>/kernel-metadata.json` on the
  launching machine, to `kernel_sources`;
- appends `--resume-run <run_id>` to the command.

Procedure, with `DEAD` as the dead run ID (e.g. `E1_20261029-1430_abc1234`):

1. Do **not** launch anything else on `vocdet-e1` until the resume finishes. A new version would replace
   the output the resume mounts.
2. Pull what the dead run saved: `make kaggle-pull RUN=$DEAD`. It downloads the logs. It exits non-zero
   when `kaggle_run.json` is missing (a killed session), and that is expected. Read
   `runs/$DEAD/command.log` and `train_log.csv` to see where and why it died. If it died from a code bug,
   fix the bug, commit, and push first. The resume runs the current HEAD, and `run_meta.json` records
   both the SHA and `resumed_from`.
3. Check that the checkpoint survived: `uv run kaggle kernels files nikaatanelishvili/vocdet-e1` must
   list `last.pt` (and `best.pt` if a validation ran). If `last.pt` is not listed, stop: there is
   nothing to resume, so rerun from scratch.
4. Check the path on CPU, which uses no GPU quota (same pattern as ADR-011):
   `uv run python scripts/kaggle_run.py --config configs/experiments/E1.yaml --resume-from $DEAD --cpu --append-args=--resolve-only`,
   then `make kaggle-pull RUN=<printed run_id>`. Its `command.log` must show `Resume from: /kaggle/input/.../$DEAD/weights/last.pt`.
5. Launch the GPU resume:
   `uv run python scripts/kaggle_run.py --config configs/experiments/E1.yaml --resume-from $DEAD`,
   then `make kaggle-pull RUN=<printed run_id>`.
6. Collect only the resumed run: list its run_id in `results/canonical_runs.yaml`, then `make results`.
   The dead run's `run_meta.json` still says `running`, so the collector would refuse it if it were
   listed. Charge both sessions in the GPU usage log in `docs/ENVIRONMENT.md`.

**What is verified and what is not.** Kernel sources mounting another kernel's output: verified (prep
data, ADR-011). A failed version keeps its `/kaggle/working` outputs when the *script exits with an
error*: verified, because the failed run `E0b_20261028-0257_5978b2b` was pulled with its logs. Whether
outputs survive when *Kaggle kills the session* (time limit, out-of-memory kill, lost worker): **not
verified**. If they do not, cross-session resume is impossible for that failure, and step 3 detects
this. The fallback is a full rerun. That costs at most the E1 estimate (§5), inside the 5 h buffer, and
PLAN.md §9 keeps every run ≤4 h, far from the session limit. The path through steps 4–5 is tested
end to end on Kaggle in P3.4 by resuming the finished E0a-frcnn smoke run on CPU (no epochs left, so it
only reloads and re-evaluates).

- **Alternatives considered:**
  - `MultiStepLR` + `LinearLR` in a `SequentialLR`: the same schedule, but a stateful object to save,
    and subtle per-step/per-epoch ordering.
  - Uploading `last.pt` as a Kaggle dataset for the new session: extra manual upload steps and ~150 MB
    through the local machine, for the same result as a kernel source.
  - A new version of the same kernel sourcing itself: impossible, because a source means its latest version.
  - Requiring the user to copy logs by hand on resume: easy to forget; a forgotten copy produced
    header-less logs.
- **Consequences:** a dead session costs at most one epoch plus the session start-up, *if* its output
  was saved. Resume kernels add extra kernels (`vocdet-e1-from-...`) to the Kaggle account. The pull
  pattern keeps weights on Kaggle, so `last.pt` lives only in the kernel output. Deleting a kernel
  deletes its checkpoints.
- **Pinned versions:** none added.

---

## ADR-013: Ultralytics val as a cross-check for our evaluator

- **Date:** 2026-10-29
- **Status:** accepted. The size of each effect is measured in every YOLO run, not assumed (see
  Consequences).
- **Context:** P4.2 asks for Ultralytics' own `model.val()` mAP next to ours for every fine-tuned YOLO
  model, with an explanation of any difference. ADR-008 makes our evaluator (torchmetrics + pycocotools)
  authoritative for every model. A second scorer catches adapter bugs (class map, box scaling, image IDs)
  that one evaluator cannot see in its own input.
- **Decision:**
  - `train/train_yolo.py::ultralytics_val_crosscheck` runs `best.val(data=<run's voc_yolo.yaml>,
    split="val")` with the run's `imgsz` and `batch`, and leaves `conf` (0.001 for val), `iou` (0.7),
    and `max_det` (300) at the Ultralytics 8.4.191 defaults. It writes four numbers to
    `run_meta.json → ultralytics_val_crosscheck`:
    - `ultralytics_map50` and `ultralytics_map50_95`, from Ultralytics;
    - `ours_map50_difficult_removed` and `ours_map50_95_difficult_removed`, which are our predictions
      scored by our evaluator against ground truth with the difficult objects removed.
  - The authoritative numbers stay in `metrics_val.json` (predict_yolo → evaluate, difficult objects
    as ignore regions).
  - A gap of more than `CROSSCHECK_TOLERANCE` (0.05 map50) between `ultralytics_map50` and
    `ours_map50_difficult_removed` logs a warning. The run is then checked by hand before its numbers
    are used. The first value, 0.02, fired on a healthy run (see the measured breakdown below). An
    adapter bug (a shifted class map, unscaled boxes, mixed-up image IDs) costs tens of points, so 0.05
    still catches it.
- **Why the two scores differ.** Candidate causes were read in the Ultralytics 8.4.191 code and ADR-008:
  1. **Difficult objects.** The YOLO labels leave them out (`data/export_yolo.py`), so for Ultralytics a
     detection on one is a false positive. Our evaluator passes them as `iscrowd`, so it ignores such a
     detection. Every run isolates this effect: `metrics_val.json map50 − ours_map50_difficult_removed`.
  2. **AP metric.** Ultralytics linearly interpolates the precision envelope and integrates with the
     trapezoid rule (`utils/metrics.py::compute_ap`). pycocotools averages step values at 101 recall
     points. Matching is the same greedy, confidence-ordered rule in both
     (`BaseValidator.match_predictions`, default path).
  3. **Detection cap.** We keep 100 per image (PLAN.md §4); Ultralytics val keeps up to 300.
  4. **Inference path.** `model.val()` loads images through its dataloader into rectangular batches,
     with half-stride padding. `predict()` letterboxes each image on its own.
- **Measured** by `scripts/diagnose_crosscheck_gap.py` on the yolo26n timing run's `best.pt` (CPU, which
  reproduces the Kaggle numbers). Output: `results/diagnostics/crosscheck_gap_E0a_yolo_timing_n_20261029-1539_860a6b4.json`.
  - Ultralytics' metric applied to *our* predictions lands very close to our evaluator, and keeping 300
    detections barely moves it. So effects 2 and 3 are small, and our adapter and evaluator agree with
    Ultralytics on the same boxes.
  - Rect batch padding (batch 16 vs 1) is also small.
  - Nearly the whole gap is effect 4: Ultralytics' val pipeline scores higher than Ultralytics' own
    `predict()` on the same weights.
  - We keep scoring the `predict()` path, because it is what the deployed service runs (Phase 7) and
    how Faster R-CNN is scored. The val-pipeline gain is a property of Ultralytics' evaluation code,
    not of the model we ship.
- **Alternatives considered:**
  - Training YOLO labels with difficult objects included. That changes the training data relative to
    Faster R-CNN, which never trains on difficult objects (`data/dataset.py`), so it was rejected.
  - Using only Ultralytics' number for YOLO models. Then the two model families would be scored by
    different code, and the comparison would be meaningless (ADR-008).
- **Consequences:**
  - Each YOLO run costs one extra val pass (seconds on a T4).
  - Effect 1 and the remaining gap are recorded per run in `run_meta.json`. They are quoted from there,
    or from the diagnostics JSON, by generated tables, never typed into this ADR.
- **Pinned versions:** none added (`ultralytics==8.4.191`, ADR-010).
