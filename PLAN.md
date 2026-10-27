# PLAN.md — Pascal VOC Object Detection: End-to-End ML Project

Execution plan for Claude Code. Read `CLAUDE.md` first. Work top to bottom and tick tasks as you finish them.
Tasks tagged **[MANUAL]** are done by the human owner; the agent writes the instructions and waits.
Every phase ends with the **Phase gate** from `CLAUDE.md`: the agent prints a phase report (work done,
deviations, results, GPU budget, manual verification checklist) and waits for approval before merging and
tagging. "Tag `phase-n`" in the exit criteria below means "after approval".

---

## 0. Goal and success criteria

**Problem statement.** Build a general-purpose 20-class object detector that runs fast enough on a **2-vCPU
machine with no GPU** to be useful in a real service (scene inventory, object counting, filtering images by
content). Accuracy is traded against CPU latency deliberately, and every trade-off is measured.

**Deliverables (from the brief), with where each one lives:**

| Brief requirement                           | Deliverable in this repo                                                    |
| ------------------------------------------- | --------------------------------------------------------------------------- |
| Data processing                             | `src/vocdet/data/`, `reports/EDA.md`                                        |
| Model development                           | `src/vocdet/models/`, `src/vocdet/train/`                                   |
| Experiments and comparison                  | `configs/experiments/`, `results/experiments.csv`, `reports/EXPERIMENTS.md` |
| Evaluation and error analysis               | `src/vocdet/eval/`, `reports/ERROR_ANALYSIS.md`                             |
| Optimization before/after                   | `src/vocdet/optimize/`, `results/benchmarks.csv`, `reports/OPTIMIZATION.md` |
| Practical application                       | `app/` (FastAPI + Gradio), live Hugging Face Space                          |
| Repo, instructions, dependencies, structure | `README.md`, `pyproject.toml`, `uv.lock`                                    |
| Trained model or reproduction steps         | Hugging Face Hub model repo + `README.md#reproduce`                         |
| Technical report (6 questions)              | `reports/TECHNICAL_REPORT.md`                                               |
| AI tool usage                               | `reports/AI_USAGE.md`                                                       |

**Done means:**

- [ ] At least 5 trained configurations compared on identical splits with one shared evaluator.
- [ ] A before/after optimization table covering accuracy, latency (p50/p95), model size, and memory.
- [ ] A live public URL serving the API and UI.
- [ ] CI is green, the README reproduces everything, and the test set was used exactly once per final model.

---

## 1. Technology stack

Pin exact versions when adding each dependency (CLAUDE.md hard rule 7). Use Kaggle's preinstalled torch and
torchvision on Kaggle; install only what is missing there.

| Layer               | Choice                                                                                                        | Why                                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Language            | Python ≥3.10 (match Kaggle)                                                                                   | Kaggle runtime compatibility                                                                        |
| Env / packaging     | `uv` + `pyproject.toml` (src layout)                                                                          | Fast, lockfile-based, reproducible                                                                  |
| DL framework        | PyTorch + torchvision                                                                                         | Standard CV ecosystem; Faster R-CNN reference implementation                                        |
| One-stage detector  | Ultralytics **YOLO26** (n, s)                                                                                 | Current Ultralytics family (Jan 2026), NMS-free end-to-end inference, clean export to ONNX/OpenVINO |
| Two-stage baseline  | torchvision `fasterrcnn_mobilenet_v3_large_fpn` (COCO weights)                                                | Different paradigm (two-stage + RPN). Own training loop gives defensible, explainable code          |
| Metrics             | `torchmetrics` `MeanAveragePrecision` (pycocotools backend)                                                   | Standard COCO-style mAP. One evaluator for all models                                               |
| Error analysis      | Own implementation ("TIDE-lite")                                                                              | Custom code we can explain. No unmaintained dependency                                              |
| Experiment tracking | Run folders + `results/experiments.csv` (primary); W&B (optional)                                             | Results live in git and need no external account                                                    |
| Optimization        | ONNX + ONNX Runtime; OpenVINO FP32 / INT8 (NNCF via Ultralytics export); optional TensorRT FP16 on Kaggle GPU | CPU deployment target; INT8 is the main lever                                                       |
| Serving             | FastAPI + Uvicorn; Gradio mounted at `/ui`                                                                    | One process gives both a REST API and a UI                                                          |
| Hosting             | Hugging Face Space (Docker SDK, free CPU) + HF Hub model repo                                                 | Free, public URL; weights versioned on the Hub                                                      |
| Training compute    | Kaggle Notebooks via `kaggle` CLI (`kernels push/status/output`)                                              | Free GPU, scriptable from the terminal                                                              |
| Quality             | ruff, pytest, pre-commit, GitHub Actions (CPU)                                                                | Automated lint and tests on every push                                                              |
| Config              | YAML + pydantic                                                                                               | Validated, diffable experiment definitions                                                          |

**License note (put in README):** Ultralytics is AGPL-3.0. That is fine for an open-source take-home. Mention
it in README and the technical report as a production consideration.

---

## 2. Architecture and repository layout

```
voc-object-detection/
├── CLAUDE.md  PLAN.md  README.md  LICENSE  Makefile
├── pyproject.toml  uv.lock  .env.example  .gitignore  .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── configs/
│   ├── data/voc2012.yaml              # paths, class list, split files
│   └── experiments/E*.yaml            # one file per experiment
├── splits/                            # committed, frozen image-ID lists
│   ├── train.txt  val.txt  test.txt
├── src/vocdet/
│   ├── config.py                      # pydantic models for YAML configs
│   ├── constants.py                   # VOC_CLASSES, COCO->VOC mapping
│   ├── data/   download.py parse_voc.py splits.py export_yolo.py export_coco.py
│   │           dataset.py transforms.py integrity.py eda.py
│   ├── models/ frcnn.py yolo.py
│   ├── train/  train_frcnn.py train_yolo.py
│   ├── eval/   schema.py predict.py evaluator.py error_analysis.py visualize.py
│   ├── optimize/ export.py benchmark.py
│   ├── inference/ detector.py         # one interface for pytorch / onnx / openvino backends
│   └── utils/  seed.py run_meta.py io.py logging.py
├── kaggle/
│   ├── templates/runner.py.j2         # clones repo at SHA, installs, runs a CLI command
│   └── kernels/                       # generated kernel folders (gitignored)
├── scripts/
│   ├── kaggle_run.py  kaggle_pull.py  collect_results.py  final_test_eval.py
│   ├── build_report_tables.py  upload_weights.py  deploy_space.py  smoke_test_live.py
├── app/   main.py  ui.py  Dockerfile  README_space.md
├── tests/  (unit tests + tiny fixtures; CI never needs the real dataset)
├── results/  experiments.csv  benchmarks.csv  runs/<run_id>/{metrics.json,config.yaml,run_meta.json}
├── reports/  EDA.md EXPERIMENTS.md ERROR_ANALYSIS.md OPTIMIZATION.md TECHNICAL_REPORT.md AI_USAGE.md figures/
├── docs/  DECISIONS.md MANUAL_STEPS.md ENVIRONMENT.md DEPLOYMENT.md defense/
└── runs/  (gitignored: weights, full predictions, logs pulled from Kaggle)
```

**Key design principle: one prediction schema, one evaluator.** Every model, whether Faster R-CNN, YOLO, or an
exported ONNX/OpenVINO variant, outputs `Prediction(image_id, boxes_xyxy_abs, scores, labels∈[0,19])`. All
metrics are computed by the same `eval/evaluator.py`. That is what makes the comparisons fair.

**Data flow:**

```
VOC tar → parse_voc → canonical annotations (parquet, 0-based xyxy)
        → splits (frozen ID lists)
        → export_yolo (labels + dataset yaml)  → Ultralytics trainer
        → export_coco (json, difficult=iscrowd) → evaluator ground truth
        → dataset.py (torch Dataset)            → Faster R-CNN trainer
models → predict adapters → Prediction JSON → evaluator / error_analysis → results/
best model → export (onnx/openvino/int8) → benchmark → Detector → FastAPI/Gradio → HF Space
```

**Run artifact contract** (every training or eval run produces this):

```
runs/<run_id>/            run_id = <exp_id>_<yyyymmdd-hhmm>_<sha7>
  config.yaml             resolved config
  run_meta.json           git sha, start/end time, GPU name, python/torch/ultralytics versions, seed, wall time
  train_log.csv           per-epoch (and FRCNN per-iter) losses, lr, val mAP
  metrics_val.json        evaluator output on val
  predictions_val.json    raw predictions (gitignored if >1 MB)
  weights/best.pt  last.pt (gitignored; uploaded to HF Hub)
```

`scripts/collect_results.py` copies the small files into `results/runs/<run_id>/` and appends one row per run to
`results/experiments.csv`.

---

## 3. Data protocol

- **Source:** VOC2012 trainval tar (`VOCtrainval_11-May-2012.tar`). Primary URL: the official Oxford host.
  Fallback: the Ultralytics assets mirror. Verify the MD5 against the value torchvision ships in
  `torchvision.datasets.voc.DATASET_YEAR_DICT["2012"]`. Read it at runtime; do not hardcode from memory.
- **Classes (20, alphabetical, index 0–19):** aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
  diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor.
- **Expected counts (verify; stop if different):** detection `ImageSets/Main/train.txt` = 5,717 images and
  `val.txt` = 5,823 images (11,540 trainval).
- **Coordinates:** VOC XML boxes are **1-based inclusive pixel indices**. Convert to 0-based float `xyxy` with
  `xmin-1, ymin-1, xmax, ymax`. Document this in an ADR and test it.
- **Splits (seed = 42, frozen after P1.4):**
  - `train` = official `train` minus a random 10% of its images.
  - `val` = that 10% (~572 images), used for model selection, early stopping, and all tuning.
  - `test` = official VOC2012 `val` (5,823 images), used only for the final evaluation.
  - Justification (ADR): VOC2012 test labels are not public, so the official val becomes an untouched test set.
- **Difficult objects:** excluded from training targets. In evaluation they are marked `iscrowd=1`, so COCO eval
  treats them as **ignore regions**: a detection on a difficult object is neither a TP nor an FP. This mirrors VOC
  protocol. Add a test for it.
- **COCO pretraining caveat:** all 20 VOC classes exist in COCO, so pretrained backbones already know them. The
  zero-shot experiment (E0b) quantifies this, and the report discusses it.

---

## 4. Evaluation protocol

- **Primary metric:** mAP@0.5 on `val` for selection and on `test` for the final report.
  **Secondary:** mAP@0.5:0.95, per-class AP@0.5, AP by object size (COCO buckets: small <32², medium 32²–96²,
  large >96² px area), recall@100.
- **Interpolation caveat:** torchmetrics/pycocotools use COCO-style 101-point interpolation, while VOC2012 uses
  all-point AP. Numbers are therefore not directly comparable to published VOC tables. State this in every
  report.
- **Inference settings for mAP:** confidence threshold 0.001, max 100 detections per image (standard for mAP).
  The demo app uses an operating threshold chosen on `val` (the F1-optimal confidence of the deployed model).
- **Speed metrics:** batch-1 latency p50/p95 (ms) after 20 warm-up runs over a fixed set of 200 `val` images,
  FPS = 1000/p50, model file size (MB), peak RSS memory (MB). Always record CPU/GPU model, thread count, and
  backend version.
- **Error analysis (TIDE-lite, own code):** at IoU 0.5 with greedy matching by score, classify every false
  positive as **Cls** (right box, wrong class), **Loc** (right class, 0.1 ≤ IoU < 0.5), **Both**, **Dupe**
  (duplicate of an already-matched GT), or **Bkg** (IoU < 0.1 with all GT). Count every unmatched GT as **Miss**.
  Report counts per category and per class, plus a class confusion matrix.
- **Seeds:** single seed (42) per configuration because of compute limits. Acknowledge this in the report. If
  the budget allows, rerun the deployed configuration with seed 43 to estimate variance.

---

## 5. Experiment matrix and GPU budget

All experiments use the same splits, the same evaluator, and COCO-pretrained weights unless stated otherwise.
Epoch numbers are starting points. The smoke runs (P3.4/P4.3) measure seconds per epoch, and the agent
**updates the estimates column before launching full runs**.

| ID  | Model                          | Change vs reference                                                       | Question answered                                     | Est. GPU h              |
| --- | ------------------------------ | ------------------------------------------------------------------------- | ----------------------------------------------------- | ----------------------- |
| E0a | both                           | 64 images, 1 epoch                                                        | Does the pipeline work end to end? Speed calibration  | 0.3                     |
| E0b | YOLO26n COCO, no training      | COCO→VOC class mapping, zero-shot                                         | How much does fine-tuning add over pretraining alone? | 0.2                     |
| E1  | Faster R-CNN MobileNetV3-L FPN | 12 epochs, SGD lr 0.01, bs 8, short side 600, AMP                         | Two-stage baseline                                    | 2.5                     |
| E2  | YOLO26n                        | 50 epochs, imgsz 640, patience 15, default aug                            | One-stage, smallest                                   | 1.5                     |
| E3  | YOLO26s                        | same as E2                                                                | Capacity vs speed                                     | 3.0                     |
| E4  | YOLO26n                        | minimal aug: mosaic=0, mixup=0, hsv=0, scale=0, translate=0 (fliplr kept) | What does augmentation buy?                           | 1.5                     |
| E5  | YOLO26n                        | freeze=10 (backbone frozen)                                               | Full fine-tune vs frozen backbone (cost/accuracy)     | 1.0                     |
| O\* | best model                     | export + GPU FP16/TensorRT bench (optional)                               | Optimization                                          | 0.5                     |
| —   | buffer                         | reruns / failures / seed 43                                               |                                                       | 5.0                     |
|     |                                |                                                                           | **Total budget**                                      | **~15.5 of ~30 weekly** |

The weekly Kaggle GPU quota is shown in Kaggle account settings (commonly about 30 h, but it floats). The human
checks it in P0.5 and writes it into `docs/ENVIRONMENT.md`. Run data preparation and EDA on **CPU** kernels so
they don't consume GPU quota.

Faster R-CNN labels: torchvision reserves 0 for background, so the model uses `num_classes=21` and labels are
`voc_idx + 1` at training time and `pred - 1` at prediction time. Add a unit test because this off-by-one is a
classic bug.

---

## 6. Phases and tasks

### Phase 0 — Setup (Day 1, morning)

- [x] **P0.1 [MANUAL]** Create a **public** GitHub repo `voc-object-detection` (empty, no template). Clone it
      locally, copy `CLAUDE.md` and `PLAN.md` into the root, then commit `chore: add agent plan` and push. The repo
      must be public so Kaggle kernels can `git clone` it without tokens. If the human wants it private until
      submission, stop and the agent will switch to the "code as Kaggle dataset" alternative (ADR).
- [x] **P0.2** Scaffold: `pyproject.toml` (uv, src layout, package `vocdet`, console script `vocdet`), ruff
      config, pytest config, `.gitignore` (data/, runs/, .env, _.pt, _.onnx, *\_openvino_model/, kaggle/kernels/),
      `.env.example`, `.pre-commit-config.yaml` (ruff format, ruff lint, end-of-file, trailing whitespace,
      check-added-large-files maxkb=1024), `Makefile` with all targets from CLAUDE.md (stubs allowed), and an empty
      package with `__init__.py` holding `__version__`.
      *Accept:\* `make setup && make check` passes on a clean clone.
- [x] **P0.3** CI: `.github/workflows/ci.yml` running on push and PR. Ubuntu, Python 3.11, uv, CPU-only torch
      (index-url for CPU wheels), then `make check`. Add a cache for uv.
      _Accept:_ green check on GitHub for the pushed commit.
- [x] **P0.4** Docs skeleton: `README.md` (title, one-paragraph summary, sections marked TODO),
      `docs/DECISIONS.md` with the ADR template and **ADR-001 Stack choice** (from §1), `docs/MANUAL_STEPS.md`,
      `docs/DEVIATIONS.md` (with the entry template), `docs/phase-reports/`,
      `docs/ENVIRONMENT.md`, and stub files in `reports/`.
- [x] **P0.5 [MANUAL]** Accounts and credentials:
  1. Kaggle account → **phone verification** (required for GPU and internet in kernels).
  2. Kaggle → Settings → API → create a token. Install it as the Kaggle CLI expects (`~/.kaggle/kaggle.json`,
     chmod 600, or the env var method in the current Kaggle docs). Verify with `kaggle kernels list --mine`.
  3. Note the weekly GPU quota shown in Kaggle settings and write it into `docs/ENVIRONMENT.md`.
  4. Hugging Face account → Settings → Access Tokens → create a **write** token. Put `HF_TOKEN` and
     `HF_USERNAME` in local `.env`.
  5. (Optional) W&B account → put `WANDB_API_KEY` in `.env`.
  6. `KAGGLE_USERNAME` in `.env`.
- [ ] **P0.6** Kaggle plumbing. Write `kaggle/templates/runner.py.j2` and `scripts/kaggle_run.py`:
  - `kaggle_run.py --config <yaml> [--gpu/--cpu] [--sources kernel-slug,...]` checks for a clean tree and a
    pushed HEAD, renders the runner script with the repo URL, SHA, and CLI command, writes
    `kernel-metadata.json` (`kernel_type: script`, `enable_gpu`, `enable_internet: true`, `kernel_sources` for
    the prepared-data kernel, `is_private: true`), and calls `kaggle kernels push`.
  - The runner clones the repo at the SHA, pip-installs the package with only the missing deps, runs the
    command, and writes everything into `/kaggle/working/<run_id>/`.
  - `scripts/kaggle_pull.py` polls `kaggle kernels status` until complete and then runs
    `kaggle kernels output <user>/<slug> -p runs/<run_id>/`.
  - Check `kaggle kernels push --help` for how to choose the accelerator (T4). If the CLI cannot select it, make
    it a [MANUAL] note: set "GPU T4 x2" in the kernel UI once. **Do not use P100**, since current PyTorch builds
    may not support its compute capability. Verify this in the smoke kernel.
  - Smoke kernel `env_check`: prints `nvidia-smi`, Python, torch, torchvision, CUDA, and installed ultralytics,
    and lists `/kaggle/input`. Run it and pull the output.
    _Accept:_ `docs/ENVIRONMENT.md` filled with real versions. Python compatibility target confirmed.

**Exit Phase 0:** CI green, smoke kernel output pulled, `ENVIRONMENT.md` complete, defense notes written, tag
`phase-0`.

### Phase 1 — Data (Day 1 afternoon → Day 2 morning)

- [ ] **P1.1** `data/download.py`: idempotent download to a target dir, primary and fallback URL, streaming with
      a progress log, MD5 verification (value read from torchvision), extraction.
      _Tests:_ MD5 verification with a tiny fake file; skip-if-exists logic.
- [ ] **P1.2** `data/parse_voc.py`: parse `Annotations/*.xml` into a canonical pandas DataFrame saved as parquet,
      with columns `image_id, width, height, class_name, class_id, x1, y1, x2, y2, difficult, truncated`. Apply the
      1-based→0-based conversion. Assert `0 ≤ x1 < x2 ≤ width` (same for y).
      _Tests:_ two hand-written fixture XMLs in `tests/fixtures/` covering the coordinate conversion, a difficult
      flag, and multiple objects.
- [ ] **P1.3** `constants.py`: `VOC_CLASSES` and `COCO_TO_VOC` mapping (airplane→aeroplane,
      motorcycle→motorbike, couch→sofa, dining table→diningtable, potted plant→pottedplant, tv→tvmonitor, plus the
      identical names). _Tests:_ 20 unique classes, mapping covers all 20.
- [ ] **P1.4** `data/splits.py`: build the train/val/test ID lists per §3 and write them to `splits/*.txt`.
      _Tests:_ disjoint sets, deterministic across two calls, sizes add up. Commit the split files with type `data`
      (they are small text). **Splits are frozen from here.** Add **ADR-002 Split protocol**.
- [ ] **P1.5** Exporters.
  - `export_yolo.py`: Ultralytics layout (`images/{train,val,test}` as symlinks, or copies on Kaggle if
    symlinks fail; `labels/*.txt` with `cls cx cy w h` normalized; difficult objects excluded) and a
    `voc_yolo.yaml` with `path`, `train`, `val`, `test`, `names`.
  - `export_coco.py`: COCO JSON per split, with categories 1..20 mapped from VOC idx, `area`, and
    `iscrowd = difficult`.
    _Tests:_ round-trip (canonical → YOLO → back) within 1e-4. COCO JSON validates with `pycocotools.COCO`.
- [ ] **P1.6** `data/dataset.py` + `transforms.py`: torch `Dataset` for Faster R-CNN returning
      `(image_tensor, {"boxes", "labels" (+1 offset), "image_id"})` with difficult excluded. Horizontal flip
      implemented manually (flip the image and set `x1' = W - x2, x2' = W - x1`). Provide a `collate_fn`.
      _Tests:_ flip twice equals identity, boxes stay valid, label offset is correct.
- [ ] **P1.7** `data/integrity.py`: the counts from §3 match, every image file exists, there are no zero-area
      boxes, every image has ≥1 non-difficult object (log and handle any that don't), and class distribution per
      split is printed. A mismatch stops with an error (hard rule 8).
- [ ] **P1.8** `data/eda.py` producing `reports/figures/eda_*.png` and `reports/EDA.md`: instances per class
      per split, objects per image histogram, size-bucket distribution per class, box aspect ratios, image
      resolution distribution, difficult/truncated share, and a grid of 12 annotated samples. Write 5–8 bullet
      findings that affect modelling (for example: person dominates, so expect class imbalance; many small bottles
      and plants, so expect low AP_small).
- [ ] **P1.9** CPU Kaggle kernel `vocdet-prepare-data`: runs download → parse → integrity → export (YOLO + COCO)
      → EDA, and writes `/kaggle/working/voc_prepared/` (images, labels, yaml, coco jsons, parquet, figures). Pull
      only the small outputs (parquet, jsons, figures, EDA.md) locally. All training kernels attach this kernel via
      `kernel_sources`. The first GPU kernel must log the actual mount path under `/kaggle/input/` and put it into
      `configs/data/voc2012.yaml`.

**Exit Phase 1:** prepared data available on Kaggle, EDA committed, splits frozen, defense notes, tag `phase-1`.

### Phase 2 — Evaluation harness, built before any training (Day 2)

- [ ] **P2.1** `eval/schema.py`: `Prediction` / `GroundTruth` dataclasses plus JSON read/write. Boxes are
      absolute 0-based xyxy floats and labels are 0–19.
- [ ] **P2.2** `eval/evaluator.py`: a wrapper over torchmetrics `MeanAveragePrecision(box_format="xyxy",
iou_type="bbox", class_metrics=True)` that passes `iscrowd` from the difficult flag (verify the installed
      version accepts `iscrowd` in targets). Returns a flat dict: `map50, map50_95, map_small, map_medium,
map_large, mar_100, ap50_per_class{...}`. CLI: `vocdet eval --pred <json> --split val`.
      _Tests (synthetic):_ perfect predictions → map50 = 1.0; no predictions → 0; a detection on a difficult GT
      leaves the score unchanged (ignore region); swapping a label lowers that class's AP.
- [ ] **P2.3** `eval/error_analysis.py` (TIDE-lite from §4), own implementation with plain loops, IoU function
      written by hand (vectorized numpy allowed). Outputs counts per error type overall and per class, plus a
      20×20 (+background) confusion matrix.
      _Tests:_ one hand-built case for each error type.
- [ ] **P2.4** `eval/visualize.py`: draw boxes with class and score. `failure_gallery()` saves the top-k
      highest-confidence FPs and the largest missed GTs per class.
- [ ] **P2.5** `utils/run_meta.py`, `utils/seed.py`, and `scripts/collect_results.py` implementing the run
      artifact contract from §2. _Tests:_ `collect_results` on a fake run directory.
- [ ] **P2.6** Experiment **E0b**: zero-shot COCO YOLO26n on `val` with the COCO→VOC mapping (drop non-VOC
      classes). Run on Kaggle (GPU, ~10 min), pull it, and collect it. This validates the whole
      predict → eval → collect chain before any training.

**Exit Phase 2:** evaluator tested, E0b in `results/experiments.csv`, defense notes, tag `phase-2`.

### Phase 3 — Faster R-CNN baseline (Day 2 evening → Day 3)

- [ ] **P3.1** `models/frcnn.py`: `build_frcnn(num_classes=21, pretrained=True)` loading torchvision COCO
      weights and replacing `roi_heads.box_predictor` with a `FastRCNNPredictor(in_features, 21)`. Set
      `min_size`/`max_size` from config (600/1000). Log parameter counts.
- [ ] **P3.2** `train/train_frcnn.py`: explicit training loop with SGD (momentum 0.9, wd 1e-4), linear warm-up
      over the first 500 iterations, `MultiStepLR` at epochs [8, 11], AMP via `torch.amp`, gradient-norm logging,
      per-iteration loss CSV (total and the four component losses), val mAP every 2 epochs plus the last epoch,
      `best.pt` saved by val map50, `last.pt` saved every epoch, and **resume** from `last.pt` (Kaggle sessions can
      die). Seed everything and set `num_workers` from config.
- [ ] **P3.3** `eval/predict.py::predict_frcnn` → `Prediction` (labels −1, score threshold 0.001, top 100).
      _Test:_ label offset round trip using a fake model output.
- [ ] **P3.4** E0a-frcnn smoke run on Kaggle: 64 train images, 1 epoch, val on 64 images. Record s/iter and
      update the §5 estimate for E1.
- [ ] **P3.5** E1 full run on Kaggle → pull → collect → run error analysis on val.
- [ ] **P3.6** Commit the E1 config and results (`exp(frcnn): E1 baseline results`), and log observations in
      `reports/EXPERIMENTS.md` (generated tables plus a hand-written interpretation paragraph that contains no typed
      numbers; reference the table instead).

**Exit Phase 3:** E1 collected, defense notes (the training loop is a top interview topic), tag `phase-3`.

### Phase 4 — YOLO26 experiments (Day 3 → Day 4)

- [ ] **P4.1** `train/train_yolo.py`: maps our YAML config onto Ultralytics `train()` args (`data, epochs, imgsz,
batch, seed=42, deterministic=True, patience, freeze, optimizer` left as the model default, augmentation
      keys, `project/name` inside the run dir). Saves the Ultralytics `results.csv` as `train_log.csv`.
- [ ] **P4.2** `predict_yolo` adapter → `Prediction` (conf 0.001, max_det 100). Also record Ultralytics' own
      `model.val()` mAP50 on val as a **cross-check**. Explain any difference (interpolation, difficult handling) in
      an ADR. Our evaluator is authoritative.
- [ ] **P4.3** E0a-yolo smoke run: 64 images, 1 epoch. Update the §5 estimates.
- [ ] **P4.4** E2 (YOLO26n) and E3 (YOLO26s). Launch E2. While it runs, work on P5.1 tooling. Then launch E3.
      Only one GPU kernel runs at a time because the quota is shared.
- [ ] **P4.5** E4 augmentation ablation and E5 frozen backbone (configs per §5).
- [ ] **P4.6** `make results` and regenerate the EXPERIMENTS tables. Add **ADR-003 Model selection** choosing the
      deployment candidate by val map50 versus CPU cost (the final latency check happens in Phase 6).

**Exit Phase 4:** E2–E5 collected, defense notes, tag `phase-4`.

### Phase 5 — Comparison, error analysis, final test evaluation (Day 4)

- [ ] **P5.1** `scripts/build_report_tables.py` generates markdown tables from `results/` (experiment
      comparison, per-class AP, size-bucket AP, error-type breakdown) plus figures (val mAP-vs-epoch curves,
      speed/accuracy scatter, per-class AP bars, confusion matrix heatmap).
- [ ] **P5.2** `scripts/final_test_eval.py`: evaluates final models E1, E2, E3, and the selected configuration
      on the **test** split exactly once, writes `results/test_metrics.csv`, and refuses to overwrite an existing
      entry without `--force-rerun-with-reason "<text>"` (the reason is logged).
- [ ] **P5.3** `reports/ERROR_ANALYSIS.md`: TIDE-lite tables for the best two models, failure galleries,
      confusions (expected: cat↔dog, cow↔sheep↔horse, chair/sofa/diningtable), small-object misses, and the link to
      EDA findings. Each finding is paired with a concrete improvement hypothesis.

**Exit Phase 5:** test metrics recorded once, analysis written, tag `phase-5`.

### Phase 6 — Optimization (Day 5)

- [ ] **P6.1** `optimize/export.py` for the selected YOLO model: ONNX FP32 (opset per Ultralytics default,
      static 640 input), OpenVINO FP32, and OpenVINO **INT8** (calibration on 300 **train** images, never
      val/test). Optional: TensorRT FP16 run on Kaggle GPU. Also export E1 (Faster R-CNN) to ONNX **only if** it
      works within 1 hour of effort; otherwise document why two-stage export is harder (dynamic proposals) as a
      finding.
- [ ] **P6.2** `inference/detector.py`: a `Detector` interface with `PyTorchBackend`, `OnnxBackend`, and
      `OpenVinoBackend`. Same preprocessing (letterbox to imgsz) and the same postprocessing to `Prediction`.
      _Tests:_ on a fixture image, all backends agree within tolerance (boxes IoU > 0.95 for matched detections) on
      a tiny exported model created in the test.
- [ ] **P6.3** `optimize/benchmark.py` per §4 (200 fixed val images, warm-up 20, batch 1, thread count fixed
      via env and config, p50/p95, FPS, size, peak RSS via `psutil`). Run it **inside the app Docker image limited
      to `--cpus=2 --memory=8g`** to emulate the Space, and also on the local machine. Record hardware info.
      Output: `results/benchmarks.csv`.
- [ ] **P6.4** Accuracy of every exported variant on `val` through the evaluator, to measure the accuracy cost
      of each optimization.
- [ ] **P6.5** Resolution sweep for the deployment model (imgsz 640 / 512 / 416 / 320, inference only):
      accuracy and latency.
- [ ] **P6.6** `reports/OPTIMIZATION.md` with a before/after table (PyTorch FP32 CPU is the "before") and
      **ADR-004 Deployment variant** using an explicit rule: _maximize val map50 subject to p50 latency ≤ target on
      2 vCPU_. Set the target from data after seeing the numbers, and justify it (e.g. interactive use).

**Exit Phase 6:** benchmarks plus ADR-004, tag `phase-6`.

### Phase 7 — Application and deployment (Day 6)

- [ ] **P7.1** `app/main.py` (FastAPI):
  - `GET /health` → status, model variant, version, git sha.
  - `GET /model-info` → classes, backend, imgsz, operating threshold, val metrics of the deployed model.
  - `POST /predict` (multipart image; query `conf`, `classes` filter, `backend`) → JSON with detections
    `{label, score, box_xyxy}`, `counts_per_class`, `inference_ms`, `image_size`.
  - `POST /predict/annotated` → PNG with boxes drawn.
  - Validation: accept only JPEG/PNG/WebP, max 10 MB, decode errors → 400, wrong type → 415, too big → 413,
    unknown class name → 422. Load the model once at startup (lifespan), and use dependency injection so tests
    can swap in a fake detector.
- [ ] **P7.2** `app/ui.py`: Gradio Blocks mounted at `/ui` (and redirect `/` → `/ui`). It has image upload,
      bundled example images (from VOC test — check the license note and use a few), a confidence slider, a class
      filter, and a **backend toggle FP32 vs INT8** so reviewers can see the optimization live. Outputs: annotated
      image, detections table, counts, latency.
- [ ] **P7.3** Tests with FastAPI `TestClient` and a fake detector: happy path, each error code, class filter,
      and the counts aggregation. CI never needs real weights.
- [ ] **P7.4** `app/Dockerfile`: python slim, non-root user, only the runtime deps (no training deps), port
      7860, weights downloaded from the HF Hub **at a pinned revision** at build time. A healthcheck is defined.
      `make docker-build && make docker-run` works locally.
- [ ] **P7.5 [MANUAL]** On Hugging Face: create the model repo `<user>/vocdet-models` (public) and the Space
      `<user>/vocdet-demo` (SDK: **Docker**, hardware: free CPU basic). Confirm `HF_TOKEN` has write access to both.
- [ ] **P7.6** `scripts/upload_weights.py`: uploads the deployed weights plus exported variants and a model card
      (training data, metrics from `results/`, limitations, license) to the model repo, and records the commit
      revision in `configs/deploy.yaml`.
- [ ] **P7.7** `scripts/deploy_space.py`: pushes `app/` + package + Space README (YAML front matter: `sdk:
docker`, `app_port: 7860`) to the Space via `huggingface_hub`. Then `scripts/smoke_test_live.py <url>` checks
      `/health` and `/predict` on a sample image and asserts sane output.
- [ ] **P7.8** CLI `vocdet predict --images <dir> --out <dir> [--backend openvino-int8]` for batch use on new
      images.
- [ ] **P7.9** `docs/DEPLOYMENT.md`: architecture diagram (Mermaid), cold start and sleep behaviour of free
      Spaces (verify the current free-tier specs and sleep policy on the HF docs and cite them), how secrets are
      handled (none needed at runtime: the model repo is public), and how to roll back (pinned revision).

**Exit Phase 7:** live URL passes the smoke test, tag `phase-7`.

### Phase 8 — Reports and final polish (Day 7)

- [ ] **P8.1** `reports/TECHNICAL_REPORT.md` (≤4 pages equivalent) answering the brief's six questions in
      order: problem; why this data, model, and approach; experiments and results (generated tables); challenges
      and fixes (pull from commit history and ADRs); strengths and weaknesses; future work (e.g. VOC07+12 training,
      multi-seed runs, RT-DETR comparison, QAT instead of PTQ, TensorRT).
- [ ] **P8.2** `README.md` final: overview, live demo link, results headline table (generated), quick start,
      **reproduce** section (exact commands per experiment), API usage with curl examples, project structure,
      dependency list, license notes (AGPL for Ultralytics, VOC data terms).
- [ ] **P8.3 [MANUAL + agent]** `reports/AI_USAGE.md`: the agent writes the skeleton (tools used: Claude Code
      for implementation under `CLAUDE.md`/`PLAN.md`; how verification worked: tests, CI, cross-checks against
      Ultralytics val, manual review of diffs). **The owner writes the honest specifics**: what he wrote or changed
      himself, what he reviewed, and what he rejected.
- [ ] **P8.4** Requirements traceability: verify every row of the §0 table points to an existing, non-empty
      artifact. Fix any gaps.
- [ ] **P8.5** `docs/defense/README.md`: an index of all phase notes plus a one-page cheat sheet of the
      pipeline: data → train → eval → optimize → serve, with key file:function references.
- [ ] **P8.6** Final: `make check`, CI green, live smoke test passes, merge, tag `v1.0.0`, and create a GitHub
      release with the results table.

---

## 7. Day-by-day schedule

| Day | Agent work                  | GPU running in background             | Human                                            |
| --- | --------------------------- | ------------------------------------- | ------------------------------------------------ |
| 1   | P0, P1.1–P1.8               | env_check, prepare-data (CPU)         | P0.1, P0.5, read defense notes                   |
| 2   | P1.9 wrap-up, P2, P3.1–P3.3 | E0b, E0a-frcnn, then **E1** overnight | review the evaluator and FRCNN loop line by line |
| 3   | P3.5–P3.6, P4.1–P4.3        | E0a-yolo, **E2**, then **E3**         | phase 3 defense exercises                        |
| 4   | P4.5–P4.6, P5               | **E4**, **E5**                        | review error analysis                            |
| 5   | P6                          | optional TensorRT bench               | review optimization numbers                      |
| 6   | P7                          | —                                     | P7.5, test the live app                          |
| 7   | P8                          | —                                     | P8.3, rehearse the defense                       |

Training jobs run on Kaggle while the agent writes the next phase's code. Keep that pipeline full.

---

## 8. Manual steps summary (human)

1. P0.1: GitHub repo (public).
2. P0.5: Kaggle phone verification, API token, quota check; HF write token; optional W&B.
3. P0.6: possibly set accelerator "GPU T4 x2" once in the Kaggle UI if the CLI can't.
4. P7.5: HF model repo and Docker Space.
5. P8.3: honest AI-usage specifics.
6. Every phase: read the diff and `docs/defense/phase-n.md`, and do the two exercises yourself without AI.

---

## 9. Risk register

| Risk                                     | Mitigation                                                                     |
| ---------------------------------------- | ------------------------------------------------------------------------------ |
| Official VOC host down                   | Fallback mirror plus MD5 check. Kaggle prep kernel output is cached and reused |
| Kaggle session dies / 12 h limit         | Resume from `last.pt`. Keep runs ≤4 h                                          |
| GPU quota exhausted                      | Budget table, CPU kernels for prep/EDA, smoke-calibrated estimates, 5 h buffer |
| P100 incompatible with current torch     | Use T4. Verified in the env_check kernel                                       |
| Ultralytics API changes between versions | Pin the version. Verify signatures against the installed package               |
| INT8 accuracy drop too large             | Report it honestly. Fall back to FP32 OpenVINO. Mention QAT as future work     |
| Free Space sleeps / slow cold start      | Documented in DEPLOYMENT.md. Smoke test retries with backoff                   |
| Metric mismatch with Ultralytics val     | Explained in an ADR. Our evaluator is authoritative                            |
| Interview without AI                     | Defense notes and exercises every phase. Explicit code in core paths           |
