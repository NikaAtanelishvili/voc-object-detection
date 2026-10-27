# Deviations from PLAN.md

Every departure from `PLAN.md` is logged here at the moment it happens (CLAUDE.md work loop).

## Entry template

```
### YYYY-MM-DD — P<x>.<y> — <short title>
- **Planned:** what PLAN.md says
- **Actual:** what was done instead
- **Reason:** why
- **Impact:** effect on scope, schedule, results, or other tasks
- **ADR:** link, or "none"
```

## Entries

### 2026-10-28 — P0.1 — Repo created by the agent, not the human
- **Planned:** [MANUAL] the human creates the GitHub repo, clones it, copies `CLAUDE.md`/`PLAN.md` in.
- **Actual:** the agent ran `git init` in the existing local folder and `gh repo create voc-object-detection --public`,
  then pushed. No clone step because the folder already held the plan files.
- **Reason:** the owner explicitly asked the agent to do it (`gh` is installed and authenticated).
- **Impact:** none on deliverables. Same public repo, same first commit.
- **ADR:** none

### 2026-10-28 — P0.1 — Minimal `.gitignore` committed before P0.2
- **Planned:** `.gitignore` is created in P0.2.
- **Actual:** a one-line `.gitignore` (`.env`) is part of the first commit; P0.2 extends it.
- **Reason:** a real `.env` with tokens already exists in the folder; it must be ignored before the first `git add`.
- **Impact:** none.
- **ADR:** none

### 2026-10-28 — P0.6 — Extra folders `configs/kaggle/` and `results/env/`
- **Planned:** `PLAN.md` §2 lists `configs/data/` and `configs/experiments/` only; smoke-kernel output only
  feeds `docs/ENVIRONMENT.md`.
- **Actual:** non-experiment kernel configs (starting with `env_check.yaml`) live in `configs/kaggle/`. The
  smoke kernel's `env.json` and `kaggle_run.json` (≈3 KB total) are committed under `results/env/`.
- **Reason:** `kaggle_run.py` needs a validated config for every kernel, and `env_check` is not an
  experiment. Committing the raw JSON means every fact in `ENVIRONMENT.md` has a checked-in source
  (hard rule 3 spirit), since `runs/` is gitignored.
- **Impact:** none on experiments; two small extra folders.
- **ADR:** none

### 2026-10-28 — P0.6 — `src/vocdet/config.py` started early
- **Planned:** `config.py` is in the layout but no task creates it before experiment configs exist.
- **Actual:** created in P0.6 with `JobConfig`/`KaggleSettings` only (`extra="allow"` at the top level until
  experiment sections get their own models).
- **Reason:** CLAUDE.md requires YAML configs validated by pydantic, and `kaggle_run.py --config` is the
  first consumer.
- **Impact:** later phases extend this module instead of creating it.
- **ADR:** none

### 2026-10-28 — P0.3 — CI matrix on Python 3.11 and 3.13
- **Planned:** CI on Python 3.11.
- **Actual:** CI runs `make check` on both 3.11 and 3.13 (`fail-fast: false`).
- **Reason:** the `env_check` kernel showed that Kaggle runs Python 3.13.15 (`docs/ENVIRONMENT.md`), so code
  that runs on Kaggle must be tested on 3.13. The owner approved this at the Phase 0 gate.
- **Impact:** about 15 s more CI time per push. Code must pass on both versions.
- **ADR:** none

### 2026-10-28 — P0.3 — CI runner pinned to `ubuntu-24.04`
- **Planned:** "Ubuntu" (implemented as `ubuntu-latest`).
- **Actual:** `runs-on: ubuntu-24.04`.
- **Reason:** GitHub warned that `ubuntu-latest` is migrating to Ubuntu 26. Pinning avoids a surprise image
  change mid-project. The owner approved this at the Phase 0 gate.
- **Impact:** none functionally. Revisit after the project ends.
- **ADR:** none

### 2026-10-28 — P7.4 — App Docker image on Python 3.13
- **Planned:** "python slim" (no version given).
- **Actual:** the Dockerfile will use a Python **3.13** slim base. PLAN.md P7.4 is updated to match.
- **Reason:** Kaggle trains on 3.13, so serving on the same minor version avoids version skew between
  training and serving. The owner approved this at the Phase 0 gate. ultralytics, onnxruntime, openvino,
  and onnx all ship 3.13-compatible Linux wheels (`docs/ENVIRONMENT.md`).
- **Impact:** CI already tests 3.13. No other change.
- **ADR:** none (ADR-004 will reference it)

### 2026-10-28 — P4.1 — Added acceptance criterion: Kaggle torch must stay untouched
- **Planned:** P4.1 maps configs onto Ultralytics `train()`; it said nothing about install side effects.
- **Actual:** P4.1 now also requires that installing the pinned ultralytics on Kaggle leaves the
  preinstalled torch/torchvision unchanged. The runner records their versions before and after the
  install and fails the run if they differ. PLAN.md P4.1 is updated to match.
- **Reason:** owner request. A pip resolver upgrade of torch would break the CUDA build and waste GPU
  quota. ultralytics 8.4.191 only requires `torch>=1.8.0`, so no change is expected, but it must be
  proven, not assumed.
- **Impact:** a small change to `kaggle/templates/runner.py.j2` plus a test, done in P4.1.
- **ADR:** none

### 2026-10-28 — P1.1 — Python floor raised from 3.10 to 3.11
- **Planned:** Python ≥3.10 (PLAN.md §1, CLAUDE.md code style, ADR-001).
- **Actual:** `requires-python = ">=3.11"`, ruff target `py311`. PLAN.md §1 and CLAUDE.md are updated to match.
- **Reason:** Kaggle runs 3.13, CI tests 3.11 and 3.13, and kaggle, pre-commit, and onnxruntime all need ≥3.11.
  Owner decision.
- **Impact:** removes the version markers from the dev dependencies. Python 3.10 is no longer supported.
- **ADR:** [ADR-005](DECISIONS.md#adr-005-python-311-floor)

### 2026-10-28 — P1.1 — Fallback mirror is pjreddie.com, not the Ultralytics assets mirror
- **Planned:** fallback = Ultralytics assets mirror, verified with torchvision's MD5 (PLAN.md §3).
- **Actual:** fallback = `https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar`, verified with the
  same torchvision MD5.
- **Reason:** the Ultralytics mirror serves a `.zip` (1,950,180,009 bytes), not the official `.tar`
  (1,999,639,040 bytes), so the MD5 check in the plan cannot apply to it. pjreddie.com serves a tar of the
  identical size. Reported under hard rule 8; the owner chose this option.
- **Impact:** none on data, provided the MD5 matches. pjreddie.com is a personal site and the MD5 is the
  only integrity guarantee (ADR-006).
- **ADR:** [ADR-006](DECISIONS.md#adr-006-voc2012-download-source-and-integrity-check)

### 2026-10-28 — P1.2 — `VOC_CLASSES` created before P1.3; only trainval XMLs parsed
- **Planned:** `constants.py` belongs to P1.3. P1.2 parses `Annotations/*.xml`.
- **Actual:** P1.2 creates `constants.py` with `VOC_CLASSES` only, because `class_id` needs it. P1.3 adds
  `COCO_TO_VOC` and its tests. P1.2 parses the images listed in `ImageSets/Main/trainval.txt`, not every
  XML in `Annotations/`.
- **Reason:** the parser cannot assign `class_id` without the class list. `Annotations/` also contains
  action- and layout-only images that are not part of the detection task, and the expected counts in §3
  refer to the Main lists.
- **Impact:** none on later tasks. P1.7 still checks the counts against §3.
- **ADR:** [ADR-007](DECISIONS.md#adr-007-canonical-annotation-table-and-coordinate-convention)

### 2026-10-28 — P1.8 — Sample grid saved as JPEG
- **Planned:** `reports/figures/eda_*.png`.
- **Actual:** the 12-image sample grid is `eda_samples.jpg` (quality 80). The other six figures are PNG.
- **Reason:** as PNG the photo grid is 2.67 MB, over the 500 KB figure limit (hard rule 2) and the
  1 MB pre-commit guard. As JPEG it is about 360 KB.
- **Impact:** none.
- **ADR:** none

### 2026-10-28 — P1.9 — Prep output under the run folder; `prepare.py` and `pull_pattern` added
- **Planned:** the kernel writes `/kaggle/working/voc_prepared/`. The layout (§2) has no `prepare.py`.
- **Actual:**
  - Output goes to `/kaggle/working/<run_id>/voc_prepared/`, so training kernels mount it at
    `/kaggle/input/vocdet-prepare-data/<run_id>/voc_prepared` (to be verified by the first GPU kernel and
    recorded in `configs/data/voc2012.yaml`).
  - The pipeline is one function, `src/vocdet/data/prepare.py::prepare_all`, exposed as `vocdet prepare`.
  - `KaggleSettings.pull_pattern` (a regex) limits what `kaggle_pull.py` downloads. `kaggle_run.py` saves
    the config into the kernel folder so the pull can read it.
- **Reason:** keeping every output under `<run_id>/` keeps the runner contract from P0.6, where
  `kaggle_pull.py` uses `<run_id>/kaggle_run.json` to tell runs apart. The run ID in the mount path also
  pins which data version a training run used. Without a pull filter, the pull would download about 2 GB
  of copied images.
- **Impact:** training configs reference a path containing the prep run ID. A new prep run means updating
  `configs/data/voc2012.yaml`.
- **ADR:** none

### 2026-10-28 — P1.4/P1.9 — Full VOC2012 download and pipeline run locally
- **Planned:** data preparation runs on a Kaggle CPU kernel (P1.9). Local work uses small fixtures.
- **Actual:** the 2 GB trainval tar was also downloaded locally with `vocdet download` (MD5 verified from
  the Oxford host). The full pipeline was then run locally: parse, splits, integrity, export (symlink
  mode), EDA, and `vocdet prepare` into the scratchpad. Local disk use on 2026-10-28: `data/` = 3.9 GB
  (`downloads/` 1.9 GB tar, `VOCdevkit/` 2.0 GB extracted, `prepared/` 96 MB, `processed/` 336 KB), plus
  97 MB of symlinks and labels in the session scratchpad.
- **Reason:** P1.4 needs the official `ImageSets/Main` lists, which exist only inside the tar. Running
  locally also caught issues before spending a Kaggle session: it validated every real box, revealed the
  2.67 MB sample figure, and gave a reference to compare the Kaggle outputs against byte for byte.
- **Impact:** about 4 GB of local disk; `data/` is gitignored. It can be deleted with `rm -rf data/` and
  rebuilt with `vocdet download` + `vocdet prepare --mode symlink`. The committed EDA outputs come from
  the local run, and the Kaggle run reproduced them byte for byte (`docs/ENVIRONMENT.md`). The owner was
  not told before the download; this entry was added at the owner's request at the Phase 1 gate.
- **ADR:** none

### 2026-10-28 — P5.1 — Bootstrap confidence intervals added (owner decision)
- **Planned:** the P5.1 comparison tables report point estimates only.
- **Actual:** P5.1 also reports a bootstrap 95% CI on val mAP50 (images resampled with replacement,
  ~200 iterations, fixed seed). PLAN.md P5.1 is updated.
- **Reason:** val has only 572 images, so differences between runs need an uncertainty estimate before
  they are interpreted. Owner request at the Phase 1 gate.
- **Impact:** extra CPU time in P5.1 (about 200 evaluator calls per run) and one more tested module. Runs
  must keep `predictions_val.json`, which the run artifact contract already requires.
- **ADR:** none (P5.1 may add one if the resampling scheme needs a decision)

### 2026-10-28 — P4.1 — Writable YOLO dataset root on Kaggle: symlinked images, copied labels (owner decision)
- **Planned:** P4.1 says nothing about where Ultralytics reads the dataset on Kaggle.
- **Actual:** P4.1 now builds a writable dataset root in `/kaggle/working/`. It symlinks the images from
  the read-only `/kaggle/input` mount, copies only the label files, and writes a fresh `voc_yolo.yaml`.
  PLAN.md P4.1 is updated.
- **Reason:** resolves Phase 1 open issue 2. The prep kernel's yaml points at its own `/kaggle/working`,
  and Ultralytics writes label `.cache` files next to the labels, which fails on the read-only mount.
  Symlinks avoid copying 2 GB of images per run.
- **Impact:** a few seconds of setup per YOLO run. It must be verified that Ultralytics maps image paths to
  label paths without resolving the symlinks.
- **ADR:** none
