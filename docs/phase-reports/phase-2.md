# Phase 2 report: Evaluation harness

Branch `phase-2-eval`. Exit criteria:
- Evaluator tested ✔ (synthetic known-answer tests, plus a real-val sanity check: ground truth as
  predictions gives 1.0)
- E0b in `results/experiments.csv` ✔
- Defense notes written ✔ (`docs/defense/phase-2.md`)
- Tag `phase-2`: pending your approval

CI is green on every push.

## Done

| Task | Summary | Commits |
|---|---|---|
| Gate notes | Local download logged (`data/` = 3.9 GB); your P5.1 bootstrap CIs and P4.1 symlink/copy decisions added to PLAN.md | `60901fb` |
| P2.1 | `Prediction` / `GroundTruth` schema, validated, JSON round trip | `1786a25` |
| P2.2 | Shared evaluator (torchmetrics, `iscrowd` = difficult, IoU-0.5 per-class instance), `vocdet eval`, ADR-008 | `de98c44` |
| P2.3 | TIDE-lite error analysis, hand-written IoU, 21×21 confusion matrix, `vocdet errors`, ADR-009 | `9eec9c1` |
| P2.4 | Box drawing and per-class failure galleries | `979923e` |
| P2.5 | `make_run_id` refactor; run metadata, seeding, `collect_results.py`, `make results`. Fix: LF line endings in CSV | `6b45b61`, `f0b8038`, `ed44646` |
| P2.6 | YOLO adapter, zero-shot run, torch guard in the runner, ADR-010. Fixes: data-location diagnosis (ADR-011), lossless predictions JSON. E0b result | `5978b2b`, `34ff675`, `3fc1694`, `587f5e9`, `9f350f7`, `fa7911f`, `a36a737` |

## Deviations

From `docs/DEVIATIONS.md`:

1. **P1.4/P1.9:** the full VOC download and pipeline also ran locally (logged at your request).
2. **P5.1:** bootstrap 95% CIs on val mAP50 added (your decision).
3. **P4.1:** writable YOLO root on Kaggle, with symlinked images and copied labels (your decision).
4. **P2.6:** the torch/torchvision guard was implemented before P4.1, and `eval/zero_shot.py` was added.

Also changed within this phase, recorded in ADRs rather than DEVIATIONS:
- Lock resolution is Linux-only (ADR-010).
- Prepared data is found by recursive search on the prep run ID (ADR-011, your instructions).
- `--resolve-only` and `--append-args` were added for CPU path checks.

## Results

From `results/runs/E0b_20261028-0321_fa7911f/metrics.json` (COCO-pretrained YOLO26n, zero-shot, val,
572 images):

- `"map50": 0.8274052143096924`, `"map50_95": 0.6493940949440002`
- `"map_small": 0.29227375984191895`, `"map_medium": 0.4934343993663788`,
  `"map_large": 0.7190988063812256`, `"mar_100": 0.7648680210113525`
- Lowest AP50: `"pottedplant": 0.4072273373603821`, `"boat": 0.6806280016899109`,
  `"diningtable": 0.6884810328483582`
- Highest AP50: `"cow": 1.0`, `"train": 0.9806529879570007`, `"aeroplane": 0.9516276717185974`

Interpretation (no new numbers): COCO pretraining alone already gives a strong VOC baseline, because all
20 VOC classes are COCO classes. This is the reference that fine-tuning (E1–E5) must beat. The weakest
classes are the ones whose VOC labelling convention differs most from COCO, or which are mostly small
(pottedplant). A perfect AP50 for cow comes from a class with few val objects (see the EDA tables), which
is the kind of small-sample effect the bootstrap CIs (P5.1) are meant to show.

Checks: re-evaluating the pulled `predictions_val.json` locally gives the in-run metrics exactly. Two GPU
runs of E0b gave identical metrics. Tests: 128 passed.

## GPU budget

| | Hours |
|---|---|
| Used this phase: 4 GPU sessions charged 0.05 h each (2 failed E0b, 1 T4 `env_check`, 1 superseded E0b), plus the collected E0b; CPU checks free | 0.25 |
| Cumulative (`docs/ENVIRONMENT.md` usage log) | 0.30 |
| Project budget (PLAN.md §5) | ~15.5 |
| Remaining against budget | ~15.2 |

The §5 estimate for E0b was 0.2 h. With the failures and the rerun it took 0.25 h.

## Manual verification checklist

1. **The evaluator gives the known answers.**
   - Run `uv run pytest tests/test_evaluator.py -v`. Expected: 7 passed.
   - Then explain why `test_detection_on_difficult_object_is_ignored` needs the control test below it.
2. **Saved predictions reproduce the reported metrics.**
   - Run `uv run vocdet eval --pred runs/E0b_20261028-0321_fa7911f/predictions_val.json --split val | head -3`.
   - Expected: `"map50": 0.8274052143096924`, matching `results/experiments.csv`.
   - A mismatch would mean the file and the reported metrics have diverged again.
3. **Each error type is classified correctly.**
   - Run `uv run pytest tests/test_error_analysis.py -v`.
   - Then read the comments in `DETECTIONS` and check one IoU by hand, e.g. the Loc case:
     `[200,0,240,100]` versus `[200,0,300,100]` gives 0.4.
4. **The torch guard held on Kaggle.**
   - Run `python3 -c "import json; k=json.load(open('runs/E0b_20261028-0321_fa7911f/kaggle_run.json')); print(k['guarded_versions_before'], k['guarded_versions_after'])"`.
   - Expected: both `{'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128'}`.
5. **The data was found at the deep mount layout.**
   - Run `grep "Prepared data found" runs/E0b_20261028-0321_fa7911f/command.log`.
   - Expected: a path under `/kaggle/input/notebooks/nikaatanelishvili/vocdet-prepare-data/prepare_data_20261028-0208_7b7605e/`.
6. **Run artifacts are complete.**
   - Run `ls results/runs/E0b_20261028-0321_fa7911f/`. Expected: `config.yaml metrics.json run_meta.json`.
   - In `run_meta.json`, check that `"git_sha"` starts with `fa7911f` and `"gpu_name"` is `"Tesla T4"`.
7. **Error analysis works on real predictions.**
   - Run `uv run vocdet errors --pred runs/E0b_20261028-0321_fa7911f/predictions_val.json --out-dir /tmp/e0b_err`.
   - Then run `python3 -c "import json; print(json.load(open('/tmp/e0b_err/errors.json'))['overall'])"`.
   - Expected: a dict with TP, Cls, Loc, Both, Dupe, Bkg, Ignored, and Miss counts, with TP the largest.
8. **CI is green.**
   - Run `gh run list --branch phase-2-eval --limit 3`. Expected: all `completed success`.

## Open issues / risks

1. **Kaggle mounts inputs at different depths in different kernels.** This is handled by searching on the
   run ID, plus a CPU `--resolve-only` check before each new GPU kernel. Every new kernel slug (E1, E2, …)
   should get that check on its first launch, at no GPU cost.
2. **YOLO26 inference runs NMS.** In ultralytics 8.4.191, `yolo26n.pt` loads with `end2end=False` (ADR-010).
   The "NMS-free" claim in PLAN.md §1 is unverified; Phases 4 and 6 must check it before the reports repeat
   it.
3. **Val is small.** Single-class APs such as cow = 1.0 rest on few objects. The bootstrap CIs (P5.1)
   address this for mAP50.
4. **The error analysis threshold is not chosen yet.** The default is 0.5. Reports must use each model's
   F1-optimal threshold on val; that selection helper doesn't exist yet and is planned for P5.3.
5. **COCO crowd semantics.** pycocotools ignores detections that mostly lie inside difficult boxes (IoU over
   detection area). This is slightly more lenient than VOC, applies equally to all models, and is
   documented in ADR-008.

## Next phase (Phase 3: Faster R-CNN baseline)

- **P3.1** `models/frcnn.py`: `fasterrcnn_mobilenet_v3_large_fpn` with COCO weights, a new 21-class
  predictor, and min/max size 600/1000 from the config.
- **P3.2** `train/train_frcnn.py`: an explicit training loop (SGD, warm-up, MultiStepLR, AMP, gradient-norm
  logging, val mAP every 2 epochs, best/last checkpoints, resume).
- **P3.3** `predict_frcnn`: the label −1 offset and the 0.001 / top-100 settings, with a round-trip test.
- No [MANUAL] steps. P3.4 (E0a-frcnn smoke, about 0.3 h GPU) runs after a CPU `--resolve-only` check of the
  new kernel. P3.5 (E1, about 2.5 h) is the first long run.
