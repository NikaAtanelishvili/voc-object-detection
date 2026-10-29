# Phase 3 report: Faster R-CNN baseline

Branch `phase-3-frcnn`. Exit criteria:
- E1 collected ✔ (`results/runs/E1_20261029-0602_da4e489/`, a row in `results/experiments.csv`)
- Defense notes written ✔ (`docs/defense/phase-3.md`; the training loop is covered in Q2–Q6)
- Tag `phase-3`: pending your approval

`make check`: 152 passed, plus format and lint. CI is green on every push of the branch.

## Done

| Task | Summary | Commits |
|---|---|---|
| P3.1 | `build_frcnn`: MobileNetV3-L FPN with COCO weights, 21-class `FastRCNNPredictor`, min/max size from config. Fix: CPU torch wheel index restored | `aeed67b`, `0d08d6e` |
| P3.2 | Explicit training loop. Resume into any folder, with logs and `best.pt` carried over and the unfinished epoch dropped. Cross-session Kaggle resume (`--resume-from` / `--resume-run`). ADR-012 | `8a35924`, `757a696`, `cf101d5` |
| P3.3 | `predict_frcnn`: label −1, score ≥ 0.001, top 100, round-trip test | `228ed9d` |
| P3.4 | E0a smoke and timing runs; per-step `data_s`/`iter_s`; `median_iter_s` in experiments.csv; shape-replay time estimator; §5 E1 estimate updated; cross-session resume tested on Kaggle | `1970920`, `66d15f4`, `0256862`, `2dd6526`, `37dea90`, `da4e489` |
| P3.5 | E1 full run on a T4, pulled, collected, TIDE-lite on val; GPU log | `f63101b` |
| P3.6 | First table of `build_report_tables.py` (`make report`), E1 results, EXPERIMENTS.md interpretation. Fixes: empty marker block, trailing newline in `error_records.json` | `1ab08d2`, `3a4ed84`, `53efe34`, `bffeb4a` |

## Deviations

From `docs/DEVIATIONS.md`:

1. **P0.3 (found in P3.2):** the CPU torch index mapping was lost from `f2ded7a` until `0d08d6e`. No
   effect on results.
2. **P3.3 before P3.2**, and one explicit `lr_at` function in place of `MultiStepLR` + warm-up. The
   schedule is identical (ADR-012).
3. **P3.4:** the E1 config was committed early (Kaggle runs only pushed commits). A CPU cross-session
   resume check was added.
4. **P3.4:** an extra GPU timing run, because the 8-step smoke run could not be extrapolated. The
   estimate comes from a shape replay, not a median.
5. **P3.6:** the experiment table of `build_report_tables.py` was written before P5.1.

## Results

From `results/runs/E1_20261029-0602_da4e489/metrics.json` (Faster R-CNN MobileNetV3-L FPN, 12 epochs,
val, 572 images):

- `"map50": 0.79451584815979`, `"map50_95": 0.5476698279380798`
- `"map_small": 0.13050149381160736`, `"map_medium": 0.35376960039138794`,
  `"map_large": 0.6382659673690796`, `"mar_100": 0.6404849886894226`
- Lowest AP50: `"pottedplant": 0.34759706258773804`, `"diningtable": 0.5783917307853699`,
  `"bottle": 0.5975550413131714`, `"chair": 0.6086148619651794`
- Highest AP50: `"horse": 0.9603960514068604`, `"cow": 0.9471624493598938`,
  `"motorbike": 0.9447545409202576`

From `results/runs/E1_20261029-0602_da4e489/error_analysis/errors.json` (score ≥ 0.5): `"TP": 991,
"Cls": 45, "Loc": 78, "Both": 23, "Dupe": 11, "Bkg": 69, "Ignored": 36, "Miss": 341`.

From `results/estimates/E1.json`: `"train_hours": 0.733`, `"distinct_shapes": 105`,
`"new_shape_s": 7.4948`, `"repeat_s": 0.2428`, `"separation": 21.104522312069484`.

Comparison (generated table in `reports/EXPERIMENTS.md`): E0b, the zero-shot YOLO26n, has
`"map50": 0.8274052143096924` (`results/runs/E0b_20261028-0321_fa7911f/metrics.json`).

Interpretation (no new numbers):
- **E1 is below the zero-shot YOLO26n.** The comparison is not like for like: different architecture,
  and far more COCO data behind E0b. It sets the bar for Phase 4.
- **Size and errors.** Small objects are E1's weakest area. Misses are the dominant error type.
- **Training curve.** Val mAP was flat at the base LR and jumped after the epoch-8 LR drop.
- **Checks.**
  - Re-evaluating the pulled `predictions_val.json` locally reproduces `metrics_val.json` exactly.
  - Regenerating the error analysis reproduces the committed files byte for byte.
  - The shape-replay estimate was close to the measured training time. Sum `iter_s` in `train_log.csv`
    to compare; the estimate excludes validation.

## GPU budget

| | Hours |
|---|---|
| Used this phase: E0a (2 runs, 0.05 h each), E0a timing (0.10 h), E1 (0.85 h); CPU checks free | 1.05 |
| Cumulative (`docs/ENVIRONMENT.md` usage log) | 1.35 |
| Project budget (PLAN.md §5, E1 lowered from 2.5 to 1.0 h) | ~14.0 |
| Remaining against budget | ~12.65 |

The §5 estimates were E0a 0.3 h and E1 2.5 h. Actual use was 0.20 h for E0a including the timing run,
and 0.85 h for E1.

## Manual verification checklist

1. **The label offset round-trips.**
   - Run `uv run pytest tests/test_predict.py tests/test_dataset.py -v`. Expected: all pass.
   - Then read `eval/predict.py::predict_frcnn` and find the `- LABEL_OFFSET`.
   - A failure would mean every class is shifted by one.
2. **Resume carries the logs and drops the unfinished epoch.**
   - Run `uv run pytest tests/test_train_frcnn.py -v`. Expected: 8 passed.
   - Then read `test_end_to_end_training_and_resume`: the planted epoch-1 row must not survive.
3. **Cross-session resume worked on Kaggle.**
   - Run `grep -E "Found .*last.pt|Resumed from" runs/E0a_frcnn_20261029-0552_2dd6526/command.log`.
   - Expected: a path under `/kaggle/input/notebooks/nikaatanelishvili/vocdet-e0a-frcnn/E0a_frcnn_20261029-0548_0256862/weights/last.pt`, then `at epoch 1, step 8`.
4. **The saved E1 predictions reproduce the reported metrics.**
   - Run `uv run vocdet eval --pred runs/E1_20261029-0602_da4e489/predictions_val.json --split val --annotations data/voc_prepared/annotations.parquet | head -3`.
   - Expected: `"map50": 0.79451584815979`.
5. **The time estimate is reproducible.**
   - Run `uv run python scripts/estimate_frcnn_time.py --timing-run results/runs/E0a_frcnn_timing_20261029-0552_2dd6526 --config configs/experiments/E1.yaml --annotations data/voc_prepared/annotations.parquet --out /tmp/e1_est.json && diff /tmp/e1_est.json results/estimates/E1.json`.
   - Expected: no diff output.
6. **The LR schedule matches the config.**
   - Run `awk -F, '$2==0||$2==500||$2==5152||$2==7084{print $1,$2,$3}' results/runs/E1_20261029-0602_da4e489/train_log.csv`.
   - Expected: `0 0 1e-05`, `0 500 0.01`, `8 5152 0.001`, `11 7084 0.0001` (644 steps per epoch).
   - Any other values would mean the warm-up or the milestones are off.
7. **The report table is generated, not typed.**
   - Run `make report && git diff --stat reports/`. Expected: no changes.
   - Do not run a bare `make results` first (see open issue 5).
8. **CI is green.**
   - Run `gh run list --branch phase-3-frcnn --limit 3`. Expected: all `completed success`.

## Open issues / risks

1. **Outputs of a session Kaggle kills are unverified** (time limit, out-of-memory kill). If they are not
   saved, resume is impossible for that failure. The fallback is a full rerun (E1 took under 1 h). ADR-012
   step 3 detects the case before any relaunch.
2. **The per-shape step cost on the T4 is not attributed to a component.** It is priced correctly, and it
   is paid once per shape. If Phase 6 benchmarks Faster R-CNN on GPU, warm-up must cover every shape.
3. **E1 lost to zero-shot YOLO26n.** This is a finding, not a defect. Its meaning depends on the Phase 4
   fine-tuned YOLO numbers, and on the bootstrap CIs (P5.1), because val is small.
4. **The error analysis threshold** is still the 0.5 default. The F1-optimal threshold is planned for P5.3
   (carried over from Phase 2).
5. **A bare `make results` also collects superseded runs** found in `runs/` (E0a 0544, the 0552 resume
   check, E0b 0318) and adds them to `experiments.csv`. The committed files are unaffected. Fix, first on
   the Phase 4 branch: a committed manifest mapping experiment ID to canonical run_id, read by
   `collect_results.py`.
6. **E1 weights live only in the `vocdet-e1` kernel output**, which a rerun would replace. Back up
   `best.pt` before Phase 4; P5.2 needs it.

## Next phase (Phase 4: YOLO26 experiments)

- **P4.1** `train/train_yolo.py`:
  - maps our YAML onto Ultralytics `train()`;
  - builds a writable dataset root in `/kaggle/working`, with symlinked images, copied labels, and a fresh
    `voc_yolo.yaml` (your decision);
  - keeps the torch guard in the runner.
- **P4.2** `predict_yolo` → `Prediction` (conf 0.001, max_det 100), with Ultralytics' own `model.val()` as
  a cross-check and an ADR on the differences.
- **P4.3** E0a-yolo smoke run (64 images, 1 epoch). It is the first launch of a new kernel, so it gets a
  CPU `--resolve-only` check first. Then the §5 YOLO estimates are updated.
- No [MANUAL] steps.
