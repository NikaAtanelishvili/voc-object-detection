# Phase 4 report: YOLO26 experiments

Branch `phase-4-yolo`. Exit criteria:
- E2–E5 collected ✔, plus E6s and E6n (owner-approved additions). All are in `results/canonical_runs.yaml`,
  `results/experiments.csv` and `results/paired_deltas.csv`.
- Defense notes written ✔ (`docs/defense/phase-4.md`).
- Tag `phase-4`: pending your approval.

`make check`: 188 passed, plus format and lint. CI is green on the branch.

## Done

| Task | Summary | Commits |
|---|---|---|
| (pre-P4.1) | Results come from a canonical-run manifest; E1 `best.pt` backed up to the private HF repo | `78cdce5`, `928c076` |
| P4.1 | `train_yolo`: config → Ultralytics `train()`, writable dataset root (symlinked images, copied labels, train/val-only YAML) | `2186f23` |
| P4.2 | Val prediction through our adapter and evaluator, Ultralytics val as a cross-check, measured gap breakdown, ADR-013 | `cac02e6`, `a3720c6`, `59ce9bc` |
| P4.3 | E0a-yolo smoke run, yolo26n/s full-data timing runs, YOLO time estimator, thread-pool dataset root, status-poll retry, §5 estimates | `6a64129`, `860a6b4`, `5f883b8`, `ef8f8a1`, `4da54a1`, `9c4f864`, `631ad4c`, `8b4e55a`, `ae67b2f` |
| P4.4 | E2 and E3 collected and backed up; head and class-order checks (ADR-015) | `2a954b6`, `9d0dd11`, `ed32a56`, `8d0e660` |
| P4.5 | E4 and E5; E6s/E6n gentle fine-tune (new settings, configs, corrected estimates); all backed up | `836c166`, `2c9ce61`, `70cdfe9`, `9d9a360`, `37e902e`, `3f5e706`, `9f2f939`, `be71add`, `4ebc807` |
| P4.6 | Quick CPU cost of the candidates; ADR-003 selects E6s | `4c606b5`, `b5c8337` |
| (P5.1, early) | Paired bootstrap of differences (ADR-014), generated delta table in `reports/EXPERIMENTS.md` | `0faa469`, `dc8cbeb`, `79d0e19`, `d80a228`, `7fb5586`, `654138d`, `0ee9680` |

## Deviations

From `docs/DEVIATIONS.md`, this phase:
1. **Before P4.1:** results are collected from a committed manifest (`results/canonical_runs.yaml`), not
   from everything in `runs/`.
2. **Before P4.1:** the HF model repo `nikaata007/vocdet-models` was created early, private, to back up
   E1's `best.pt` (P7.5 becomes "make public + model card").
3. **P4.1:** the dataset root lives in the run folder and is deleted after training; the torch guard
   already existed in the runner.
4. **P4.3:** two extra full-data timing runs (yolo26n, yolo26s), because the 4-iteration smoke run
   could not be extrapolated.
5. **P4.3:** the dataset root is built with a thread pool (the serial build cost ~11 GPU minutes).
6. **P5.1 (done early):** paired bootstrap of differences replaces per-run CI comparison (ADR-014).
7. **P4.4:** head-initialisation checks before E4/E5 (ADR-015).
8. **P4.5:** E6 gentle fine-tune added (E6s, E6n).
9. **P4.5:** E6 estimates corrected before launch (start-up priced from a cold run).

## Results

Val metrics, each from `results/runs/<run_id>/metrics.json`:

| Exp | Run | map50 | map50_95 |
|---|---|---|---|
| E0b | `E0b_20261028-0321_fa7911f` | `0.8274052143096924` | `0.6493940949440002` |
| E1 | `E1_20261029-0602_da4e489` | `0.79451584815979` | `0.5476698279380798` |
| E2 | `E2_20261029-1808_ae67b2f` | `0.8120008707046509` | `0.6434926986694336` |
| E3 | `E3_20261029-1933_dc8cbeb` | `0.8188428282737732` | `0.6587539911270142` |
| E4 | `E4_20261031-2228_654138d` | `0.7627784609794617` | `0.5903745889663696` |
| E5 | `E5_20261101-1538_2c9ce61` | `0.8160077929496765` | `0.642431914806366` |
| E6n | `E6n_20261105-2334_9f2f939` | `0.834827184677124` | `0.6696570515632629` |
| E6s | `E6s_20261102-1659_3f5e706` | `0.8690264225006104` | `0.7145063877105713` |

Paired differences (`results/paired_deltas.csv`, 200 shared resamples, Δ [95% CI]; the full table is
generated in `reports/EXPERIMENTS.md`):

| Comparison | Δ map50 | Δ map50_95 |
|---|---|---|
| E2 − E0b | −0.015 [−0.032, +0.006] | −0.006 [−0.025, +0.013] |
| E3 − E0b | −0.009 [−0.033, +0.012] | +0.009 [−0.013, +0.031] |
| E2 − E1 | +0.017 [−0.005, +0.038] | **+0.096 [+0.075, +0.115]** |
| E3 − E2 | +0.007 [−0.013, +0.025] | +0.015 [−0.004, +0.032] |
| E4 − E2 | **−0.049 [−0.078, −0.021]** | **−0.053 [−0.076, −0.027]** |
| E4 − E0b | **−0.065 [−0.086, −0.038]** | **−0.059 [−0.080, −0.036]** |
| E5 − E2 | +0.004 [−0.013, +0.022] | −0.001 [−0.018, +0.014] |
| E5 − E0b | −0.011 [−0.031, +0.012] | −0.007 [−0.025, +0.013] |
| E6s − E0b | **+0.042 [+0.023, +0.062]** | **+0.065 [+0.049, +0.086]** |
| E6s − E2 | **+0.057 [+0.040, +0.077]** | **+0.071 [+0.057, +0.089]** |
| E6s − E3 | **+0.050 [+0.035, +0.072]** | **+0.056 [+0.040, +0.074]** |
| E6n − E0b | +0.007 [−0.006, +0.022] | **+0.020 [+0.004, +0.038]** |
| E6n − E2 | **+0.023 [+0.005, +0.038]** | **+0.026 [+0.012, +0.038]** |
| E6s − E6n | **+0.034 [+0.019, +0.052]** | **+0.045 [+0.031, +0.060]** |

Bold: the CI excludes zero.

- **Head inheritance** (`results/diagnostics/finetune_vs_zeroshot.json`): mean AP50 change against E0b
  for the 6 fresh-row classes vs the 14 copied ones. E2: `-0.039133861660957336` vs
  `-0.005234556538718087`. E3: `-0.031081661581993103` vs `0.0010887725012642996`.
- **CPU cost** (`results/diagnostics/cpu_cost_quick.json`, development laptop, 2 torch threads, median
  batch-1 ms):

  | Model | median ms | params |
  |---|---|---|
  | E1 | `279.9743919986213` | `19052004` |
  | E6s | `141.65856500039808` | `9963344` |
  | E6n | `52.752880999832996` | `2543120` |

**Interpretation** (no new numbers):
- Standard 50-epoch fine-tuning (E2, E3) is not shown to beat the zero-shot model.
- Augmentation matters (E4 is worse than everything). A frozen backbone costs nothing measurable (E5).
- The gentle fine-tune E6s is the best model, ahead of every other run with CIs excluding zero, so
  ADR-003 selects it, with E6n as the fallback.
- **E6 was designed after viewing val results** (the 3-epoch timing run), so its val score is optimistic
  as a selection. The P5.2 test-split results decide. If E6s does not beat E0b and E3 on test, ADR-003 is
  revisited.

## GPU budget

From the usage log in `docs/ENVIRONMENT.md`:

| | Hours |
|---|---|
| Used this phase: E0a-yolo 0.05, timing-n 0.45, timing-s 0.15, E2 0.95, E3 1.20, E4 0.45, E5 0.80, E6s 0.40, E6n 0.30 (CPU checks free) | 4.75 |
| Cumulative | 6.10 |
| Project budget (PLAN.md §5, including E6) | ~12.6 |
| Remaining against budget | ~6.5 |

E6 used 0.70 h of the owner's 1.5 h pre-approval. E6s exceeded its own 0.363 h estimate (0.40 h logged).

## Manual verification checklist

1. **Ultralytics finds labels through symlinks, in a read-only source.**
   - Run `uv run pytest tests/test_train_yolo.py -k symlinked -v`. Expected: 1 passed.
   - A failure would mean the dataset root cannot be used on Kaggle.
2. **E6 copies all 20 COCO head rows.**
   - Run `grep -a "Remapped" runs/E6s_20261102-1659_3f5e706/command.log runs/E2_20261029-1808_ae67b2f/command.log`.
   - Expected: `20/20` for E6s and `14/20` for E2.
   - Anything else would mean ADR-015's explanation is wrong.
3. **A backup is the right checkpoint.**
   - Run `sha256sum runs/E6s_20261102-1659_3f5e706/weights/best.pt`.
   - Expected: `abca25a6…eb66`, the value in the Weight backups table in `docs/ENVIRONMENT.md`.
   - Then compare it with the LFS sha256 of that file on huggingface.co/nikaata007/vocdet-models.
4. **Pairing makes intervals tighter.**
   - Run `uv run pytest tests/test_bootstrap_ci.py -v`. Expected: 5 passed.
   - Then read `test_pairing_is_tighter_than_separate_draws`.
5. **The delta table is generated, not typed.**
   - Run `make report && git diff --stat reports/`. Expected: no changes.
6. **The collector ignores superseded runs.**
   - Run `make results && git status --short results/`. Expected: no changes, although `runs/` holds
     resolve-only and superseded folders.
7. **The Kaggle image change.**
   - Run `grep -h '"torch"' runs/E6s_20261102-1659_3f5e706/run_meta.json runs/E6n_20261105-2334_9f2f939/run_meta.json`.
   - Expected: `+cu128` for E6s, `+cu130` for E6n.
8. **CI.**
   - Run `gh run list --branch phase-4-yolo --limit 3`. Expected: `completed success`.

## Open issues / risks

1. **E6's val score is optimistic** (designed after viewing val). Handling: P5.2 test results decide;
   ADR-003 is revisited if E6s does not lead there.
2. **One seed per configuration.** The CIs ignore training randomness. Handling: if budget remains, a
   seed-43 rerun of E6s (~0.4 h) after P5.2.
3. **Kaggle changed its image** (Python 3.13.16, torch `+cu130`) between E6s and E6n. Handling: recorded;
   any further GPU run should note which image it got. A guard against a changed base image could compare
   `run_meta.json` packages with the previous run.
4. **The first E4 pre-check stalled for 26+ minutes**, for an unknown reason. The retry passed in 57 s. It
   was not deleted. The CLI has no cancel, and status only shows the latest version.
5. **Background jobs do not survive a lost session**, so long local jobs (bootstrap, pulls) are now run in
   bounded foreground chunks. The scripts are idempotent, so a rerun only does what is missing.
6. **The HF model repo is private.** Handling: P7.5 makes it public with a model card.

## Next phase, and a proposed reorder

Your stated priority is a live demo and a readable README soon. **Proposal: Phase 5 core → Phase 7 →
Phase 8 → Phase 6**, with these changes:

1. **P5.2 first.** Evaluate E1, E2, E3 and E6s on test once (CPU, locally; about 15 min per YOLO model).
   The README needs test numbers, and it settles the val-optimism question before the demo ships E6s.
2. **Phase 7 next, PyTorch backend only.**
   - Pull P6.2's `Detector` interface forward with just `PyTorchBackend`. FastAPI + Gradio (P7.1–P7.3),
     Dockerfile (P7.4), and the Space deploy (P7.5 [MANUAL], P7.6, P7.7).
   - P7.8's `--backend openvino-int8` option waits for Phase 6.
   - E6s runs at roughly the same CPU latency as measured here (the free Space has 2 vCPUs), which is
     acceptable for a demo.
3. **Phase 8 next.** Technical report, README with generated tables and the live link, traceability and
   the defense index. P5.1's remaining tables and figures and P5.3's error analysis feed it, so they are
   done here.
4. **Phase 6 last.** ONNX / OpenVINO export and benchmarks, then swap the Space to the fastest backend
   that keeps accuracy, and update the README optimization section.

**Cost:**
- The README's optimization section and the deployed backend change once more at the end.
- Phase 6's "before/after" table measures against PyTorch, which is what will already be deployed.
- No GPU impact. This is a reordering of PLAN.md, so it would get a DEVIATIONS entry when you approve it.

**Upcoming [MANUAL] steps:**
- **P7.5:** create the Hugging Face Space (Docker SDK) and switch `nikaata007/vocdet-models` to public.
- Nothing manual is needed for P5.2.

If you approve without the reorder, Phase 5 runs in its planned order: P5.1 (remaining tables and figures),
P5.2, P5.3.
