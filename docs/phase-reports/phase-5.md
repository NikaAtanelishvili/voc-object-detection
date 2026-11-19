# Phase 5 report: final test evaluation (Phase 5 core)

Branch `phase-5-eval`. Scope, per the owner's reorder at the Phase 4 gate: P5.2 with the owner's
adjustments, the E6s seed-43 rerun, and 1000-draw comparisons. P5.1's remaining tables and figures and
P5.3's error analysis move to Phase 8 (DEVIATIONS 2026-11-06).

Exit criteria for this scope:
- Test metrics recorded once ✔ (`results/test_metrics.csv`).
- Defense notes ✔ (`docs/defense/phase-5.md`).
- Tag `phase-5`: pending your approval.

`make check`: 198 passed, plus format and lint. CI is green on the branch head.

## Done

| Task | Summary | Commits |
|---|---|---|
| Setup | Phase 4 merged and tagged; reorder and adjustments logged; E6s seed-43 config | `38ab143`, `6a7dc2e`, `ba39ce5` |
| P5.2 | `final_test_eval.py` (Kaggle CPU kernel `predict`, local once-only `record`); test results for E0b, E1, E2, E3, E6s, E6n; generated test tables; ADR-003 confirmed | `cfc221d`, `a29763c`, `b3d464d`, `d991253`, `49eadf6`, `ecbc2ec` |
| Bootstrap | Fast resample scoring (`eval/resample.py`), 1000 draws on val and test, `--split test` | `e19d4cd`, `3656a89` |
| Seed spread | E6s_seed43 collected (val only), backed up, paired against E6s | `df97087` |

## Deviations

From `docs/DEVIATIONS.md`, this phase:
1. **Phases 5–8 reordered** to Phase 5 core → 7 → 8 → 6, with the owner's adjustments: test models E0b,
   E1, E2, E3, E6s and E6n; paired test deltas for E6s; ≥1000 draws for final comparisons; the seed-43
   rerun; E6 described as post-hoc in the README and the technical report.
2. **The test evaluation ran in a Kaggle CPU kernel**, not locally. Weights came from the training kernels'
   outputs, and their sha256 was checked against the Hub backups. It survives a lost session and costs no
   GPU quota.
3. **Process slip:** `a29763c` and `b3d464d` were committed while `make check` failed (a helper named
   `test_*` was collected by pytest), against the "never commit with failing tests" rule. `d991253` fixed
   it, and the test results were unaffected.

## Results

**Test split, once per model** (`results/test_metrics.csv`, 5823 images):

| Exp | map50 | map50_95 |
|---|---|---|
| E0b | `0.805066704750061` | `0.6264790892601013` |
| E1 | `0.7623130083084106` | `0.5039623379707336` |
| E2 | `0.7821313738822937` | `0.5998595952987671` |
| E3 | `0.8061645030975342` | `0.63302081823349` |
| E6s | `0.85569167137146` | `0.688808023929596` |
| E6n | `0.8110044002532959` | `0.6317331790924072` |

**Test paired deltas** (`results/paired_deltas_test.csv`, 1000 shared draws, Δ [95% CI]):

| Comparison | Δ map50 | Δ map50_95 |
|---|---|---|
| E6s − E0b | +0.051 [+0.043, +0.058] | +0.062 [+0.056, +0.069] |
| E6s − E3 | +0.050 [+0.042, +0.057] | +0.056 [+0.049, +0.062] |
| E6s − E6n | +0.045 [+0.038, +0.051] | +0.057 [+0.051, +0.063] |
| E6s − E1 | +0.093 [+0.085, +0.102] | +0.185 [+0.177, +0.191] |

Every test interval excludes zero. **ADR-003's revisit condition is not triggered**: E6s beats E0b and E3
on test, so it stays the deployment candidate.

**E6 is post-hoc.** It was designed after viewing val, so its val score is optimistic. The claim rests on
the test results above, which were produced once, after selection. On test, E6s's margin over E0b and E3
is about as large as on val.

**Seed spread** (val only, `results/paired_deltas.csv`, 1000 draws): E6s_seed43 − E6s = +0.013
[−0.003, +0.028] on map50 and +0.014 [−0.001, +0.027] on map50_95. E6s_seed43's val map50 is
`0.8821157813072205` (`results/runs/E6s_seed43_20261106-0014_ba39ce5/metrics.json`).
- **Confound:** E6s_seed43 ran on Kaggle's newer image (Python 3.13.16, torch `+cu130`), E6s on the older
  one (`+cu128`), so this spread mixes the seed and the environment. Separating them would need a
  seed-42 rerun on the new image.
- **Reading:** this spread is about the size of the smaller Phase 4 differences (for example E6n − E2,
  +0.023 [+0.007, +0.040] on val map50). Single-seed differences of that size are weak evidence. E6s's
  test margins are several times larger.

**Val comparisons with 1000 draws** (`results/paired_deltas.csv`): every Phase 4 conclusion holds; no
interval changed sides of zero. For example, E6s − E0b is +0.042 [+0.020, +0.065] and E2 − E0b is −0.015
[−0.035, +0.006] on map50.

## GPU budget

From the usage log in `docs/ENVIRONMENT.md`:

| | Hours |
|---|---|
| Used this phase: E6s_seed43 0.30 (the test evaluation ran on CPU, 11290 s, no quota) | 0.30 |
| Cumulative | 6.40 |
| Project budget (PLAN.md §5; seed 43 falls in its buffer) | ~12.6 |
| Remaining against budget | ~6.2 |

## Manual verification checklist

1. **The test results were recorded once and cannot be overwritten silently.**
   - Run `uv run python scripts/final_test_eval.py record --from runs/final_test_eval_20261106-0017_cfc221d`.
   - Expected: `AlreadyEvaluatedError ... once per model`, and `git status` shows no change.
2. **The test checkpoints are the backed-up ones.**
   - Run `grep -h weights_sha256 runs/final_test_eval_20261106-0017_cfc221d/E6s/model.json`.
   - Expected: `abca25a6…eb66`, the value in the Weight backups table of `docs/ENVIRONMENT.md` and on the Hub.
3. **The fast bootstrap equals the evaluator.**
   - Run `uv run pytest tests/test_resample.py -v`. Expected: 4 passed.
4. **The test tables are generated, not typed.**
   - Run `make report && git diff --stat reports/`. Expected: no changes.
5. **Every model covered the whole test split.**
   - Run `python3 -c "import json;[print(e, json.load(open(f'runs/final_test_eval_20261106-0017_cfc221d/{e}/metrics_test.json'))['num_images']) for e in ['E0b','E1','E2','E3','E6s','E6n']]"`.
   - Expected: 5823 for each.
6. **Where the test split is read.**
   - Run `grep -rn '"test"\]' src scripts`.
   - Expected: `scripts/final_test_eval.py` (the evaluation), plus `data/integrity.py` and `data/eda.py`
     (Phase 1 split checks and counts, no predictions).
   - Note: the `vocdet eval` and `vocdet errors` CLIs still accept `--split test` (open issue 5). No
     committed result used them on test.
7. **CI.**
   - Run `gh run list --branch phase-5-eval --limit 3`. Expected: `completed success` on the head commit.

## Open issues / risks

1. **The seed spread is confounded with the Kaggle image change.** Handling: the report states it; a
   seed-42 rerun on the new image (~0.4 h) would separate the two, if you want it.
2. **A 1000-draw test bootstrap takes about 10 minutes per model** on this laptop, because the test split
   has 10× more images. Results are cached per model, so it reruns only for new models.
3. **P5.1's remaining tables and figures and P5.3's error analysis are still open.** Handling: they are
   done in Phase 8, where the README and the technical report use them.
4. **The HF model repo is still private.** Handling: P7.5 makes it public.
5. **The CLI does not enforce hard rule 4.** `vocdet eval --split test` and `vocdet errors --split test`
   would score test without the once-only guard. Proposal: drop `test` from their `--split` choices, so
   `final_test_eval.py` is the only way in. It is a two-line change plus a test; I can do it at the start
   of Phase 7 if you agree.

## Next phase (Phase 7: application and deployment, PyTorch backend)

- **P7.1** FastAPI app: `/health`, `/model-info`, `/predict`, `/predict/annotated`, input validation, and
  dependency injection of a `Detector`. The P6.2 interface is pulled forward with `PyTorchBackend` only;
  E6s's COCO-spelled names map back through `build_class_map`. New dependencies (FastAPI, Gradio,
  uvicorn, multipart) are pinned and their APIs checked.
- **P7.2** Gradio UI at `/ui` with examples, confidence and class filters. The FP32/INT8 toggle waits for
  Phase 6.
- **P7.3** `TestClient` tests with a fake detector, so CI never needs weights.
- **[MANUAL] P7.5:** create the Space `nikaata007/vocdet-demo` (Docker SDK, free CPU), and switch
  `nikaata007/vocdet-models` to public. I will write the exact steps into `docs/MANUAL_STEPS.md` when we
  reach it; P7.1–P7.4 do not need it.
