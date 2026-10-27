# Phase 0 report: Setup

Branch `phase-0-setup`. Exit criteria: CI green ✔, smoke kernel output pulled ✔, `docs/ENVIRONMENT.md`
complete ✔, defense notes written ✔ (`docs/defense/phase-0.md`), tag `phase-0` pending your approval.

## Done

| Task | Summary | Commits |
|---|---|---|
| P0.1 | Public repo `NikaAtanelishvili/voc-object-detection` created with `gh`; `.env` ignored before the first `git add` | `5d12365` (on `main`) |
| P0.2 | uv src-layout package `vocdet` + console script, ruff, pytest, pre-commit (1 MB guard), Makefile with every target | `74e4d17` |
| P0.3 | GitHub Actions: Ubuntu, Python 3.11, uv cache, `uv sync --locked`, `make check`; CPU torch index mapped | `c207d9d` |
| P0.4 | README, ADR-001 stack choice, MANUAL_STEPS, ENVIRONMENT, phase-report/defense folders, report stubs | `46c486e` |
| P0.5 | Kaggle and HF credentials verified (no secrets recorded); weekly quota written to ENVIRONMENT.md | `13c30c8` |
| P0.6 | `kaggle_run.py` / runner template / `kaggle_pull.py` + tests; `env_check` smoke kernel run on T4 and pulled | `d4cbed4`, `e7684b1` |

## Deviations

From `docs/DEVIATIONS.md`:

1. **P0.1:** the agent created the repo at the owner's request (no clone; the folder already held the plan).
2. **P0.1:** a one-line `.gitignore` (`.env`) was committed in the first commit instead of in P0.2.
3. **P0.6:** extra folders `configs/kaggle/` (non-experiment kernel configs) and `results/env/` (raw smoke
   kernel JSON, the committed source for ENVIRONMENT.md).
4. **P0.6:** `src/vocdet/config.py` was started early (`JobConfig`/`KaggleSettings`) because `kaggle_run.py`
   is its first consumer.

## Results

No model metrics in this phase. Environment facts, quoted from `results/env/kaggle_env_check.json` and
`results/env/kaggle_env_check_run.json`:

- `"python": "3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0]"`
- `"torch": "2.11.0+cu128"`, `"torchvision": "0.26.0+cu128"`, `"ultralytics": null`
- `"arch_list": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]`, two `"Tesla T4"` devices,
  `"cuda_matmul_ok": true`
- `"commit_sha": "d4cbed4f5412918b9a14f53d3ba818535e685a3d"`, `"wall_time_s": 21.8`, `"returncode": 0`

Tests: 23 passed (`make check`). CI: green on every push of `phase-0-setup`.

## GPU budget

| | Hours |
|---|---|
| Used this phase (`env_check`, charged conservatively; script time 27 s from the kernel log) | 0.05 |
| Cumulative | 0.05 |
| Project budget (PLAN.md §5) | ~15.5 |
| Remaining against budget | ~15.45 |
| Weekly Kaggle quota (owner-reported) | 30 |

`env_check` is not a row in the §5 table; it is charged to the 5 h buffer.

## Manual verification checklist

1. **`.env` is not in git.** `git log --all --format= --name-only | grep -c '^\.env$'` → expected `0`.
   Also `git check-ignore -v .env` → `.gitignore:2:.env  .env`. A non-zero count means a secret was
   committed: rotate both tokens immediately.
2. **Clean-clone setup works.** `git clone https://github.com/NikaAtanelishvili/voc-object-detection /tmp/v
   && cd /tmp/v && git switch phase-0-setup && make setup && make check` → ends with `23 passed`. A failure
   means something only works on this machine.
3. **CI is green.** `gh run list --branch phase-0-setup --limit 3` → all `completed success`. Or open the
   Actions tab on GitHub.
4. **The dirty-tree guard works.** `touch scratch.txt && make kaggle-run RUN=configs/kaggle/env_check.yaml`
   → fails with `GitStateError: Working tree is dirty` and pushes nothing. Then `rm scratch.txt`. If it
   pushes, the reproducibility guarantee (hard rule 6) is broken.
5. **The kernel ran our exact commit.** Open <https://www.kaggle.com/code/nikaatanelishvili/vocdet-env-check>
   → Logs → the second line is `git checkout --quiet d4cbed4…`. Compare with
   `git log --oneline -1 d4cbed4`.
6. **The GPU charge is plausible.** Kaggle → Settings → quota. Expected: GPU used is a few minutes at most.
   If it is much higher, tell me so I can correct the usage log in `docs/ENVIRONMENT.md`.
7. **Read the runner.** Open `kaggle/templates/runner.py.j2` and explain out loud why it uses
   `pip install --no-deps` and why it exits with the command's return code (see `docs/defense/phase-0.md`).

## Open issues / risks

1. **Kaggle runs Python 3.13; CI tests 3.11 (as PLAN.md P0.3 specifies).** Code that runs on Kaggle is never
   tested on 3.13 in CI. Proposal: make CI a matrix of 3.11 and 3.13 (≈+15 s per run). This changes P0.3,
   so I need your OK; if approved, I log it as a deviation and do it at the start of Phase 1.
2. **Version skew on Kaggle.** The repo is installed with `--no-deps`, so Kaggle's pydantic 2.13.5 runs
   instead of our 2.14.1 pin. That is harmless for the current code. Rule: never rely on features newer
   than the Kaggle versions in ENVIRONMENT.md.
3. **`ubuntu-latest` is migrating to Ubuntu 26** (GitHub annotation on every CI run). Proposal: pin
   `runs-on: ubuntu-24.04` to avoid a surprise image change mid-project. This is a small change; I'll do
   it with your approval.
4. **GPU charge is an estimate.** Kaggle's session overhead isn't visible in kernel outputs. Check 6 above
   lets you confirm it.
5. **ultralytics, onnxruntime, and openvino aren't on Kaggle.** Every YOLO config must list ultralytics
   (pinned) in `kaggle.pip`. That gets handled in P4.1.

## Next phase (Phase 1: Data)

- **P1.1** `data/download.py`: idempotent VOC2012 download, primary and fallback URL, MD5 read from
  torchvision at runtime, extraction.
- **P1.2** `data/parse_voc.py`: XML → canonical parquet with the 1-based → 0-based coordinate conversion.
- **P1.3** `constants.py`: `VOC_CLASSES` and `COCO_TO_VOC`.
- No [MANUAL] steps in Phase 1. P1.9 runs on a **CPU** Kaggle kernel, so it uses no GPU quota.
