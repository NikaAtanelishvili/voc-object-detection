# CLAUDE.md — Agent Operating Rules

This file is read by Claude Code at the start of every session. Follow it exactly.

## Project in one paragraph

End-to-end object detection on **Pascal VOC 2012** (20 classes) with **PyTorch**. We compare a two-stage detector
(torchvision Faster R-CNN, own training loop) against one-stage Ultralytics YOLO26 models, run controlled ablations,
optimize the best model for **CPU inference** (ONNX / OpenVINO FP32 / INT8), and ship it as a **FastAPI + Gradio**
service on a free Hugging Face Space. Training runs on **Kaggle free GPUs** via the Kaggle CLI; all code is written
and tested locally. Deadline: **7 days**. The owner (Nikoloz) must defend every line of code in a live interview
**without AI tools**, so clarity beats cleverness everywhere.

## Sources of truth

| What | Where |
|---|---|
| Tasks, order, acceptance criteria | `PLAN.md` (checkboxes = progress) |
| Design decisions | `docs/DECISIONS.md` (ADR log) |
| Things the human must do | `docs/MANUAL_STEPS.md` |
| Every metric that appears anywhere | files under `results/` produced by scripts |
| Environment facts (Kaggle Python/torch/GPU) | `docs/ENVIRONMENT.md` |

## The work loop (repeat for every task)

1. Open `PLAN.md`. Take the **first unchecked task** (`- [ ] Px.y`) of the current phase. Never skip ahead.
2. If the task is tagged **[MANUAL]**: append exact step-by-step instructions to `docs/MANUAL_STEPS.md`
   (numbered, with links, what to paste where, how to verify), print them to the user, and **stop**. Resume only
   after the user confirms.
3. Implement the smallest change that satisfies the task's acceptance criteria. Add or update tests.
4. Run `make check` (ruff format check + ruff lint + pytest). It must pass. Do not weaken or delete tests to pass.
5. Update docs the change touches (README section, ADR if a choice was made, `docs/ENVIRONMENT.md` if relevant).
6. Tick the task's checkbox in `PLAN.md` **in the same commit**.
7. Commit (format below) and `git push`.
8. At the end of a phase: run the **Phase gate** below. Do not start the next phase until the user approves.

Any time you deviate from `PLAN.md` (different library, parameter, file layout, skipped or reordered task, changed
estimate), append an entry to `docs/DEVIATIONS.md` **at that moment**: date, task ID, planned, actual, reason,
impact, and a link to an ADR if one was written. Never deviate silently.

## Phase gate (mandatory stop after every phase)

1. Verify the phase **Exit criteria** in `PLAN.md`. Run `make check` and confirm CI is green on the phase branch.
2. Write `docs/defense/phase-<n>.md` (see below).
3. Write `docs/phase-reports/phase-<n>.md` and print it in full to the user. Required sections:
   - **Done:** each task ID with a one-line summary and its commit hashes.
   - **Deviations:** every `DEVIATIONS.md` entry from this phase, or "None".
   - **Results:** key numbers from this phase, quoted from `results/` files with paths. No hand-typed metrics.
   - **GPU budget:** hours used this phase, cumulative total, and remaining against the §5 budget.
   - **Manual verification checklist:** 3–8 concrete checks the user performs, each with the exact command or
     action, the expected output, and what a failure would mean. Examples: run a test file, open a figure,
     inspect a split file, compare a count, read a specific function.
   - **Open issues / risks:** anything unresolved and the proposed handling.
   - **Next phase:** its first 3 tasks and any [MANUAL] steps coming up.
4. **Stop and wait.** Continue only when the user replies with approval (e.g. "approved" or "go"). If the user
   reports a failed check, fix it on the phase branch, re-run the gate from step 1, and stop again.
5. After approval: merge with `--no-ff`, tag `phase-<n>`, push, then start the next phase.

## Git rules

- Branch per phase: `phase-<n>-<slug>` (e.g. `phase-1-data`). `main` is always green.
- Merge with `git merge --no-ff phase-<n>-<slug>` then tag `phase-<n>`. Final release tag: `v1.0.0`.
- **Conventional Commits**: `type(scope): imperative summary` (≤72 chars).
  Types: `feat fix test docs refactor perf chore ci build data exp`.
  `exp` = experiment configs/results; `data` = data pipeline outputs such as split files.
- Commit body: what changed and **why**, then a trailer line `Task: P<x>.<y>`.
- One logical change per commit. Never mix refactors with features.
- Forbidden: `--no-verify`, force-push to `main`, amending or rebasing pushed commits, committing with failing tests.

## Hard rules

1. **No secrets in git.** Real values live in `.env` (gitignored) and `~/.kaggle/`. Only `.env.example` is committed.
2. **No data, weights, or binaries >1 MB in git.** Datasets live on Kaggle (kernel outputs), weights on the
   Hugging Face Hub, run artifacts in `runs/` (gitignored). Small text results (`metrics.json`, CSVs, split ID
   lists, figures ≤500 KB) are committed under `results/`, `splits/`, `reports/figures/`.
3. **Never type a metric by hand.** Every number in any report or README is generated from `results/` by
   `scripts/build_report_tables.py`. If a number is needed and does not exist in `results/`, run the script that
   produces it.
4. **Test-set discipline.** The test split (official VOC2012 `val`) is used **only** in Phase 5 by
   `scripts/final_test_eval.py`, once per final model. All model selection, threshold choice, and tuning use the
   `val` split.
5. **Frozen splits.** After P1.4 the split files are frozen. Changing them requires a new ADR and rerunning all
   experiments.
6. **Kaggle runs only from a pushed, clean commit.** `scripts/kaggle_run.py` refuses a dirty tree or unpushed
   HEAD and records the commit SHA in the run metadata.
7. **Verify APIs, don't recall them.** Before using a library function, check it against the installed version
   (`python -c "import x, inspect; print(inspect.signature(x.f))"` or the docs). Pin every dependency version in
   `pyproject.toml` at the moment you add it and note it in the relevant ADR.
8. **Facts in PLAN.md are claims to verify, not ground truth.** If an expected count, URL, checksum, path, or
   platform limit disagrees with reality: stop, report the discrepancy to the user, and do not silently adjust
   the code to match.
9. **GPU budget.** Before launching any Kaggle GPU run, check the budget table in `PLAN.md` §5. If the run would
   push the cumulative total past the budget, stop and ask.

## Code style

- Python **≥3.10 compatible** (match Kaggle; see `docs/ENVIRONMENT.md`). Package lives in `src/vocdet/`.
- Type hints on all public functions. Docstrings explain **why**, not just what.
- `ruff` for formatting and linting (config in `pyproject.toml`), line length 100.
- Use `pathlib`, the `logging` module (no `print` in library code), and YAML configs validated by pydantic models.
- Core logic stays **explicit and readable**: the Faster R-CNN training loop, box conversions, the evaluator
  wrapper, and the error analysis are written as plain loops and functions, not hidden behind frameworks.
- No premature abstraction. If a helper is used once, inline it.
- Every module starts with a short docstring stating its responsibility.
- Raise specific exceptions with actionable messages. Never use bare `except:`.

## Defense notes (required at the end of every phase)

Write `docs/defense/phase-<n>.md` containing:

1. What was built (5–10 lines).
2. Key code paths as `file.py::function` with a one-line explanation each.
3. Design decisions with links to their ADRs.
4. 5–8 likely interview questions with concise answers.
5. Two practice exercises for the owner: one **bug to find** (describe a plausible bug that could be introduced
   into this phase's code) and one **small feature to add**, each with hints. Do not write solutions in code.

## Stop and ask the user when

- A phase is finished (Phase gate above).
- A task is tagged [MANUAL] or needs credentials, accounts, or payment.
- The same failure persists after 2 fix attempts with an unclear root cause.
- A GPU run would exceed the budget, or a Kaggle session fails twice.
- A decision not covered by `PLAN.md` would change scope, metrics, splits, or the deliverables.
- Reality contradicts a fact in `PLAN.md` (hard rule 8).

## Makefile targets (create in P0.2, keep up to date)

```
make setup            # uv sync + pre-commit install
make fmt              # ruff format
make lint             # ruff check
make test             # pytest -q
make check            # fmt-check + lint + test   (must pass before every commit)
make kaggle-run RUN=<config>   # push a Kaggle kernel for an experiment config
make kaggle-pull RUN=<run_id>  # download kernel outputs into runs/<run_id>/
make results          # collect runs/ -> results/experiments.csv
make report           # regenerate all report tables and figures
make bench            # optimization benchmarks
make serve            # run FastAPI + Gradio locally
make docker-build / make docker-run
make deploy           # push weights + app to Hugging Face
```
