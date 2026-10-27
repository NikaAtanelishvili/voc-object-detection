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
