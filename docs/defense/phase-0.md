# Phase 0 defense notes: setup and Kaggle plumbing

## 1. What was built

- A public GitHub repo with a `uv`-managed, src-layout Python package `vocdet` (Python ≥3.10) and a
  committed lockfile, so every machine resolves identical versions.
- Quality gates: ruff format + lint (line length 100, `print` banned in library code), pytest, pre-commit
  hooks (incl. a 1 MB large-file guard), and a `Makefile` exposing every project command.
- GitHub Actions CI (Ubuntu, Python 3.11, uv cache, `uv sync --locked`, `make check`). torch/torchvision
  are mapped to the CPU wheel index so CI never downloads CUDA.
- Docs skeleton: README, ADR log with ADR-001 (stack), deviations log, manual steps, environment facts,
  report stubs.
- Kaggle plumbing: `kaggle_run.py` turns a validated YAML config into a Kaggle script kernel that clones the
  repo **at an exact pushed SHA**, runs one command, and records what ran; `kaggle_pull.py` waits for the
  kernel and downloads outputs into `runs/<run_id>/`. A smoke kernel (`env_check`) recorded the real Kaggle
  runtime into `docs/ENVIRONMENT.md`.

## 2. Key code paths

- `src/vocdet/config.py::KaggleSettings` / `JobConfig`: pydantic models; `extra="forbid"` turns a typo like
  `gpus:` into an error before any GPU time is spent.
- `src/vocdet/config.py::load_job_config`: YAML → validated model; rejects non-mapping YAML.
- `scripts/kaggle_run.py::ensure_clean_and_pushed`: `git status --porcelain` must be empty and
  `git branch -r --contains HEAD` non-empty, otherwise the kernel would run different code than we think.
- `scripts/kaggle_run.py::https_repo_url`: converts the SSH remote to HTTPS because Kaggle clones anonymously.
- `scripts/kaggle_run.py::make_run_id`: `<exp_id>_<yyyymmdd-hhmm>_<sha7>`, ties every artifact to a commit.
- `scripts/kaggle_run.py::build_metadata`: writes `kernel-metadata.json` (script kernel, private, internet
  on, GPU flag, `kernel_sources` for mounting other kernels' outputs).
- `scripts/kaggle_run.py::render_runner` + `kaggle/templates/runner.py.j2`: values are injected with
  `tojson`, which produces valid Python string/list literals with correct escaping.
- `runner.py.j2::main`: clone → checkout SHA → `pip install --no-deps -e .` → explicit extra packages → run
  command with output teed to `command.log` → write `kaggle_run.json` → exit with the command's return code
  (so a failed command shows as a failed kernel).
- `scripts/kaggle_pull.py::wait_for_kernel` / `parse_status`: polls `kaggle kernels status` and parses the
  `KernelWorkerStatus.X` enum name.
- `scripts/kaggle_pull.py::main`: downloads outputs and refuses them if `runs/<run_id>/kaggle_run.json` is
  missing (a newer version of the same kernel would have a different run folder).
- `scripts/env_check.py::torch_info`: besides versions, runs a CUDA matmul, because an unsupported GPU
  architecture only fails at the first real kernel launch.

## 3. Design decisions

- [ADR-001 Stack choice](../DECISIONS.md#adr-001-stack-choice), including every pinned version so far.
- Deviations (repo created by the agent; `.gitignore` committed early to protect `.env`):
  [DEVIATIONS.md](../DEVIATIONS.md).

## 4. Likely interview questions

1. **Why clone the repo at a SHA inside the kernel instead of uploading the code?** The SHA in
   `kaggle_run.json` and in the run ID proves exactly which code produced a result; uploading a working copy
   could include uncommitted edits. That is also why `kaggle_run.py` refuses a dirty or unpushed tree.
2. **Why `pip install --no-deps`?** Our `pyproject.toml` pins exact versions. A normal install would replace
   Kaggle's CUDA-matched torch with a different build. Missing packages are listed explicitly in the config.
3. **Why does the runner write into `/kaggle/working/<run_id>/` and not `/kaggle/working/` directly?** Only
   `/kaggle/working` is saved; the run-ID subfolder lets `kaggle_pull.py` detect that downloaded outputs
   belong to this run and not to a later version of the same kernel.
4. **Why `uv sync --locked` in CI?** It fails if `uv.lock` is out of date with `pyproject.toml`, so a
   dependency change can't land without its lockfile.
5. **Why are some dev tools marked `python_version >= '3.11'`?** pre-commit 4.7 and kaggle 2.2.5 require
   3.11, but the package must stay installable on 3.10 (Kaggle compatibility). The marker lets uv resolve
   for the whole `>=3.10` range.
6. **Why pydantic for configs?** Validation fails fast with the field name. A misspelled key would
   otherwise be silently ignored and the run would use a default, wasting GPU hours.
7. **How do you avoid committing secrets?** `.env` was gitignored before the first `git add`; only
   `.env.example` is tracked; Kaggle's token lives in `~/.kaggle/` outside the repo.

## 5. Practice exercises (do these without AI)

**Bug to find.** Suppose `ensure_clean_and_pushed` were changed to compare `git rev-parse HEAD` with
`git rev-parse origin/main`. Describe a workflow on the `phase-0-setup` branch where this check gives the
wrong answer, and which of the two failure modes (false refusal or false acceptance) is worse here.
*Hints:* which branch do we push during a phase? What does `git branch -r --contains` check that an equality
test does not? Does the check fetch first?

**Feature to add.** Add a `--timeout-min` option to `kaggle_run.py` that limits a kernel's runtime (protects
the GPU budget from a hung run). *Hints:* look at `kaggle kernels push --help` for the relevant flag and its
unit; decide whether it belongs in `KaggleSettings` (per-config default) or only on the command line; add
a test in `tests/test_config.py` for the new field's validation.
