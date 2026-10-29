# Environment

Facts about every machine the project runs on. Only verified values; each section says how it was checked.

## Local development machine

Checked 2026-10-28 with `python3 --version`, `uv --version`, `uname -r`.

| Item | Value |
|---|---|
| OS | Linux (kernel 6.8.0-147-generic) |
| System Python | 3.12.3 |
| uv | 0.13.5 |

## Python 3.13 support of later-phase packages

Checked 2026-10-28 against the PyPI JSON API (`https://pypi.org/pypi/<pkg>/json`, latest release, Linux
x86_64 wheels). Versions are re-checked and pinned when each package is added.

| Package | Latest | `requires_python` | Python 3.13 on Linux x86_64 |
|---|---|---|---|
| ultralytics | 8.4.191 | >=3.8 | pure-Python `py3` wheel, 3.13 classifier; needs only `torch>=1.8.0`, `torchvision>=0.9.0` |
| onnxruntime | 1.31.1 | **>=3.11** | `cp313` manylinux_2_28 wheel |
| openvino | 2026.4.1 | >=3.10 | `cp313` manylinux_2_28 wheel |
| onnx | 1.23.2 | >=3.10 | `cp312-abi3` wheel (stable ABI, so it also installs on 3.13) |
| torch / torchvision (CPU) | 2.14.1 / 0.29.1 | >=3.10 | `cp313` wheels |

Consequences: the app Docker image uses Python 3.13 (owner decision, `docs/DEVIATIONS.md`). onnxruntime
needs ≥3.11, which the project floor now matches (ADR-005).

## Kaggle account

Quota as shown in Kaggle account settings, reported by the owner on 2026-10-28 (the CLI cannot read it).
It floats week to week, so re-check before large runs.

| Item | Value |
|---|---|
| Username | `nikaatanelishvili` |
| GPU quota | 30 h/week (0 h used on 2026-10-28) |
| TPU quota | 20 h/week (not used by this project) |
| Private datasets storage | 214.75 GB |

The project GPU budget (`PLAN.md` §5) is ~11.8 h of the 30 h weekly quota (updated in P4.3).

## Credentials

Checked 2026-10-28. No secret values are recorded here.

| Service | Where the secret lives | Verification command | Result |
|---|---|---|---|
| Kaggle | `~/.kaggle/access_token` (mode 600), read by Kaggle CLI 2.2.5 | `uv run kaggle kernels list --mine` | Authenticated as `nikaatanelishvili`; prints `Not found` because the account has no kernels yet. A bogus `KAGGLE_API_TOKEN` gives `Authentication required`, so the real token is accepted. |
| Hugging Face | `HF_TOKEN` in `.env` | `set -a; . ./.env; set +a; uv run hf auth whoami` | `user=nikaata007`; `huggingface_hub.whoami()` reports token role `write`. |
| GitHub | `gh` keyring login | `gh auth status` | Logged in as `NikaAtanelishvili`, scopes include `repo`. |

### GPU usage log

Script time comes from the pulled `kernel.log`; Kaggle's session start-up overhead is not in the outputs,
so each run is charged conservatively. The Kaggle settings page is the authority if they disagree.

| Date | Run ID | Accelerator | Script time | Charged | Cumulative |
|---|---|---|---|---|---|
| 2026-10-28 | `env_check_20261028-0054_d4cbed4` | GPU T4 x2 | 27 s | 0.05 h | 0.05 h |
| 2026-10-28 | `prepare_data_20261028-0208_7b7605e` | CPU only (no GPU quota) | 187 s | 0 h | 0.05 h |
| 2026-10-28 | `E0b_20261028-0257_5978b2b` (failed: data not found) | GPU T4 x2 | 51 s | 0.05 h | 0.10 h |
| 2026-10-28 | `env_check_20261028-0300_34ff675` (mount layout, CPU) | CPU only (no GPU quota) | 16 s | 0 h | 0.10 h |
| 2026-10-28 | `env_check_20261028-0301_34ff675` (mount layout, T4) | GPU T4 x2 | 38 s | 0.05 h | 0.15 h |
| 2026-10-28 | `E0b_20261028-0305_3fc1694` (failed: data not found) | GPU T4 x2 | 172 s | 0.05 h | 0.20 h |
| 2026-10-28 | `E0b_20261028-0316_9f350f7` (`--resolve-only` path check) | CPU only (no GPU quota) | 50 s | 0 h | 0.20 h |
| 2026-10-28 | `E0b_20261028-0318_9f350f7` (complete; superseded, rounded predictions JSON) | GPU T4 x2 | 110 s | 0.05 h | 0.25 h |
| 2026-10-28 | `E0b_20261028-0321_fa7911f` (complete; **collected**) | GPU T4 x2 | 104 s | 0.05 h | 0.30 h |
| 2026-10-29 | `E0a_frcnn_20261029-0543_1970920` (`--resolve-only` path check) | CPU only (no GPU quota) | 49 s | 0 h | 0.30 h |
| 2026-10-29 | `E0a_frcnn_20261029-0544_1970920` (complete; superseded, no step timing) | GPU T4 x2 | 101 s | 0.05 h | 0.35 h |
| 2026-10-29 | `E0a_frcnn_20261029-0548_0256862` (complete; **collected**) | GPU T4 x2 | 106 s | 0.05 h | 0.40 h |
| 2026-10-29 | `E0a_frcnn_20261029-0552_2dd6526` (cross-session resume check of `0548`) | CPU only (no GPU quota) | 123 s | 0 h | 0.40 h |
| 2026-10-29 | `E0a_frcnn_timing_20261029-0552_2dd6526` (complete; **collected**) | GPU T4 x2 | 253 s | 0.10 h | 0.50 h |
| 2026-10-29 | `E1_20261029-0601_da4e489` (`--resolve-only` path check) | CPU only (no GPU quota) | 48 s | 0 h | 0.50 h |
| 2026-10-29 | `E1_20261029-0602_da4e489` (complete; **collected**) | GPU T4 x2 | 3023 s | 0.85 h | 1.35 h |
| 2026-10-29 | `E0a_yolo_20261029-1531_6a64129` (`--resolve-only` path check) | CPU only (no GPU quota) | 62 s | 0 h | 1.35 h |
| 2026-10-29 | `E0a_yolo_20261029-1534_6a64129` (complete; **collected**) | GPU T4 x2 | 145 s | 0.05 h | 1.40 h |
| 2026-10-29 | `E0a_yolo_timing_n_20261029-1539_860a6b4` (complete; **collected**) | GPU T4 x2 | 1559 s | 0.45 h | 1.85 h |
| 2026-10-29 | `E0a_yolo_timing_s_20261029-1749_631ad4c` (complete; **collected**) | GPU T4 x2 | 419 s | 0.15 h | 2.00 h |

### Weight backups

A kernel's output is replaced when that kernel runs again, and pull patterns skip weights. So the
`best.pt` of every run used later is copied off Kaggle before its kernel is relaunched. **Never rerun a
kernel whose weights are not listed here.** Hub repo: `nikaata007/vocdet-models` (private until P7.5),
path `<run_id>/weights/best.pt`, with a second copy in `runs/<run_id>/weights/` (gitignored).

| Date | Run | Checkpoint | sha256 (local = Hub LFS) | Hub commit | Check |
|---|---|---|---|---|---|
| 2026-10-29 | `E1_20261029-0602_da4e489` | `best.pt`, epoch 11, 76 395 617 B | `faa404c82337f5f75ce91c63ecf014bac8345f20fa06b66ed9254951e3eefe86` | `b6b13d6` | Re-predicting val on CPU reproduces `metrics_val.json` exactly (map50 and map50_95) |

## Kaggle kernel runtime

Recorded 2026-10-28 by the `env_check` smoke kernel (`configs/kaggle/env_check.yaml`, commit `d4cbed4`).
Source files: `results/env/kaggle_env_check.json` (from `scripts/env_check.py`) and
`results/env/kaggle_env_check_run.json` (runner metadata).

| Item | Value |
|---|---|
| Python | **3.13.15** |
| OS | Linux 6.18.55, glibc 2.39 |
| CPU cores | 4 |
| GPU | 2 × Tesla T4 (sm_75, 15360 MiB each), driver 580.178.04 |
| torch | 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19.0) |
| torch CUDA arch list | sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 |
| CUDA matmul smoke test | passed |
| torchvision | 0.26.0+cu128 |
| torchmetrics | 1.9.0 |
| pycocotools | 2.0.11 |
| numpy / pandas | 2.1.3 / 2.3.3 |
| pydantic / pyyaml | 2.13.5 / 6.0.3 |
| opencv-python | 4.14.0.94 |
| onnx | 1.22.0 |
| ultralytics | **not installed** (must be listed in a config's `kaggle.pip`) |
| onnxruntime, openvino | not installed |
| `/kaggle/input` | empty (no sources attached) |

**Faster R-CNN step time on the T4 is bimodal** (P3.4, 2026-10-29). A training step whose padded
batch shape has not been seen before in the session costs several seconds more than a repeat step.
`scripts/estimate_frcnn_time.py` replays the seeded shuffle and predicts exactly which steps are slow,
with no exceptions in `E0a_frcnn_timing_20261029-0552_2dd6526`. The fitted times are in
`results/estimates/E1.json`. `cudnn.benchmark` is off (the PyTorch default), and the CUDA arch list
includes sm_75, so this is neither autotuning nor PTX JIT. Which component pays the per-shape cost is
not identified. The cost is paid once per shape, so it does not grow with the number of epochs.

Consequences:

- **Python compatibility target:** the code must run on Kaggle's 3.13 and stay ≥3.11 compatible
  (`requires-python = ">=3.11"`, ruff `target-version = "py311"`, ADR-005). CI tests 3.11 and 3.13.
- **P100 is unusable**, confirmed: the torch build's arch list starts at sm_75, and the P100 is sm_60.
  `scripts/kaggle_run.py` always requests `NvidiaTeslaT4`.
- The repo is installed with `--no-deps`, so Kaggle's own versions win (e.g. pydantic 2.13.5, not our
  local 2.14.1 pin). Code must not rely on features newer than the Kaggle versions listed here.
- Accelerator selection works from the CLI (`kaggle kernels push --accelerator NvidiaTeslaT4`); no manual
  UI step is needed.

## Kaggle CPU kernel (prepare-data)

Recorded 2026-10-28 by `vocdet-prepare-data` (`configs/kaggle/prepare_data.yaml`, commit `7b7605e`).
Source files: `results/env/kaggle_prepare_data_versions.json` and `results/env/kaggle_prepare_data_run.json`.

| Item | Value |
|---|---|
| Python | 3.13.15 (same image family as the GPU kernel) |
| torch | 2.11.0+cpu (the CPU kernel gets a CPU build) |
| pandas / numpy | 2.3.3 / 2.1.3 (match our pins) |
| pycocotools | 2.0.11 (matches our pin) |
| pyarrow | **23.0.1** (our local pin is 26.0.0) |
| matplotlib | **3.10.0** (our local pin is 3.11.2) |
| tabulate | 0.10.0 (installed from the config's `kaggle.pip`) |
| Script wall time | 187 s for download + full prepare |

Consequences:

- The parquet table, all three COCO JSONs, and every EDA table and summary are **byte-identical** to the
  local run, despite the pyarrow and matplotlib version differences. The only difference was the row
  order of `resolutions.csv`, which was fixed in `0415ff6`.
- Training kernels mount the prepared data through `kernel_sources: [vocdet-prepare-data]`. The mount
  depth **differs between kernels**: `vocdet-env-check` sees `/kaggle/input/vocdet-prepare-data/...`,
  while `vocdet-e0b` sees `/kaggle/input/notebooks/nikaatanelishvili/vocdet-prepare-data/...`. Code
  therefore searches for `<prep_run_id>/voc_prepared/` recursively. The run ID is in
  `configs/data/voc2012.yaml` (ADR-011).
- `kaggle kernels output` lists outputs 20 per page and filters client-side. The prep kernel's ~23k
  outputs hit HTTP 429 until `kaggle_pull.py` switched to `--page-size 200` with one retry.
