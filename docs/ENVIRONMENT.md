# Environment

Facts about every machine the project runs on. Only verified values; each section says how it was checked.

## Local development machine

Checked 2026-10-28 with `python3 --version`, `uv --version`, `uname -r`.

| Item | Value |
|---|---|
| OS | Linux (kernel 6.8.0-147-generic) |
| System Python | 3.12.3 |
| uv | 0.13.5 |

## Kaggle account

Quota as shown in Kaggle account settings, reported by the owner on 2026-10-28 (the CLI cannot read it).
It floats week to week, so re-check before large runs.

| Item | Value |
|---|---|
| Username | `nikaatanelishvili` |
| GPU quota | 30 h/week (0 h used on 2026-10-28) |
| TPU quota | 20 h/week (not used by this project) |
| Private datasets storage | 214.75 GB |

The project GPU budget (`PLAN.md` §5) is ~15.5 h of the 30 h weekly quota.

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

Consequences:

- **Python compatibility target:** the code must run on Kaggle's 3.13 and stay ≥3.10 compatible
  (`requires-python = ">=3.10"`, ruff `target-version = "py310"`). CI currently tests 3.11 (PLAN.md P0.3).
- **P100 is unusable**, confirmed: the torch build's arch list starts at sm_75, and the P100 is sm_60.
  `scripts/kaggle_run.py` always requests `NvidiaTeslaT4`.
- The repo is installed with `--no-deps`, so Kaggle's own versions win (e.g. pydantic 2.13.5, not our
  local 2.14.1 pin). Code must not rely on features newer than the Kaggle versions listed here.
- Accelerator selection works from the CLI (`kaggle kernels push --accelerator NvidiaTeslaT4`); no manual
  UI step is needed.
