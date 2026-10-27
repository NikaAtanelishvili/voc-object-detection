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

## Kaggle kernel runtime

TODO (P0.6): Python, torch, torchvision, CUDA, GPU, ultralytics, from the `env_check` smoke kernel.
