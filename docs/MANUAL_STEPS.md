# Manual steps

Steps the human owner performs. The agent writes the instructions here and waits for confirmation.

## P0.1: GitHub repository (done by the agent at the owner's request)

Done on 2026-10-28 with `gh repo create voc-object-detection --public --source=. --push`.
See `docs/DEVIATIONS.md`. Verify: `gh repo view NikaAtanelishvili/voc-object-detection --json visibility`
prints `"visibility":"PUBLIC"`.

## P0.5: Accounts and credentials (done by the owner before the session)

1. Kaggle account with **phone verification** (needed for GPU and internet in kernels):
   <https://www.kaggle.com/settings> → Phone verification.
2. Kaggle API token: <https://www.kaggle.com/settings> → API → create a token. Store it in `~/.kaggle/`
   with mode 600 (the current CLI reads `~/.kaggle/access_token`; the legacy format is
   `~/.kaggle/kaggle.json`). Verify: `uv run kaggle kernels list --mine` prints a table (possibly empty)
   and no 401 error.
3. Write the weekly GPU quota shown in Kaggle settings into `docs/ENVIRONMENT.md`.
4. Hugging Face → <https://huggingface.co/settings/tokens> → create a **write** token. Put `HF_TOKEN` and
   `HF_USERNAME` in `.env`. Verify: `uv run hf auth whoami` (with `HF_TOKEN` exported from `.env`)
   prints your username.
5. (Optional) W&B: put `WANDB_API_KEY` in `.env`.
6. Put `KAGGLE_USERNAME` in `.env`.
