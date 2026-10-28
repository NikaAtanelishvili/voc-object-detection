"""Seed every random number generator a run touches (Python, NumPy, torch CPU and CUDA)."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Make runs repeatable.

    deterministic=True also forces deterministic cuDNN kernels. That is slower and some torchvision
    detection ops have no deterministic CUDA kernel, so we only warn about those instead of failing:
    the run stays seeded but may not be bit-identical on GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses such as DataLoader workers
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader workers: derive Python/NumPy seeds from torch's per-worker seed.

    torch already seeds each worker differently; without this, NumPy and `random` would start from
    the same state in every worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
