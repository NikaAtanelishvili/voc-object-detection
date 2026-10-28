"""Record the runtime environment (Python, torch, CUDA, GPU, packages, /kaggle/input) to env.json.

Runs inside the env_check Kaggle kernel to fill docs/ENVIRONMENT.md with real versions. Also runs a
small CUDA matmul, because a GPU that torch can see but cannot run kernels on (unsupported compute
capability) only fails at the first real operation.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = ["torch", "torchvision", "ultralytics", "torchmetrics", "pycocotools", "numpy", "pandas",
            "pydantic", "pyyaml", "opencv-python", "onnx", "onnxruntime", "openvino"]  # fmt: skip


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False}
    info: dict = {
        "available": True,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "devices": [],
    }
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        info["devices"].append(
            {"name": torch.cuda.get_device_name(i), "capability": f"sm_{major}{minor}"}
        )
    if info["cuda_available"]:
        try:
            x = torch.randn(256, 256, device="cuda")
            info["cuda_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
        except RuntimeError as e:
            info["cuda_matmul_ok"] = False
            info["cuda_matmul_error"] = str(e)
    return info


def command_output(args: list[str]) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    return subprocess.run(args, capture_output=True, text=True).stdout


def kaggle_input_listing(max_depth: int = 4, max_entries: int = 200) -> list[str]:
    """Directories (and a few files) under /kaggle/input, to see how kernel_sources are mounted."""
    root = Path("/kaggle/input")
    if not root.exists():
        return []
    entries = []
    for depth in range(1, max_depth + 1):
        entries.extend(sorted(str(p) for p in root.glob("/".join(["*"] * depth)))[:50])
    return entries[:max_entries]


def collect() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "torch": torch_info(),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "kaggle_input": kaggle_input_listing(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="Directory for env.json")
    args = parser.parse_args()
    env = collect()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(json.dumps(env, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
