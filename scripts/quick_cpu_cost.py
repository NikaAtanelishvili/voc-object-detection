"""Quick CPU cost of the deployment candidates for model selection (P4.6, ADR-003).

Usage: uv run python scripts/quick_cpu_cost.py --out results/diagnostics/cpu_cost_quick.json

Parameter count and median batch-1 latency (decode + preprocess + forward + postprocess) over the
first 50 val images after 5 warm-up images, on this machine's CPU with torch's default threads. It
only ranks candidates; Phase 6 measures the exported, optimised model properly. One run per
architecture is enough: E2/E3/E5 share E6n/E6s's network, so their cost is the same.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, decode_image
from ultralytics import YOLO

from collect_results import MANIFEST, load_manifest
from vocdet.data.splits import load_splits
from vocdet.models.frcnn import build_frcnn

REPO_ROOT = Path(__file__).resolve().parents[1]
WARMUP, TIMED = 5, 50


def median_ms(predict_one, images: list[Path]) -> float:
    for path in images[:WARMUP]:
        predict_one(path)
    times = []
    for path in images[WARMUP : WARMUP + TIMED]:
        start = time.perf_counter()
        predict_one(path)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "voc_prepared")
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST)
    val_dir = args.data_dir / "yolo" / "images" / "val"
    images = [val_dir / f"{i}.jpg" for i in load_splits(REPO_ROOT / "splits")["val"]]
    result = {
        "cpu": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
    }

    frcnn = build_frcnn(pretrained=False)
    weights = REPO_ROOT / "runs" / manifest["E1"] / "weights" / "best.pt"
    frcnn.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"])
    frcnn.eval()

    def frcnn_one(path: Path) -> None:
        image = decode_image(str(path), mode=ImageReadMode.RGB).float().div(255)
        with torch.inference_mode():
            frcnn([image])

    result["E1"] = {
        "params": sum(p.numel() for p in frcnn.parameters()),
        "median_ms": median_ms(frcnn_one, images),
    }
    for exp_id in ("E6n", "E6s"):
        model = YOLO(str(REPO_ROOT / "runs" / manifest[exp_id] / "weights" / "best.pt"))
        result[exp_id] = {
            "params": sum(p.numel() for p in model.model.parameters()),
            "median_ms": median_ms(
                lambda p, m=model: m.predict(str(p), imgsz=640, device="cpu", verbose=False), images
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))  # noqa: T201  (script output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
