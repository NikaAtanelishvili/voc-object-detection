"""Fine-tune vs zero-shot checks: head init, class order, and per-class AP50 deltas (ADR-015).

Usage: uv run python scripts/finetune_vs_zeroshot.py
       --out results/diagnostics/finetune_vs_zeroshot.json

Answers why fine-tuned YOLO does not beat the COCO model. Needs runs/ (logs and best.pt of E2, E3).
- init: what each run's Ultralytics log says about the starting weights and the class head.
- class order: names stored in best.pt, and class IDs of the YOLO val labels vs the parquet table.
- per-class AP50 of E2/E3 minus E0b, split by whether Ultralytics could copy the COCO head row:
  cls_remap (on by default) matches names exactly, so renamed VOC classes get fresh rows.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from collect_results import MANIFEST, load_manifest
from vocdet.constants import COCO_TO_VOC, VOC_CLASSES
from vocdet.data.splits import load_splits

REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_LINES = re.compile(
    r"Downloading .*?\.pt|Overriding model\.yaml.*|Remapped \d+/\d+.*|Transferred .*"
)


def fresh_head_classes() -> list[str]:
    """VOC classes whose COCO name differs, so cls_remap cannot match them."""
    return sorted(voc for coco, voc in COCO_TO_VOC.items() if coco != voc)


def per_class_deltas(run: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {name: run[name] - base[name] for name in VOC_CLASSES}


def group_means(deltas: dict[str, float], fresh: list[str]) -> dict[str, float]:
    """Mean delta over the classes with a fresh head row and over the copied ones."""
    groups = {"fresh_head_row": fresh, "copied_head_row": [c for c in deltas if c not in fresh]}
    return {k: sum(deltas[c] for c in names) / len(names) for k, names in groups.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "voc_prepared")
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST)
    runs = {e: REPO_ROOT / "runs" / manifest[e] for e in ("E2", "E3")}
    ap50 = {
        e: json.loads((REPO_ROOT / "results/runs" / manifest[e] / "metrics.json").read_text())[
            "ap50_per_class"
        ]
        for e in ("E0b", "E2", "E3")
    }
    result = {"init": {}, "names_equal_voc_classes": {}}
    for exp_id, run in runs.items():
        log = (run / "command.log").read_text(errors="replace")
        lines = dict.fromkeys(m.group(0) for m in INIT_LINES.finditer(log))
        result["init"][exp_id] = {
            "weights": json.loads((run / "run_meta.json").read_text())["weights"],
            "log": list(lines),
        }
        names = YOLO(str(run / "weights" / "best.pt")).names
        result["names_equal_voc_classes"][exp_id] = [names[i] for i in range(20)] == list(
            VOC_CLASSES
        )

    # Label files vs the annotation table: same multiset of class IDs per image, difficult excluded.
    annotations = pd.read_parquet(args.data_dir / "annotations.parquet")
    mismatched = 0
    val_ids = load_splits(REPO_ROOT / "splits")["val"]
    kept = annotations[~annotations["difficult"]].groupby("image_id")["class_id"].apply(list)
    for image_id in val_ids:
        text = (args.data_dir / "yolo" / "labels" / "val" / f"{image_id}.txt").read_text()
        labels = Counter(int(line.split()[0]) for line in text.splitlines())
        mismatched += labels != Counter(kept.get(image_id, []))
    result["val_images_with_label_class_mismatch"] = mismatched

    fresh = fresh_head_classes()
    deltas = {e: per_class_deltas(ap50[e], ap50["E0b"]) for e in ("E2", "E3")}
    result["fresh_head_classes"] = fresh
    result["ap50_delta_vs_E0b"] = deltas
    result["ap50_delta_vs_E0b_group_means"] = {e: group_means(d, fresh) for e, d in deltas.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
