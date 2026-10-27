"""Command-line entry point `vocdet`. Subcommands are added as their phases are implemented."""

import argparse
import logging
from pathlib import Path

from vocdet import __version__

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch.

    Plain argparse on purpose: the CLI has no hidden framework magic to explain.
    """
    parser = argparse.ArgumentParser(prog="vocdet", description=__doc__)
    parser.add_argument("--version", action="version", version=f"vocdet {__version__}")
    sub = parser.add_subparsers(dest="command")

    download = sub.add_parser("download", help="Download, verify, and extract VOC2012 trainval.")
    download.add_argument("--root", type=Path, default=Path("data"), help="Data root folder.")

    parse = sub.add_parser("parse", help="Parse VOC XML annotations into a parquet table.")
    parse.add_argument("--voc-dir", type=Path, default=Path("data/VOCdevkit/VOC2012"))
    parse.add_argument("--out", type=Path, default=Path("data/processed/annotations.parquet"))

    splits = sub.add_parser("splits", help="Write the frozen train/val/test ID lists.")
    splits.add_argument("--voc-dir", type=Path, default=Path("data/VOCdevkit/VOC2012"))
    splits.add_argument("--out-dir", type=Path, default=Path("splits"))

    export = sub.add_parser("export", help="Export YOLO layout and COCO JSON for every split.")
    export.add_argument(
        "--annotations", type=Path, default=Path("data/processed/annotations.parquet")
    )
    export.add_argument("--voc-dir", type=Path, default=Path("data/VOCdevkit/VOC2012"))
    export.add_argument("--splits-dir", type=Path, default=Path("splits"))
    export.add_argument("--out-dir", type=Path, default=Path("data/prepared"))
    export.add_argument("--mode", choices=["symlink", "copy"], default="symlink")

    integrity = sub.add_parser("integrity", help="Check counts, files, boxes; log class counts.")
    integrity.add_argument(
        "--annotations", type=Path, default=Path("data/processed/annotations.parquet")
    )
    integrity.add_argument("--voc-dir", type=Path, default=Path("data/VOCdevkit/VOC2012"))
    integrity.add_argument("--splits-dir", type=Path, default=Path("splits"))

    eda = sub.add_parser("eda", help="EDA tables (results/eda), figures, and reports/EDA.md.")
    eda.add_argument("--annotations", type=Path, default=Path("data/processed/annotations.parquet"))
    eda.add_argument("--voc-dir", type=Path, default=Path("data/VOCdevkit/VOC2012"))
    eda.add_argument("--splits-dir", type=Path, default=Path("splits"))
    eda.add_argument("--results-dir", type=Path, default=Path("results/eda"))
    eda.add_argument("--report", type=Path, default=Path("reports/EDA.md"))

    prepare = sub.add_parser(
        "prepare", help="Full data pipeline (the Kaggle prep kernel runs this)."
    )
    prepare.add_argument("--download-root", type=Path, default=Path("data"))
    prepare.add_argument("--splits-dir", type=Path, default=Path("splits"))
    prepare.add_argument("--out-dir", type=Path, default=Path("data/voc_prepared"))
    prepare.add_argument("--mode", choices=["symlink", "copy"], default="copy")

    ev = sub.add_parser("eval", help="Evaluate a predictions JSON on a split (the one evaluator).")
    ev.add_argument("--pred", type=Path, required=True, help="Predictions JSON (eval/schema.py).")
    ev.add_argument("--split", choices=["train", "val", "test"], default="val")
    ev.add_argument("--annotations", type=Path, default=Path("data/processed/annotations.parquet"))
    ev.add_argument("--splits-dir", type=Path, default=Path("splits"))
    ev.add_argument("--out", type=Path, help="Write metrics JSON here (default: print only).")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.command == "download":
        from vocdet.data.download import download_voc2012

        voc_dir = download_voc2012(args.root)
        logger.info("VOC2012 ready at %s", voc_dir)
        return 0
    if args.command == "parse":
        from vocdet.data.parse_voc import build_annotations

        build_annotations(args.voc_dir, args.out)
        return 0
    if args.command == "splits":
        from vocdet.data.splits import build_splits

        build_splits(args.voc_dir, args.out_dir)
        return 0
    if args.command == "export":
        import pandas as pd

        from vocdet.data.export_coco import export_coco
        from vocdet.data.export_yolo import export_yolo
        from vocdet.data.splits import load_splits

        annotations = pd.read_parquet(args.annotations)
        split_ids = load_splits(args.splits_dir)
        image_dir = args.voc_dir / "JPEGImages"
        export_yolo(annotations, split_ids, image_dir, args.out_dir / "yolo", mode=args.mode)
        export_coco(annotations, split_ids, args.out_dir / "coco")
        return 0
    if args.command == "integrity":
        import pandas as pd

        from vocdet.data.integrity import check_integrity
        from vocdet.data.splits import load_splits

        annotations = pd.read_parquet(args.annotations)
        check_integrity(annotations, load_splits(args.splits_dir), args.voc_dir)
        return 0
    if args.command == "eda":
        import pandas as pd

        from vocdet.data.eda import run_eda
        from vocdet.data.splits import load_splits

        annotations = pd.read_parquet(args.annotations)
        splits_ = load_splits(args.splits_dir)
        run_eda(annotations, splits_, args.voc_dir, args.results_dir, args.report)
        return 0
    if args.command == "prepare":
        from vocdet.data.download import download_voc2012
        from vocdet.data.prepare import prepare_all

        voc_dir = download_voc2012(args.download_root)
        prepare_all(voc_dir, args.splits_dir, args.out_dir, image_mode=args.mode)
        return 0
    if args.command == "eval":
        import json

        import pandas as pd

        from vocdet.data.splits import load_splits
        from vocdet.eval.evaluator import evaluate, write_metrics
        from vocdet.eval.schema import ground_truth_from_annotations, read_predictions

        image_ids = load_splits(args.splits_dir)[args.split]
        gt = ground_truth_from_annotations(pd.read_parquet(args.annotations), image_ids)
        metrics = evaluate(read_predictions(args.pred), gt)
        if args.out:
            write_metrics(metrics, args.out)
        print(json.dumps(metrics, indent=2))  # noqa: T201  (CLI output)
        return 0
    parser.print_help()
    return 0
