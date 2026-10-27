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
    parser.print_help()
    return 0
