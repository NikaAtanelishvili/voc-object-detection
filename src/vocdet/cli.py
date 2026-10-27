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
    parser.print_help()
    return 0
