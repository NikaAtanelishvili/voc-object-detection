"""Command-line entry point `vocdet`. Subcommands are added as their phases are implemented."""

import argparse
import logging

from vocdet import __version__

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch.

    Plain argparse on purpose: the CLI has no hidden framework magic to explain.
    """
    parser = argparse.ArgumentParser(prog="vocdet", description=__doc__)
    parser.add_argument("--version", action="version", version=f"vocdet {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0
