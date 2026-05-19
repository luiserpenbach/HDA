"""``python -m hda`` launches the HDA desktop application.

Usage:
    python -m hda                          # open with last-used test root
    python -m hda --log-dir /var/log/hda   # log directory override
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hda", description="Hopper Data Studio")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for hda.log (default: ~/.hda/logs)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    from hda.ui.main import main as ui_main

    return ui_main(log_dir=args.log_dir)


if __name__ == "__main__":
    raise SystemExit(main())
