"""``python -m hda`` launches the desktop app.

Usage:
    python -m hda                                     # default db at ~/.hda/hda.db
    python -m hda --db /tmp/test.db                   # override db location
    python -m hda --campaign INJ-CF-C1                # pick the active campaign
    python -m hda --log-dir /var/log/hda              # log directory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hda", description="Hopper Data Studio v3")
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to hda.db (default: ~/.hda/hda.db)",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for hda.log (default: ~/.hda/logs)",
    )
    p.add_argument(
        "--campaign",
        type=str,
        default="DEMO-C1",
        help="Active campaign id (created if it does not exist)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    from hda.ui.main import main as ui_main

    return ui_main(
        db_path=args.db,
        log_dir=args.log_dir,
        campaign_id=args.campaign,
    )


if __name__ == "__main__":
    raise SystemExit(main())
