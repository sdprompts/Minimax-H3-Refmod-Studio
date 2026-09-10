"""CLI: rank stills in a folder by identity-ref usability."""

from __future__ import annotations

import sys
from pathlib import Path

from backend.quality import _print_table, rank_dir, score_image


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if target.is_dir():
        _print_table(rank_dir(target))
        return 0
    if target.is_file():
        _print_table([score_image(target)])
        return 0
    print(f"not found: {target}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
