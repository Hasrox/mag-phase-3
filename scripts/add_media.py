"""Import whatever was dropped in assets or assets/inbox. No JSON manifest.

    python scripts/add_media.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.config import load_config
from mag.db import connect, init_db
from mag.inbox import absorb
from mag.importing import ensure_layout
from mag.paths import ROOT


def main() -> None:
    ensure_layout(ROOT)
    cfg = load_config()
    conn = connect()
    init_db(conn)
    result = absorb(conn, cfg)
    print(result.line())
    for item in result.rejected:
        print(f"skip {item}")


if __name__ == "__main__":
    main()
