"""Load the store from operator manifests. Idempotent. Never calls the model.

    python scripts/import_store.py --images images.json --sounds sounds.json
    python scripts/import_store.py --gold data/gold/manifest.json

Image and sound manifests are JSON lists. Each row needs file and license_note.
Sounds also need class and intensity. Gold rows are flagged is_gold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.config import load_config
from mag.db import connect, init_db
from mag.importing import ensure_layout, import_batch, import_gold
from mag.paths import ROOT
from mag.phase1 import format_status, operator_status


def _load(path: str | None) -> list[dict]:
    if not path:
        return []
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="MAG Phase 1 store import")
    parser.add_argument("--images", help="JSON list of image rows")
    parser.add_argument("--sounds", help="JSON list of sound rows")
    parser.add_argument("--gold", help="Gold manifest.json")
    args = parser.parse_args()
    ensure_layout(ROOT)
    cfg = load_config()
    conn = connect()
    init_db(conn)
    if args.gold:
        result = import_gold(conn, Path(args.gold), cfg)
        print(f"gold added={result.added} reused={result.reused}")
    if args.images or args.sounds:
        result = import_batch(conn, _load(args.images), _load(args.sounds), cfg)
        print(f"import added={result.added} reused={result.reused}")
    print(format_status(operator_status(conn)))


if __name__ == "__main__":
    main()
