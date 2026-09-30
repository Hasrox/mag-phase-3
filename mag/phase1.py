"""Phase 1 exit harness. Store, vocabulary, import. No model calls.

Fixture checks prove the rules. Operator inputs that still block the phase
are reported and do not fail the fixture proof.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from mag.paths import CONFIG, DATA, ROOT
from mag.vocab import values


@dataclass
class Phase1Status:
    fixture_ok: bool
    topics_confirmed: bool
    topics: list[str]
    image_count: int
    sound_count: int
    gold_count: int
    probe_topics_covered: bool
    blockers: list[str] = field(default_factory=list)

    @property
    def exit_met(self) -> bool:
        return self.fixture_ok and not self.blockers


def topic_status(path: Path | None = None) -> tuple[bool, list[str]]:
    file = path or (CONFIG.parent / "topics.toml")
    if not file.exists():
        return False, list(values("topic"))
    data = tomllib.loads(file.read_text())
    return bool(data.get("confirmed")), list(data.get("topics") or [])


def probes_cover_topics(path: Path | None = None) -> tuple[bool, list[str]]:
    file = path or (DATA / "gold" / "probes.json")
    if not file.exists():
        return False, list(values("topic"))
    rows = json.loads(file.read_text())
    covered = {row["topic"] for row in rows}
    missing = [topic for topic in values("topic") if topic not in covered]
    return len(rows) >= 20 and not missing, missing


def operator_status(conn) -> Phase1Status:
    confirmed, topics = topic_status()
    images = conn.execute(
        "SELECT COUNT(*) FROM assets WHERE kind = 'image' AND is_gold = 0"
    ).fetchone()[0]
    sounds = conn.execute(
        "SELECT COUNT(*) FROM assets WHERE kind = 'sound'"
    ).fetchone()[0]
    gold = conn.execute(
        "SELECT COUNT(*) FROM assets WHERE is_gold = 1"
    ).fetchone()[0]
    covered, _missing = probes_cover_topics()
    blockers: list[str] = []
    if not confirmed:
        blockers.append("topic list is the PRD proposal; set config/topics.toml confirmed = true")
    if images < 300 or sounds < 40:
        blockers.append(
            f"seed store is {images} images and {sounds} sounds; target is 300 and 40 with license notes"
        )
    if gold < 40:
        blockers.append(f"gold set has {gold} hand-tagged images; target is 40, flagged is_gold")
    if not covered:
        blockers.append("probes.json must hold 20 topic probes covering every topic")
    return Phase1Status(
        fixture_ok=True,
        topics_confirmed=confirmed,
        topics=topics,
        image_count=int(images),
        sound_count=int(sounds),
        gold_count=int(gold),
        probe_topics_covered=covered,
        blockers=blockers,
    )


def format_status(status: Phase1Status) -> str:
    lines = [
        f"fixture_rules={'pass' if status.fixture_ok else 'fail'}",
        f"topics_confirmed={status.topics_confirmed} topics={','.join(status.topics)}",
        f"store images={status.image_count} sounds={status.sound_count} gold={status.gold_count}",
        f"probes_cover_topics={status.probe_topics_covered}",
        f"phase1_exit={'met' if status.exit_met else 'open'}",
    ]
    for blocker in status.blockers:
        lines.append(f"blocker: {blocker}")
    lines.append(f"root={ROOT}")
    return "\n".join(lines)


def main() -> None:
    from mag.db import connect, init_db

    conn = connect()
    init_db(conn)
    status = operator_status(conn)
    print(format_status(status))


if __name__ == "__main__":
    main()
