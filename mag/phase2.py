"""Phase 2 exit harness. Model acceptance and the offline tagger.

Fixture checks prove the rules. The 12B model, the gate record, the gold
set, and the 300 active images are operator inputs. They block exit, not
the fixture proof. No UI in this phase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mag.paths import DATA, MODELS, ROOT, GATES
from mag.review import deferred_ids, queue


@dataclass
class Phase2Status:
    fixture_ok: bool
    model_present: bool
    card_ready: bool
    gate_pass: bool
    gold_count: int
    s4_pass: bool
    s5_pass: bool
    run_accepted: bool
    active_images: int
    review_open: int
    review_deferred: bool
    blockers: list[str] = field(default_factory=list)

    @property
    def exit_met(self) -> bool:
        return self.fixture_ok and not self.blockers


def _card_ready(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text()
    needed = ("emotion accuracy", "family accuracy", "topics micro-F1", "license:")
    return all(item in text for item in needed) and "license:\n" not in text


def _latest_gate(model_sha: str) -> bool:
    if not GATES.exists():
        return False
    for path in sorted(GATES.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("pass") and data.get("model_sha256") == model_sha:
            return True
    return False


def operator_status(conn, *, model_sha: str = "") -> Phase2Status:
    models = list(MODELS.glob("*.gguf")) if MODELS.exists() else []
    card = MODELS / "model-card.txt"
    gold = int(conn.execute("SELECT COUNT(*) FROM assets WHERE is_gold = 1").fetchone()[0])
    active = int(conn.execute(
        "SELECT COUNT(*) FROM assets WHERE kind = 'image' AND state = 'active' AND is_gold = 0"
    ).fetchone()[0])
    run = conn.execute(
        "SELECT accepted, gold_scores FROM tag_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    s4 = s5 = accepted = False
    if run and run["gold_scores"]:
        scores = json.loads(run["gold_scores"])
        s4 = bool(scores.get("s4_pass"))
        s5 = bool(scores.get("s5_pass"))
        accepted = bool(run["accepted"])
    review_rows = queue(conn)
    deferred = deferred_ids()
    review_open = [row for row in review_rows if int(row["id"]) not in deferred]
    blockers: list[str] = []
    if len(models) < 2:
        blockers.append("models/ needs the GGUF and the matching mmproj; model choice blocks Phase 2")
    if not _card_ready(card):
        blockers.append("model card draft is missing a license and the frozen gold thresholds")
    if not model_sha or not _latest_gate(model_sha):
        blockers.append("no passing gate record for this model hash")
    if gold < 40:
        blockers.append(f"gold set has {gold} hand-tagged images; S-4 needs 40")
    if not s4 or not s5:
        blockers.append("S-4 and S-5 are not recorded as pass")
    if not accepted:
        blockers.append("tag run is not accepted")
    if active < 300:
        blockers.append(f"active images={active}; exit needs 300 with accepted-run or human tags")
    if review_open:
        blockers.append(f"review queue has {len(review_open)} rows; empty it or defer explicitly")
    return Phase2Status(
        fixture_ok=True,
        model_present=len(models) >= 2,
        card_ready=_card_ready(card),
        gate_pass=bool(model_sha) and _latest_gate(model_sha),
        gold_count=gold,
        s4_pass=s4,
        s5_pass=s5,
        run_accepted=accepted,
        active_images=active,
        review_open=len(review_open),
        review_deferred=bool(deferred) and not review_open,
        blockers=blockers,
    )


def format_status(status: Phase2Status) -> str:
    lines = [
        f"fixture_rules={'pass' if status.fixture_ok else 'fail'}",
        f"model_present={status.model_present} card_ready={status.card_ready} gate_pass={status.gate_pass}",
        f"gold={status.gold_count} s4={status.s4_pass} s5={status.s5_pass} accepted={status.run_accepted}",
        f"active_images={status.active_images} review_open={status.review_open}",
        f"phase2_exit={'met' if status.exit_met else 'open'}",
    ]
    for blocker in status.blockers:
        lines.append(f"blocker: {blocker}")
    lines.append(f"root={ROOT} data={DATA}")
    return "\n".join(lines)


def main() -> None:
    from mag.db import connect, init_db

    conn = connect()
    init_db(conn)
    print(format_status(operator_status(conn)))


if __name__ == "__main__":
    main()
