"""Phase 4 exit harness. UI loop and the uniform baseline. No weight update."""

from __future__ import annotations

import json
from pathlib import Path

from mag.audit import run_audit
from mag.config import load_config
from mag.paths import ROOT
from mag.serve import status_line

S1_PATH = ROOT / "config" / "s1_record.toml"
AUDIT_SQL = ROOT / "scripts" / "audit.sql"
LOCK_PATH = ROOT / "config" / "phase4_lock.toml"


def s1_verdict(stars: list[int], flags: list[str]) -> dict:
    """S-1. Failure does not block Phase 5. It blocks a claim that the product is fun."""
    if len(stars) != 10:
        raise ValueError("S-1 is a 10-meme run")
    bad = [item for item in flags if item in {"unreadable_caption", "broken_asset"}]
    rated_3 = sum(1 for star in stars if star >= 3)
    rated_4 = sum(1 for star in stars if star >= 4)
    passed = rated_3 >= 5 and rated_4 >= 1 and not bad
    return {
        "attempted": True,
        "passed": passed,
        "rated_3_or_higher": rated_3,
        "rated_4_or_higher": rated_4,
        "bad_flags": len(bad),
        "stars": stars,
    }


def record_s1(stars: list[int], flags: list[str], path: Path = S1_PATH) -> dict:
    verdict = s1_verdict(stars, flags)
    lines = [
        "# S-1 smoke. Failure does not block Phase 5.",
        "attempted = true",
        f"passed = {'true' if verdict['passed'] else 'false'}",
        f"rated_3_or_higher = {verdict['rated_3_or_higher']}",
        f"rated_4_or_higher = {verdict['rated_4_or_higher']}",
        f"bad_flags = {verdict['bad_flags']}",
        f"stars = {json.dumps(verdict['stars'])}",
    ]
    path.write_text("\n".join(lines) + "\n")
    return verdict


def audit_clean(conn) -> list[tuple[str, list[tuple]]]:
    return run_audit(conn, AUDIT_SQL)


def format_status(topic: str | None, rated: int, pool_size: int) -> str:
    line = status_line(topic, rated, pool_size)
    if "policy" in line or "holdout" in line or "bandit" in line:
        raise RuntimeError("status line leaked the policy")
    return line


def main() -> None:
    cfg = load_config()
    print("phase4_entry_pin=aacc33aca183e5d73aac8cd43d789b536c1d46b3")
    print(f"config_hash={cfg.config_hash[:12]}")
    print("weight_update=false")
    print("http_client=none")
    print("ui_host=127.0.0.1 ui_port=7860 llama_port=8080")
    print(f"s1_record={'present' if S1_PATH.exists() else 'open'}")
    print(f"phase4_lock={'present' if LOCK_PATH.exists() else 'missing'}")
    print("phase4_exit=open until ten manual rows, S-1 recorded, and audit A-1..A-8 are empty")


if __name__ == "__main__":
    main()
