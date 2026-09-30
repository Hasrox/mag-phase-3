"""SQLite store. WAL, foreign keys, constraints live in the schema. Weights are a cache."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mag.paths import CONFIG, DB_PATH
from mag.vocab import VOCAB, load_vocab_file

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS vocab (
    axis TEXT NOT NULL,
    value TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    since_version TEXT NOT NULL DEFAULT '1.1',
    PRIMARY KEY (axis, value)
);

CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('image', 'sound')),
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    license_note TEXT NOT NULL CHECK (length(trim(license_note)) > 0),
    source TEXT,
    state TEXT NOT NULL CHECK (state IN ('new', 'tagged', 'active', 'review', 'retired', 'quarantine')),
    is_gold INTEGER NOT NULL DEFAULT 0 CHECK (is_gold IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS image_info (
    asset_id INTEGER PRIMARY KEY REFERENCES assets(id),
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    subject TEXT,
    action TEXT,
    subject_plural INTEGER NOT NULL DEFAULT 0 CHECK (subject_plural IN (0, 1)),
    text_in_image INTEGER NOT NULL DEFAULT 0 CHECK (text_in_image IN (0, 1)),
    profanity_in_image INTEGER NOT NULL DEFAULT 0 CHECK (profanity_in_image IN (0, 1)),
    tag_conf REAL
);

CREATE TABLE IF NOT EXISTS sound_info (
    asset_id INTEGER PRIMARY KEY REFERENCES assets(id),
    duration_ms INTEGER NOT NULL,
    onset_ms INTEGER,
    loudness_lufs REAL,
    -- Kept NOT NULL for existing rows. Loudnorm was rejected by the operator on
    -- 2026-09-30, so this column is written with the ORIGINAL path and is no
    -- longer read by the play loop. Serve reads assets.path / original_path.
    norm_path TEXT NOT NULL,
    -- Repo-relative path of the stored original file. Added additively; a
    -- pre-existing row is back-filled from assets.path by the migration.
    original_path TEXT,
    warn_long INTEGER NOT NULL DEFAULT 0 CHECK (warn_long IN (0, 1))
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY,
    asset_id INTEGER NOT NULL REFERENCES assets(id),
    axis TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('model', 'human', 'import')),
    run_id INTEGER,
    superseded INTEGER NOT NULL DEFAULT 0 CHECK (superseded IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS tag_runs (
    id INTEGER PRIMARY KEY,
    model_sha256 TEXT NOT NULL,
    mmproj_sha256 TEXT NOT NULL,
    llama_commit TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    rubric_hash TEXT NOT NULL,
    vocab_version TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    gate_ref TEXT NOT NULL,
    gold_scores TEXT,
    accepted INTEGER NOT NULL DEFAULT 0 CHECK (accepted IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS templates (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    patterns TEXT NOT NULL,
    render TEXT NOT NULL CHECK (render IN ('bar', 'impact')),
    requires TEXT NOT NULL DEFAULT '{}',
    max_chars INTEGER NOT NULL,
    max_lines INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS template_tags (
    template_id INTEGER NOT NULL REFERENCES templates(id),
    axis TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (template_id, axis)
);

CREATE TABLE IF NOT EXISTS pair_priors (
    id INTEGER PRIMARY KEY,
    feature_a TEXT NOT NULL,
    feature_b TEXT NOT NULL,
    mean REAL NOT NULL,
    strength REAL NOT NULL,
    note TEXT,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    authored_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS captions (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES assets(id),
    template_id INTEGER NOT NULL REFERENCES templates(id),
    variant INTEGER NOT NULL,
    slots TEXT NOT NULL,
    text TEXT NOT NULL,
    text_sha TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS compositions (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES assets(id),
    caption_id INTEGER NOT NULL REFERENCES captions(id),
    sound_id INTEGER REFERENCES assets(id),
    UNIQUE (image_id, caption_id, sound_id)
);

CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    archived_at TEXT,
    config_hash TEXT NOT NULL,
    weights_path TEXT,
    last_applied_impression INTEGER,
    note TEXT,
    rng_seed INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES profiles(id),
    topic_filter TEXT,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS impressions (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    profile_id INTEGER NOT NULL REFERENCES profiles(id),
    composition_id INTEGER REFERENCES compositions(id),
    image_id INTEGER REFERENCES assets(id),
    template_id INTEGER REFERENCES templates(id),
    variant INTEGER,
    sound_id INTEGER,
    topic_filter TEXT,
    served_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    resolved_at TEXT,
    policy TEXT CHECK (policy IN ('coverage', 'bandit', 'holdout')),
    block_id INTEGER,
    block_pos INTEGER,
    capped INTEGER NOT NULL DEFAULT 0 CHECK (capped IN (0, 1)),
    bonus_won INTEGER NOT NULL DEFAULT 0 CHECK (bonus_won IN (0, 1)),
    propensity REAL,
    pred_mean REAL,
    pred_bonus REAL,
    candidate_count INTEGER,
    seed INTEGER,
    config_hash TEXT NOT NULL,
    weights_version INTEGER,
    features_json TEXT,
    outcome TEXT NOT NULL DEFAULT 'pending'
        CHECK (outcome IN ('pending', 'rated', 'skipped', 'abandoned', 'flagged')),
    stars INTEGER CHECK (stars IS NULL OR stars BETWEEN 1 AND 5),
    reward REAL,
    train_value REAL,
    train_weight REAL,
    attribution TEXT,
    latency_ms INTEGER,
    cooldown_k INTEGER
);

CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY,
    impression_id INTEGER NOT NULL REFERENCES impressions(id),
    kind TEXT NOT NULL CHECK (kind IN (
        'bad_tags', 'unreadable_caption', 'sound_clash', 'unsafe', 'broken_asset'
    )),
    a_id INTEGER,
    b_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS pair_stats (
    profile_id INTEGER NOT NULL,
    level TEXT NOT NULL CHECK (level IN ('attr', 'item')),
    a_axis TEXT NOT NULL,
    a_value TEXT NOT NULL,
    b_axis TEXT NOT NULL,
    b_value TEXT NOT NULL,
    n INTEGER NOT NULL,
    mean REAL NOT NULL,
    sd REAL NOT NULL,
    built_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, level, a_axis, a_value, b_axis, b_value)
);

CREATE TABLE IF NOT EXISTS render_rejects (
    id INTEGER PRIMARY KEY,
    session_id INTEGER,
    image_id INTEGER,
    template_id INTEGER,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS feature_registry (
    name TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    lambda REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tags_live ON tags(asset_id, axis) WHERE superseded = 0;
CREATE INDEX IF NOT EXISTS idx_impr_profile ON impressions(profile_id, id);
CREATE INDEX IF NOT EXISTS idx_impr_image ON impressions(image_id, id);
CREATE INDEX IF NOT EXISTS idx_flags_open ON flags(a_id, b_id) WHERE resolved_at IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_comp_unique
    ON compositions(image_id, caption_id, ifnull(sound_id, 0));

CREATE VIEW IF NOT EXISTS pool_images AS
SELECT a.id, a.path, a.sha256, a.license_note, a.source, a.state
FROM assets a
WHERE a.kind = 'image'
  AND a.state = 'active'
  AND a.is_gold = 0
  AND length(trim(a.license_note)) > 0;

CREATE VIEW IF NOT EXISTS pool_sounds AS
SELECT a.id, a.path, a.sha256, a.license_note, s.duration_ms, s.onset_ms,
       s.loudness_lufs, s.original_path, s.warn_long
FROM assets a
JOIN sound_info s ON s.asset_id = a.id
WHERE a.kind = 'sound'
  AND a.state = 'active'
  AND a.is_gold = 0
  AND length(trim(a.license_note)) > 0;
"""


def _materialize(path: Path) -> None:
    """Build the schema on local disk, then copy. The project mount rejects
    a large DDL transaction with disk I/O; a copied file then opens in WAL."""
    import shutil
    import tempfile

    handle, name = tempfile.mkstemp(suffix=".sqlite")
    os_close = __import__("os").close
    os_close(handle)
    tmp = Path(name)
    try:
        built = sqlite3.connect(tmp)
        built.execute("PRAGMA foreign_keys = ON")
        built.execute("PRAGMA journal_mode = WAL")
        _apply_schema(built)
        built.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        built.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
        Path(str(tmp) + "-wal").unlink(missing_ok=True)
        Path(str(tmp) + "-shm").unlink(missing_ok=True)


def _schema_ready(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    probe = sqlite3.connect(path)
    try:
        row = probe.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'assets'"
        ).fetchone()
        return row is not None
    except sqlite3.DatabaseError:
        return False
    finally:
        probe.close()


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _schema_ready(path):
        _materialize(path)
    # Gradio runs clicks on worker threads. One connection, serialized by PlaySession.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing store up to SCHEMA_VERSION. Additive and idempotent."""
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if current < SCHEMA_VERSION:
        migrate_original_path(conn)


SCHEMA_VERSION = 2


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def migrate_original_path(conn: sqlite3.Connection) -> dict[str, int]:
    """Additive, idempotent. Adds sound_info.original_path and back-fills it.

    Loudnorm was rejected by the operator, so the play loop serves the stored
    original file. The column is added rather than repointing norm_path alone so
    that the measurement columns keep their meaning and the intent is recorded
    in the schema. norm_path is repointed to the original as well so it cannot
    keep pointing at a wav nobody serves any more.
    """
    result = {"added_column": 0, "backfilled": 0, "repointed": 0, "dropped_view": 0}
    if "sound_info" not in {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }:
        return result
    if "original_path" not in _columns(conn, "sound_info"):
        conn.execute("ALTER TABLE sound_info ADD COLUMN original_path TEXT")
        result["added_column"] = 1
    cur = conn.execute(
        """UPDATE sound_info
           SET original_path = (SELECT a.path FROM assets a WHERE a.id = sound_info.asset_id)
           WHERE original_path IS NULL OR trim(original_path) = ''"""
    )
    result["backfilled"] = int(cur.rowcount)
    cur = conn.execute(
        """UPDATE sound_info
           SET norm_path = original_path
           WHERE original_path IS NOT NULL
             AND trim(original_path) <> ''
             AND norm_path <> original_path"""
    )
    result["repointed"] = int(cur.rowcount)
    # The view was created with the old column list. SQLite will not redefine it.
    conn.execute("DROP VIEW IF EXISTS pool_sounds")
    result["dropped_view"] = 1
    _apply_schema(conn)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return result


def _apply_schema(conn: sqlite3.Connection) -> None:
    buffer: list[str] = []
    for line in SCHEMA.splitlines():
        buffer.append(line)
        if line.strip().endswith(";"):
            statement = "\n".join(buffer).strip()
            buffer = []
            if statement:
                conn.execute(statement)
    conn.commit()


def init_db(conn: sqlite3.Connection, vocab_path: Path | None = None) -> None:
    _apply_schema(conn)
    path = vocab_path or (CONFIG.parent / "vocab.json")
    rows = load_vocab_file(path) if path else VOCAB
    conn.executemany(
        """INSERT INTO vocab(axis, value, description, active, since_version)
           VALUES (?, ?, ?, 1, '1.1')
           ON CONFLICT(axis, value) DO UPDATE SET description = excluded.description""",
        rows,
    )
    conn.commit()


def live_tags(conn: sqlite3.Connection, asset_id: int) -> dict[str, list[str]]:
    rows = conn.execute(
        "SELECT axis, value FROM tags WHERE asset_id = ? AND superseded = 0",
        (asset_id,),
    ).fetchall()
    out: dict[str, list[str]] = {}
    for row in rows:
        out.setdefault(row["axis"], []).append(row["value"])
    return out
