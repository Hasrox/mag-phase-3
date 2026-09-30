# Implementation Plan

## Overview

Close Phase 4 on the operator's Windows machine and open Phase 2 honestly: serve the
**original** sound file instead of the loudnorm wav, stop the same sound repeating, stop
writing loudnorm copies, clear the fake human placeholder tags that would make a real tag
run a no-op, install the Gemma files and model card, build `llama.cpp` with `GGML_CUDA=ON`
behind a Windows-native gate, run the gate, gold-score, and batch tag, then record S-1, run
audit A-1 to A-8, and flip `config/phase4_lock.toml` only when those are actually true.

The Gradio screen is **not** redesigned in this plan (operator decision). `pick_candidate`
and `predict` in `mag/serve.py` are **untouched** so Phase 5 still has a clean seam. No
Python dependency is added; the allow-list plus the `huggingface_hub<1`, `starlette<1`,
`fastapi<0.116` pins stay exactly as they are.

Ordering principle: the play loop must be honest before the ranker learns anything. Part A
lands and is pushed before Part C starts, because a tag run over 600 images takes hours and
must not be repeated to fix a serve-path bug.

## Investigation findings

| Finding | Evidence |
| --- | --- |
| `main` was `6130a65` with one uncommitted change | `git diff HEAD` showed only the `time.strftime` to `datetime.now(timezone.utc)` change |
| Loudnorm wav is what plays | `mag/serve.py:318-327` `_sound_path` read `sound_info.norm_path`; 40 rows pointed at `assets\sounds\norm\<sha>.wav`; log line `sound=audio 2a1a36e0....wav` |
| `recent_sounds` never passed | `mag/serve.py:369-372` passed `previous_sound` and `blocked_sounds` only; `mag/composer.py:204` already accepted `recent_sounds` |
| All images carry fake `human` tags | `mag/inbox.py:75-88` via `write_human_tags`; DB shows `human` on every axis, `tag_runs` has 0 rows |
| A model tag run would write nothing | `mag/importing.py:275-281` and `mag/promote.py:82-87` skip any axis with a live `human` tag; `scripts/tag_batch.py:51` selects `state='new'` (0 rows) |
| Renderer has no font | `assets/fonts/` did not exist; `mag/render.py:18-22` fell back to `ImageFont.load_default()`, ignoring the requested size |
| Render cache keyed by RNG seed | `mag/serve.py:464` used `cache/{seed}{suffix}`; `mag/render.py:88` `cache_path()` existed but was never called |
| Tests failed | `test_phase1` raised no `ImportRejected` for an over-cap sound (`accept_over_cap=true`); `test_gate` raised `FileNotFoundError` for `bash` |
| Model not wired | `models/` held only `model-card.txt`; GGUFs in `C:\Users\Hasan\Downloads` (9.11 GB + 0.15 GB) |
| Build toolchain absent | no `cmake`, `ninja`, MSVC, or CUDA Toolkit; no `runtime/llama.cpp`. GPU present: RTX 4080, 16376 MiB, driver 610.88 |
| Gold set absent | `data/gold/` held only `probes.json`; `is_gold` count 0 |
| No exit CLIs | `mag/audit.py` had no `main`; `mag/phase4.record_s1` had no script wrapper |
| Sound labels degenerate | `mag/inbox.py:68-72` read class from a folder name; all 40 sounds sit in `assets/sounds/` root so every `sound_class` is `other` |

## Types

```python
# mag/importing.py — new
@dataclass
class SoundImport:
    asset_id: int
    original_path: str            # repo-relative path to the stored original file
    duration_ms: int
    onset_ms: int
    loudness_lufs: float | None   # measurement only, never a gate
    warn_long: int                # 0/1, informational
```

```python
# mag/gatecli.py — new (pydantic, mirrors mag/gate.py)
class GateOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Path
    mmproj: Path
    log: Path
    mem_before_mib: float
    mem_after_mib: float
    tokens_per_second: float
    server_pid: int
    llama_commit: str
    gpu_name: str = "unknown"
    min_vram_gb: float = 16.0
    host_buffer_max_mib: float = 64.0
    memory_delta_ratio: float = 0.9
    min_decode_tps: float = 15.0

class GateVerdict(BaseModel):
    ok: bool
    checks: dict[str, bool]
    failures: list[str]
    run_id: str
    model_sha256: str
    mmproj_sha256: str
```

```python
# mag/placeholders.py — new (pydantic)
class PlaceholderSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    created_at: str
    asset_count: int
    tag_count: int
    assets: list[PlaceholderAsset]

class PlaceholderAsset(BaseModel):
    asset_id: int
    path: str
    subject: str | None
    action: str | None
    tags: dict[str, list[str]]
```

DB shape change, additive only:

```sql
-- mag/db.py SCHEMA, sound_info gains one nullable column
    original_path TEXT,   -- repo-relative path of the stored original file
-- norm_path stays TEXT NOT NULL and is written with the original path so the
-- column never dangles. loudness_lufs keeps the measurement.
```

`norm_path` is `NOT NULL` and 40 rows exist. Rather than rebuild the table to drop
`NOT NULL`, we add `original_path`, back-fill it from `assets.path`, and repoint
`norm_path` at the same value.

## Files

### New files

| Path | Purpose |
| --- | --- |
| `scripts/migrate_original_path.py` | Idempotent additive migration; prints row counts; safe to run twice |
| `scripts/clear_placeholder_tags.py` | Snapshots inbox placeholders, then supersedes them and sets images to `state='new'`; `--restore` puts them back; `--dry-run` prints |
| `scripts/label_sounds.py` | Bulk operator labelling from CSV `id,class,intensity,topics,fit_emotion`; `--dry-run` first |
| `scripts/make_gold_manifest.py` | Deterministic stratified sample of 40 active images into `data/gold/manifest.json` with empty label fields |
| `scripts/score_gold.py` | Tags the 40 gold images twice, scores S-4/S-5, records `gold_scores`; `--accept` calls `accept_run` |
| `scripts/audit.py` | CLI over `mag.audit.run_audit`; exits 1 if any A-* query returns rows |
| `scripts/record_s1.py` | `--from-db` or `--stars`; calls `mag.phase4.record_s1`; writes `config/s1_record.toml` |
| `scripts/status.py` | Prints phase 1, 2, 3 and 4 status blocks in one run |
| `mag/gatecli.py` | Windows-native gate; all eight section-7.3 checks; the only writer of `runtime/gates/<run_id>.json` on Windows |
| `mag/gatetest.py` | Eight negative cases in Python so `tests/test_gate.py` runs without bash |
| `mag/placeholders.py` | Snapshot, clear and restore logic for inbox placeholder tags |
| `scripts/tag_run.ps1` | PowerShell driver: start server, capture log, sample VRAM, probe, gate, tag, stop |
| `scripts/build_llama.ps1` | Clone llama.cpp at a pinned commit, configure `-DGGML_CUDA=ON -DGGML_NATIVE=OFF`, build `llama-server` |
| `docs/phase4_handover.md` | Operator runbook and the silence rule |
| `tests/test_phase4.py` | Phase 4 exit rules on a synthetic store |
| `tests/test_render.py` | Font resolution and composition-hash cache |
| `tests/test_placeholders.py` | Snapshot, clear, restore round trip |
| `tests/test_gate_negatives.py` | Runs `mag.gatetest` without bash |
| `models/.gitkeep` | Keeps `models/` tracked while `models/*.gguf` stays ignored |

### Modified files

| Path | Change |
| --- | --- |
| `mag/serve.py` | `serve_sound_path` reads `assets.path`; `candidates_for` passes `recent_sounds`; `score_and_pick` uses `render.cache_path`; missing sound files drop out of candidates |
| `mag/composer.py` | `sound_candidates` verifies each file exists and drops missing ones |
| `mag/importing.py` | `import_sound` stops calling `_normalize` and stops writing `assets/sounds/norm/`; writes `original_path`; keeps duration, onset and loudness as measurement |
| `mag/db.py` | `sound_info` gains `original_path`; `pool_sounds` exposes it; `connect()` runs the additive migration once |
| `mag/render.py` | `_font` resolves `font_file` then `font_fallbacks` then raises; `cache_path` is actually used |
| `mag/inbox.py` | Docstring states placeholder tags are provisional and must be cleared before a tag run |
| `mag/promote.py` | `write_model_tags` sets `state='tagged'` so a batch run resumes |
| `scripts/tag_batch.py` | Selects `state IN ('new','tagged')`; prints progress per image |
| `tests/test_phase1.py` | Over-cap sound is accepted, not rejected; assert no norm file and a resolvable `original_path` |
| `tests/test_gate.py` | `bash` test becomes `skipUnless`; add `test_python_negatives` |
| `config/mag.toml` | Add `font_fallbacks` and `write_normalized_copy = false` |
| `config/phase2_lock.toml` | Add `model_files_moved`, `gate_passed`, `s4_claimed`, `s5_claimed`, `run_accepted`, all false until recorded |
| `config/phase4_lock.toml` | Stays `locked=false` until audit and S-1 are true; records `sound_source` and `recent_sounds_exclusion` |
| `models/model-card.txt` | Fill name, source, license, quant, parameter_count, mmproj; thresholds and S-4/S-5 stay blank |
| `.gitignore` | Add `runtime/llama.cpp-commit.txt`, `data/placeholder_tags.json`, `assets/fonts/*.ttf` |
| `requirements.in` | **No change** |

### Deleted files

- `mag/importing.py::_normalize`, `_mean_volume_db`, `_ffmpeg_json` are removed.
- `assets/sounds/norm/*.wav` (40 gitignored files) are deleted from disk after the migration
  and a spot-checked mp3 serve.
- No repository files are moved.

## Functions

### New

| Function | File | Purpose |
| --- | --- | --- |
| `migrate_original_path` | `scripts/migrate_original_path.py` | `(conn) -> dict[str,int]` additive, idempotent |
| `snapshot_placeholders` | `mag/placeholders.py` | `(conn) -> PlaceholderSnapshot` |
| `clear_placeholders` | `mag/placeholders.py` | `(conn, snapshot, *, dry_run) -> int` |
| `restore_placeholders` | `mag/placeholders.py` | `(conn, snapshot) -> int` |
| `label_sounds_from_csv` | `scripts/label_sounds.py` | `(conn, path, *, dry_run) -> list[tuple[int,str]]` |
| `build_gold_manifest` | `scripts/make_gold_manifest.py` | `(conn, n=40, seed=1103) -> list[dict]` |
| `run_checks` | `mag/gatecli.py` | `(opts: GateOptions) -> GateVerdict` |
| `write_record` | `mag/gatecli.py` | `(verdict, path) -> Path` |
| `negative_cases` | `mag/gatetest.py` | `() -> list[tuple[str,bool]]` |
| `serve_sound_path` | `mag/serve.py` | `(conn, sound_id) -> str`, replaces `_sound_path` |
| `recent_sound_ids` | `mag/serve.py` | `(recent, limit=8) -> list[int]` |

### Modified

| Function | File | Change |
| --- | --- | --- |
| `_sound_path` | `mag/serve.py:318` | Deleted, replaced by `serve_sound_path` |
| `candidates_for` | `mag/serve.py:330` | Build `recent_sounds`; pass it in; drop missing files |
| `score_and_pick` | `mag/serve.py:397` | `dest = cache_path(cache, image_sha, text, layout)` |
| `_font` | `mag/render.py:18` | Resolve configured font then fallbacks, else raise |
| `import_sound` | `mag/importing.py:194` | No normalize, no norm file, write `original_path` |
| `write_model_tags` | `mag/promote.py:53` | Set `state='tagged'` |
| `promote_asset` | `mag/promote.py:120` | Treat `tagged` as a valid current state |

### Removed

- `mag/importing.py::_normalize`, `_mean_volume_db`, `_ffmpeg_json`.
- `mag/sound.py::import_decision` is left in place, unused, and flagged for a later cleanup.

## Classes

No new or removed classes on the play side. `ServeResult`, `Caption`, `SoundPick` and
`Composition` keep their current fields; `ServeResult.sound_path` already carries the
resolved path. `PlaySession` is unchanged. New support types are `SoundImport`,
`GateOptions`, `GateVerdict`, `PlaceholderSnapshot` and `PlaceholderAsset`.

## Dependencies

**Python: no change.** `requirements.in` stays exactly as pinned. `mag/gatecli.py`,
`mag/gatetest.py` and `mag/placeholders.py` use only `pydantic`, `hashlib`, `json`,
`subprocess` and `httpx`. No `llama-cpp-python`.

**Non-Python toolchain, installed outside the venv:**

| Tool | Why | Version |
| --- | --- | --- |
| Visual Studio Build Tools 2022, C++ workload | `cl.exe` | 17.x latest |
| CUDA Toolkit | `nvcc` for `GGML_CUDA=ON` | 12.x; driver 610.88 reports UMD 13.3 |
| CMake | build system | >= 3.22 |
| Ninja | faster configure | latest |

**Model files** copied from `C:\Users\Hasan\Downloads\` into `models\` (not moved):
`gemma-4-12B-it-uncensored-heretic.Q6_K.gguf` (9.11 GB) and
`gemma-4-12B-it-uncensored-heretic.mmproj-Q8_0.gguf` (0.15 GB). `models/*.gguf` is ignored.
Disk: 24.3 GB free against 9.3 GB of model plus a build tree.

**PowerShell** has no `export`; use `$env:GRADIO_ANALYTICS_ENABLED = "False"`. The process
sets it itself in `mag/ui.py` and `scripts/serve_ui.py`, so the screen is safe regardless.

## Testing

### Existing tests to fix

1. `tests/test_phase1.py::test_exit_rules` — the `assertRaises(ImportRejected)` around a
   5.2 s sound is stale because `accept_over_cap = true`. Replace with: accepted, no file
   under `assets/sounds/norm`, `original_path` resolves to the stored original.
2. `tests/test_gate.py::test_shell_negatives_exit_1` — `FileNotFoundError` because bash is
   not on PATH. Add `skipUnless(shutil.which("bash"))` and add `test_python_negatives`.

### New tests

`tests/test_phase4.py` on a throwaway store with 25 images and 6 ffmpeg-made sounds:
`sound_path` ends in the original suffix and never contains `sounds\norm`; no sound repeats
inside any window of 8 serves; consecutive impressions never share a `sound_id`; silence
returns `""` and the deliberate-silence note while a missing file returns a different note;
the render lands on the composition-hash path and is reused; `status_line` leaks no policy
word; `run_audit` returns `[]` after 12 serves; a flagged image leaves the pool.

`tests/test_render.py`: the font size requested is the size used; a missing font raises
`RenderRejected` rather than falling back to `load_default()`; `cache_path` is stable for
equal inputs and changes with text or layout; an oversized caption still raises.

`tests/test_placeholders.py`: snapshot, clear, assert superseded and `state='new'` and a
shrunken pool, then restore and assert the tags and states come back.

`tests/test_gate_negatives.py`: `mag.gatetest.negative_cases()`, every case must fail.

### Validation

Run the unit suite; `scripts/status.py`; the placeholder dry run and real run; the model
copy; `scripts/build_llama.ps1`; `python -m mag.gatecli`; a negative gate run with `-ngl 10`
that must exit 1 and write no record; gold manifest and scoring; batch tagging; a
`serve_ui.py` session checked in `data/mag.log` for `.mp3` paths and no repeat inside 8
serves; `scripts/record_s1.py`; `scripts/audit.py`; then flip `config/phase4_lock.toml`.

## Implementation Order

Part A, play-side truth, pushed after each step:

1. Push the pending `datetime` timestamp fix.
2. `serve_sound_path` in `mag/serve.py`, reading `assets.path`.
3. Schema addition plus `scripts/migrate_original_path.py`; run it against the operator DB.
4. Stop loudnorm in `mag/importing.py`; add `write_normalized_copy = false`.
5. Wire `recent_sounds` through `candidates_for`; drop missing sound files. Must land with
   step 2.
6. Renderer font resolution and `cache_path`; delete the stale render cache.
7. Fix the two failing tests; add `tests/test_phase4.py` and `tests/test_render.py`.
8. `scripts/audit.py`, `scripts/record_s1.py`, `scripts/status.py`, `scripts/label_sounds.py`;
   run the audit against the operator DB and record what it reports.

Part B, make a tag run possible:

9. `mag/placeholders.py`, `scripts/clear_placeholder_tags.py`, `tests/test_placeholders.py`;
   dry run, then real run.
10. Copy the two GGUFs into `models/`; fill the model card fields; set
    `model_files_moved = true` only.
11. `scripts/make_gold_manifest.py` produces the 40-image manifest for the operator to label.
12. `mag/gatecli.py`, `mag/gatetest.py`, `scripts/build_llama.ps1`, `scripts/tag_run.ps1`,
    `tests/test_gate_negatives.py`; run the negative tests before any model load.

Part C, GPU:

13. Install VS Build Tools 2022, CUDA Toolkit 12.x, CMake, Ninja.
14. Build llama.cpp; assert the binary reports CUDA; write the commit file.
15. Run the gate. If it fails, stop. Change quant or model, never layers.
16. Gold labels filled, `scripts/score_gold.py`, accept only when S-4 and S-5 pass.
17. Batch tag, resumable, with progress; then the review queue.
18. Confirm the pool is back and captions read from model tags; label sounds.

Part D, close Phase 4:

19. Operator session of at least 10 generate-rate cycles; check the log.
20. `scripts/record_s1.py`; record the verdict either way.
21. `scripts/audit.py` clean; then set `locked = true` and push.
22. `docs/phase4_handover.md`, final `scripts/status.py` output, and the documented Phase 5
    seam.

## Assumptions and limitations

- Building CUDA llama.cpp is the longest pole and the only step that cannot be fully
  guaranteed: no MSVC, no CUDA toolkit, no cmake, 24 GB free against a 9.1 GB model. Steps
  13 to 15 fail loudly and early rather than half-succeed.
- Gold-40 hand labelling is operator labour. Steps 11 and 16 build everything around it, but
  S-4 cannot be claimed until the labels exist. Nothing here fabricates them.
- The stale `test_phase1` assertion and the missing `bash` are pre-existing failures, not
  regressions; fixing them is in scope so a red suite does not hide real failures.
- `pick_candidate`, `predict`, the `ui.py` layout and `requirements.in` are untouched. Phase 5
  and the UI product pass stay available as separate, clean changes.



