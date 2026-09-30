# Phase 4 handover

State as of 2026-09-30. Read this before changing anything in the play loop.

## What was wrong, and what fixed it

**The same wav kept playing.** `mag/serve.py::_sound_path` read
`sound_info.norm_path`, which pointed at `assets/sounds/norm/<sha>.wav`. It now
reads `assets.path`. Two things had to land together: the serve path, and
`recent_sounds` being passed into `sound_candidates`. `composer.py` already
excluded recent sounds from its prior slots, but `serve.py` never supplied the
argument, so with placeholder tags giving every image one prior the same file
won every other draw.

**A second, quieter bug behind it.** Every `assets.path` for a sound pointed at
`assets/sounds/<sha256>.mp3`, which does not exist. The operator's files are on
disk under their original names (`akh.mp3` and so on) and all 40 match the
store's sha256. The loudnorm wav was the only path that resolved, which is why
fixing the serve path alone did not change what you heard.
`scripts/repair_sound_paths.py` repoints the rows; it has been run.

**Captions were not model output.** `mag/inbox.py` writes placeholder tags with
`source='human'` so a dropped file enters the pool. Both `importing._write_tag`
and `promote._write_model` refuse to overwrite a live human tag, and
`tag_batch.py` only selected `state='new'`. A real tag run against that store
would have written nothing at all: 295 images, 1770 tags, all fake. Those tags
are now cleared and the images are back to `state='new'`, which is why the pool
is 0.

**The caption bar.** `assets/fonts/` was empty, so `render._font` fell back to
`ImageFont.load_default()`, which ignores the requested size. Every caption drew
at one fixed tiny size. It now resolves `font_file` then `font_fallbacks` and
raises if neither exists. The render cache is keyed by composition hash rather
than the per-serve RNG seed, so stale files are replaced instead of accumulating.

## Rules that are not negotiable

- **Silence is not a bug.** `sound_id IS NULL` is a legal draw and the ranker is
  meant to learn it. A *missing file* is a bug. The status line says which.
- **Loudnorm is not wanted.** Import measures duration, onset and loudness for
  reporting. It never re-encodes and never gates on a measurement. The 38 wav
  files under `assets/sounds/norm/` have been deleted.
- **No weights on the play side.** `score_and_pick` does not open a socket, load
  a model, or update a weight. Phase 5 replaces `pick_candidate` and `predict`
  only.
- **The status line never shows the policy.** Showing whether the last item was
  a holdout unblinds the rater and biases the S-2 comparison.
- **No partial offload.** If the gate fails, change the quant or the model. Never
  reduce layers, never fall back to CPU.

## Run order on this machine

```powershell
$env:GRADIO_ANALYTICS_ENABLED = "False"   # the process also sets this itself
python scripts/status.py                  # where every phase stands
python scripts/audit.py                   # A-1..A-8; exits 1 on any row
python scripts/serve_ui.py                # 127.0.0.1:7860, no share
```

ffmpeg and ffprobe must be on `PATH` or sounds are not imported. Drop files in
`assets/` or `assets/inbox/` and restart; there is no manifest.

## What is still open, in order

1. **Build llama.cpp with CUDA.** Nothing is tagged. Install VS 2022 Build Tools
   (C++ workload), the CUDA Toolkit 12.x, and CMake, then
   `powershell -ExecutionPolicy Bypass -File scripts/build_llama.ps1`.
2. **Run the gate.**
   `powershell -ExecutionPolicy Bypass -File scripts/tag_run.ps1 -Model ... -Mmproj ...`
   If it fails, stop. Do not tag with fewer layers.
3. **Label the gold set.** `data/gold/manifest.json` has 40 sampled images with
   empty `labels`. Fill them in by hand, then
   `python scripts/import_store.py --gold data/gold/manifest.json`.
   The sample is not stratified yet because the store has no real tags; re-run
   `scripts/make_gold_manifest.py --force` after a tag run.
4. **Score and accept.** `python scripts/score_gold.py`, then `--accept` only if
   S-4 and S-5 both pass.
5. **Batch tag.** `python scripts/tag_batch.py ...`. It resumes: an interrupted
   run picks up where it stopped.
6. **Label the sounds.** `python scripts/label_sounds.py --template` prints a CSV
   to fill. All 40 sounds currently have `sound_class = other` and no
   `fit_emotion`, so the pairing features are degenerate.
7. **Ten manual generate-rate cycles, then S-1.**
   `python scripts/record_s1.py --from-db`. S-1 failing does not block Phase 5.
8. **Then** set `locked = true` in `config/phase4_lock.toml`.

## To undo the placeholder clear

```powershell
python scripts/clear_placeholder_tags.py --restore data\placeholder_tags.json
```

That puts the 288 images back in the pool with their filename-subject tags. It
also puts back the no-op-for-the-tagger state, so only do it if you need the
screen to serve something before the model is ready.

## The Phase 5 seam

`pick_candidate(policy, candidates, rng) -> dict` and
`predict(features) -> (mean, bonus)` in `mag/serve.py`. Phase 5 replaces those
two bodies and nothing else. The candidate generator in `candidates_for`, the
three-stage uniform draw, the propensity formula, and the UI all stay as they
are.
