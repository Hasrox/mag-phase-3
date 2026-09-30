#!/usr/bin/env bash
# Start llama-server, gate it, run the tagger, stop the server. VRAM is free during play.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="127.0.0.1"
PORT="8080"
N_GPU_LAYERS="all"
CTX="4096"
SERVER="${LLAMA_SERVER:-$ROOT/runtime/llama.cpp/build/bin/llama-server}"
MODEL="${1:-}"
MMPROJ="${2:-}"
if [[ -z "$MODEL" || -z "$MMPROJ" ]]; then
  echo "usage: tag_run.sh <model.gguf> <mmproj.gguf>" >&2
  exit 1
fi
if [[ ! -x "$SERVER" ]]; then
  echo "tag run fail: llama-server missing at $SERVER" >&2
  exit 1
fi
if ! "$SERVER" --help 2>&1 | grep -q "gpu-layers"; then
  echo "tag run fail: binary has no GPU layer flag; discarded" >&2
  exit 1
fi
LOG="$(mktemp)"
"$SERVER" --host "$HOST" --port "$PORT" --model "$MODEL" --mmproj "$MMPROJ" \
  --n-gpu-layers "$N_GPU_LAYERS" --parallel 1 --ctx-size "$CTX" >"$LOG" 2>&1 &
PID=$!
cleanup() { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; }
trap cleanup EXIT
MEM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
for _ in $(seq 1 30); do
  if grep -q "offloaded" "$LOG"; then
    break
  fi
  sleep 1
done
MEM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if curl -sf "http://${HOST}:${PORT}/health" >/dev/null; then
  echo "health ok" >> "$LOG"
fi
bash "$ROOT/scripts/gpu_gate.sh" --model "$MODEL" --mmproj "$MMPROJ" \
  --n-gpu-layers all --bind "$HOST" --host "$HOST" \
  --fixture-log "$LOG" --mem-before "$MEM_BEFORE" --mem-after "$MEM_AFTER" \
  --tps "${MIN_TPS:-15}" --pid "$PID" --commit "${LLAMA_COMMIT:-unpinned}"
RECORD="$(ls -1t "$ROOT/runtime/gates/"*.json | head -1)"
echo "gate passed; tagging with $RECORD"
python3 "$ROOT/scripts/tag_batch.py" --model "$MODEL" --mmproj "$MMPROJ" --gate-record "$RECORD" \
  --rubric "$ROOT/config/safety_rubric.txt"
