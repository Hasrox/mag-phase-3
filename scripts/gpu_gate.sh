#!/usr/bin/env bash
# MAG-PRD-1.1 section 7.3. All eight checks must pass. This script is the only
# writer of runtime/gates/<run_id>.json. There is no override flag.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GATES="$ROOT/runtime/gates"
mkdir -p "$GATES"

HOST="127.0.0.1"
PORT="8080"
MIN_VRAM_GB="16"
MIN_TPS="15"
HOST_MAX_MIB="64"
MEM_RATIO="0.9"
FIXTURE_LOG=""
MODEL=""
MMPROJ=""
COMMIT=""
MEM_BEFORE=""
MEM_AFTER=""
TPS=""
PID_VALUE="0"
BIND="127.0.0.1"
N_GPU_LAYERS=""
NO_MMPROJ_OFFLOAD="0"
CPU_BUILD="0"
EXPECT_FAIL="0"

fail() {
  echo "gate fail: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --mmproj) MMPROJ="$2"; shift 2 ;;
    --commit) COMMIT="$2"; shift 2 ;;
    --fixture-log) FIXTURE_LOG="$2"; shift 2 ;;
    --mem-before) MEM_BEFORE="$2"; shift 2 ;;
    --mem-after) MEM_AFTER="$2"; shift 2 ;;
    --tps) TPS="$2"; shift 2 ;;
    --pid) PID_VALUE="$2"; shift 2 ;;
    --bind) BIND="$2"; shift 2 ;;
    --n-gpu-layers) N_GPU_LAYERS="$2"; shift 2 ;;
    --no-mmproj-offload) NO_MMPROJ_OFFLOAD="1"; shift ;;
    --cpu-build) CPU_BUILD="1"; shift ;;
    --min-tps) MIN_TPS="$2"; shift 2 ;;
    --expect-fail) EXPECT_FAIL="1"; shift ;;
    *) fail "unknown argument $1" ;;
  esac
done

# Preflight. These are the negative tests that must exit 1 before a server starts.
[[ "$BIND" == "127.0.0.1" && "$HOST" == "127.0.0.1" ]] || fail "server must bind 127.0.0.1, not ${BIND}/${HOST}"
[[ "$NO_MMPROJ_OFFLOAD" == "0" ]] || fail "--no-mmproj-offload is forbidden"
[[ "$CPU_BUILD" == "0" ]] || fail "CPU-only build is discarded"
[[ -n "$N_GPU_LAYERS" ]] || fail "missing --n-gpu-layers all"
[[ "$N_GPU_LAYERS" == "all" || "$N_GPU_LAYERS" == "999" ]] || fail "partial offload --n-gpu-layers ${N_GPU_LAYERS}"
[[ -n "$MODEL" && -n "$MMPROJ" ]] || fail "--model and --mmproj are required"
[[ -f "$MODEL" && -f "$MMPROJ" ]] || fail "model or mmproj missing"

LOG=""
if [[ -n "$FIXTURE_LOG" ]]; then
  [[ -f "$FIXTURE_LOG" ]] || fail "fixture log missing"
  LOG="$(cat "$FIXTURE_LOG")"
else
  command -v nvidia-smi >/dev/null || fail "nvidia-smi missing"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
  python3 - "$VRAM_MIB" "$MIN_VRAM_GB" <<'PY' || fail "VRAM below 16 GB"
import sys
mib = float(sys.argv[1]); need = float(sys.argv[2])
sys.exit(0 if mib / 1024.0 >= need else 1)
PY
  fail "live server log was not passed; tag_run.sh must start the server and pass --fixture-log of that run, or this host has no gate log"
fi

# Log checks 2, 3, 4 live in Python so the regexes have fixtures.
export MAG_GATE_LOG="$LOG"
python3 - <<PY || fail "log checks 2-4"
import os, sys
sys.path.insert(0, "$ROOT")
from mag.gate import check_log
fails = check_log(os.environ["MAG_GATE_LOG"], float("$HOST_MAX_MIB"))
if fails:
    print("; ".join(fails), file=sys.stderr)
    sys.exit(1)
PY

# Check 5, memory delta. Fixture mode passes the two samples.
[[ -n "$MEM_BEFORE" && -n "$MEM_AFTER" ]] || fail "memory samples missing"
MODEL_BYTES=$(wc -c < "$MODEL")
MMPROJ_BYTES=$(wc -c < "$MMPROJ")
python3 - "$MEM_BEFORE" "$MEM_AFTER" "$MODEL_BYTES" "$MMPROJ_BYTES" "$MEM_RATIO" <<'PY' || fail "GPU memory delta"
import sys
before, after, model, mmproj, ratio = map(float, sys.argv[1:])
delta = after - before
need = ratio * (model + mmproj)
sys.exit(0 if delta >= need else 1)
PY

# Check 6 is health. Fixture mode records it as a required token in the log.
grep -q "health ok" <<<"$LOG" || fail "health not ok"

# Check 7, probe.
grep -q "probe json ok" <<<"$LOG" || fail "probe did not return valid JSON"

# Check 8, decode speed.
[[ -n "$TPS" ]] || fail "decode speed missing"
python3 - "$TPS" "$MIN_TPS" <<'PY' || fail "decode speed below min_decode_tps"
import sys
sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
MODEL_SHA=$(sha256sum "$MODEL" | awk '{print $1}')
MMPROJ_SHA=$(sha256sum "$MMPROJ" | awk '{print $1}')
UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LAYER=$(python3 - <<PY
import os, re
m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", os.environ["MAG_GATE_LOG"])
print(f"{m.group(1)}/{m.group(2)}")
PY
)
DELTA=$((MEM_AFTER - MEM_BEFORE))
GPU_NAME="${GPU_NAME:-fixture}"
cat > "$GATES/${RUN_ID}.json" <<EOF
{
  "pass": true,
  "layer_string": "$LAYER",
  "gpu_name": "$GPU_NAME",
  "model_sha256": "$MODEL_SHA",
  "mmproj_sha256": "$MMPROJ_SHA",
  "llama_commit": "${COMMIT:-unpinned}",
  "server_pid": $PID_VALUE,
  "memory_delta_bytes": $DELTA,
  "tokens_per_second": $TPS,
  "utc": "$UTC",
  "run_id": "$RUN_ID"
}
EOF
echo "gate pass $GATES/${RUN_ID}.json"
exit 0
