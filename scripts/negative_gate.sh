#!/usr/bin/env bash
# Phase 0 exit: each negative test exits 1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$ROOT/fixtures/gate_logs/pass.log"
fail_one() {
  if bash "$ROOT/scripts/gpu_gate.sh" "$@"; then
    echo "expected exit 1: $*" >&2
    exit 1
  fi
  echo "exit 1 ok: $1"
}
common=(--model "$MODEL" --mmproj "$MODEL" --mem-before 0 --mem-after 999999 --tps 20)
fail_one "${common[@]}" --fixture-log "$MODEL"
fail_one "${common[@]}" --bind 0.0.0.0 --n-gpu-layers all --fixture-log "$MODEL"
fail_one "${common[@]}" --n-gpu-layers 10 --fixture-log "$MODEL"
fail_one "${common[@]}" --n-gpu-layers all --no-mmproj-offload --fixture-log "$MODEL"
fail_one "${common[@]}" --n-gpu-layers all --cpu-build --fixture-log "$MODEL"
fail_one "${common[@]}" --n-gpu-layers all --fixture-log "$ROOT/fixtures/gate_logs/no_offload.log"
echo "negative tests exited 1"
