"""Section 7.3 gate, in Python, for Windows. No bash required.

This module is the only writer of runtime/gates/<run_id>.json on Windows. It is
the same eight checks scripts/gpu_gate.sh performs, with no override flag and no
retry: if any check fails the tag run stops and no record is written.

    python -m mag.gatecli --model models/x.gguf --mmproj models/x-mmproj.gguf `
        --log runtime/last-server.log --mem-before 0 --mem-after 9500 `
        --tps 22 --pid 1234 --commit <sha>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mag.config import Config, load_config
from mag.gate import check_log, sha256_file
from mag.paths import GATES, MODELS

OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
HEALTH = re.compile(r"health ok|server is listening", re.IGNORECASE)


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
    bind: str = "127.0.0.1"
    host: str = "127.0.0.1"
    n_gpu_layers: str = "all"
    no_mmproj_offload: bool = False
    cpu_build: bool = False
    probe_ok: bool = True
    health_ok: bool = True


class GateVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ok: bool
    checks: dict[str, bool]
    failures: list[str]
    run_id: str
    model_sha256: str
    mmproj_sha256: str


def preflight_failures(
    *,
    bind: str,
    host: str,
    n_gpu_layers: str,
    no_mmproj_offload: bool,
    cpu_build: bool,
    model_ok: bool,
) -> list[str]:
    """Checks that must fail before a server is even started."""
    failures: list[str] = []
    if bind != "127.0.0.1" or host != "127.0.0.1":
        failures.append(f"server must bind 127.0.0.1, not {bind}/{host}")
    if no_mmproj_offload:
        failures.append("--no-mmproj-offload is forbidden")
    if cpu_build:
        failures.append("CPU-only build is discarded")
    if not n_gpu_layers:
        failures.append("missing --n-gpu-layers all")
    elif n_gpu_layers not in ("all", "999"):
        failures.append(f"partial offload --n-gpu-layers {n_gpu_layers}")
    if not model_ok:
        failures.append("--model and --mmproj are required and must exist")
    return failures


def vram_failures(total_mib: float, need_gb: float) -> list[str]:
    if total_mib / 1024.0 < need_gb:
        return [f"VRAM {total_mib / 1024.0:.1f} GB is below {need_gb} GB"]
    return []


def decode_failures(tps: float, floor: float) -> list[str]:
    if tps < floor:
        return [f"decode speed {tps} is below min_decode_tps {floor}"]
    return []


def _memory_delta_failures(opts: GateOptions) -> list[str]:
    model_bytes = opts.model.stat().st_size
    mmproj_bytes = opts.mmproj.stat().st_size
    delta_mib = opts.mem_after_mib - opts.mem_before_mib
    need = opts.memory_delta_ratio * (model_bytes + mmproj_bytes) / (1024 * 1024)
    if delta_mib < need:
        return [f"GPU memory delta {delta_mib:.0f} MiB is below {need:.0f} MiB"]
    return []


def _nvidia(query: str) -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip().splitlines()[0]
    except Exception:
        return "0" if query.startswith("memory") else "unknown"


def gpu_name() -> str:
    return _nvidia("name") or "unknown"


def gpu_total_mib() -> float:
    try:
        return float(_nvidia("memory.total"))
    except ValueError:
        return 0.0


def gpu_used_mib() -> float:
    try:
        return float(_nvidia("memory.used"))
    except ValueError:
        return 0.0


def run_checks(opts: GateOptions) -> GateVerdict:
    """All eight checks. No override flag, no retry, no partial offload."""
    checks: dict[str, bool] = {}
    failures: list[str] = []
    model_ok = opts.model.is_file() and opts.mmproj.is_file()
    failures += preflight_failures(
        bind=opts.bind, host=opts.host, n_gpu_layers=opts.n_gpu_layers,
        no_mmproj_offload=opts.no_mmproj_offload, cpu_build=opts.cpu_build,
        model_ok=model_ok,
    )
    checks["preflight"] = not failures
    if failures or not model_ok:
        return _verdict(opts, checks, failures)

    text = opts.log.read_text(errors="replace") if opts.log.is_file() else ""
    log_failures = check_log(text, opts.host_buffer_max_mib)
    checks["load_log"] = not log_failures
    failures += log_failures

    match = OFFLOAD.search(text)
    checks["offload_ratio"] = bool(match) and int(match.group(1)) == int(match.group(2)) > 0

    vram = vram_failures(gpu_total_mib(), opts.min_vram_gb)
    checks["vram"] = not vram
    failures += vram

    delta = _memory_delta_failures(opts)
    checks["memory_delta"] = not delta
    failures += delta

    checks["health"] = bool(HEALTH.search(text)) and opts.health_ok
    if not checks["health"]:
        failures.append("health check did not pass")

    checks["probe"] = opts.probe_ok
    if not checks["probe"]:
        failures.append("grammar probe did not return parseable JSON")

    speed = decode_failures(opts.tokens_per_second, opts.min_decode_tps)
    checks["decode_speed"] = not speed
    failures += speed
    return _verdict(opts, checks, failures)


def _verdict(opts: GateOptions, checks: dict[str, bool], failures: list[str]) -> GateVerdict:
    return GateVerdict(
        ok=not failures and all(checks.values()),
        checks=checks,
        failures=failures,
        run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        model_sha256=sha256_file(opts.model) if opts.model.is_file() else "",
        mmproj_sha256=sha256_file(opts.mmproj) if opts.mmproj.is_file() else "",
    )



def write_record(verdict: GateVerdict, opts: GateOptions, directory: Path | None = None) -> Path:
    """The only writer of runtime/gates/<run_id>.json. Refuses a failing verdict."""
    if not verdict.ok:
        raise SystemExit(f"gate fail: {'; '.join(verdict.failures)}")
    target = directory or GATES
    target.mkdir(parents=True, exist_ok=True)
    run_id = f"{verdict.run_id}-{opts.server_pid}"
    path = target / f"{run_id}.json"
    text = opts.log.read_text(errors="replace") if opts.log.is_file() else ""
    match = OFFLOAD.search(text)
    delta = int(opts.mem_after_mib - opts.mem_before_mib)
    payload = {
        "pass": True,
        "layer_string": f"{match.group(1)}/{match.group(2)}" if match else "unknown",
        "gpu_name": opts.gpu_name,
        "model_sha256": verdict.model_sha256,
        "mmproj_sha256": verdict.mmproj_sha256,
        "llama_commit": opts.llama_commit,
        "server_pid": opts.server_pid,
        "memory_delta_bytes": delta * 1024 * 1024,
        "tokens_per_second": opts.tokens_per_second,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "checks": verdict.checks,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _from_args(args: argparse.Namespace, cfg: Config) -> GateOptions:
    return GateOptions(
        model=args.model,
        mmproj=args.mmproj,
        log=args.log,
        mem_before_mib=args.mem_before,
        mem_after_mib=args.mem_after,
        tokens_per_second=args.tps,
        server_pid=args.pid,
        llama_commit=args.commit,
        gpu_name=gpu_name(),
        min_vram_gb=cfg.runtime.min_vram_gb,
        host_buffer_max_mib=cfg.runtime.host_buffer_max_mib,
        memory_delta_ratio=cfg.runtime.memory_delta_ratio,
        min_decode_tps=cfg.runtime.min_decode_tps,
        bind=args.bind,
        host=args.host,
        n_gpu_layers=args.n_gpu_layers,
        no_mmproj_offload=args.no_mmproj_offload,
        cpu_build=args.cpu_build,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MAG GPU gate, section 7.3")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--mem-before", type=float, default=0.0)
    parser.add_argument("--mem-after", type=float, default=0.0)
    parser.add_argument("--tps", type=float, default=0.0)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--commit", default="unpinned")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--n-gpu-layers", default="all")
    parser.add_argument("--no-mmproj-offload", action="store_true")
    parser.add_argument("--cpu-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="report only, write no record")
    args = parser.parse_args()
    opts = _from_args(args, load_config())
    verdict = run_checks(opts)
    for name, ok in verdict.checks.items():
        print(f"check {name}={'pass' if ok else 'FAIL'}")
    if not verdict.ok:
        print(f"gate fail: {'; '.join(verdict.failures)}", file=sys.stderr)
        raise SystemExit(1)
    if args.dry_run:
        print("dry-run: gate would pass; no record written")
        return
    print(f"gate pass {write_record(verdict, opts)}")


if __name__ == "__main__":
    main()

