"""Gate record checks. The shell script is the only writer. This module refuses to tag otherwise."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
CPU_KV = re.compile(r"CPU KV buffer", re.IGNORECASE)
HOST_BUFFER = re.compile(r"(?:CPU|host)[^\n]{0,40}model buffer size\s*=\s*([0-9.]+)\s*MiB", re.IGNORECASE)
PROJECTOR = re.compile(r"(mmproj|clip|projector).{0,80}CUDA", re.IGNORECASE)


class GateRecord(BaseModel):
    # model_sha256 and mmproj_sha256 shadow pydantic's model_ namespace.
    model_config = ConfigDict(extra="ignore", protected_namespaces=())
    pass_: bool
    layer_string: str
    gpu_name: str
    model_sha256: str
    mmproj_sha256: str
    llama_commit: str
    server_pid: int
    memory_delta_bytes: int
    tokens_per_second: float
    utc: str
    run_id: str

    @property
    def ok(self) -> bool:
        return self.pass_


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_log(text: str, host_buffer_max_mib: float) -> list[str]:
    failures = []
    match = OFFLOAD.search(text)
    if not match:
        failures.append("missing offloaded N/N layers to GPU")
    else:
        left, right = int(match.group(1)), int(match.group(2))
        if left == 0 or left != right:
            failures.append(f"offload is {left}/{right}, not N/N")
    if CPU_KV.search(text):
        failures.append("CPU KV buffer allocation")
    for host in HOST_BUFFER.finditer(text):
        size = float(host.group(1))
        if size >= host_buffer_max_mib:
            failures.append(f"host model buffer {size} MiB")
    if not PROJECTOR.search(text):
        failures.append("multimodal projector backend is not CUDA")
    return failures


def load_record(path: Path) -> GateRecord:
    data = json.loads(path.read_text())
    if "pass" in data and "pass_" not in data:
        data["pass_"] = data["pass"]
    return GateRecord.model_validate(data)


def assert_gate_alive(record_path: Path, model: Path, mmproj: Path) -> GateRecord:
    if not record_path.exists():
        raise SystemExit("tagger refuse: no passing gate record")
    record = load_record(record_path)
    if not record.pass_:
        raise SystemExit("tagger refuse: gate record is not a pass")
    if sha256_file(model) != record.model_sha256 or sha256_file(mmproj) != record.mmproj_sha256:
        raise SystemExit("tagger refuse: model hash does not match the gate record")
    try:
        os.kill(record.server_pid, 0)
    except OSError as exc:
        raise SystemExit("tagger refuse: gate server pid is not alive") from exc
    return record
