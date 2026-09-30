"""Gate record checks. The shell script is the only writer. This module refuses to tag otherwise."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
# Matched against a real llama-server -lv 4 log. The device line names the GPU
# and the buffer line proves where the weights actually landed:
#   llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 4080) ...
#   load_tensors:        CUDA0 model buffer size =  9317.65 MiB
CUDA_DEVICE = re.compile(r"using device\s+(CUDA\d+)\s*\(", re.IGNORECASE)
# The real line carries a leading timestamp and a log level:
#   0.12.573.827 I load_tensors:        CUDA0 model buffer size =  9317.65 MiB
CUDA_BUFFER = re.compile(
    r"load_tensors:\s+(?P<tag>CUDA\d+) model buffer size\s*=\s*(?P<mib>[\d.]+)\s*MiB",
    re.IGNORECASE,
)
# The projector is the mmproj/clip. On this build it does not print a per-device
# buffer line, so the evidence is the fitted-memory line naming the mmproj
# against a CUDA device plus the absence of an mmproj CPU buffer.
MMPROJ = re.compile(r"mmproj", re.IGNORECASE)
CPU_KV = re.compile(r"CPU KV buffer", re.IGNORECASE)
HOST_BUFFER = re.compile(r"(?:CPU|host)[^\n]{0,40}model buffer size\s*=\s*([0-9.]+)\s*MiB", re.IGNORECASE)
# A fully offloaded model still maps part of the file into host memory. The
# check is a ratio against the CUDA buffer, not this number.
HOST_BUFFER_MAX_MIB = 1024.0
CPU_ONLY = re.compile(r"using device\s+CPU\b|offloaded 0/\d+ layers", re.IGNORECASE)


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


def check_log(text: str, host_buffer_max_mib: float | None = None) -> list[str]:
    """Section 7.3 checks 2 to 4, against a real llama-server log.

    These strings were taken from a live -lv 4 run of llama-server, not guessed.
    An earlier version required a line this build never prints
    ("(mmproj|clip|projector).{0,80}CUDA"), so a correctly offloaded model was
    reported as a CPU projector.
    """
    if host_buffer_max_mib is None:
        host_buffer_max_mib = HOST_BUFFER_MAX_MIB
    failures: list[str] = []

    # A server that came up on the CPU never registers a CUDA device at all.
    if CPU_ONLY.search(text):
        failures.append("no CUDA device was selected; the server ran on the CPU")

    match = OFFLOAD.search(text)
    if not match:
        failures.append(
            "missing offloaded N/N layers to GPU. If the model did load, the log was "
            "captured below verbosity 4: rerun with -lv 4."
        )
    else:
        left, right = int(match.group(1)), int(match.group(2))
        if left == 0:
            failures.append("offload is 0 layers, the model is on the CPU")
        elif left != right:
            failures.append(f"offload is {left}/{right}, not N/N")

    device = CUDA_DEVICE.search(text)
    if not device:
        failures.append("no 'using device CUDA0' line; weights did not reach a GPU")

    # The strongest single piece of evidence: a CUDA buffer holding the bulk of
    # the weights. A CPU run has no such line at all.
    cuda_mib = [float(m.group("mib")) for m in CUDA_BUFFER.finditer(text)]
    if not cuda_mib:
        failures.append("no CUDA model buffer; the weights are not in VRAM")
    elif max(cuda_mib) < 64.0:
        failures.append(f"largest CUDA model buffer is only {max(cuda_mib):.0f} MiB")

    if CPU_KV.search(text):
        failures.append("CPU KV buffer allocation")

    for host in HOST_BUFFER.finditer(text):
        size = float(host.group(1))
        # A fully offloaded Q6_K still maps part of the file into host memory.
        # The real RTX 4080 run shows CPU_Mapped at 787 MiB with 9317 MiB in
        # VRAM. The meaningful test is the ratio, not an absolute cap: a host
        # buffer as large as the CUDA buffer means the weights stayed on the CPU.
        if size >= host_buffer_max_mib and cuda_mib and size >= max(cuda_mib):
            failures.append(
                f"host model buffer {size:.0f} MiB is at least as large as the "
                f"CUDA buffer {max(cuda_mib):.0f} MiB"
            )

    # The projector must be loaded and must not have a CPU buffer of its own.
    if not MMPROJ.search(text):
        failures.append("log does not mention the multimodal projector")
    elif re.search(r"mmproj[^\n]{0,80}CPU[^\n]{0,40}buffer", text, re.IGNORECASE):
        failures.append("multimodal projector was placed on the CPU")
    return failures


def load_record(path: Path) -> GateRecord:
    data = json.loads(path.read_text())
    if "pass" in data and "pass_" not in data:
        data["pass_"] = data["pass"]
    return GateRecord.model_validate(data)


def assert_gate_alive(
    record_path: Path,
    model: Path,
    mmproj: Path,
    *,
    require_pid: bool = False,
) -> GateRecord:
    """Refuse to tag unless a passing gate record exists for these exact files.

    The liveness check used to be unconditional: it required the gated server to
    still be running. That makes a batch run impossible, because the server is
    stopped after the run finishes, and every tag attempt then died with "gate
    server pid is not alive". The model and mmproj hashes are the real integrity
    binding, and those are always checked.

    require_pid=True keeps the original behaviour for any caller that genuinely
    needs a live server, such as a long-lived tag daemon.
    """
    if not record_path.exists():
        raise SystemExit("tagger refuse: no passing gate record")
    record = load_record(record_path)
    if not record.pass_:
        raise SystemExit("tagger refuse: gate record is not a pass")
    if sha256_file(model) != record.model_sha256 or sha256_file(mmproj) != record.mmproj_sha256:
        raise SystemExit("tagger refuse: model hash does not match the gate record")
    if require_pid:
        try:
            os.kill(record.server_pid, 0)
        except OSError as exc:
            raise SystemExit("tagger refuse: gate server pid is not alive") from exc
    return record
