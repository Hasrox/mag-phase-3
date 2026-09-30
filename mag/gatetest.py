"""The eight gate checks, in Python, so they run on Windows without bash.

Every case here must be REJECTED. They exist so a CUDA gate cannot silently
pass on a partial offload, a CPU KV cache, a host model buffer, a projector on
the CPU, a public bind, a small GPU, a slow decode, or a missing offload line.
scripts/gpu_gate.sh is the POSIX equivalent and writes the same record shape.
"""

from __future__ import annotations

from mag.gate import check_log

PASS_LOG = """\
ggml_cuda_init: found 1 CUDA devices:
load_tensors: offloaded 48/48 layers to GPU
load_tensors: CPU KV buffer size = 512.00 MiB
clip_model: mmproj (projector) loaded on CUDA
main: server is listening on http://127.0.0.1:8080
"""

NO_OFFLOAD = PASS_LOG.replace("offloaded 48/48 layers to GPU", "offloaded 0/48 layers to GPU")
PARTIAL = PASS_LOG.replace("offloaded 48/48 layers to GPU", "offloaded 10/48 layers to GPU")
CPU_KV = PASS_LOG.replace(
    "load_tensors: CPU KV buffer size = 512.00 MiB",
    "llama_context: CPU KV buffer size = 512.00 MiB, CPU model buffer size = 900.00 MiB",
)
HOST_BUFFER = PASS_LOG.replace(
    "load_tensors: CPU KV buffer size = 512.00 MiB",
    "load_tensors: host model buffer size = 900.00 MiB",
)
NO_PROJECTOR = PASS_LOG.replace(
    "clip_model: mmproj (projector) loaded on CUDA",
    "clip_model: mmproj (projector) loaded on CPU",
)
STRIPPED = "main: server is listening on http://127.0.0.1:8080\n"

HOST_MAX_MIB = 64.0


def negative_cases() -> list[tuple[str, bool]]:
    """Every case that MUST be rejected, log-based and preflight-based."""
    cases = [
        ("partial_offload", check_log(PARTIAL, HOST_MAX_MIB)),
        ("zero_offload", check_log(NO_OFFLOAD, HOST_MAX_MIB)),
        ("cpu_kv_buffer", check_log(CPU_KV, HOST_MAX_MIB)),
        ("host_model_buffer", check_log(HOST_BUFFER, HOST_MAX_MIB)),
        ("projector_not_cuda", check_log(NO_PROJECTOR, HOST_MAX_MIB)),
        ("offload_line_removed", check_log(STRIPPED, HOST_MAX_MIB)),
    ]
    named = [(name, bool(failures)) for name, failures in cases]
    named += [
        (name, rejected)
        for name, rejected in preflight_negative_cases()
        if name not in MUST_PASS
    ]
    return named


def preflight_negative_cases() -> list[tuple[str, bool]]:
    """(name, was_rejected) for the preflight and hardware checks."""
    from mag.gatecli import decode_failures, preflight_failures, vram_failures

    rejected = [
        ("bind_0.0.0.0", preflight_failures(
            bind="0.0.0.0", host="127.0.0.1", n_gpu_layers="all",
            no_mmproj_offload=False, cpu_build=False, model_ok=True)),
        ("partial_offload_flag", preflight_failures(
            bind="127.0.0.1", host="127.0.0.1", n_gpu_layers="10",
            no_mmproj_offload=False, cpu_build=False, model_ok=True)),
        ("missing_n_gpu_layers", preflight_failures(
            bind="127.0.0.1", host="127.0.0.1", n_gpu_layers="",
            no_mmproj_offload=False, cpu_build=False, model_ok=True)),
        ("no_mmproj_offload", preflight_failures(
            bind="127.0.0.1", host="127.0.0.1", n_gpu_layers="all",
            no_mmproj_offload=True, cpu_build=False, model_ok=True)),
        ("cpu_only_build", preflight_failures(
            bind="127.0.0.1", host="127.0.0.1", n_gpu_layers="all",
            no_mmproj_offload=False, cpu_build=True, model_ok=True)),
        ("vram_under_16gb", vram_failures(15.0 * 1024, 16.0)),
        ("decode_below_floor", decode_failures(9.0, 15.0)),
        ("clean_preflight", preflight_failures(
            bind="127.0.0.1", host="127.0.0.1", n_gpu_layers="all",
            no_mmproj_offload=False, cpu_build=False, model_ok=True)),
        ("vram_exactly_16gb", vram_failures(16.0 * 1024, 16.0)),
        ("decode_at_floor", decode_failures(15.0, 15.0)),
    ]
    return [(name, bool(failures)) for name, failures in rejected]


def must_pass_cases() -> list[tuple[str, bool]]:
    """The clean cases. Each must be ACCEPTED, or the gate is over-strict."""
    return [(name, not rejected) for name, rejected in preflight_negative_cases()
            if name in MUST_PASS]


MUST_PASS = {"clean_preflight", "vram_exactly_16gb", "decode_at_floor"}
