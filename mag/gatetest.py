"""The eight gate checks, in Python, so they run on Windows without bash.

Every case here must be REJECTED. They exist so a CUDA gate cannot silently
pass on a partial offload, a CPU KV cache, a host model buffer, a projector on
the CPU, a public bind, a small GPU, a slow decode, or a missing offload line.
scripts/gpu_gate.sh is the POSIX equivalent and writes the same record shape.

The PASS_LOG below is a real llama-server -lv 4 capture from an RTX 4080 with
gemma-4-12B-it-uncensored-heretic.Q6_K, trimmed to the lines the gate reads.
The strings are what this build actually prints. An earlier version of this file
used invented strings and passed a log no real server produces.
"""

from __future__ import annotations

from mag.gate import HOST_BUFFER_MAX_MIB, check_log

# Verbatim from runtime/last-server.log.err of a real -lv 4 run.
PASS_LOG = """\
common_param: device_info:
common_init_: fitting params to device memory ...
common_params_fit_impl: projected to use 9988 MiB of device memory vs. 15074 MiB of free device memory
llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 4080) (0000:01:00.0) - 15074 MiB free
srv    load_model: [mtmd] estimated worst-case memory usage of mmproj is 339.08 MiB (took 53.14 ms)
load_tensors: offloading output layer to GPU
load_tensors: offloading 47 repeating layers to GPU
load_tensors: offloaded 49/49 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   787.50 MiB
load_tensors:        CUDA0 model buffer size =  9317.65 MiB
srv  llama_server: model loaded
srv  llama_server: listening on http://127.0.0.1:8080
"""

# What the operator's failed run looked like: no CUDA device line, no offload
# line, no CUDA buffer. The model loaded entirely on the CPU in 16 seconds.
CPU_ONLY_LOG = """\
srv  llama_server: initializing ...
cmn  common_param: verbosity = 3
srv    load_model: loading model 'gemma-4-12B-it-uncensored-heretic.Q6_K.gguf'
cmn          init: llama threadpool init, n_threads = 14
srv    load_model: loaded multimodal model, 'gemma-4-12B-it-uncensored-heretic.mmproj-Q8_0.gguf'
srv    load_model: initializing, n_slots = 1, n_ctx_slot = 4096
srv  llama_server: model loaded
srv  llama_server: listening on http://127.0.0.1:8080
"""

NO_OFFLOAD = PASS_LOG.replace("offloaded 49/49 layers to GPU", "offloaded 0/49 layers to GPU")
PARTIAL = PASS_LOG.replace("offloaded 49/49 layers to GPU", "offloaded 10/49 layers to GPU")
CPU_KV = PASS_LOG.replace(
    "load_tensors:   CPU_Mapped model buffer size =   787.50 MiB",
    "llama_context: CPU KV buffer size = 512.00 MiB, CPU model buffer size = 9000.00 MiB",
)
HOST_BUFFER = PASS_LOG.replace(
    "load_tensors:   CPU_Mapped model buffer size =   787.50 MiB",
    "load_tensors:   CPU model buffer size = 9317.65 MiB",
)
NO_PROJECTOR = "\n".join(
    line for line in PASS_LOG.splitlines() if "mmproj" not in line
) + "\n"
PROJECTOR_ON_CPU = PASS_LOG.replace(
    "[mtmd] estimated worst-case memory usage of mmproj is 339.08 MiB",
    "[mtmd] mmproj loaded on CPU, model buffer size = 339.08 MiB",
)
NO_CUDA_BUFFER = "\n".join(
    line for line in PASS_LOG.splitlines() if "CUDA0 model buffer size" not in line
) + "\n"
DEVICE_IS_CPU = PASS_LOG.replace(
    "llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 4080) (0000:01:00.0) - 15074 MiB free",
    "llama_prepare_model_devices: using device CPU",
)
STRIPPED = "srv  llama_server: listening on http://127.0.0.1:8080\n"


def negative_cases() -> list[tuple[str, bool]]:
    """Every case that MUST be rejected, log-based and preflight-based."""
    cases = [
        ("cpu_only_run", check_log(CPU_ONLY_LOG)),
        ("zero_offload", check_log(NO_OFFLOAD)),
        ("partial_offload", check_log(PARTIAL)),
        ("cpu_kv_buffer", check_log(CPU_KV)),
        ("host_model_buffer", check_log(HOST_BUFFER)),
        ("projector_on_cpu", check_log(PROJECTOR_ON_CPU)),
        ("no_projector_line", check_log(NO_PROJECTOR)),
        ("no_cuda_buffer", check_log(NO_CUDA_BUFFER)),
        ("device_is_cpu", check_log(DEVICE_IS_CPU)),
        ("offload_line_removed", check_log(STRIPPED)),
    ]
    named = [(name, bool(failures)) for name, failures in cases]
    named += [
        (name, rejected)
        for name, rejected in preflight_negative_cases()
        if name not in MUST_PASS
    ]
    return named


def real_log_is_accepted() -> tuple[str, list[str]]:
    """The real capture must pass. Returns (name, failures)."""
    return ("real_lv4_log", check_log(PASS_LOG, HOST_BUFFER_MAX_MIB))


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
        ("vram_12gb_card", vram_failures(12 * 1024, 16.0)),
        ("clean_preflight", preflight_failures(
            bind="127.0.0.1", host="127.0.0.1", n_gpu_layers="all",
            no_mmproj_offload=False, cpu_build=False, model_ok=True)),
        ("vram_exactly_16gb", vram_failures(16.0 * 1024, 16.0)),
        # A card sold as 16 GB reports 16376 MiB. Rejecting it would reject the
        # exact hardware the spec targets.
        ("vram_16gb_card_reported_as_mib", vram_failures(16376, 16.0)),
        ("decode_at_floor", decode_failures(15.0, 15.0)),
    ]
    return [(name, bool(failures)) for name, failures in rejected]

def must_pass_cases() -> list[tuple[str, bool]]:
    """The clean cases. Each must be ACCEPTED, or the gate is over-strict."""
    return [(name, not rejected) for name, rejected in preflight_negative_cases()
            if name in MUST_PASS]


MUST_PASS = {
    "clean_preflight", "vram_exactly_16gb", "vram_16gb_card_reported_as_mib", "decode_at_floor",
}
