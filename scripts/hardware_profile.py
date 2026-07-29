#!/usr/bin/env python3
"""Detect conservative first-boot defaults for RunGLM's supported TP2 path."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def mem_gib() -> float:
    limits = []
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            limits.append(int(line.split()[1]) * 1024)
            break
    # Containers may expose the host's /proc/meminfo while enforcing a lower
    # cgroup limit. Use the effective limit so preflight cannot false-pass and
    # later die while staging the all-CPU-backed adaptive expert store.
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            value = path.read_text().strip()
            if value != "max":
                parsed = int(value)
                if 0 < parsed < (1 << 60):
                    limits.append(parsed)
        except (FileNotFoundError, ValueError):
            pass
    return min(limits, default=0) / 1024**3


def available_cpus() -> int:
    limits = [os.cpu_count() or 1]
    try:
        limits.append(len(os.sched_getaffinity(0)))
    except AttributeError:
        pass
    # Respect a cgroup CPU quota when it is stricter than the visible/affine
    # CPU set (common on rented pods).
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limits.append(max(1, int(quota) // int(period)))
    except (FileNotFoundError, ValueError):
        pass
    return max(1, min(limits))


def cpu_flags() -> set[str]:
    match = re.search(r"^flags\s*:\s*(.*)$", Path("/proc/cpuinfo").read_text(), re.MULTILINE)
    return set(match.group(1).split()) if match else set()


def env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _gpu_query() -> list[tuple[str, str, int, float]]:
    """(index, name, MiB, compute capability) for every visible GPU."""
    text = command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    )
    rows = []
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 4:
            continue
        index = parts[0].strip()
        name = ",".join(parts[1:-2]).strip()
        try:
            rows.append((index, name, int(parts[-2].strip()), float(parts[-1].strip())))
        except ValueError:
            pass
    # Honor CUDA_VISIBLE_DEVICES so a masked single-card run (e.g. exporting
    # CUDA_VISIBLE_DEVICES=0 on a 2-GPU host) is profiled as one GPU, not two.
    # nvidia-smi ignores the mask, so we filter here. UUID masks are left
    # untouched (we only match plain integer indices).
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != "":
        wanted = [tok.strip() for tok in visible.split(",") if tok.strip() != ""]
        if wanted and all(tok.isdigit() for tok in wanted):
            selected = [row for row in rows if row[0] in wanted]
            if selected:
                rows = selected
    return rows


def gpu_rows() -> list[tuple[str, int]]:
    return [(name, memory) for _, name, memory, _ in _gpu_query()]


def min_compute_cap() -> float:
    """Lowest SM version present; 0.0 when no GPU is visible.

    The MINIMUM matters, not the maximum: every rank must run the same kernels,
    so a mixed host is limited by its oldest card.
    """
    return min((cap for _, _, _, cap in _gpu_query()), default=0.0)


# --- calibrated per-card VRAM footprint model (measured on 2xH100, TP2) -------
# footprint/card ≈ TRUNK_TOTAL/TP + N * MOE_LAYERS * EXPERT_FULL/TP + KV + headroom
# where N is the per-layer resident expert count (--kt-num-gpu-experts). The
# dense trunk and each INT4 expert are both TP-sharded along the intermediate
# dim, so per-card cost ≈ full/TP. Constants: 17.5 GiB/card base at TP2 -> 35 GiB
# unsharded trunk; 9.45 MiB/card/expert at TP2 -> 18.9 MiB unsharded; 78 layers
# minus 3 dense = 75 MoE layers; fp8 MLA KV ~44 KB/token (replicated per rank).
_TRUNK_TOTAL_MIB = 35_000.0
_EXPERT_FULL_MIB = 18.9
_MOE_LAYERS = 75
_KV_MIB_PER_TOKEN = 44.0 / 1024.0
_HEADROOM_MIB = 12_000.0  # cuda graphs + draft(MTP) + prefill scratch + slack
# The smallest resident expert count this repo has ever booted. N=0 is not
# "no GPU experts", it is an empty-resident crash, and the measured low-VRAM
# ladder starts at 4. A host that cannot hold 4 cannot run the model, so the
# preflight must say so rather than recommend a count nothing has ever run.
_MIN_GPU_EXPERTS = 4


def card_floor_mib(tp: int, mem_fraction: float, max_tokens: int) -> float:
    """Per-card VRAM a host must have before it can run this model at all."""
    tp = max(1, tp)
    return (
        _TRUNK_TOTAL_MIB / tp
        + max_tokens * _KV_MIB_PER_TOKEN
        + _HEADROOM_MIB
        + _MIN_GPU_EXPERTS * _MOE_LAYERS * _EXPERT_FULL_MIB / tp
    )


def valid_tp(count: int) -> int:
    """Largest tensor-parallel size that is actually launchable for this model.

    GLM-5.2 has 64 attention heads, so TP must be a power of two (1/2/4/8/...);
    an odd or non-power-of-two GPU count (3, 5, 6, 7) would make sglang abort at
    startup. For complete adaptability we clamp DOWN to the largest usable power
    of two and leave the surplus card(s) idle rather than fail to launch.
    """
    tp = 1
    while tp * 2 <= max(1, count):
        tp *= 2
    return tp


def fit_gpu_experts(minimum_mib: int, tp: int, mem_fraction: float, max_tokens: int) -> int:
    """Largest per-layer resident expert count that fits one card's VRAM budget.

    Conservative: a fat headroom absorbs the CUDA-graph pool, the MTP draft
    model, and prefill scratch so a first boot does not OOM. Values remain
    env-overridable, so this only needs to be safe, not optimal.
    """
    tp = max(1, tp)
    budget = minimum_mib * mem_fraction
    trunk = _TRUNK_TOTAL_MIB / tp
    kv = max_tokens * _KV_MIB_PER_TOKEN
    avail = budget - trunk - kv - _HEADROOM_MIB
    per_n = _MOE_LAYERS * _EXPERT_FULL_MIB / tp
    if avail <= 0 or per_n <= 0:
        # Floor at the smallest count that has ever booted rather than at 1.
        # This value is only reached on a host the preflight rejects anyway, and
        # returning a never-run 1 made the failure look like a tuning problem.
        return _MIN_GPU_EXPERTS
    return max(_MIN_GPU_EXPERTS, min(256, int(avail / per_n)))


def choose_profile(rows: list[tuple[str, int]], adaptive: bool = False) -> dict[str, str]:
    count = len(rows)
    minimum_mib = min((memory for _, memory in rows), default=0)
    names = " ".join(name.lower() for name, _ in rows)
    cpus = available_cpus()
    tp = valid_tp(count)
    profile, gpu_experts, mem_fraction, max_tokens = f"generic-tp{tp}", 16, "0.85", 8192
    max_running = "1"
    context_length = str(max_tokens)

    if count == 2 and ("l40" in names or "6000 ada" in names) and minimum_mib >= 45_000:
        # 30 experts/layer at mem_fraction 0.88 leaves ~7.7 GB/card free and is
        # the largest validated point: 32 @ 0.91 OOMs during CUDA-graph capture
        # (fills 44.29 of 44.31 GiB). Note this is a weak lever under safe2
        # routing -- CPU traffic scales as 2*(1-residency), so 24 -> 30 only cuts
        # it 1.81 -> 1.77 experts/token, worth ~3% (18.9 -> 19.6 tok/s).
        profile, gpu_experts, mem_fraction = "2xl40", 30, "0.88"
    elif count == 2 and "h100" in names and minimum_mib >= 75_000:
        profile = "2xh100"
        gpu_experts = 96 if adaptive or minimum_mib < 90_000 else 104
        mem_fraction = "0.85" if adaptive else ("0.95" if minimum_mib >= 90_000 else "0.94")
        max_tokens = 8192 if adaptive else (16384 if minimum_mib < 90_000 else 81920)
        # Preserve the validated H100 launcher's derived 1M context in adaptive
        # mode and its existing two-request scheduler capacity.
        context_length = "" if adaptive else str(max_tokens)
        max_running = "2"
    elif count == 2 and minimum_mib >= 45_000:
        profile, gpu_experts = "2x48gb", 24
    else:
        # Any other topology (1 GPU, 3+, or an unrecognized 2-GPU host): size the
        # resident expert count from measured per-card VRAM instead of a fixed
        # table. TP = GPU count; extra GPUs shard the same expert set thinner.
        gpu_experts = fit_gpu_experts(minimum_mib, tp, float(mem_fraction), max_tokens)

    numa_nodes = sorted(
        path.name.removeprefix("node")
        for path in Path("/sys/devices/system/node").glob("node[0-9]*")
    ) or ["0"]
    selected_numa = numa_nodes[:2]
    cpuinfer = 72 if profile == "2xh100" and cpus >= 80 else max(
        1, min(72, cpus - max(4, cpus // 8))
    )
    # The CPU-expert path is DRAM-bandwidth-bound, not FLOP-bound, so the right
    # thread count is the one that saturates memory bandwidth -- not "all cores".
    # Measured streaming-read sweep on the 2xL40 pod (dual EPYC 7773X, DDR4):
    #   8 thr 163 GB/s | 16 thr 248 | 28 thr 296 | 56 thr 356 | 112 thr 271
    # i.e. bandwidth peaks near 56 and *regresses* past it. 56 cut decode step
    # time 170ms -> 152ms vs the previous hardcoded 28. Re-measure per host with
    # a streaming-read benchmark before changing this; do not assume more=better.
    # (The prior value of 28 assumed a 32-vCPU pod allocation; measured parallel
    # capacity on this host is ~40 cores, and the bandwidth curve peaks higher.)
    if profile == "2xl40":
        cpuinfer = min(56, max(1, cpus - max(4, cpus // 8)))
    selected = {
        "RUNGLM_PROFILE": profile,
        "TP_SIZE": str(tp),
        "GPU_EXPERTS": str(gpu_experts),
        "MEM_FRACTION": mem_fraction,
        "MAX_TOTAL_TOKENS": str(max_tokens),
        "CONTEXT_LENGTH": context_length,
        "CPUINFER": str(cpuinfer),
        "NUMA_NODES": " ".join(selected_numa),
        "KT_THREADPOOL_COUNT": str(len(selected_numa)),
        "MAX_RUNNING": max_running,
        "CUDA_GRAPH_MAX_BS": "1",
        "KT_GPU_PREFILL_THRESHOLD": "0" if minimum_mib < 75_000 else "2048",
    }
    # GPU kernel backends are chosen by COMPUTE CAPABILITY, not by the two-card
    # profile table. Kernel availability is an architecture property: FlashMLA
    # and the CUTLASS W4A8 MoE are Hopper-only, and on Ada they fail as "no
    # kernel image for SM89" and TMA descriptor error 801 respectively.
    #
    # Keying this off the profile name left every host that fell through to the
    # generic branch -- notably ANY single-GPU host, which is the common
    # small-machine case -- with no backend pinned at all, so run_server_int4.sh
    # applied its Hopper defaults and a single L40/A6000 would boot straight
    # into those two failures. Selecting on SM fixes that class of host without
    # changing either validated profile: L40 is SM 8.9 -> the same marlin/triton
    # trio the 2xl40 table pinned, H100 is SM 9.0 -> the same cutlass/flashmla.
    cap = min_compute_cap()
    if cap >= 9.0:
        selected["KT_W4AFP8_GPU_BACKEND"] = "cutlass_sm90"
        selected["FP8_GEMM_BACKEND"] = "cutlass"
        selected["ATTENTION_BACKEND"] = "flashmla"
    elif cap >= 8.0:
        selected["KT_W4AFP8_GPU_BACKEND"] = "marlin_sm80"
        selected["FP8_GEMM_BACKEND"] = "triton"
        selected["ATTENTION_BACKEND"] = "triton"
    # fp8 KV cache is only validated on the flashmla path. On every other
    # attention backend (the Triton fallback that non-Hopper cards land on) an
    # fp8_e4m3 KV cache measurably degrades MTP/NEXTN acceptance, because the
    # draft head's agreement with the target is sensitive to KV precision.
    # Measured on 2xL40 switching fp8_e4m3 -> bf16 (accept length, greedy):
    #   technical prose 2.2 -> 2.63 | structured list -> 3.21
    #   code generation -> 3.27     | reasoning -> 3.36  | repetitive -> 3.75
    # Net decode throughput went 12.9 -> 16.5 tok/s (and ~22.5 on code), i.e.
    # the "MTP accept length collapsed on this box" symptom was this setting.
    # MLA makes it nearly free: kv_lora_rank=512 means bf16 KV for 8192 tokens
    # costs well under 1 GB. Default anything that is not flashmla to bf16.
    if selected.get("ATTENTION_BACKEND") != "flashmla":
        selected["KV_CACHE_DTYPE"] = "auto"
    # Per-layer hot-core placement. `uniform` fills every layer with experts
    # 0..N-1 BY INDEX, so top-2 coverage is only N/256 (12.7% at N=30) -- nothing
    # about it is informed by routing. `hotcore` gives each layer its own
    # hottest-N from the committed ranking and is strictly better on every host
    # (it self-falls-back to uniform if the ranking does not match the model).
    selected["PLACEMENT"] = "hotcore"

    # Top-K routing mode -- THE portability trap. `sub2` keeps the genuine top-2
    # and substitutes the other six slots with GPU-RESIDENT experts, so its
    # output quality depends entirely on how good the resident set is:
    #   validated coherent at 96-104 experts/layer (2xH100, ~90% top-2 coverage)
    #   measured INCOHERENT at 24-30 experts/layer (2xL40, ~41% coverage)
    # `safe2` instead sends a non-resident genuine expert to the CPU, so it is
    # always correct and only ever slower. Anything below the validated-coherent
    # point therefore defaults to safe2; the 30..96 range is untested, and the
    # conservative choice there is the correct-by-construction one.
    selected["RUNGLM_TOPK_MODE"] = "sub2" if gpu_experts >= 96 else "safe2"

    # The measured path remains AVX-512 VNNI. An explicitly allowed AVX2 host
    # must not inherit run_fast.sh's avx512_packed default or it will SIGILL.
    if "avx512_vnni" not in cpu_flags() and "avx2" in cpu_flags():
        selected["KT_RAWINT4_BACKEND"] = "avx2_packed"
        selected["KT_KERNEL_CPU_VARIANT"] = "avx2"
    return selected


def problems(rows: list[tuple[str, int]], adaptive: bool, weights_dir: Path | None) -> list[str]:
    found = []
    memory = mem_gib()
    count = len(rows)
    minimum_mib = min((value for _, value in rows), default=0)
    tp = valid_tp(count)
    if count == 0:
        found.append("no NVIDIA GPUs detected")
    else:
        # The dense trunk (~35 GiB unsharded) is TP-sharded across the cards and
        # is not optional. A card whose budget cannot hold its trunk shard plus a
        # little room for KV/graphs/>=1 expert cannot run this model at all.
        chosen = choose_profile(rows, adaptive)
        selected_mem = chosen["MEM_FRACTION"] or "0.85"
        budget = minimum_mib * float(selected_mem)
        # The old check only required the dense trunk plus 2 GiB, which passed
        # hosts that then died at warm-up: a 46 GiB card clears trunk+2 GiB but
        # has nothing left for the CUDA-graph pool, the MTP draft model, prefill
        # scratch AND a usable resident expert set. Require the whole floor.
        floor = card_floor_mib(tp, float(selected_mem), int(chosen["MAX_TOTAL_TOKENS"]))
        if budget < floor:
            found.append(
                f"per-card budget ~{budget:.0f} MiB (VRAM {minimum_mib} MiB x "
                f"mem_fraction {selected_mem}) is below the ~{floor:.0f} MiB floor "
                f"at TP={tp} (dense trunk {_TRUNK_TOTAL_MIB / tp:.0f} + KV + "
                f"{_HEADROOM_MIB:.0f} runtime headroom + {_MIN_GPU_EXPERTS} resident "
                "experts); use a bigger card or more GPUs"
            )
        # Below Ampere there is no INT4 GPU expert path at all: the Marlin
        # W4A16 kernel is the oldest one this repo carries and it requires
        # SM80+. Say so here rather than letting the launch die inside a kernel
        # dispatch with an unrelated-looking error.
        cap = min_compute_cap()
        if 0.0 < cap < 8.0:
            found.append(
                f"oldest GPU is compute capability {cap:.1f}; the INT4 expert "
                "path needs SM 8.0+ (Ampere or newer)"
            )
    flags = cpu_flags()
    if "avx512_vnni" not in flags:
        if not env_enabled("RUNGLM_ALLOW_AVX2"):
            found.append(
                "CPU lacks AVX-512 VNNI; set RUNGLM_ALLOW_AVX2=1 to explicitly "
                "try the slower AVX2 fallback"
            )
        elif not {"avx2", "fma"}.issubset(flags):
            found.append("AVX2 fallback requested, but the CPU lacks AVX2 and FMA")
    selected = choose_profile(rows, adaptive)
    # All-CPU-backed INT4 is roughly 400 GiB. Static placement avoids retaining
    # about 1.38 GiB of host weights per GPU expert/layer under TP2.
    static_ram = max(240, int(410 - 1.38 * int(selected["GPU_EXPERTS"])))
    required_ram = 430 if adaptive else static_ram
    if memory < required_ram:
        found.append(
            f"{memory:.0f} GiB RAM detected; {'adaptive' if adaptive else 'static'} "
            f"mode conservatively requires {required_ram} GiB"
        )
    if weights_dir:
        target = weights_dir if weights_dir.exists() else weights_dir.parent
        while not target.exists() and target != target.parent:
            target = target.parent
        free = shutil.disk_usage(target).free / 1024**3
        if free < 390 and not weights_dir.exists():
            found.append(f"only {free:.0f} GiB free for the approximately 373 GB checkpoint")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--weights-dir", type=Path)
    args = parser.parse_args()
    rows = gpu_rows()
    profile = choose_profile(rows, args.adaptive)

    if args.shell:
        for key, value in profile.items():
            print(f"{key}={value}")
    else:
        print("GPUs:", ", ".join(f"{name} ({memory} MiB)" for name, memory in rows) or "none")
        print(f"RAM: {mem_gib():.1f} GiB; CPUs: {available_cpus()}")
        print("Profile:", profile["RUNGLM_PROFILE"])
        print("Defaults:", " ".join(f"{k}={v}" for k, v in profile.items() if k != "RUNGLM_PROFILE"))

    if args.check:
        tp = valid_tp(len(rows))
        if tp < len(rows):
            print(
                f"WARNING: {len(rows)} GPUs detected but tensor-parallel size must"
                f" divide 64 attention heads; using TP={tp} and leaving"
                f" {len(rows) - tp} card(s) idle."
            )
        if len(rows) != 2:
            print(
                f"WARNING: {len(rows)} GPU(s) detected; the measured reference is 2x"
                " GPU. Single/other-count runs are auto-sized from VRAM and are"
                " functional but unvalidated for peak throughput."
            )
        if (
            env_enabled("RUNGLM_ALLOW_AVX2")
            and "avx512_vnni" not in cpu_flags()
            and {"avx2", "fma"}.issubset(cpu_flags())
        ):
            print("WARNING: using experimental packed AVX2 fallback; expect lower throughput")
        found = problems(rows, args.adaptive, args.weights_dir)
        for item in found:
            print(f"ERROR: {item}")
        if found:
            return 1
        print("Preflight checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
