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


def gpu_rows() -> list[tuple[str, int]]:
    text = command(
        "nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"
    )
    rows = []
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        index = parts[0].strip()
        name = ",".join(parts[1:-1]).strip()
        try:
            rows.append((index, name, int(parts[-1].strip())))
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
    return [(name, memory) for _, name, memory in rows]


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
    headroom = 12_000.0  # cuda graphs + draft(MTP) + prefill scratch + slack
    avail = budget - trunk - kv - headroom
    per_n = _MOE_LAYERS * _EXPERT_FULL_MIB / tp
    if avail <= 0 or per_n <= 0:
        return 1
    return max(1, min(256, int(avail / per_n)))


def choose_profile(rows: list[tuple[str, int]], adaptive: bool = False) -> dict[str, str]:
    count = len(rows)
    minimum_mib = min((memory for _, memory in rows), default=0)
    names = " ".join(name.lower() for name, _ in rows)
    cpus = available_cpus()
    tp = max(1, count)
    profile, gpu_experts, mem_fraction, max_tokens = f"generic-tp{tp}", 16, "0.85", 8192
    max_running = "1"
    context_length = str(max_tokens)

    if count == 2 and ("l40" in names or "6000 ada" in names) and minimum_mib >= 45_000:
        profile, gpu_experts = "2xl40", 24
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
    selected = {
        "RUNGLM_PROFILE": profile,
        "TP_SIZE": str(max(1, count)),
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
    # The measured path remains AVX-512 VNNI. An explicitly allowed AVX2 host
    # must not inherit run_fast.sh's avx512_packed default or it will SIGILL.
    if "avx512_vnni" not in cpu_flags() and "avx2" in cpu_flags():
        selected["KT_RAWINT4_BACKEND"] = "avx2"
        selected["KT_KERNEL_CPU_VARIANT"] = "avx2"
    return selected


def problems(rows: list[tuple[str, int]], adaptive: bool, weights_dir: Path | None) -> list[str]:
    found = []
    memory = mem_gib()
    count = len(rows)
    minimum_mib = min((value for _, value in rows), default=0)
    tp = max(1, count)
    if count == 0:
        found.append("no NVIDIA GPUs detected")
    else:
        # The dense trunk (~35 GiB unsharded) is TP-sharded across the cards and
        # is not optional. A card whose budget cannot hold its trunk shard plus a
        # little room for KV/graphs/>=1 expert cannot run this model at all.
        selected_mem = choose_profile(rows, adaptive)["MEM_FRACTION"] or "0.85"
        trunk_shard = _TRUNK_TOTAL_MIB / tp
        budget = minimum_mib * float(selected_mem)
        if budget < trunk_shard + 2_000:
            found.append(
                f"per-card budget ~{budget:.0f} MiB (VRAM {minimum_mib} MiB x "
                f"mem_fraction {selected_mem}) cannot hold the dense-trunk shard "
                f"~{trunk_shard:.0f} MiB at TP={tp}; use a bigger card or more GPUs"
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
            print("WARNING: using unvalidated AVX2 CPU fallback; expect lower throughput")
        found = problems(rows, args.adaptive, args.weights_dir)
        for item in found:
            print(f"ERROR: {item}")
        if found:
            return 1
        print("Preflight checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
