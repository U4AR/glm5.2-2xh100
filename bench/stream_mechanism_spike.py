#!/usr/bin/env python3
"""Phase 3: which transfer mechanism, measured rather than preferred.

Two problems, and the second is the one that decides the design.

(i) GRAPH-SAFE DYNAMIC ADDRESSING. Decode runs inside a captured CUDA graph, and
    a captured copy node has its addresses baked in at capture time. But which
    experts a layer needs is only known at replay. Two ways out:

      A. UVA gather kernel -- register a window of host expert memory, and have a
         captured kernel read it through device-computed indices. The indices
         live in a device tensor the scheduler writes before replay, so the same
         graph moves different experts every step.
      B. Host-node staging -- a host callback memcpys the chosen experts into a
         fixed pinned staging buffer, then a captured copy with fixed addresses
         moves it. Simple, but it touches every byte twice on the host.

(ii) LAYOUT. kt's CPU store holds RAWINT4 in the CPU kernel's packed layout; the
     GPU's cutlass W4A8 path wants a different one, and today's whole-layer
     interleave costs ~136 ms -- unusable per step. So either the host keeps a
     second copy already in GPU layout (costs RAM), or the bytes go over raw and
     a GPU kernel repacks them (costs GPU time, which Phase 1 showed is ~67-80%
     idle at safe8).

This measures all of it on the machine it runs on, so the recommendation is a
number, not a preference.

    .venv/bin/python bench/stream_mechanism_spike.py --experts 4 --iters 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

EXPERT_MB = 9.7

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// Gather `k` experts out of a host-resident window into contiguous VRAM slots.
// `src_base` is a raw host pointer that the device can read directly under
// unified addressing once the pages are registered; `idx` is a DEVICE tensor,
// which is the whole point -- it can change between graph replays without
// recapturing anything.
__global__ void gather_experts_kernel(
    const uint4* __restrict__ src_base,
    uint4* __restrict__ dst,
    const long* __restrict__ idx,
    long vec_per_expert) {
  long e = blockIdx.y;
  const uint4* src = src_base + idx[e] * vec_per_expert;
  uint4* out = dst + e * vec_per_expert;
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_per_expert;
       i += (long)gridDim.x * blockDim.x) {
    out[i] = src[i];
  }
}

void gather_experts(long src_ptr, torch::Tensor dst, torch::Tensor idx,
                    long bytes_per_expert, long blocks) {
  TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA tensor");
  TORCH_CHECK(idx.is_cuda(), "idx must live on the device or the graph cannot "
                             "pick experts at replay time");
  long vec_per_expert = bytes_per_expert / sizeof(uint4);
  dim3 grid(blocks, idx.numel());
  gather_experts_kernel<<<grid, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(src_ptr),
      reinterpret_cast<uint4*>(dst.data_ptr()),
      idx.data_ptr<long>(), vec_per_expert);
}
"""

# load_inline generates its own module definition, so the declaration goes in the
# C++ half and the kernel stays in the .cu -- one PYBIND11_MODULE, not two.
CPP_SRC = (
    "void gather_experts(long src_ptr, torch::Tensor dst, torch::Tensor idx, "
    "long bytes_per_expert, long blocks);"
)


def build_ext():
    from torch.utils.cpp_extension import load_inline
    return load_inline(
        name="kt_stream_spike",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["gather_experts"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def measure_uva_gather(ext, window_experts: int, k: int, iters: int,
                       device: int) -> dict:
    """Mechanism A: captured kernel, device-side indices, host-resident source."""
    torch.cuda.set_device(device)
    bytes_per_expert = int(EXPERT_MB * 1024 * 1024) // 16 * 16
    host = torch.empty(window_experts * bytes_per_expert, dtype=torch.uint8,
                       pin_memory=True)
    host.random_(0, 255)
    dst = torch.empty(k * bytes_per_expert, dtype=torch.uint8, device=f"cuda:{device}")
    idx = torch.zeros(k, dtype=torch.long, device=f"cuda:{device}")
    src_ptr = host.data_ptr()

    def run(blocks=1024):
        ext.gather_experts(src_ptr, dst, idx, bytes_per_expert, blocks)

    # warm up outside the graph, then capture
    for _ in range(3):
        run()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        run()

    # Does replay honour a *changed* index vector? That is the entire claim.
    picks = torch.randint(0, window_experts, (k,), dtype=torch.long)
    idx.copy_(picks.cuda(device))
    g.replay()
    torch.cuda.synchronize()
    ok = True
    for i, p in enumerate(picks.tolist()):
        want = host[p * bytes_per_expert:(p + 1) * bytes_per_expert]
        got = dst[i * bytes_per_expert:(i + 1) * bytes_per_expert].cpu()
        if not torch.equal(want, got):
            ok = False
            break

    times = []
    for _ in range(iters):
        idx.copy_(torch.randint(0, window_experts, (k,), dtype=torch.long).cuda(device))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    med = statistics.median(times)
    gb = k * bytes_per_expert / 1e9
    return {
        "mechanism": "A: UVA gather kernel in-graph",
        "correct_after_index_change": ok,
        "k": k, "ms_median": med, "ms_per_expert": med / k,
        "gbs": gb / (med / 1000.0),
        "window_experts": window_experts,
        "window_gb": window_experts * bytes_per_expert / 1e9,
    }


def measure_staged_copy(k: int, iters: int, device: int, threads: int) -> dict:
    """Mechanism B: host memcpy into fixed staging, then a captured H2D."""
    torch.cuda.set_device(device)
    bytes_per_expert = int(EXPERT_MB * 1024 * 1024)
    window_experts = 64
    src = torch.empty(window_experts * bytes_per_expert, dtype=torch.uint8)
    src.random_(0, 255)
    staging = torch.empty(k * bytes_per_expert, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty_like(staging, device=f"cuda:{device}")

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        dst.copy_(staging, non_blocking=True)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        dst.copy_(staging, non_blocking=True)

    torch.set_num_threads(threads)
    host_ms, copy_ms = [], []
    for _ in range(iters):
        picks = torch.randint(0, window_experts, (k,)).tolist()
        t0 = time.perf_counter()
        for i, p in enumerate(picks):
            staging[i * bytes_per_expert:(i + 1) * bytes_per_expert].copy_(
                src[p * bytes_per_expert:(p + 1) * bytes_per_expert])
        host_ms.append((time.perf_counter() - t0) * 1000.0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        copy_ms.append((time.perf_counter() - t1) * 1000.0)

    mh, mc = statistics.median(host_ms), statistics.median(copy_ms)
    return {
        "mechanism": f"B: host staging ({threads} threads) + captured H2D",
        "k": k,
        "host_stage_ms": mh, "h2d_ms": mc, "ms_median": mh + mc,
        "ms_per_expert": (mh + mc) / k,
        "gbs": (k * bytes_per_expert / 1e9) / ((mh + mc) / 1000.0),
    }


def measure_repack(iters: int, device: int) -> dict:
    """Layout (ii): what does a GPU-side repack of one expert actually cost?

    The real cutlass interleave is a permutation, and a permutation of 9.7 MB is
    bounded by HBM bandwidth, not by arithmetic. This measures that bound with a
    representative shuffle so the answer does not depend on writing the final
    kernel first.
    """
    torch.cuda.set_device(device)
    n = int(EXPERT_MB * 1024 * 1024) // 4
    src = torch.randint(0, 2**31 - 1, (n,), dtype=torch.int32, device=f"cuda:{device}")
    dst = torch.empty_like(src)
    perm = torch.randperm(n, device=f"cuda:{device}")

    for _ in range(3):
        torch.index_select(src, 0, perm, out=dst)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        torch.index_select(src, 0, perm, out=dst)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    med = statistics.median(times)
    return {
        "op": "GPU repack (worst-case full random permutation of one expert)",
        "ms_median": med,
        "note": "a real interleave is far more regular than a random gather, "
                "so this is an upper bound",
    }


def measure_host_register(ext, gb: float, k: int, iters: int, device: int) -> dict:
    """Can the gather read kt's EXISTING store instead of a pinned copy?

    The window analysis assumes streaming needs its own page-locked copy of the
    weights, which costs real RAM (22.7 GB for 55% coverage here). But kt already
    holds every non-resident expert in host memory. cudaHostRegister page-locks
    pages *in place* -- no copy, no extra RAM -- and under unified addressing the
    device can then read them. If that works and registration is not too slow,
    coverage is 100% and the window problem disappears.

    Measures: registration throughput (a boot-time cost) and whether a captured
    gather reads registered-but-not-torch-pinned memory correctly.
    """
    torch.cuda.set_device(device)
    bytes_per_expert = int(EXPERT_MB * 1024 * 1024) // 16 * 16
    n_experts = max(int(gb * 1e9) // bytes_per_expert, k)
    # Ordinary pageable memory, exactly like a C++ store's allocation.
    host = torch.empty(n_experts * bytes_per_expert, dtype=torch.uint8)
    host.random_(0, 255)

    cudart = torch.cuda.cudart()
    t0 = time.perf_counter()
    err = cudart.cudaHostRegister(host.data_ptr(), host.numel(), 0)
    reg_s = time.perf_counter() - t0
    if int(err) != 0:
        return {"registered": False, "cuda_error": int(err),
                "note": "cudaHostRegister failed; streaming would need its own "
                        "pinned window after all"}

    dst = torch.empty(k * bytes_per_expert, dtype=torch.uint8, device=f"cuda:{device}")
    idx = torch.zeros(k, dtype=torch.long, device=f"cuda:{device}")
    picks = torch.randint(0, n_experts, (k,), dtype=torch.long)
    idx.copy_(picks.cuda(device))
    ext.gather_experts(host.data_ptr(), dst, idx, bytes_per_expert, 1024)
    torch.cuda.synchronize()
    ok = all(
        torch.equal(host[p * bytes_per_expert:(p + 1) * bytes_per_expert],
                    dst[i * bytes_per_expert:(i + 1) * bytes_per_expert].cpu())
        for i, p in enumerate(picks.tolist())
    )

    times = []
    for _ in range(iters):
        idx.copy_(torch.randint(0, n_experts, (k,), dtype=torch.long).cuda(device))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ext.gather_experts(host.data_ptr(), dst, idx, bytes_per_expert, 1024)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t1) * 1000.0)
    med = statistics.median(times)

    cudart.cudaHostUnregister(host.data_ptr())
    return {
        "registered": True,
        "correct": ok,
        "gb_registered": host.numel() / 1e9,
        "register_s": reg_s,
        "register_gbs": (host.numel() / 1e9) / reg_s,
        "ms_median": med,
        "ms_per_expert": med / k,
        "gbs": (k * bytes_per_expert / 1e9) / (med / 1000.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=4, help="k streamed per layer")
    ap.add_argument("--window", type=int, default=64,
                    help="host-resident experts addressable by the gather")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--register-gb", type=float, default=4.0,
                    help="pageable memory to page-lock in place for the test")
    ap.add_argument("--out", default="bench/profile_out/phase3_mechanism.json")
    args = ap.parse_args()

    results = {}
    print("building the gather kernel...")
    ext = build_ext()

    print("\n--- (i) graph-safe dynamic addressing ---")
    a = measure_uva_gather(ext, args.window, args.experts, args.iters, args.device)
    results["A"] = a
    print(f"  A  UVA gather in-graph : {a['ms_median']:.3f} ms for {a['k']} experts "
          f"({a['ms_per_expert']:.3f} ms each, {a['gbs']:.1f} GB/s)")
    print(f"     picks new experts after capture: "
          f"{'YES' if a['correct_after_index_change'] else 'NO -- disqualifying'}")

    b = measure_staged_copy(args.experts, max(args.iters // 4, 10), args.device,
                            args.threads)
    results["B"] = b
    print(f"  B  staging + captured  : {b['ms_median']:.3f} ms "
          f"({b['host_stage_ms']:.3f} host + {b['h2d_ms']:.3f} H2D, "
          f"{b['ms_per_expert']:.3f} ms each, {b['gbs']:.1f} GB/s)")

    print("\n--- (i-b) can the gather read kt's existing store in place? ---")
    reg = measure_host_register(ext, args.register_gb, args.experts,
                                max(args.iters // 4, 10), args.device)
    results["register"] = reg
    if reg.get("registered"):
        print(f"  cudaHostRegister      : {reg['gb_registered']:.1f} GB in "
              f"{reg['register_s']:.2f} s ({reg['register_gbs']:.1f} GB/s), "
              f"gather correct: {'YES' if reg['correct'] else 'NO'}")
        print(f"  gather from registered: {reg['ms_per_expert']:.3f} ms/expert "
              f"({reg['gbs']:.1f} GB/s)")
    else:
        print(f"  cudaHostRegister FAILED: {reg}")

    print("\n--- (ii) layout ---")
    r = measure_repack(max(args.iters // 4, 10), args.device)
    results["repack"] = r
    print(f"  GPU repack per expert  : {r['ms_median']:.3f} ms (upper bound)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
