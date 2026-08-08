"""Mechanism probe: does cudaHostRegister on the kt store break the CPU kernel?

The in-server bisect narrowed the corruption to the pinning step alone -- no
gather, no stream fork, no event, no routing. This reproduces it offline in
seconds: run the CPU expert forward, pin the store, run the identical forward
again, and compare. Same inputs, same weights, so any difference is caused by
the registration itself.

    LD_LIBRARY_PATH=.venv/lib .venv/bin/python bench/pin_breaks_cpu_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("KT_RAWINT4_BACKEND", "avx512_packed")
# Reproduce the in-server condition: shm-backed store, registered as ONE SPAN
# per expert (what pf_build_table does) rather than region by region.
SPAN = os.environ.get("PROBE_SPAN", "0") == "1"
if os.environ.get("PROBE_SHM", "0") == "1":
    os.environ["KT_STORE_SHM"] = "1"

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "int4_scripts"))
from _paths import W4  # noqa: E402
from expert_stream_kernels import host_register  # noqa: E402

from kt_kernel.utils.amx import NativeMoEWrapper  # noqa: E402

L, HID, MOE, NE, K = 3, 6144, 2048, 256, 8
TOKENS = 4


def cpu_forward(w, x, ids, wts):
    out = w.forward(x, ids, wts, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return out.float().cpu().clone()


def main() -> int:
    w = NativeMoEWrapper(
        layer_idx=L, num_experts=NE, num_experts_per_tok=K,
        hidden_size=HID, moe_intermediate_size=MOE,
        gpu_experts_mask=torch.zeros(NE, dtype=torch.bool),
        cpuinfer_threads=int(os.getenv("CPUINFER", "28")),
        threadpool_count=2, weight_path=W4, chunked_prefill_size=2048,
        method="RAWINT4", numa_nodes=[0, 1],
    )
    w.load_weights(torch.arange(NE, dtype=torch.long))
    moe = w.moe

    g = torch.Generator(device="cuda").manual_seed(1234)
    x = (torch.randn(TOKENS, HID, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(NE, generator=g, device="cuda")[:K]
                       for _ in range(TOKENS)]).to(torch.int64)
    wts = torch.softmax(torch.randn(TOKENS, K, generator=g, device="cuda"), -1).float()

    before = cpu_forward(w, x, ids, wts)
    print(f"before pinning: finite={bool(torch.isfinite(before).all())} "
          f"absmax={before.abs().max():.6f} sum={before.double().sum():.6f}")

    ptrs = moe.expert_store_ptrs()
    sizes = list(moe.expert_store_bytes())
    n_tp = len(ptrs)
    pinned = 0
    for tp in range(n_tp):
        for e in range(NE):
            row = ptrs[tp][e]
            if not row:
                continue
            if SPAN:
                lo = min(row)
                hi = max(o + sizes[i] for i, o in enumerate(row))
                host_register(lo, hi - lo)
                pinned += hi - lo
            else:
                for i, off in enumerate(row):
                    host_register(off, sizes[i])
                    pinned += sizes[i]
    print(f"pinned {pinned / 1e9:.2f} GB across {n_tp} partitions")

    after = cpu_forward(w, x, ids, wts)
    print(f"after  pinning: finite={bool(torch.isfinite(after).all())} "
          f"absmax={after.abs().max():.6f} sum={after.double().sum():.6f}")

    same = torch.equal(before, after)
    print(f"\nCPU expert output identical after pinning: {same}")
    if not same:
        d = (before - after).abs()
        n = int((before != after).sum())
        print(f"  {n}/{before.numel()} elements differ, max |delta| = {d.max():.6f}")
        print("  => cudaHostRegister on the kt store CORRUPTS the CPU expert path.")
    else:
        print("  => pinning is innocent here; the in-server fault is elsewhere.")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
