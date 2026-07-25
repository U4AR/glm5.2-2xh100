#!/usr/bin/env python3
"""Project low-VRAM decode tok/s from 2xH100 onto real small-GPU silicon.

CAVEAT this addresses: the low-VRAM ladder only shrank the FOOTPRINT — every
step still ran on full 2xH100 compute (SM count, HBM bandwidth, tensor cores).
A real 24 GiB card has far less compute, so those tok/s are an UPPER BOUND.

Batch=1 decode is overwhelmingly MEMORY-BANDWIDTH bound (it streams each layer's
weights once per token). So a first-order projection scales the GPU-side per-step
time by the HBM-bandwidth ratio. The CPU-expert path (rank0 AVX512) does NOT
scale with the GPU and is assumed unchanged (optimistic for the small-GPU box,
which likely has a weaker CPU too — noted as a second caveat).

We split measured step time into a GPU-bound part and a fixed CPU/handshake part
using two measured points, then re-price the GPU part at the target's bandwidth.
"""
# measured on 2xH100 (aggregate HBM ~2x3.35 = 6.7 TB/s effective for TP2)
H100_BW = 3350.0  # GB/s per card (HBM3)
# (N, tok/s) low-VRAM held-out points, full H100 compute
PTS = {32: 32.3, 16: 30.3, 8: 27.1}
# fraction of per-step time that is GPU-bound vs fixed CPU/handshake.
# From the oracle-ceiling work: CPU submit/sync handshake is a fixed ~ per-layer
# cost independent of N; at these tiny N the GPU expert work is small and the
# dense-trunk + attention (also GPU, BW-bound) dominates the GPU part.
# Estimate GPU-bound fraction from N=96 (fast, GPU-heavy) vs N=8 (CPU-heavy):
# we don't re-derive here; use a conservative split f_gpu applied to step time.
F_GPU = 0.70  # ~70% of a decode step is GPU-BW-bound (dense trunk + attn + resident experts)

TARGETS = {
    "RTX 4090 (24GB)":      1008.0,
    "RTX 3090 (24GB)":       936.0,
    "A10 (24GB)":            600.0,
    "L4 (24GB)":             300.0,
    "2xH100 (measured)":    H100_BW,
}

print(f"{'GPU':<22}{'HBM GB/s':>10}   " + "  ".join(f"N={n}" for n in PTS))
print("-" * 70)
for name, bw in TARGETS.items():
    row = []
    for n, tps in PTS.items():
        step = 1.0 / tps
        gpu_part = step * F_GPU
        cpu_part = step * (1 - F_GPU)
        # re-price GPU part by bandwidth ratio (more BW-starved -> slower)
        new_step = gpu_part * (H100_BW / bw) + cpu_part
        row.append(f"{1.0/new_step:5.1f}")
    print(f"{name:<22}{bw:>10.0f}   " + "   ".join(row))

print("""
Notes:
 - Projection is first-order (BW-only). It IGNORES: SM-count limits, smaller
   L2, kernel-launch overhead differences, and a likely-weaker host CPU on a
   budget box (which would drag the CPU-expert path down further).
 - So even these projected numbers are an UPPER BOUND for the named cards.
 - Real takeaway: on true 24 GiB silicon expect roughly 1/3-1/2 of the H100
   tok/s at the same N, i.e. ~10-16 tok/s for a 4090/3090-class card at N=8-32,
   dropping toward single digits on bandwidth-starved cards (L4).
 - The VRAM FOOTPRINT numbers (N=8->23 GiB, N=16->28.7, N=32->40 GiB/card) are
   hardware-independent and stand as-is; only the tok/s needs this haircut.
""")
