#!/usr/bin/env python3
"""Trace-aware probe for streaming routed experts RAM->GPU at decode time.

This does not modify the server. It answers the first-order question for the
proposed experiment:

  If true top-1/top-2 routed experts that are not already GPU-resident are copied
  to GPU before expert computation, how much PCIe time is added per token?

The current GLM-5.2 W4AFP8 launch uses uniform placement, which means logical
experts [0, GPU_EXPERTS) are resident on GPU for every MoE layer. The saved trace
contains logical top-k ids per decode token/layer.
"""

from __future__ import annotations

import argparse
import statistics as stats
import time
from collections import defaultdict

import torch


HIDDEN = 6144
INTER = 2048
MOE_LAYERS = 75
EXPERT_BYTES_W4AFP8 = 3 * HIDDEN * INTER // 2


def measure_h2d(nbytes: int, device: torch.device, iters: int) -> tuple[float, float]:
    cpu = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True, device="cpu")
    gpu = torch.empty(nbytes, dtype=torch.uint8, device=device)
    for _ in range(8):
        gpu.copy_(cpu, non_blocking=True)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        gpu.copy_(cpu, non_blocking=True)
    torch.cuda.synchronize(device)
    dt = (time.perf_counter() - t0) / iters
    return dt, nbytes / dt / 1e9


def load_decode_rows(path: str) -> dict[int, list[list[int]]]:
    trace = torch.load(path, map_location="cpu", weights_only=False)
    by_layer: dict[int, list[list[int]]] = defaultdict(list)
    for layer_idx, topk in trace:
        topk = topk.view(-1, topk.shape[-1])
        for row in topk:
            by_layer[int(layer_idx)].append([int(x) for x in row.tolist()])
    return by_layer


def summarize(values: list[float]) -> str:
    return (
        f"mean={stats.mean(values):.3f} median={stats.median(values):.3f} "
        f"min={min(values):.0f} max={max(values):.0f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="experiments/top2_transfer/topk_trace.pt")
    ap.add_argument("--gpu-experts", type=int, default=104)
    ap.add_argument("--baseline-tok-s", type=float, default=15.0)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device)
    by_layer = load_decode_rows(args.trace)
    layers = sorted(by_layer)
    if not layers:
        raise SystemExit("empty trace")

    one_dt, one_bw = measure_h2d(EXPERT_BYTES_W4AFP8, dev, args.iters)
    two_dt, two_bw = measure_h2d(2 * EXPERT_BYTES_W4AFP8, dev, args.iters)

    top1_cpu_counts: list[int] = []
    top2_cpu_counts: list[int] = []
    move_one_counts: list[int] = []
    move_two_counts: list[int] = []

    steps = min(len(v) for v in by_layer.values())
    for step in range(steps):
        step_top1_cpu = 0
        step_top2_cpu = 0
        step_move_one = 0
        step_move_two = 0
        for layer in layers:
            ids = by_layer[layer][step]
            top2 = ids[:2]
            top1_cpu = int(top2[0] >= args.gpu_experts)
            top2_cpu = sum(int(e >= args.gpu_experts) for e in top2)
            step_top1_cpu += top1_cpu
            step_top2_cpu += top2_cpu
            # Case B: move only one of the top-2 to GPU, leave the other top-2
            # expert on its original path. If both are CPU, move one; if one is
            # CPU, move that one; if none are CPU, no transfer is needed.
            step_move_one += int(top2_cpu > 0)
            # Case A: both top-2 should compute on GPU, so every nonresident
            # top-2 expert must transfer.
            step_move_two += top2_cpu
        top1_cpu_counts.append(step_top1_cpu)
        top2_cpu_counts.append(step_top2_cpu)
        move_one_counts.append(step_move_one)
        move_two_counts.append(step_move_two)

    base_ms = 1000.0 / args.baseline_tok_s
    mean_one = stats.mean(move_one_counts)
    mean_two = stats.mean(move_two_counts)
    # One expert transfers are the right unit because dynamic top-2 can be 0, 1,
    # or 2 nonresident experts per layer/token.
    one_extra_ms = mean_one * one_dt * 1000.0
    two_extra_ms = mean_two * one_dt * 1000.0

    print(f"trace={args.trace}")
    print(f"layers={len(layers)} decode_steps={steps} gpu_resident=experts[0:{args.gpu_experts})")
    print(f"expert_payload={EXPERT_BYTES_W4AFP8 / 1e6:.2f} MB (W4AFP8 gate+up+down)")
    print(f"h2d one_expert={one_dt * 1000:.3f} ms {one_bw:.1f} GB/s")
    print(f"h2d two_experts_contiguous={two_dt * 1000:.3f} ms {two_bw:.1f} GB/s")
    print()
    print("nonresident top expert counts per generated token across all MoE layers:")
    print(f"  true top-1 CPU-resident: {summarize(top1_cpu_counts)} / {len(layers)}")
    print(f"  true top-2 CPU-resident: {summarize(top2_cpu_counts)} / {2 * len(layers)}")
    print()
    print("dynamic transfer cases:")
    print(f"  move one of top-2 when needed: {summarize(move_one_counts)} transfers/token")
    print(f"    extra transfer time ~= {one_extra_ms:.1f} ms/token")
    print(f"    projected throughput from {args.baseline_tok_s:.2f} tok/s baseline: "
          f"{1000.0 / (base_ms + one_extra_ms):.2f} tok/s before any GPU-compute win")
    print(f"  move both top-2 when needed:   {summarize(move_two_counts)} transfers/token")
    print(f"    extra transfer time ~= {two_extra_ms:.1f} ms/token")
    print(f"    projected throughput from {args.baseline_tok_s:.2f} tok/s baseline: "
          f"{1000.0 / (base_ms + two_extra_ms):.2f} tok/s before any GPU-compute win")
    print()
    print("break-even CPU time that must be removed:")
    print(f"  move-one case must save > {one_extra_ms:.1f} ms/token from CPU expert path")
    print(f"  move-two case must save > {two_extra_ms:.1f} ms/token from CPU expert path")


if __name__ == "__main__":
    main()
