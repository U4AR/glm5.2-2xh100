# Streaming True Top-K Experts to GPU: Probe Results

Date: 2026-06-29

## Question

Test the idea the user intended: keep the true routed experts, but move selected
CPU-resident top experts from RAM to GPU before computation, then compute them on
GPU. This is different from the already-validated substitution trick, which keeps
top-K and replaces the tail with already-GPU experts.

Cases requested:

- `move-one`: of the true top-2, move one CPU-resident expert to GPU and leave
  the other on its normal path.
- `move-two`: move both true top-2 experts to GPU when they are CPU-resident.

In principle, this should preserve model quality because it computes the same
experts, just on a different device.

## What I measured

Added `stream_topk_probe.py`, a no-server probe that combines:

- the existing real decode routing trace:
  `experiments/top2_transfer/topk_trace.pt`
- current uniform placement with `GPU_EXPERTS=104`, meaning logical experts
  `[0, 104)` are already GPU-resident for each MoE layer
- real W4AFP8 expert payload size:
  `3 * 6144 * 2048 / 2 = 18.87 MB` per expert
- measured pinned H2D bandwidth on `cuda:0`

Run:

```bash
source .venv/bin/activate
python experiments/top2_transfer/stream_topk_probe.py --baseline-tok-s 15.0
```

## Results

H2D copy speed:

- one W4AFP8 expert: `0.329 ms`, `57.3 GB/s`
- two contiguous experts: `0.656 ms`, `57.5 GB/s`

Trace-aware transfer demand across 75 MoE layers:

- true top-1 CPU-resident: mean `44.0` layers/token
- true top-2 CPU-resident: mean `88.7` expert transfers/token
- `move-one`: mean `62.6` transfers/token
- `move-two`: mean `88.7` transfers/token

Projected transfer overhead:

- `move-one`: `~20.6 ms/token`
- `move-two`: `~29.2 ms/token`

With a 15 tok/s baseline (`~66.7 ms/token`), that projects to:

- `move-one`: `~11.5 tok/s` before any CPU-compute/host-sync savings
- `move-two`: `~10.4 tok/s` before any CPU-compute/host-sync savings

Break-even:

- `move-one` must remove more than `20.6 ms/token`
- `move-two` must remove more than `29.2 ms/token`

The earlier per-layer sync study measured the full CPU submit/sync bubble at
about `44 ms/token`, so the idea is not ruled out by PCIe bandwidth alone.

## Important implementation constraint

The serving decode path runs as a static CUDA graph. Python in
`KTEPWrapperMethod.apply()` runs during capture, not on every graph replay. A
literal per-token "look at this token's top-2, copy those expert weights to a GPU
slot, update `gpu_experts_mask`, then compute them on GPU" cannot be implemented
as an ordinary Python branch in the decode hot path.

A real server implementation would need one of:

- a C++/CUDA-graph-aware kt-kernel change that performs dynamic expert slot
  copies and routing during replay, or
- disable CUDA graphs for this experiment, which would validate functionality but
  almost certainly destroy the 15 tok/s regime, or
- a static approximation such as pre-promoting top experts before capture, which
  is not the user's requested per-token RAM-to-GPU movement.

## Current recommendation

Do not spend a full server restart on another substitution/reroute benchmark for
this question. The next meaningful step is a focused C++/CUDA-graph design for a
small per-layer streaming cache, then benchmark `move-one` and `move-two` in the
actual captured graph.

The probe suggests the idea has theoretical headroom (`44 ms` bubble versus
`29 ms` top-2 transfer), but it is only worthwhile if the implementation can
avoid reintroducing per-layer host-node overhead and can preserve CUDA graph
replay speed.
