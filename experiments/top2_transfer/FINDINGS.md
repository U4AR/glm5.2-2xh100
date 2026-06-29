> ## ⚠️ SUPERSEDED — see [BLOG_TOP2_EXPERTS.md](../../BLOG_TOP2_EXPERTS.md)
>
> This file's headline conclusion ("expert placement is decode-speed-neutral") was
> **wrong**. The `_kt_reroute_to_gpu` hook documented below lives in `glm4_moe.py`,
> but GLM-5.2 (`GlmMoeDsaForCausalLM`) runs its MoE through **`deepseek_v2.py`** —
> so this hook was **dead code that never executed**, and the "neutral" result was
> baseline-vs-baseline.
>
> A working reroute (in `deepseek_v2.py`, `_kt_topk_experiment`) shows placement
> **does** matter: keeping the true top-K experts and **substituting** the tail with
> the best GPU-resident experts gives **1.25× (KEEP=4) to 1.5× (KEEP=2)** faster
> decode, coherently. Use [`run_fast.sh`](../../run_fast.sh). Dropping the tail
> (rather than substituting) is garbage; substituting all 8 (KEEP=0) hits ~2× but
> degenerates. Full story + the methodology that caught the degeneration:
> **[BLOG_TOP2_EXPERTS.md](../../BLOG_TOP2_EXPERTS.md)**. Original notes below for history.

---

# Experiment: "move the top-2 experts to GPU, route the other 6 from GPU"

**Branch:** `experiment/top2-expert-transfer`  •  **Date:** 2026-06-29  •  **Model:** GLM-5.2 W4AFP8
(256 routed experts/layer, top-8, 75 MoE layers), 2× H100 NVL, `GPU_EXPERTS=104`.

## The idea under test
A paper noted the top-2 routed experts dominate. Proposal: per token, move the top-2 (if RAM-resident)
to the GPU and let the router fill the other 6 from GPU-resident experts — paying for accuracy with
speed, on the theory that the CPU expert path is the decode bottleneck.

## Key implementation insight: no weight transfer is needed
The kt hybrid path **already computes CPU-resident experts in place** and overlaps them with the GPU
experts (`per-layer = max(cpu, gpu)`). So the idea reduces to a **pure router-logit mask**: keep the
top-K experts (any device) and force the remaining `8-K` slots to GPU-resident experts. The CPU then
computes only the ≤K kept experts it owns; no PCIe transfer of weights at all.

Implemented as `_kt_reroute_to_gpu()` in `sglang/srt/models/glm4_moe.py`, gated by:
- `KT_REROUTE_GPU=1` — enable
- `KT_REROUTE_KEEP=K` — K=2 is the idea (keep top-2, fill 6 from GPU); K=0 = pure GPU-only routing.

(Shape-static, CUDA-graph safe. Reads the per-layer GPU mask via `experts.quant_method.gpu_experts_mask_cuda`.)

## Results (decode tok/s, 5-run median, temp 0; coherence verified each run)

| Config | CPU experts/layer | tok/s | Coherent? |
|---|---|---|---|
| Baseline (reroute off) | ~5 of 8 | **14.93** | yes |
| KEEP=2 (keep top-2, fill 6 from GPU) | ~1.25 | **14.89** | yes |
| KEEP=0 (pure GPU-only, CPU idle) | 0 | **14.94** | yes |

**Placement is speed-neutral.** Reroute is confirmed active (KEEP=2's greedy output *differed* from
baseline — at temp 0 that can only happen if routing changed).

## Why — decode is not expert-compute bound
The decode critical path is **per-step launch/serialization + per-layer fixed overhead** (NCCL
all-reduce, attention, the kt submit/sync scaffolding), not MoE expert math, which is already hidden
under the CPU/GPU overlap. Corroborating evidence:
- **CUDA-graph vs eager: 14.9 vs 4.0 tok/s (3.7×).** Graphs only remove kernel-launch/dispatch
  overhead — a 3.7× win means most eager wall-time was overhead, not compute. (Matches the historical
  2.6× CUDA-graph win.)
- Moving 6–8 experts CPU→GPU (zero transfer) changed nothing.

## Supporting microbenchmarks
- **PCIe** (`pcie_bench.py`): 18.87 MB/int4 expert, 46–58 GB/s, transfers overlap compute on a copy
  stream. Even worst-case cold transfer of the top-2 every layer → ceiling **20–32 tok/s**, *above*
  the current 14.6. So the original "PCIe halves throughput" objection was also wrong.
- **Expert reuse** (`analyze_trace.py`, 244 decode steps): working set **208/256** distinct experts
  per layer; top-8 temporal reuse **0.33**; LRU cache C=104 still misses **0.22** → ~1.77
  transfers/layer even cached. Locality is low, so caching would *not* make per-token transfer cheap —
  but this is moot given placement is speed-neutral.

## Verdict
The idea is sound computer science and the implementation works and stays coherent — but it targets
the wrong bottleneck for **this** system. Expert placement does not gate decode throughput here.
Real decode levers remain speculative decoding / MTP and cutting per-step overhead.

## Repro
`launch.sh` boots the server with env overrides (no foreground `sleep` — the agent harness kills
those). `evaluate.sh` runs the tok/s + coherence suite. See `../../README.md` §2a for the base recipe.
Add `KT_REROUTE_GPU=1 KT_REROUTE_KEEP=2` to reproduce the idea.
