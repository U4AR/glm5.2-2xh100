# Agent task: kill the per-MoE-layer GPU stall in the kt hybrid-MoE decode path

You are working on **GLM-5.2 W4AFP8 (int4 experts) served on 2× H100 NVL (TP=2)** via
**ktransformers (kt) heterogeneous CPU/GPU MoE + sglang**, in repo `/data/models/RunGLM`.
Single-stream (batch-size-1) decode runs at **~14.74 tok/s**. Your job: reduce the per-step
GPU-idle time by attacking the per-layer CPU↔GPU stream synchronization in the kt MoE path,
**without breaking output coherence and without regressing tok/s**. Measure everything; this
project has a history of plausible-but-wrong perf claims, so prove each result.

## What is already known (do NOT re-derive — it's measured and solid)

Decode runs as **ONE fused CUDA graph per step** (verified: `cudaGraphLaunch == 1.00/step`;
`enable_torch_compile=False`; the overlap scheduler is ALREADY ON — `disable_overlap_schedule=False`
in the live `server_args`). Per ~70 ms step: **GPU busy ~26 ms (37%), GPU idle ~44 ms (63%)**.

The idle is **~75 gaps/step of ~0.1–1.7 ms each — exactly one per MoE layer (the model has 75 MoE
layers: 78 total − 3 dense)**. The gaps are *inside* the single graph. They are **identical when CPU
experts compute nothing** (a routing experiment that sends all top-8 to GPU-resident experts gave
the same idle), so the stall is **NOT expert matmul, NOT graph segmentation, NOT scheduler Python**
(the scheduler overlaps the graph fine). It is the **fixed per-layer cost of the kt hybrid MoE's
cross-stream handoff**: in `KTEPWrapperMethod.apply()` every MoE layer does
`stage-copy(main stream) → fork cpu_stream + submit CPU host-node → GPU expert GEMM(main) →
main stream wait_event(cpu_stream done) → merge`. That captured per-layer `wait_event` /
host-node dispatch bubbles the GPU main stream ~0.5–1 ms × 75 layers ≈ the 44 ms.

### ⚠️ Hard constraint that shapes BOTH approaches: decode is a STATIC CUDA graph
During graph **replay**, Python does NOT run — the captured ops execute as-is. So you **cannot** add a
data-dependent Python `if "are any experts CPU-routed this token?"` to skip the submit/sync at
decode: that branch only runs at *capture*, and the decision must be valid for ALL future replays.
Any "skip the CPU path" decision must be **static per layer** (known at capture time, true for every
token), or implemented as a real CUDA-graph conditional node. Plan around this — don't waste a day on
a Python branch that silently bakes one capture-time decision into every step.

## The two approaches to try

### Lever 1 — remove/cheapen the per-layer cross-stream sync
Start with the **ZERO-COST env-var probes first (no code, no rebuild)**, because they may already
answer whether the bubble is the 2-stream machinery vs something else:
- `SGLANG_KT_HYBRID_NO_CPU_STREAM=1` — collapses the CPU-expert stream onto the main stream
  (removes the fork + `wait_event`). Measure idle & tok/s. If idle drops, the 2-stream event
  handshake is the cost and you've found the lever cheaply. If tok/s regresses (CPU compute now
  inline-serializes), it confirms the stream split is load-bearing and you need a smarter fix.
- `SGLANG_KT_HYBRID_TIMING=1` (+ optionally `SGLANG_KT_HYBRID_TIMING_DEEP=1` for one-shot triage)
  — logs per-stage wall times (submit/mask/gpu/sync/merge/cpu_wait) for layers 0,5,20,35 on TP0.
  Use it to see whether `cpu_wait` is ~0 (→ bubble is pure dispatch overhead, not compute) which
  the KEEP=0 result already implies.

Then, if a code change is warranted: the static, replay-safe version is to **skip the submit/sync
for any layer whose `num_gpu_experts == global_num_experts`** (all experts GPU-resident for that
layer → no CPU path needed, valid for every token). With the current *uniform* placement no layer
hits this, so to exercise it you'd pair it with **non-uniform placement** (put 100% of a subset of
layers' experts on GPU, fewer on others, same VRAM budget). Evaluate whether concentrating full-GPU
coverage on some layers (zero bubble there) beats uniform (small bubble everywhere). The mask is
`self.gpu_experts_mask` / `self.num_gpu_experts` in the wrapper.

### Lever 2 — deepen the pipeline so a layer's GPU work overlaps the previous layer's CPU sync
Today `apply()` overlaps GPU GEMM with CPU compute *within one layer*, then waits. The residual
stream forbids deferring layer N's MoE merge past layer N+1's attention, so a naive cross-layer
defer breaks numerics — **don't** do that. The tractable angle: reduce how long the main stream
waits by **issuing the CPU submit earlier** (e.g., right after the router/topk is known, before the
GPU GEMM) and **moving the `wait_event`/merge as late as legally possible within the same layer** so
more GPU work hides the CPU latency. Check whether the current order already does this (it submits in
Step 1 and syncs in Step 4 — see code) and whether the bubble is dispatch latency (then pipelining
won't help and you should report that and stop) vs genuine cpu_wait (then there's headroom). The
TIMING probe above tells you which; **if `cpu_wait≈0`, Lever 2 cannot help — say so and don't ship a
no-op.**

## Files

- Hot path: `/data/models/RunGLM/.venv/lib/python3.12/site-packages/sglang/srt/layers/moe/kt_ep_wrapper.py`
  - `KTEPWrapperMethod.apply()` ≈ line 2948. Step 1 stage-copy + submit on `_cpu_stream` ≈ lines
    3060–3086; Step 3 GPU GEMM ≈ 3120–3167; Step 4 sync + `wait_event` + merge ≈ 3173–3192.
  - `submit()` ≈ 2861, `sync()` ≈ 2889, `_submit_cpu_forward` ≈ 2830, `_sync_cpu_forward` ≈ 2851.
  - Env hooks already present: `SGLANG_KT_HYBRID_NO_CPU_STREAM` (≈3072/3175),
    `SGLANG_KT_HYBRID_TIMING` (≈2984), `SGLANG_KT_BYPASS_GPU_MOE`.
  - NOTE there are 3 copies of this file in the tree; the **live** one is the `.venv` path above.
    Edit that one. (The repo's `ktransformers/third_party/sglang/...` copy is not what runs.)
- Launcher: `/data/models/RunGLM/run_server_int4.sh` (read the header comments — knobs like
  `GPU_EXPERTS`, `MEM_FRACTION`, `DISABLE_NSA`, `KT_GPU_PREFILL_THRESHOLD`).
- If you change C++/kernels you must rebuild kt-kernel (see memory + `install.sh build` with
  `CPUINFER_USE_CUDA=1`); a manual cmake produces a CPU-only `.so` that drops
  `submit_with_cuda_stream` — don't do that. Most of Levers 1–2 are Python-only.

## How to boot, measure, and validate (scripts in `bench/perf_probe/`)

Boot wrapper (survives this harness; uses GPU_EXPERTS=104, RAWINT4 avx512_packed):
```
cd /data/models/RunGLM
nohup bash bench/perf_probe/launch.sh /data/models/RunGLM/run.log [ENV=VAL ...] >/dev/null 2>&1 &
# e.g. add  SGLANG_KT_HYBRID_NO_CPU_STREAM=1  as a trailing KEY=VAL to test Lever-1 probe
# wait for readiness via /health (do NOT use foreground `sleep` — the harness kills it):
( for i in $(seq 1 60); do curl -s -m2 localhost:8000/health >/dev/null && break; sleep 5; done )
```
Boot takes ~3 min (weights on /cache nvme). Confirm effective flags by grepping the `server_args=`
line in the log.

Measure:
- **tok/s** (decode, temp 0, 5 runs, single stream): `python bench/perf_probe/decbench.py 200 5`
  Baseline to beat: **14.74 tok/s**. A change that lowers tok/s is a regression even if "cleaner".
- **GPU idle anatomy**: capture a fresh profiler trace then analyze:
  `python bench/perf_probe/profile_decode.py /tmp/prof 25` (starts a gen, profiles 25 decode steps),
  then `python bench/perf_probe/idle_vs_graph.py /tmp/prof/<TP-0 trace>.json.gz`
  (reports busy/idle %, gaps/step, where the idle sits). `whatfills.py` and `step_anatomy.py`
  give per-step host-API and thread breakdowns.
- **Coherence (MANDATORY for any change that alters numerics)**: send a few prompts and verify
  sensible output — e.g. "list the first 10 prime numbers", "what is 17*23", "write a python
  function for fibonacci". Garbage/repetition = your change broke the merge math; revert.

## Success criteria
A win = **GPU idle/step measurably down AND tok/s ≥ 14.74 (ideally up) AND output coherent.**
Report the before/after idle ms, gaps/step, tok/s (5-run median), and a coherence sample for each
config you try. A negative result is a valid, valuable outcome — if the bubble is pure dispatch
overhead that neither lever can touch, prove it (TIMING `cpu_wait≈0`, NO_CPU_STREAM doesn't help)
and recommend stopping.

## Constraints / landmines
- **Do NOT touch MTP / speculative decode.** It's broken on this flashmla+kt+spec-v2 stack (two
  sglang bugs: `FlashMLAMultiStepDraftBackend` missing `on_after_cuda_graph_warmup_pass` for
  num_steps>1; `PrefillMetadata has no block_kv_indices` at decode for num_steps=1) and is
  explicitly out of scope.
- This box is shared. Before trusting any "it got slow" verdict, run `ps --sort=-pcpu -eo pid,pcpu,comm | head`
  — a runaway process once cratered all kt perf (decode 14→0.77) and faked a regression.
- `GPU_EXPERTS=104` uses ~88 GB/card (≈8 GB free). Don't push it higher blindly (112 OOMs). Spec/
  draft buffers are NOT in play here.
- Always free the GPUs between runs: `pkill -9 -f sglang.launch_server`, then poll
  `nvidia-smi --query-gpu=memory.used --format=csv,noheader` until ~0 (teardown lags a few sec).
- The harness kills foreground `sleep` (exit 144); use background until-loops for waits.
- Restore the working baseline when done (no env overrides = the 14.74 tok/s config).

## Deliverable
A short report: which configs you tried, the measured idle-ms/gaps/tok-s/coherence for each, whether
Lever 1 and/or Lever 2 helped, the root mechanism you confirmed (dispatch overhead vs cpu_wait), and
a recommendation. If a code change wins, leave it applied to the live `.venv` file and note exactly
what you changed.

---

# CONCLUSIONS (run 2026-06-29 — RESULTS)

## TL;DR
The 44 ms/step GPU idle IS the kt per-MoE-layer CPU-expert **submit/sync host-node**, and removing it
gives a real **2.3× decode speedup (14.74 → 34.10 tok/s)**. **BUT the CPU path is LOAD-BEARING — it
computes the *majority* of every layer's routed-expert output** — so the submit/sync cannot be skipped
without producing garbage. Both Python-level levers (1 cheapen/remove the sync; 2 deepen the pipeline)
are **DEAD**. The win is only reachable via a kt-kernel **C++** change or **MTP**; placement/stream
tricks do nothing.

## ⚠️ CORRECTION to the "known" section above — the CPU path *is* load-bearing
This prompt's "What is already known" said the stall is "identical when CPU experts compute nothing (a
routing experiment that sends all top-8 to GPU-resident experts gave the same idle)" and concluded the
bubble is "NOT expert matmul." **The first half is right (idle is fixed dispatch overhead, not
cpu_wait), but it led to a wrong corollary that the CPU path is droppable.** Two measured corrections:

1. **`KT_REROUTE_GPU=1 KEEP=0` never actually rerouted compute to the GPU.** On GLM's GROUPED-topk +
   `e_score_correction_bias` routing, masking `router_logits` to `-inf` is ignored by the group/bias
   selection, so the CPU still computes its experts. The "same idle with CPU doing nothing" experiment
   was NOT actually making the CPU do nothing — it just changed routing slightly while the CPU path ran
   in full. (Direct proof below.)
2. **When you genuinely remove the CPU submit/sync, output collapses**, because the CPU contributes the
   dominant share of routed-expert magnitude every layer (measured cpu/gpu abs-sum ratio **1–100×**).

So: the per-layer host-node bubble is the bottleneck (2.3× ceiling) AND the work it gates is essential.
Earlier sessions that called placement "speed-neutral" were correct *for tok/s*, but they never tested
TRUE GPU-only routing, and they should not be read as "the CPU experts are skippable."

## Measured results (5-run median, temp 0, single stream, GPU_EXPERTS=104)
| config | wall/step | GPU busy | GPU idle | big gaps/step | tok/s | coherent? |
|---|---|---|---|---|---|---|
| baseline (no env) | 70.6 ms | 26.3 ms (37%) | 44.3 ms (63%) | 78 | **14.74** | ✅ |
| `SGLANG_KT_HYBRID_NO_CPU_STREAM=1` | 71.3 ms | 26.8 ms | 44.5 ms | 151 | 14.67 | ✅ |
| static skip submit/sync + reroute | **29.9 ms** | 25.6 ms (86%) | **4.2 ms** | **1** | **34.10** | ❌ garbage |
| `KEEP=0` reroute, no skip (control) | — | — | — | — | 14.75 | ✅ |

## What each probe proved
- **Lever 1a — `NO_CPU_STREAM=1` (collapse cpu_stream onto main):** idle UNCHANGED (44.5 vs 44.3 ms),
  tok/s flat. It only shattered the ~75 big per-layer gaps into 75 small + 24 big — same total. ⇒ the
  **two-stream event handshake is NOT the cost.** Also: `SGLANG_KT_HYBRID_TIMING_DEEP=1` is unusable —
  its `torch.cuda.synchronize()` is illegal during graph capture and crashes the server
  (`cudaErrorStreamCaptureUnsupported`); plain `TIMING` only logs at capture because `apply()` Python
  never runs during graph replay.
- **Lever 1b — static skip of Step-1 submit + Step-4 sync/merge** (env `SGLANG_KT_SKIP_CPU_SUBMIT`,
  added to `apply()` then REVERTED): idle collapsed **44.3 → 4.2 ms**, GPU 37% → 86% busy, 78 → 1
  gap/step, wall 70.6 → 29.9 ms, **34.10 tok/s**. ⇒ **CONFIRMED the entire 44 ms idle = the captured
  per-MoE-layer CPU-expert submit/sync host-node dispatch.** Same GPU work; the idle was pure
  host-node serialization inside the single fused graph.
- **Coherence kill / load-bearing proof:** the skip degenerated to repetition on fib/bat-ball (the one
  "coherent" short prompt was luck). Added a one-shot `KT_DEBUG_CPU_NORM` log of `cpu_output` vs
  `gpu_output` abs-sum at capture (= decode bs=1, `cuda_graph_max_bs=1`). Under `KEEP=0` reroute the
  CPU DOMINATES: ratios 1–100× (layer 6 gpu≈0 / cpu=43; layer 8 cpu=860 / gpu=704; layer 5 cpu=12.6 /
  gpu=1.8). ⇒ kt CPU experts produce most of each layer's routed output; submit/sync is load-bearing.
- **Lever 2 — deepen the pipeline:** cannot help. The bubble is fixed host-node DISPATCH latency (not
  `cpu_wait`), and you cannot overlap-away a host-node that is *captured inside* the static graph and
  whose output (the dominant CPU contribution) is consumed in the same layer's residual. No-op; not shipped.

## Online research
CUDA-graph **conditional nodes** (device-evaluated branch to skip the CPU path per token inside the
static graph) need **CUDA 12.8** and have **no PyTorch authoring API** → not a usable lever here.
Refs: developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes,
docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html.

## Recommendation (do NOT chase Levers 1/2 in Python again)
The 2.3× is real but gated by load-bearing CPU work. Coherent paths to it, in order of tractability:
1. **kt-kernel C++:** cut the per-layer host-node round-trip — one host-node per *step* instead of per
   *layer*, or replace the host callback in `submit_with_cuda_stream` with a GPU-side semaphore the
   captured graph can wait on without a host bounce. This is THE lever (2.3× ceiling). Needs rebuild
   (`install.sh build` + `CPUINFER_USE_CUDA=1`).
2. **MTP / spec-decode** to amortize the 75 bubbles over multiple tokens/step — out of scope and
   currently broken on this flashmla+kt+spec-v2 stack.
3. **Fit all 256 experts on GPU** (no CPU path at all) — VRAM-blocked on 2×H100 (112 experts OOMs).

## State left behind
All probe edits REVERTED; live `kt_ep_wrapper.py` parses clean, no experiment hooks. Baseline restored
and verified: **14.71 tok/s, no env overrides, coherent.** Scripts in `bench/perf_probe/`. Memory:
`glm52-top2-transfer-experiment.md` (6th validation).
