# Expert prefetch: eliminating the CPU expert path at top2

Successor to `EXPERT_STREAMING_PLAN.md`, which parked at its own gate because the
predicted payoff at the shipped tier was 1.000x. Two findings since then reopened
it, both from questioning the measurement rather than the mechanism:

1. **The repack is ~6% of what I costed it at.** `w4afp8.py:318-349`
   (`process_weights_after_loading`) interleaves *scales only*. The int4 weight
   bytes are never permuted. The CPU kernel header
   (`rawint4_packed_avx512vnni-moe.hpp:23,117`) calls its layout the "Kimi native
   INT4 layout" and states the encoding was "verified against the coherent GPU
   cutlass path". Both sides appear to read the same weight bytes. Scales are
   0.59 MB against 9.44 MB of weights, they are static per expert, and they can
   be interleaved **once at boot** -- driving the per-step repack to zero.

2. **The contention constant was calibrated in the wrong regime.** The
   1.21 ms/GB came from a run moving 17.26 GB/step at ~93% of link saturation.
   The streaming workload moves ~3.55 GB/step aggregate. Across the Phase 1b
   artifacts the coefficient swings 3.26 / 3.64 / 1.07 ms/GB -- it is measuring
   queueing structure, not bytes, and extrapolating it linearly into a
   half-empty link overstates the penalty.

## The target, and the hard ceiling

Measured same-boot ladder, safe mode, GPU_EXPERTS=104, MTP depth-3
(`bench/profile_out/phase1_e104_safemode_tiers.json`):

| tier | ms/step | accept | tok/s |
|---|---:|---:|---:|
| top0 | 53.57 | 3.571 | 67.84 |
| top2 | 65.81 | 2.778 | **41.89** |
| top8 | 147.78 | 3.125 | 21.20 |

The CPU expert path at top2 is `65.81 - 53.57 = 12.24 ms/step`. Deleting all of
it lands at the 53.57 ms floor:

| | ms/step | tok/s | vs today |
|---|---:|---:|---:|
| top2 today | 65.81 | 41.89 | -- |
| repack hoisted, contention ~0.5 ms | 54.07 | 51.38 | **1.23x** |
| zero contention (asymptote) | 53.57 | 51.85 | 1.24x |

**51.85 tok/s is the hard ceiling and the plan must not claim more.** top0's
67.84 is unreachable at honest routing: at the floor both tiers have *identical*
step times, and the entire remaining 1.29x is accept length (3.571 vs 2.778).
top0 accepts more draft tokens because it ignores the router and emits
degenerate repetitive text that MTP predicts almost perfectly. That bonus is not
purchasable without the breakage.

1.23x clears the 10% gate that the previous plan failed.

## Routing constants (`bench/profile_out/phase2_distinct_experts.json`)

At top2, a verify step carries 4 draft tokens x 2 genuine experts = 8 slots/layer:

- **U = 3.60** slots/layer land on non-GPU-resident experts (the CPU's work)
- **D = 2.50** distinct experts serve those slots
- **reuse = 1.44** -- one transfer removes 1.44 slots of CPU work

`reuse` exists *only* because MTP puts 4 tokens in a step. At batch 1 without
MTP it is exactly 1.0. **MTP and streaming are complements**: MTP owns the
accept-length dimension, streaming owns the step-time dimension, and MTP makes
each transfer 1.44x more valuable. Concurrent batching restores reuse the same way.

---

> **STATUS 2026-08-04.** Stage 0 BUILT and passing (`bench/placement_score.py`,
> retrodicts the 6-point ladder to 0.4%). Stage A2 PASSES. Stage D measured
> ahead of schedule because it turned out to be the binding constraint. The
> routing constants below were re-measured **in decode, in safe mode**, and
> several of them were wrong; see "Corrected constants". The honest payoff on
> this machine is **~1.14x at top2, not 1.23x**, and the limiter is prediction
> accuracy, not transport. Details at the end.

## Stage 0 -- the placement quality score (the instrument)

Everything downstream needs a cheap, deterministic verdict on "did the right
experts end up in the right tier", because tok/s is noisy and confounded
(accept length, thermals, background load -- cf. the `ugrep` runaway that faked
a 14 -> 0.77 tok/s regression).

**Do not use hand-picked weights (10 / 2 / 1).** Two reasons:

- *Non-linearity.* A layer costs `max(cpu, gpu) + fixed submit/sync`. The first
  non-resident expert in a layer costs nearly the whole penalty; the 2nd-8th
  ride the same submit/sync. A linear per-expert score reports large gains where
  the clock does not move -- exactly the trap behind "oracle coverage 69% -> 92%
  barely moves tok/s" and the voided SSD ladder.
- *Two currencies.* GPU-vs-RAM is latency. RAM-vs-SSD is **fidelity**, because
  an SSD-tier expert is never fetched on the critical path -- it is dropped and
  substituted (`kt_ep_wrapper.py:189`). Summing them lets a config buy score by
  degrading output.

**Build instead:** `bench/placement_score.py`, reporting a pair.

1. `predicted_ms_per_step` -- derived from the measured tier ladder, denominated
   in milliseconds, dominated by *layers forced onto the CPU path at all* rather
   than raw expert-hit counts.
2. `routing_fidelity` -- fraction of genuine top-K demand actually honored
   (separately: dropped-to-SSD, substituted).

Inputs already exist: `KT_DUMP_TOPK` routing dumps plus the sibling `.masks.pt`.

**Gate:** score must reproduce the measured top0/1/2/4/6/8 ladder within ~10%.
An instrument that cannot retrodict the known ladder cannot referee new work.

---

## Stage A -- verify the two load-bearing facts (kill gate)

Cheap, and if either fails the ceiling drops back toward 1.06x.

**A1. Does cutlass consume kt's host-store weight bytes with no permutation?**
This is the assumption the whole plan rests on, and it is currently an inference
from two files. Take one expert, stream its bytes from the kt store into a
scratch GPU layer, interleave only the scales, run `cutlass_w4a8_moe`, and
compare against the same expert's resident output. Bit-exact or it is false.

*If false:* measure the real weight permutation cost (the 0.0376 ms in
`phase3_mechanism.json` is a deliberate random-permutation upper bound; a
structured interleave is HBM-bound at ~0.008 ms). Re-price before continuing.

**A2. What does contention actually cost at the real volume?** -- **DONE
2026-08-04, gate PASSES with room to spare.** Measured with
`bench/pcie_contention.py` at 9.7 MB rounds, queue depth 1, at the design
volume. Kill criterion was >4 ms/step; the answer is ~0.33 ms/step.

| run | tier | GB/step | decode | ms/GB | artifact |
|---|---|---:|---:|---:|---|
| duty 0.5 | top2 (CPU path active) | 3.19 | +9.5% | **1.98** | `phase1b_design_d50.json` |
| duty 0.5 | top0 (no CPU path) | 2.49 | +0.6% | **0.14** | `phase1b_design_d50_tier0.json` |
| duty 0.7 | top0, full design rate 61.3 GB/s | 3.31 | +0.6% | **0.10** | `phase1b_design_d70_tier0.json` |

**The contention is 20x worse when the CPU expert path is running.** That is the
final confirmation of Phase 1's head-of-line diagnosis: the bulk traffic is not
saturating PCIe, it is blocking *kt's own per-layer activation round-trips*.
Eliminate the CPU path and there is nothing left to block -- +0.6% even at
61.3 GB/s aggregate, which is above the ~66 GB/s the design needs.

**Consequence that reshapes Stages C and D: the payoff is CONVEX in per-layer
coverage.** A layer that streams *all* its non-resident experts leaves the
contention regime entirely (0.10 ms/GB). A layer that streams *some* keeps the
CPU path alive and pays the expensive regime (1.98 ms/GB) *plus* the retained
submit/sync. Break-even per-expert accuracy is therefore **~59% for partial
streaming** (0.038 ms cost vs 0.065 ms saved) but the real prize only arrives at
whole-layer elimination. Decide at layer granularity, not expert granularity --
which is exactly where a router-confidence gate earns its place.

---

## Stage B -- hoist the scale interleave to boot (repack -> 0)

Store a second, pre-interleaved copy of the **scales only** for CPU-tier experts.
~6.3% of a ~222 GB store is **~14 GB**, against 374 GB available (`free -g`).
Weights are not duplicated -- both paths read the same bytes.

Every expert then has both a CPU path and a stream path, so a prefetch miss
degrades gracefully to the CPU instead of stalling. That is what makes the
prediction work in Stage D optional rather than load-bearing.

**Gate:** streamed-expert output bit-exact vs the CPU path; boot time +<60 s.

---

## Stage C -- exact-schedule streaming, with depth-1 lookahead

An earlier draft of this stage claimed "prediction may not be needed at all"
because layer L's router names its experts exactly, at layer L. **That was
wrong, and the A2 measurement is what shows it.**

Within a layer the order is attention -> router -> MoE, and the MoE consumes
exactly the experts the router just named. So an exact-schedule fetch has **zero
slack**: it is issued after the router and awaited before the MoE, serially.
2.5 experts x 0.183 ms = 0.46 ms of stall per layer, 34 ms per step -- which
would cost more than the 12.24 ms it is trying to save.

The ~0.72 ms/layer figure is the layer's *entire wall time*, not slack, and A2's
+0.6% was measured with an **asynchronous background** worker -- the transfers
were overlapped with decode, not serialised into it. Overlap is the whole
mechanism, and overlap requires lead time.

So lookahead is not an optimisation on top of streaming here; it is what makes
streaming work at all. Depth 1 is the natural unit and the arithmetic fits:
predict layer L+1's experts during layer L, giving a full layer (~0.72 ms) of
shadow for 0.46 ms of transfer -- ~36% margin. Deeper lookahead buys more shadow
at falling accuracy, which is what Stage D measures.

Build:
1. In-graph UVA gather with device-side indices -- **already built and
   validated** (`bench/stream_mechanism_spike.py`, mechanism A, 0.197 ms/expert,
   `correct_after_index_change: true`). Reuse it.
2. `cudaHostRegister` over the kt store at boot -- already validated at 8 GB/s,
   in place, no extra RAM.
3. Landing slots: ~5 experts' worth double-buffered, ~50 MB VRAM.
4. Mark streamed experts `-1` so kt's `should_skip_expert` drops them.
   **Highest-risk step in the plan** -- the `-1` sentinel is exactly what
   produced the original W4AFP8 NaN corruption. Assert that no expert is
   computed twice and none is dropped.
5. Per-layer corner: when `k == D` the layer routes nothing to the CPU and its
   submit/sync must go away too, or most of the prize is left on the table.

**Gate:** coherent output (66/66 accuracy harness), and ms/step measured against
the 54.07 prediction. If it lands at ~54-56 ms, the plan is essentially done and
Stage D is unnecessary.

---

## Stage D -- lookahead prediction (only if Stage C is transfer-latency-bound)

Enter only if Stage C shows the fetch failing to complete before the MoE needs it.

The mechanism is *not* predicting the next token -- that requires finishing the
current step, and MTP already computes its 4 verify positions together, so there
is no lead time there. The mechanism with real lead time is **predicting layer
L+n's routing from layer L's hidden state**, since the residual stream changes
slowly. A router is 6144x256 (~3 MB); running a future layer's router early
costs microseconds.

Measure offline first, from dumps we already have -- no server changes:

1. **Persistence** -- how often step t+1 routes what step t routed. This is the
   free predictor. The adaptive cache already harvests long-window persistence
   via decayed top-2 counters, so this bounds what a new predictor can add.
2. **Accuracy of layer-L-hidden -> layer-L+n router, as a function of n.**
   Directly gives the usable lookahead depth.
3. **Confidence gating** from router softmax mass. Caveat: the GLM router is
   flat (p=0.9 nucleus -> ~7 of 8 experts), so expect little dynamic range.

Because Stage B keeps a CPU fallback, a miss costs a wasted transfer, not a
stall. Feed measured accuracy through `bench/hybrid_policy.py` to price the
runtime work **before** building it.

**Adverse selection warning:** at top2 the adaptive cache has already promoted
the hot experts into VRAM. What still reaches the CPU is the residual the cache
judged too cold or unstable to hold -- structurally the hardest to predict.
Expect accuracy to come in low here, and note that top8 (D=12.4) is a far
friendlier regime.

---

## Stage E -- portability

`bench/hybrid_policy.py` and `bench/test_hybrid_policy.py` already prove the
policy adapts rather than hardcoding this box: k is monotonic in link bandwidth
and in CPU pole, and a missing or zero-bandwidth profile falls back to all-CPU.

Re-run `bench/calibrate_hybrid.py` after Stage B so the emitted
`hybrid_profile.<hostname>.json` carries the corrected repack (~0) and the
Stage A2 contention. Extend `test_hybrid_policy.py` with the corrected constants.

The regime where this pays far more than 1.23x is the **RAM-starved / three-tier**
case: an SSD-tier expert is dropped today, and prefetch converts a *fidelity*
loss into a latency that lookahead can hide. That is the portability case worth
targeting after this box works.

---

## Risk register

| risk | severity | mitigation |
|---|---|---|
| A1 false: weights need permutation | ceiling 1.23x -> ~1.16x | Stage A gate before any build |
| `-1` sentinel corruption (silent NaN/garbage) | correctness | bit-exact per-layer assert; the known W4AFP8 failure mode |
| submit/sync survives at k == D | most of the prize lost | explicit per-layer corner + timing assert |
| contention worse than extrapolated | ceiling | Stage A2 measures it directly; kill at >4 ms/step |
| prediction accuracy below break-even | Stage D only | CPU fallback preserved by Stage B; price offline first |
| score drifts from reality | wrong verdicts everywhere | Stage 0 gate: must retrodict the measured ladder |

---

## Corrected constants (measured 2026-08-04, decode + safe mode)

Everything above this line that quotes `D`, `U` or a tier ladder was derived
from `bench/measure_distinct_experts.py` over a `KT_DUMP_TOPK` dump. Two defects
made those numbers wrong in the same direction:

1. **The dump is prefill-only.** `_kt_dump_topk` is Python inside the captured
   region, and Python does not re-execute on CUDA-graph replay, so it never saw
   a decode step. Decode is the regime the ladder is measured in, and it routes
   differently: the resident set holds the experts the router most wants, so
   residency correlates with router rank in a way prefill understates.
2. **It sliced `ids[:, :K]`, which is not the top-K.** The dumped weights are
   not sorted descending (47 of 876 sampled rows were), so this took an
   arbitrary K of 8. Because residency correlates with rank, an arbitrary K
   lands on the CPU far more often than the genuine top-K does.

Both are fixed by measuring in-graph instead (`KT_PLACE_SCORE=1`). Per layer per
step at top2, decode, safe mode, GPU_EXPERTS=104:

| quantity | plan (prefill, positional) | measured (decode, genuine) |
|---|---:|---:|
| U -- CPU expert-token units | 3.60 | **2.19** |
| D -- distinct CPU experts | 2.50 | **1.69** |
| reuse U/D | 1.44 | **1.30** |
| layers touching the CPU at all | -- | **80%** (60.9 of 75.9 calls/step) |

The transport picture improves accordingly: full elimination at top2 needs
**2.42 GB/step**, which is 39.4 ms at the measured 61.3 GB/s aggregate, or
**~66% duty of a 59.5 ms floor step**. It fits. Transport is not the limiter.

Fitted cost model (`bench/profile_out/placement_model.json`), 6-point ladder,
two free parameters, worst retrodiction error **0.4%**:

    step_ms = 59.54 + 0.0323 * active_layer_calls + 0.0691 * cpu_units

Note what the fit says about the convexity assumption: at top2 the fixed
per-layer term is only **12-19%** of the CPU cost and the marginal per-unit term
carries the rest. The "first non-resident expert costs nearly the whole penalty"
model is not supported for the CPU pole. Convexity survives only in the
*contention* term measured in A2, which is a smaller effect.

## Stage D result -- prediction is the binding constraint

Measured in-graph with `KT_PRED_LOOKAHEAD`, scored against the router's genuine
top-2, with a depth-0 self-check that reads **100.0%** (so the numbers below are
hidden-state drift, not a harness artefact):

At the shipped tier (top2):

| predictor | top-2 recall |
|---|---:|
| "fetch what this layer needed last step" (persistence) | **27.1%** |
| run layer L+1's router early, on layer L's hidden state | **75.2%** |
| ... L+2 | 67.4% |
| ... L+4 | 57.3% |
| ... L+8 | 43.4% |

The per-expert miss rate is **flat at ~24% across every tier** (top1 23.1%,
top2 24.8%, top4 24.4%, top6 24.1%, top8 23.9%) -- it is a property of how fast
the residual stream moves, not of the configuration, which is a good
portability signal. What collapses with tier is WHOLE-LAYER coverage, because
more experts must all hit at once: 69% of layers fully covered at top1, **55%
at top2**, 28% at top4, 12% at top6, 4% at top8.

Do not read the top0 row (35.6% miss): top0 emits degenerate text, so its
hidden-state trajectory is not representative of anything shippable.

**The user's proposal is the right mechanism and it is 2.4x better than the
obvious alternative.** Running the next layer's router early beats recent-history
prefetch decisively, and it is nearly free (a 6144x256 GEMM on four tokens).

An earlier version of this measurement used raw gate logits and reported 55.7%
at depth 1 -- but also 71% at depth 0, i.e. most of the apparent error was the
sigmoid/bias/group scoring path, not the residual stream. Using the target
layer's own `topk` module removed that confound entirely.

### What 75.2% is worth, per layer, at top2

Per ACTIVE layer-call (61.11 of 75.94 calls/step touch the CPU), 2.09 distinct
CPU experts and 2.70 expert-token units:

- a layer that routes to the CPU costs `0.0255 + 2.70 x 0.0721 = 0.220 ms`
- fully covered (`0.752^2.09` = **55% of layers**): saves the whole 0.220 ms and
  pays only the cheap contention regime, `0.040 GB x 0.10 = 0.004 ms`
- partially covered (45%): hits 1.57 of 2.09 experts, saving
  `1.57 x 0.0721 = 0.113 ms`, but the layer still submits/syncs AND pays the
  expensive regime, `0.040 GB x 1.98 = 0.078 ms` -> only +0.035 ms

Partial coverage is barely above break-even, which is the ~59% threshold from
A2 restated with real numbers. The win is concentrated in the layers covered
*completely*.

Blending: `0.55 x 0.216 + 0.45 x 0.035 = 0.135 ms` per active layer-call, times
61.11 = **~8.2 ms/step. 67.42 -> ~59.2 ms, 41.66 -> ~47.5 tok/s = 1.14x**,
against a 1.26x ceiling (the 53.59 ms floor). Above the 10% gate, and about
half of what full elimination would buy.

Raising it requires better recall, not more bandwidth. Transferring a predicted
superset does not help here: the link is already at ~66% duty at exact coverage,
and 3 experts/layer would need 4.26 GB/step = 69 ms against a 59.5 ms step.

## Honest summary

Ceiling **~51.5 tok/s (1.23x)** at top2 if the CPU path were eliminated
outright. Not 67.84 -- that number requires degenerate routing.

Achievable with the best predictor measured: **~1.14x** (41.66 -> ~47.5 tok/s).
Transport passes its gate with room to spare; prediction is the limiter. A
depth-1 early-router prefetch hits 75.2% of the experts it needs, but a miss
keeps the layer's CPU submit/sync alive, so only the 55% of fully-covered
layers pay properly. Stage 0 and the Stage D measurement together cost far less
than the Stage B/C build and priced it before any of it was written -- which is
what they were for.

**More bandwidth does not help.** Raising the payoff means raising recall:
better lookahead (predict from a later point in layer L, or blend the depth-1
prediction with persistence), or fewer experts that have to hit at once (a
lower tier, or a larger resident set).
