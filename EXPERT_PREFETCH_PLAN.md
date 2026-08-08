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

### Stage D2 -- superset, blend, and DIRECTLY measured whole-layer coverage

The section above infers whole-layer coverage as `recall ** D`, which assumes a
layer's misses are independent. They are not. The instrument now measures the
quantity itself: per active layer-call, was EVERY non-resident expert that layer
needed present in the predicted set. It also scores a predicted SUPERSET (fetch
the predicted top-P for P > K) and a BLEND with persistence (union the
prediction with what this same layer needed last step).

Measured 2026-08-04, decode, safe mode, GPU_EXPERTS=104, prediction point
`pre` (layer L's MoE input), scored against the genuine top-2, depth-0 self
check exactly 100.0% in every cell
(`bench/profile_out/placement_model_pred.json`):

| tier | cell | recall | layer coverage | blend coverage |
|---|---|---:|---:|---:|
| top2 | depth 1, P=2 | 77.0% | **60.8%** | 67.4% |
| top2 | depth 1, P=3 | 85.8% | **77.6%** | 80.9% |
| top2 | depth 1, P=4 | 89.1% | **84.8%** | 86.9% |
| top2 | depth 2, P=2 | 69.1% | 50.9% | 58.8% |
| top2 | depth 2, P=3 | 78.7% | 66.7% | 71.8% |
| top2 | depth 2, P=4 | 82.7% | 74.8% | 78.5% |

Three findings, all of which move the number:

1. **The independence approximation was pessimistic.** Directly measured
   whole-layer coverage at depth 1 / P=2 is **60.8%**, not the `0.752**2.09` =
   55% the section above assumed. A layer's misses are positively correlated --
   which makes sense, since a layer that has drifted has drifted for all of its
   experts at once.
2. **A superset is a real lever, and it is bigger than the blend.** Going from
   P=2 to P=4 takes whole-layer coverage from 60.8% to **84.8%**. Nothing else
   measured moves it that far.
3. **The blend with persistence is worth ~7 points on its own** (60.8 -> 67.4)
   and is nearly free in compute, but it is dominated by P=3 at a similar byte
   cost. It stops mattering once P >= 3 (84.8 -> 86.9).

The per-expert miss rate is again **flat across tiers** (recall 77.0 / 77.1 /
77.1% at top2 / top4 / top8 for depth 1, P=2), confirming the portability
signal: drift is a property of the residual stream, not of the configuration.

### Stage D3 -- the corrected fetch set closes the superset question

The fetch columns now intersect the predicted set with the residency mask, so
they count experts that actually cross the link. Sanity check: the depth-0 cell
reports fetch == `distinct_need` == 1.99 exactly, as it must.

Measured, top2, depth 1 (`bench/profile_out/placement_model_fetchfix.json`):

| cell | recall | layer coverage | experts FETCHED | vs exact |
|---|---:|---:|---:|---:|
| exact (depth 0) | 100% | 100% | 1.99 | 1.00x |
| P=2 | 76.4% | 60.6% | **2.30** | 1.16x |
| P=3 | 85.3% | 78.1% | **3.83** | 1.92x |
| P=4 | 88.8% | 84.7% | **5.55** | 2.79x |
| P=2 + persistence blend | -- | 66.7% | **3.74** | 1.88x |

The 27%-residency scaling used to estimate these in Stage D2 was optimistic by
32% at P=3 and 46% at P=4 -- which is exactly why it was not priced on.

Priced against the production boot (67.42 ms/step, floor 53.59, 62 active
layer-calls, 0.223 ms of CPU per active call = 0.040 fixed + 0.183 marginal,
9.7 MB/expert/card aggregated over both cards, 61.3 GB/s, contention 0.10
ms/GB when the layer is fully covered and 1.98 ms/GB when it is not):

| cell | GB/step | link ms | **link duty** | ms/step | tok/s | speedup |
|---|---:|---:|---:|---:|---:|---:|
| exact (unreachable) | 2.34 | 38.1 | 71% | 53.82 | 51.61 | 1.253x |
| P=2 | 2.70 | 44.1 | **76%** | 57.92 | 47.97 | **1.164x** |
| P=3 | 4.50 | 73.4 | **129%** | 56.75 | 48.95 | 1.188x |
| P=4 | 6.52 | 106.3 | **188%** | 56.65 | 49.04 | 1.190x |
| P=2 + blend | 4.39 | 71.6 | **123%** | 58.41 | 47.56 | 1.154x |

**The superset is priced out, twice over.** It fails the link outright -- P=3
needs 129% of the step and P=4 needs 188%, so neither is schedulable at any
policy. And even with an infinite link it would only move 1.164x -> 1.190x,
because the extra bytes are paid on EVERY active layer-call while the extra
coverage only helps the layers it converts. The persistence blend fails the same
way: +6 points of coverage for +63% bytes is a net loss (1.154x).

**So Stage D2's reopening of "more bandwidth does not help" closes again.** The
original claim was right; it just had not been measured. The operating point is
**P=2, depth 1, no blend: ~1.164x (41.2 -> 48.0 tok/s), at 76% link duty**,
against a 1.258x ceiling. That is the target the Stage B/C build must be judged
against, and 76% duty is close enough to saturation that the A2 head-of-line
regime is a live risk in the partially-covered layers.

One policy idea this does NOT rule out, because it costs bytes only where they
pay: fetch **selectively** rather than on every active layer-call -- skip the
layers whose prediction looks unlikely to cover, spending the freed link budget
on a superset only where it converts a layer. `bench/hybrid_policy.py` already
sweeps k per layer and is the right home for it. Unmeasured.

**Superseded note (Stage D2): the instrument counted the WRONG fetch set.** `fetch_per_call` is `|predicted|` -- 7.31 experts at P=2 -- but
most of those are already GPU-resident and cost nothing to "fetch". The bytes
that actually move are `|predicted AND non-resident|`. Scaling by the observed
non-resident fraction (1.98 needed of ~7.36 distinct routed, 27%) estimates
~2.0 / ~2.9 / ~3.8 experts actually transferred at P = 2 / 3 / 4, i.e. P=4
costs ~1.9x the bytes of exact coverage. At top2 exact coverage is 2.52 GB/step
= 41.2 ms at 61.3 GB/s against a 67.4 ms production step (61% duty), so P=4
would need ~78 ms and does NOT fit, while P=3 at ~60 ms is marginal. **Fix the
instrument to intersect with the residency mask before believing any of this**
-- the 27% scaling assumes predicted-but-not-routed experts are resident at the
same rate as routed ones, which is exactly the kind of assumption this plan has
already been burned by twice.

Note also that the instrumented boot's fitted floor is 109.35 ms, not 59.54:
nine scoring cells per layer per step is ~50 ms of extra kernel launches. The
ACCURACY figures are unaffected (they are ratios of counters), but that boot's
`a_active` / `a_unit` are not a production cost model -- keep using
`placement_model.json` for that.

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

> **AMENDED by Stage D2, then RESTORED by Stage D3.** D2 reopened this: a
> predicted superset takes whole-layer coverage from 60.8% to 84.8%, so recall
> looked buyable with bandwidth. D3 measured the bytes properly and closed it
> again -- P=3 needs 129% of the step's link budget and P=4 needs 188%, and even
> at infinite bandwidth the superset is only worth 1.164x -> 1.190x. The
> statement stands, now on measurement rather than assumption.

---

## Where this stands, 2026-08-04 EOD

**Done and trustworthy:** Stage 0 (instrument, retrodicts 1.25% on a second
independent boot), Stage A2 (contention, passes), Stage D (recall by depth),
Stage D2 (superset / blend / direct whole-layer coverage).

**Next session, in order.**

1. ~~Fix the fetch columns to intersect with `~allow_vec`~~ -- **DONE**, and
   ~~price P = 2,3,4 against the link~~ -- **DONE, see Stage D3.** The operating
   point is **P=2, depth 1, no blend, ~1.164x**; the superset is priced out.
2. **Measure `KT_PRED_POINT=post`** (already implemented, never run): predict
   from layer L's OUTPUT residual stream instead of its MoE input. Higher recall
   -- the only drift left is layer L+1's attention -- but the transfer's shadow
   shrinks to that attention block. Two numbers, one tradeoff, one boot.
3. **Selective fetching** -- the one lever D3 did not rule out. Spend link
   budget only where it converts a layer, instead of on every active layer-call.
   `bench/hybrid_policy.py` already sweeps k per layer.
4. **Then, and only then, Stage B/C**, judged against the 1.164x operating
   point. The build has not started.

**Loose ends, all still open.** `bench/measure_distinct_experts.py` still
carries both defects that produced the wrong constants (prefill-only dump,
positional `ids[:, :K]`) and should be deleted or rewired to the instrument;
`bench/calibrate_hybrid.py` still sources `--distinct` from it, and
`hybrid_profile.H100-VM1.json` / `hybrid_policy.py` / `test_hybrid_policy.py`
still carry D=2.50 / U=3.60 / contention 1.21 ms/GB. The whole instrument lives
in the vendored sglang tree under `.venv`, which is gitignored and therefore
**not version controlled** -- a venv reinstall loses it.

---

## Stage F -- the gather's root cause, found and fixed (2026-08-06)

Stage C shipped, was coherent, and lost: the full path measured **84.67 ms/step
against a 66.64 baseline (0.79x)**, with the gather contributing **22.46 ms and
getting essentially ZERO overlap** despite being forked a whole layer ahead of
its consumer. Five explanations were proposed and each was killed by
measurement or by reading the source:

| hypothesis | verdict | evidence |
|---|---|---|
| link capacity | no | 38% duty, 2.6x headroom |
| fence placement | no | ~2 ms of shadow for a 0.3 ms gather, verified in source |
| host DRAM contention | no | DRAM at 38% of 379 GB/s; asserted for several turns before checking, and wrong |
| SM occupancy / block count | no | throttling to `BLOCKS=8` made it **worse**, 94 -> 102 ms |
| CUDA host nodes | no | kt's `cudaLaunchHostFunc` submit/sync do land blocking host nodes in the graph, but `bench/graph_hostnode_probe.py` shows **99% of the gather still hides** with them present |

**The actual cause was in the kernel, visible the whole time.** In
`bench/expert_stream_kernels.py`, `k_w13_weights` moved `uint4` -- 16 bytes per
thread-iteration, no division. `k_w2_weights`, directly beneath it, moved
`uint8_t`: **one byte per thread-iteration, with an emulated 64-bit
`i / src_row_bytes` on every byte.** Half the bytes, eight times the loop trips.

That made the transfer **latency-bound rather than bandwidth-bound**, which has
two consequences that are fatal together:

1. It reaches bandwidth only by brute-force occupancy -- 512 blocks of warps
   that spend their lives stalled on PCIe. Stalled warps still hold their SM
   slots, so a gather that needs the whole GPU to move bytes locks the main
   stream's compute off the machine and cannot overlap with anything.
2. It cannot be made polite, because narrowing a latency-bound loop just makes
   it proportionally slower. Measured standalone at the real shapes:

| kernel | 8 blocks | 64 blocks |
|---|---:|---:|
| `w2` current, 1 B + 64-bit divide | 1.418 ms / 4.1 GB/s | 0.193 ms / 30.4 GB/s |
| `w2` vectorised, `uint4` | **0.127 ms / 46.2 GB/s** | 0.127 ms / 46.1 GB/s |
| `w13` (already `uint4`, reference) | 0.244 ms / 48.1 GB/s | 0.244 ms / 48.0 GB/s |

This is why the `BLOCKS=8` experiment read as a refutation of the occupancy
hypothesis when it was actually confirming it: the diagnosis was right and the
remedy was unavailable until the kernel changed.

### The fix is two changes and neither works alone

Vectorising makes it bandwidth-bound; being bandwidth-bound is what makes a low
block count survivable; a low block count is what stops it holding every SM
slot. Verified **byte-identical** to the old kernel at 8/64/256 blocks for
n_tp 1 and 2 (`bench/gather_kernel_probe.py --verify`). Full path, numerically
exact, same box state:

| | ms/step | tok/s | gather exposed | hidden |
|---|---:|---:|---:|---:|
| old kernel, `BLOCKS=64` (grid 512) | 84.67 | 33.74 | 22.46 | 0% |
| fixed kernel, `BLOCKS=64` | 82.77 | 34.84 | 20.29 | 10% |
| fixed kernel, `BLOCKS=16` (grid 128) | 72.97 | 39.32 | 10.81 | 52% |
| **fixed kernel, `BLOCKS=8`** (grid 64) | **70.91** | **40.38** | **8.70** | **61%** |

Kernel rewrite alone is worth 1.9 ms; narrowing the grid on top of it a further
11.9 ms. `KT_PREFETCH_BLOCKS` default changed **64 -> 8**: the knob now wants to
go DOWN, the opposite of what the old kernel wanted.

Cross-check: the `ROUTE=0` bytes-only rows give exposed = 8.77 ms at 8 blocks,
against 8.70 derived from the full path. Two independent routes agree.

### It still does not ship on this machine

70.91 against a 66.85 baseline is **0.94x**. The mechanism is now correct and
the transfer is largely hidden; what remains is that the costs still exceed the
saving. Decomposed, all measured:

    predictor + pf_issue select ops    +5.03 ms/step
    gather still exposed at 8 blocks   +8.70
    CPU experts skipped                -9.46
    net                                +4.27  ->  0.94x

The predictor's 5.03 ms is **not arithmetic** -- per layer it is a gate GEMM on
at most 8 tokens plus a topk scoring path plus `pf_issue`'s ~28 small tensor
ops, i.e. ~2100 extra nodes in the captured graph at a couple of microseconds
each. The fix is fusion, drafted in `bench/pred_select_kernel.py` (one launch,
one block, replaces the ~28). `KT_PREFETCH_SELECT=0` splits the 5.03 between
the router and the select ops so the right half gets fused.

**Not** the fix: predicting from the previous step's routing instead of running
a lookahead router. Stage D already measured persistence at **27.1%** whole-layer
coverage against the router's **75.2%** -- it would save ~4 ms and give back ~6.

## Stage E result -- where this DOES pay

`bench/prefetch_portability.py` models the payoff from measured constants and
reproduces the server to within 0.7 ms (predicts 83.95 where the old path
measured 84.67; 0.794x vs 0.79x), so it is calibrated rather than notional.

The decisive quantity is the composition of the CPU pole: **0.89 ms/layer, of
which only 0.13 ms is expert arithmetic and 86% is kt's fixed submit/sync
latency.** The fixed part is synchronisation, so it does NOT grow on a slower
host, while the compute the prefetcher eliminates does. Link speed stops
mattering once the transfer fits the idle window, which it does with 2x margin
here (F = 0.45).

| CPU vs this box | 1x | 2x | 3x | 4x | 6x | 8x |
|---|---:|---:|---:|---:|---:|---:|
| speedup at 61% hidden | 0.94 | 1.08 | 1.21 | 1.33 | 1.55 | 1.74 |

This host is close to the worst case in circulation for the technique: an
80-core EPYC 9V84 with AVX-512-VNNI and 379 GB/s of DRAM behind an ordinary
PCIe Gen5 link. It computes an expert almost as fast as the link can ship one
(R = 0.38), so there is little to win. That is a property of this machine, not
of the mechanism.

### Method note

Five hypotheses died because each probe tested one variable while holding the
others at their broken values, and because the microbenchmarks did not match
the real access pattern -- a 64 MB single-kernel pull said "throttle it", and
throttling cost 8 ms/step. What finally worked was reading two kernels sitting
next to each other in the same file and noticing that one used `uint4` and the
other `uint8_t`.

## Stage G -- chain prediction: walk the stream forward through resident experts

The predictor shipped in Stage C/D shows layer L's hidden state to layer L+d's
router and eats d layers of drift for it. The alternative proposed here is to
spend the window in which the CPU is computing this layer's non-resident
experts running an APPROXIMATE layer on the GPU-resident experts only -- route
normally, substitute every non-resident expert with the best resident one, add
the update to the residual stream, renormalise, run the next layer's router --
and repeat, walking a few layers ahead. The prediction is then made from a
stream that has actually moved.

The chain costs a real MoE forward per layer of lookahead, so it is only worth
building if it is MORE ACCURATE. That is a purely statistical question and it is
settled first, in `bench/chain_predict.py` (instrument), `chain_predict_run.sh`
(boot), `chain_predict_report.py` (table).

Measured 2026-08-06: eager (DISABLE_CUDA_GRAPH=1, the chain runs arbitrary
Python per layer so it cannot be captured), MTP off, GPU_EXPERTS=104, safe2,
592 decode steps, chains started every 6th layer, scored against the router's
genuine top-2. **Both depth-0 self-checks read 100.0%**: `direct` validates the
scoring path, `renorm` validates that the residual stream this harness stitched
together is the one the model actually normalised.

### The arms, and what each isolates

| arm | signal | available when? |
|---|---|---|
| `persist` | what this layer needed last step | free |
| `direct` | h_L -> gate_{L+d} (SHIPPED) | in the CPU shadow |
| `renorm` | post_ln_{L+d}(r_L) | in the CPU shadow |
| `post` | + layer L's REAL MoE update | only AFTER the CPU returns |
| `chain_shared` | walk, shared expert only | in the CPU shadow |
| `chain` | walk, resident experts (PROPOSAL) | in the CPU shadow |

`post` is the odd one out and must not be read as a competitor: it needs layer
L's true MoE output, which is exactly what the prefetch exists to avoid waiting
for. It is in the table as the CEILING for any depth-1 chain.

### Result -- recall / whole-layer coverage, P=2 (the exact fetch set)

| arm | d=1 | d=2 | d=3 | d=4 |
|---|---:|---:|---:|---:|
| persist | 22.5 / 18.0 | -- | -- | -- |
| direct | 80.0 / 72.6 | 69.5 / 62.0 | 66.5 / 59.7 | 65.4 / 61.3 |
| renorm | 80.4 / 71.6 | 69.9 / 62.3 | 67.4 / 59.8 | 65.3 / 61.4 |
| chain_shared | 81.5 / 72.5 | 72.1 / 64.1 | 68.2 / 59.0 | 67.2 / 61.1 |
| **chain** | **83.7 / 75.3** | **75.4 / 67.6** | **72.3 / 63.2** | **71.0 / 66.1** |
| post (ceiling) | 84.7 / 77.4 | 73.5 / 65.5 | 69.3 / 60.3 | 67.4 / 62.5 |

At P=3 the same ordering holds with everything shifted up ~8 points (chain
91.8 / 87.7 at d=1, 80.7 / 78.8 at d=4).

Five findings:

1. **The chain is better, and it is better at every depth.** +3.7 recall /
   +2.7 whole-layer coverage at d=1, growing to +5.6 / +4.8 at d=4.
2. **What it actually buys is DEPTH, not accuracy.** `direct` falls off a cliff
   between d=1 and d=2 (80.0 -> 69.5) and then flattens; the chain decays
   gently and stays 5-6 points clear from d=2 on. chain@d=4 (71.0 / 66.1) beats
   direct@d=2 (69.5 / 62.0) -- **the chain is worth about two extra layers of
   lookahead reach.**
3. **Renormalisation was never the problem.** `renorm` is within noise of
   `direct` at every depth. That variable is dead.
4. **At d=1 the chain recovers 56% of the available headroom.** The ceiling is
   `post` at 77.4% coverage (knowing layer L's MoE exactly); direct sits at
   72.6, chain at 75.3. So the resident-only approximation is only ~1 point
   worse than knowing the true update -- the substitution is not the weak link,
   the skipped attention is.
5. **It is not bought with bytes.** Fetch cost is 0.55 non-resident experts per
   layer-call for the chain against 0.59 for direct at d=1, i.e. slightly
   CHEAPER. The gain is a genuinely better prediction at the same link duty.

`persist` at 22.5% is the harness's external cross-check: Stage D measured the
same predictor at 27.1% in a different configuration (MTP on, 4 tokens/call),
and it remains an order of magnitude behind either lookahead.

### What it is worth, and where

Cost was NOT measured -- the question asked here was accuracy. Structurally the
chain costs one resident-only MoE forward per layer per depth: gate, topk,
substitution, the grouped cutlass GEMM, the shared expert, and a TP all-reduce.
That is the same order of work as a real expert layer, against a `direct`
predictor whose router half measured 2.25 ms/step for all 75 layers (Stage F).

On THIS box that trade is not close. Depth 1 is all the shadow the transfer
needs (F = 0.45, Stage E), and +2.7 points of whole-layer coverage is worth
about 0.3 ms/step against the Stage D2 blend -- roughly 0.45% of the step --
while the chain would spend far more than that. Add it to the 4.27 ms the
predictor is already underwater by (Stage F) and it is not shippable here.

Where it pays is the case this project is actually building for. A machine whose
CPU is 4-8x slower has a proportionally larger idle window for the approximate
forward to hide in, and a slower link needs MORE than one layer of shadow to
land the transfer -- which is precisely the regime where `direct` has collapsed
to 69.5/62.0 and the chain still reads 75.4/67.6. **The chain is not an accuracy
optimisation; it is what makes deep lookahead viable at all.**

`chain_shared` is the cheap variant worth remembering: shared expert only, no
routed experts, and it still takes 60% of the chain's d=2 gain. If the chain is
ever built, that is the first thing to try.

### Stage G2 -- how far up can you prefetch from? (depths 1..16, every arm)

Same instrument, ladder extended to d = 1,2,3,4,6,8,12,16, chains started every
8th layer, `chain_exact` (walk with GENUINE full routing, CPU experts included)
added as the ceiling. 592 decode steps, both depth-0 self-checks 100.0%.
`bench/profile_out/chain_deep.rank0.json`.

Whole-layer coverage -- the fraction of CPU-touching layer-calls covered
COMPLETELY, which is what a prefetch is actually paid in:

| P=2 (exact set) | d=1 | d=2 | d=3 | d=4 | d=6 | d=8 | d=12 | d=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| persist | 19.6 | | | | | | | |
| direct | 75.8 | 65.3 | 60.8 | 51.8 | 51.2 | 45.8 | 30.7 | 29.7 |
| **chain** | **81.9** | **72.8** | **66.3** | **59.0** | 51.1 | 47.9 | 30.7 | 28.9 |
| chain_exact | 79.2 | 72.7 | 64.0 | 57.1 | 50.7 | 45.0 | 30.4 | 29.0 |
| chain_shared | 77.9 | 68.3 | 61.8 | 52.5 | 46.1 | 41.1 | 24.0 | 19.5 |
| post (needs CPU) | 83.1 | 69.9 | 64.5 | 53.2 | 51.8 | 47.1 | 31.8 | 30.7 |

| P=4 (superset) | d=1 | d=2 | d=3 | d=4 | d=6 | d=8 | d=12 | d=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 91.6 | 84.7 | 79.8 | 72.2 | 69.9 | 62.9 | 44.8 | 41.6 |
| chain | 94.8 | 89.8 | 84.9 | 78.4 | 70.5 | 62.1 | 45.2 | 39.4 |
| chain_shared | 93.2 | 86.5 | 81.6 | 75.5 | 66.0 | 55.5 | 34.4 | 27.4 |

**1. The chain's advantage has a hard horizon at depth ~5.** chain - direct is
+6.1, +7.5, +5.5, +7.2 at d=1..4, then -0.1, +2.1, 0.0, -0.8 at d=6..16. Past
depth 5 the walk buys nothing at all.

**2. The reason is that its error COMPOUNDS while direct's SATURATES.** Direct's
error is "how far has the stream moved in d layers", which flattens (65.3 ->
51.2 from d=2 to d=6, then only slowly). The chain's error is its own
approximation, re-applied d times. Around depth 5 the accumulated approximation
is as large as the drift it was correcting.

**3. It is NOT the resident-expert substitution that compounds.** `chain_exact`
walks with genuine full routing -- CPU experts and all, unshippable by
construction -- and traces the same curve, in fact slightly BELOW `chain` at
d=1 (79.2 vs 81.9). Two consequences: building a better approximation of the
expert update is pointless, and the substituted update tracks production
slightly better than the genuine one does, which is coherent because production
itself runs safe2. What compounds is the SKIPPED ATTENTION, d times over.

**4. Deep prefetch does not work with either method.** At P=2 nothing clears
~60% coverage past d=4; at d=12 every arm is at ~30%, well under the ~59%
break-even from Stage A2/D2. Lead time bought this way is worthless.

**5. If you need reach, widen the fetch set -- do not walk the stream.** Taking
the ladder at >=60% coverage: P=2 reaches d=3 (direct) / d=4 (chain); P=3
reaches d=6; P=4 reaches d=8. **The superset buys ~5 layers of reach, the chain
buys 1.** The catch is that it buys them with link duty -- P=4 names 1.70
non-resident experts per layer-call against 0.62 genuinely needed, ~2.7x the
bytes -- and Stage D3 already priced supersets out on THIS box (P=3 = 129% link
duty). The machine that needs deep lookahead is the machine with the slow link
that cannot afford the superset. That tension is the real finding.

**6. CORRECTION to Stage G.** `chain_shared` was described there as taking "60%
of the chain's d=2 gain"; recomputed on this larger sweep it takes 40% at d=2
and 34% at d=1, and it DIVERGES with depth (19.5% at d=16, far below every
other arm) because omitting the routed experts biases every step of the walk in
the same direction. It is not the cheap substitute that section suggested.

### Stage H -- deep prefetch, for real (tok/s, not accuracy)

Stage G/G2 measured PREDICTION. This measures the fetch actually firing at
depth. Two forces pull opposite ways: coverage falls with depth (75.8 -> 51.8%
from d=1 to d=4) while lead time rises, so more of the 8.70 ms/step exposed
gather can hide instead of blocking.

Same-boot ladder, GPU_EXPERTS=100, 4 landing slots, 8 blocks, safe2, MTP on:

| | ms/step | accept | tok/s |
|---|---:|---:|---:|
| baseline (no predictor, no gather) | 67.08 | 2.857 | 42.59 |
| direct, depth 1 | 73.54 | 2.857 | 39.21 |
| direct, depth 2 | 71.43 | 2.703 | 37.84 |

Read the step rate and the accept length separately: `KT_PREFETCH_CPUSKIP=1`
moves a prefetched expert from the CPU kernel to the GPU one and the two are not
bit-identical, so the draft accepts slightly differently. Depth 2 has the better
step rate and the worse tok/s for that reason alone.

#### The NEXTN draft-capture landmine (fixed)

Depths 3 and 4 did not boot. `cudaErrorStreamCaptureUnjoined`, and the traceback
lands in `eagle_draft_cuda_graph_runner.capture` -- the DRAFT graph, not the
target model's.

The NEXTN/MTP draft is a single layer captured into its own graph, and it runs
the same `DeepseekV2DecoderLayer.forward`, so it fired a prefetch too, aimed at
`its_layer_id + depth`. `pf_issue` forks `pf_stream` and records `pf_done`; the
rejoining `wait_event` lives in the TARGET layer's `apply()`, which never runs
inside the draft's capture. So capture ended with a forked stream outstanding.

It only bites when `draft_lid + depth` lands on a real routed-MoE layer, which
is exactly why depths 1 and 2 booted and depth 3 did not -- a silent
depth-dependent trap, not a general prefetch failure.

Fixed by refusing to emit from a NextN layer at all (`is_nextn` threaded into
`_kt_pred_emit` / `_kt_pred_post_hook` / `chain_prefetch.emit`). That is correct
on the merits regardless: the target model re-runs every layer immediately
afterwards, so the draft's predictions are worthless.

**Rule for anything built on this path: a predictor that fires from the draft
model must either target a layer inside the draft's own graph or not fire.**

#### The capture-safe chain, driving a real fetch (`bench/chain_prefetch.py`)

Built and measured 2026-08-06. It walks inside the decode CUDA graph, so unlike
the Stage G instrument it runs during decode at production speed. Three design
choices, all load-bearing:

- **Step 0 is free and exact.** At the post hook layer L's real MoE update is
  already in hand (`hidden_states + residual` IS the stream entering L+1 minus
  its attention), so the walk starts from truth. At depth 1 it therefore does
  NO expert math at all -- it is purely "predict from layer L's output instead
  of its input".
- **It never touches the CPU path.** It calls `gpu_method.apply` directly rather
  than the kt wrapper's `apply`, which submits and syncs a CPU expert batch on
  every call.
- **Modules are registered at construction**, because capture precedes the first
  request and a registry filled on first forward is captured as nothing.

Same ladder, same box state, baseline 67.08 ms / 42.59 tok/s:

| predictor | depth | walk | ms/step | accept | tok/s | vs base |
|---|---|---|---:|---:|---:|---:|
| direct (shipped) | 1 | -- | 73.54 | 2.857 | 39.21 | 0.921 |
| direct | 2 | -- | 71.43 | 2.703 | 37.84 | 0.888 |
| **chain** | **1** | **none (free)** | **70.43** | 2.899 | **41.58** | **0.977** |
| chain | 2 | 1 layer, 2 slots | 88.63 | 2.817 | 31.83 | 0.747 |
| chain | 2 | 1 layer, 8 slots | 92.04 | 2.857 | 31.04 | 0.729 |

**1. The free half of the chain is worth 3.1 ms/step over the shipped
predictor** (70.43 vs 73.54) and closes the gap to break-even from 8% to 2.3%.
Same predictor cost, better prediction: the counters show 1.83 experts wanted
per layer-call against direct's 2.02, i.e. more layers were covered completely
and dropped their CPU round-trip.

**2. The walk is unaffordable. One walked layer costs 18.2 ms/step** -- 27% of
the whole step, for coverage that is LOWER than depth 1 (72.8% vs 81.9%). Worse
on both axes simultaneously.

**3. Widening the expert slots is NOT where the cost is.** 8 slots costs only
3.4 ms more than 2, so of depth 2's ~21.6 ms overhead only ~3.4 ms moves with
the amount of expert GEMM.

**THE CAUSE OF THE REMAINING ~18 ms IS NOT YET ATTRIBUTED.** It is one extra
approximate layer per real layer, and the candidate terms are the norm, the
gate, the topk scoring path, the ~15-op substitution, the id remap, the shared
expert, the GPU expert kernel's fixed cost, and one extra NCCL all-reduce -- but
which of those carries it has not been measured. Stage F is the standing warning
here: the gather also looked "structurally expensive" until the real cause
turned out to be a byte-granular copy kernel with an emulated 64-bit divide per
byte, and it got 2.6x faster once that was fixed. A large number plus a
plausible mechanism is not a root cause.

Depth 3 hit the VRAM cliff and was not re-measured; it is not evidence either
way until the 18 ms is understood.

**No verdict on the walk yet.** What IS established: the free half of the chain
(exact first step, no walk) beats the shipped predictor by 3.1 ms/step, and the
walk as currently written costs 18.2 ms/step per walked layer. Whether that
figure is intrinsic or a defect is the next thing to decompose -- see Stage H2.

### Your two extra predictors (measured, 240 steps, provisional)

**Cross-layer carry (`prevlayer`) is dead: 1.0% recall, 0.5% whole-layer
coverage.** Predicting layer L+d's experts as the ones layer L just used is
worth nothing -- adjacent MoE layers share almost no experts -- and it names
1.25 non-resident experts per layer-call against `direct`'s 0.58, so it costs
more than twice the bytes to be wrong. If this survives an independent check
against the routing dump it is a real fact worth keeping: expert identity
carries signal across TIME (persistence, 22.8%) but not across DEPTH.

**The persistence blend is free and it works, and it helps MORE with depth.**
Whole-layer coverage, unioning each arm's prediction with what that layer needed
on the previous token:

| arm | d=1 | d=2 | d=3 | d=4 |
|---|---:|---:|---:|---:|
| direct | 74.3 -> **77.0** | 62.9 -> **68.6** | 61.1 -> **67.8** | 58.3 -> **63.8** |
| chain | 74.8 -> **78.0** | 68.3 -> **74.1** | 66.3 -> **73.8** | 63.9 -> **70.2** |

+2.7 points at depth 1 rising to +6.7 at depth 3 for `direct`. That shape makes
sense: persistence does not decay with lookahead depth while the router
prediction does, so the two are most complementary exactly where the lookahead
is weakest. It costs no compute at all -- only the larger union to fetch.

⚠️ The `chain_post` row in that run is VOID: the instrument's walk loop
double-applies layer L's update to an arm that already starts from layer L's
real output. See `TODO_ROOT_CAUSES.md` item 6. The capture-safe mechanism does
it correctly; only the instrument arm is wrong.

## Stage H2 -- the walk's cost, ATTRIBUTED

TODO item 1, closed. `bench/walk_decomp.sh` turns the walk's components on
cumulatively with the prefetch's EFFECT off (`GATHER=0 ROUTE=0 CPUSKIP=0`), so a
stage's degraded prediction cannot feed back into coverage and move the clock.
Eight boots, same box state, accept length identical (2.857) in every row.

| stage | component added | ms/step | delta |
|---|---|---:|---:|
| s0 | chain off (reference) | 66.59 | -- |
| s1 | prediction only, NO walk | 72.85 | +6.26 |
| s2 | + walked layer's gate and topk | 73.42 | +0.57 |
| s3 | + resident substitution, remap, slot mask | 79.97 | **+6.55** |
| s4 | + the GPU expert kernel | 86.58 | **+6.61** |
| s5 | + the shared expert | 89.92 | +3.34 |
| s6 | + the TP all-reduce | 90.30 | +0.38 |
| s0b | reference, repeated | 66.67 | -- |

**The decomposition closes.** Walk total `s6 - s1 = 17.45 ms`; the parts sum to
17.45 ms exactly. **Noise floor `|s0b - s0| = 0.08 ms`**, so every delta above
0.4 ms is real. (Yesterday's 18.2 ms was measured with the prefetch effect on;
17.45 with it off is the same number.)

### My prime suspect was wrong

I named the 75 extra NCCL all-reduces as the likely driver and estimated 2-4 ms.
**They are 0.38 ms -- 2.2% of the walk.** A 4-token all-reduce on NVLink is
simply not expensive. Refuted.

### What actually carries it: two fixed per-invocation costs, ~38% each

- **The substitution search, 6.55 ms (87 us/layer).** `_resident_ids` runs about
  twenty kernels on tiny tensors per layer: a `where` over (T,256), a `clone`, a
  `scatter_`, a `topk` over all 256 experts, an `argsort`, two `gather`s, two
  more `where`s, a `sigmoid`, then the id remap and the slot-mask block with a
  second argsort. ~4.4 us per kernel is graph-node dispatch plus a small launch,
  not arithmetic.
- **The GPU expert kernel, 6.61 ms (88 us/layer)** -- and this is at only TWO
  routed slots on FOUR tokens, i.e. a few hundred microseconds of actual MACs.
  It is the grouped-MoE setup: sorting tokens by expert, building the
  problem-size array, launching the grouped GEMM. This also explains yesterday's
  puzzle that 2 -> 8 slots cost only 3.4 ms: the fixed part dominates and
  widening moved the small part.
- The shared expert adds 3.34 ms, gate and topk 0.57 ms.

**So the walk is dominated by work that does not scale with the useful work
being done.** Both large terms are fixed costs paid 75 times a step to move four
tokens through two experts.

### What this does and does not license

The substitution search is a QUALITY choice, not a requirement -- "keep the
resident slots, zero the rest, renormalise" would delete the topk-over-256, both
argsorts and the gathers, and should recover most of 6.55 ms. Its accuracy cost
is unmeasured. The expert-kernel setup is harder; the walk cannot batch across
layers because the dependency is sequential.

But cost is not why depth 2 loses. **Depth 2 predicts WORSE than depth 1** (72.8%
vs 81.9% whole-layer coverage, Stage G2). Even a walk that cost zero would not
make depth 2 beat depth 1 on this box, so no amount of the optimisation above
changes the shipped choice. The cost work matters only for a machine that NEEDS
two layers of lead time to land the transfer -- and there the walk could plausibly
be made ~4x cheaper before it is judged.

## Stage H3 -- the cheap substitution, BUILT and measured (2026-08-07)

`KT_CHAIN_PF_SUB=drop` in `bench/chain_prefetch.py`: keep the routed slots that
are already GPU-resident, zero the rest, renormalise. No topk over 256, no
argsorts, no gathers -- `mask_and_remap_expert_ids` already writes -1 (the
kernel's skip sentinel) into every non-resident slot, so dropping them is nearly
free. Note it keeps ALL resident slots rather than a top-N of them, precisely
because ranking is the cost being removed.

Five boots, `bench/walk_fix.sh`, same box state. The `-s3`/`-s6` rows have the
prefetch EFFECT off so they read against Stage H2's ladder directly; the
`-real` rows have it fully on, so they are the shippable comparison.

| row | ms/step | tok/s | accept | vs search |
|---|---:|---:|---:|---|
| FIX-base (no chain) | 66.78 | 42.78 | 2.857 | reference |
| DROP-s3 (substitution isolated) | 74.97 | 38.11 | 2.857 | vs 79.97 |
| DROP-s6 (whole walk) | 87.70 | 32.58 | 2.857 | vs 90.30 |
| SEARCH-real (shippable) | 88.56 | 31.70 | 2.817 | -- |
| DROP-real (shippable) | 85.55 | 33.00 | 2.817 | **-3.01 ms, +4.1%** |

**The substitution itself: 6.55 -> 1.55 ms, a 76% cut.** (`DROP-s3 - s2` =
74.97 - 73.42.)

**The whole walk: 17.45 -> 14.85 ms, only a 15% cut.** The difference is not
noise and it is not a measurement problem -- it is the trade being paid back.
Dropping the top-2 cap leaves ~3 live slots per token instead of 2, and the
grouped-MoE setup term (Stage H2's other 6.61 ms) grows with live slots. I
bought 5.00 ms on the substitution and returned ~2.40 ms to the expert kernel.

**End to end, prefetch on: 3.01 ms/step, +4.1% tok/s, accept length identical
(2.817 in both rows).** A clean step-rate win with no numerics artefact.

**It does not change the verdict.** Same-boot baseline is 66.78 ms/step; the
depth-2 walk still costs 18.77 ms/step after the fix (0.771x, against search's
0.741x). And depth 2 predicts worse than depth 1 regardless of cost. The value
of this is for a machine that NEEDS two layers of lead time -- there the walk is
now 15% cheaper, and the fidelity question below decides whether it is also
worse.

Cross-check on box stability: FIX-base 66.78 against Stage H2's s0 66.59 and s0b
66.67, spread 0.19 ms across two days. The cross-day ladder reads are valid.

### The open question this creates

The walk is PAID FOR its fidelity -- a worse hidden state means a worse guess at
the next router, which is the entire product. `search` puts a stand-in with
similar router mass where the missing expert was; `drop` removes it and rescales
what is left. Which lands closer to the true hidden state is not obvious.
`bench/chain_predict.py` now carries a `chain_drop` arm so both are scored in
the SAME eager boot, same trajectory, same tokens (`bench/drop_acc.sh`). Until
that lands, the 3.01 ms is measured but not bankable.

### The fidelity answer: DROP LOSES, and loses to nothing at all

`bench/drop_acc.sh`, 608 decode steps, eager, GPU_EXPERTS=104, safe2, all arms
in ONE boot so the trajectory and tokens are identical. Whole-layer coverage at
P=2 (the payoff metric); the raw run is archived as
`bench/profile_out/chain_dropacc.rank0.json`.

| arm | what it does | d=1 | d=2 | d=3 | d=4 |
|---|---|---:|---:|---:|---:|
| `chain_post` | walk, step 1 REAL, search | **79.5** | **71.0** | **67.6** | **67.6** |
| `post` | layer L's real update, no walk | 79.5 | 68.1 | 63.9 | 62.6 |
| `chain` | walk, SEARCH for substitutes | 78.0 | 70.1 | 67.1 | 67.3 |
| `chain_shared` | walk, shared expert ONLY | 75.1 | 67.1 | 63.4 | 62.4 |
| `direct` | no walk at all (shipped) | 74.3 | 64.7 | 63.1 | 61.3 |
| `chain_drop` | walk, DROP + renormalise | 72.3 | 64.1 | 60.6 | 60.9 |
| `prevlayer` | layer L's expert ids carried | 0.5 | 0.7 | 1.0 | 0.8 |

**Drop is strictly dominated.** It is ~6 points below the search at every depth,
below the arm that runs NO routed experts, and below not walking at all. So the
3.01 ms/step it saves buys a walk that is worse than the thing it replaces --
the cost win is real and unbankable. **The search stays.** `KT_CHAIN_PF_SUB`
keeps both modes; `search` remains the default.

**Cause not attributed.** Why dropping is worse than OMITTING is not
established. The only thing drop does that shared-only does not is renormalise
-- it rescales the two or three surviving experts up to carry the full routed
weight, producing an update of roughly the right magnitude in the wrong
direction, where shared-only produces a small one. That is a hypothesis. The
`chain_drop_raw` arm (drop, do NOT renormalise) isolates exactly that term and
nothing else; until it runs, this is "drop is 6 points worse, cause unknown".

**`chain_post` is validated and is the best arm.** TODO item 6's defect (its
walk double-applied layer L) is fixed, and the fix carries its own proof: at
d=1 `chain_post` must degenerate to `post`, and it reads 79.5 against 79.5.
It beats `chain` at every depth, which is what starting from the layer's REAL
output should do. It is also exactly what `bench/chain_prefetch.py` implements.

Two other readings from the same run: `prevlayer` is at the chance floor at
every depth, independently confirming TODO item 5 -- expert identity carries no
signal across layer depth. And `post` (free -- layer L's real update is already
computed at the hook) matches `chain_post` at d=1 and only falls behind from
d=2, so **at depth 1 the walk is worth nothing at all**; its entire value is at
depth 2+, which is the regime this box does not need.

## Stage H4 -- the prefetch pays for itself, if you show the router the right state

TODO item 3, closed. `bench/item3_attrib.sh`: three boots with the prediction
computed but nothing moved (`GATHER=0 ROUTE=0 CPUSKIP=0`), which isolates each
predictor's fixed cost from what its bytes buy.

| row | ms/step | predictor fixed cost |
|---|---:|---:|
| I3-base, no predictor | 66.79 | -- |
| I3-direct-off | 72.25 | +5.47 |
| I3-chain-off, depth 1 | 72.32 | +5.53 |

**The two predictors cost the same** (0.07 ms apart, noise floor 0.08). My
leading explanation -- that `chain_prefetch.py` is simply lighter code than the
shipped predictor -- is refuted. Subtracting the fixed cost from each row's
total overhead leaves the fetch effect alone:

| | total overhead | fixed | fetch effect |
|---|---:|---:|---:|
| direct predictor (DEEP-d1, 73.54) | +6.46 | 5.47 | **+0.99 ms, a COST** |
| post-state predictor (CPF-d1, 70.43) | +3.35 | 5.53 | **-2.18 ms, a SAVING** |

3.17 ms of difference against a 3.11 ms observed gap: **0.06 ms apart, inside
the noise floor. The decomposition closes.**

### The first configuration where the bytes have ever paid

Every prefetch row before this one was underwater: the gather cost more than the
CPU-expert work it removed (Stage C, 0.79x; Stage F, still +0.99 ms here). At
depth 1 from the POST-layer state the fetch returns 2.18 ms/step more than it
costs. Same bytes, same gather kernel, same landing slots -- the only difference
is which hidden state the lookahead router was shown.

The accuracy instrument says the same thing independently and was measured in a
different regime (eager, MTP off, 104 experts): predicting from the pre-MoE
input (`direct`) covers 74.3% of CPU-touching layer-calls completely at d=1;
predicting from the post-layer residual stream (`post`) covers 79.5%. The post
state is already computed at the hook, so the +5.2 points are free in compute.
What they cost is LEAD TIME -- the fetch now starts after layer L's experts
instead of before them -- and on this box that trade is clearly worth taking.

`KT_PRED_POINT=pre|post` already exists in `deepseek_v2.py`, and the post hook
feeds `_kt_pred_emit` -> `pf_issue`, so this is a switch and not a rewrite.
`bench/predpoint.sh` tests it on the shipped path.

### Why it is still not a win, in arithmetic

`66.79 + 5.53 - 2.18 = 70.14`. The predictor's own 5.47 ms is the whole problem
now, and it is not arithmetic -- Stage F split it into 2.25 ms of lookahead
router and 2.45 ms of selection ops, both dispatch-bound. To beat the 66.79
baseline the predictor must come in under ~2.2 ms.

The fused kernel (`bench/pred_select_kernel.py`) replaces the selection half in
one launch. If it recovered ALL of it, the predictor would be ~2.35 ms and the
full path `66.79 + 2.35 - 2.18 = 66.96` -- a wash, not a win. **The lookahead
router's 2.25 ms has to come down as well.** That is the honest position: two
measured terms and a subtraction, not a projection.

## Stage H5 -- the drop is fine; the RENORMALISE was the defect

TODO item 9, closed. `bench/drop_raw_acc.sh` adds one arm that changes exactly
one term: `chain_drop_raw` zeroes the non-resident slots and does NOT rescale
what is left. 608 steps, every arm in ONE boot, whole-layer coverage at P=2.

| arm | substitution cost | d=1 | d=2 | d=3 | d=4 |
|---|---:|---:|---:|---:|---:|
| `chain` (SEARCH) | 6.55 ms | 77.0 | 70.0 | 66.0 | 67.6 |
| `chain_drop_raw` (drop, no renorm) | 1.55 ms | **75.0** | **67.4** | **65.0** | **66.4** |
| `chain_shared` (no routed experts) | ~0 | 74.8 | 65.7 | 61.1 | 63.0 |
| `direct` (no walk at all) | ~0 | 74.0 | 63.2 | 61.3 | 62.3 |
| `chain_drop` (drop + renorm) | 1.55 ms | 71.4 | 62.3 | 59.7 | 61.7 |

**Removing the renormalisation recovers 3.6 / 5.1 / 5.3 / 4.7 points** and moves
the arm from BELOW the do-nothing baseline to above it. One division, isolated
by a single-variable change, accounts for the entire deficit Stage H3 recorded.

Why it hurts: zeroing the non-resident slots leaves the routed weights summing
to less than one, and renormalising hands the two or three survivors the FULL
routed magnitude -- an update of roughly the right size pointing the wrong way,
which then compounds through the walk. Leaving the sum short is the honest
statement that the missing experts contributed nothing. (That is the mechanism;
what is MEASURED is that this term accounts for all of it.)

`search` still needs the renormalise -- with `_WALK_K < E` its top-K mask zeroes
real slots whose weight genuinely does belong to the survivors. So the division
is correct where it started and wrong where it was copied to.

### Revised operating point for a machine that needs depth 2

`drop_raw` gives up ~2 coverage points against the search and saves 5.0 ms of
the substitution (Stage H3 measured 6.55 -> 1.55 ms). That is the mode to use;
`KT_CHAIN_PF_SUB=drop_raw` is wired into `bench/chain_prefetch.py`, with `drop`
kept only so the comparison stays reproducible. **This does not change the
verdict on THIS box** -- the depth-2 walk is still ~18 ms/step underwater, and
depth 1 needs no walk at all.

### Reading these tables

Within a run every arm shares the trajectory token for token, so those
comparisons are exact. ACROSS runs there is about a point of drift: `chain` at
d=1 read 78.0 in the Stage H3 boot and 77.0 here, `direct` 74.3 and 74.0. Never
compare a number in one table against a number in another.

## Stage H6 -- RETRACTION of Stage H4, and what actually survives

Stage H4 claimed the prediction POINT (pre-MoE input vs post-layer residual) was
worth 3.1 ms/step and flipped the fetch from a cost into a saving. **That is
wrong.** `bench/predpoint.sh` tested it on the shipped path, same day, same
box state, and the point is worth **0.32 ms**, not 3.1.

| row | ms/step | overhead | predictor fixed | fetch effect |
|---|---:|---:|---:|---:|
| PP-base | 66.96 | -- | -- | -- |
| PP-pre (pre-MoE input) | 70.31 | +3.34 | 5.47 | **-2.12 ms** |
| PP-post (post-layer residual) | 69.98 | +3.02 | 5.53 | **-2.51 ms** |

Today's baseline spread across three boots (66.78 / 66.79 / 66.96) is 0.18 ms,
so the 0.32 ms between the two points is real but small -- and 0.39 ms of fetch
effect, not 3.1.

### Where the error came from: a cross-day subtraction

Stage H4 computed the direct predictor's fetch effect as
`(DEEP-d1 - DEEP-baseline) - (I3-direct-off - I3-base)` -- yesterday's prefetch
row minus today's fixed cost. I checked that the two BASELINES agreed (67.08 vs
66.79) and treated that as licence to mix days. It was not. The baselines agreed
while the prefetch rows did not: **DEEP-d1 read 73.54 where today's identical
configuration reads 70.31, a 3.2 ms gap that is entirely inside the prefetch
path.** The whole "+0.99 ms, a cost" figure was that gap, not a property of the
pre-MoE state.

Rule this earns: a baseline agreeing across boots licenses NOTHING about the
rows measured beside it. Subtract only within a run.

### DEEP-d1's 3.2 ms is now itself unexplained

Only one configuration difference is visible: the DEEP and CPF ladders ran at
`run_fast.sh`'s default MEM_FRACTION (0.95), which leaves ~150 MiB of VRAM free
once four landing slots are allocated -- two boots on that ladder died of CUDA
OOM. Today's rows all use 0.94. VRAM pressure on the gather is a CANDIDATE, not
a cause. It also means **every prefetch tok/s number measured at 0.95 is
suspect**, including the Stage F and Stage G2 ladders. See TODO item 7.

### What survives, all measured same-day

1. **The two predictors cost the same fixed amount**: +5.47 (direct) vs +5.53
   (chain depth 1), 0.06 apart. Predictor overhead explains none of any gap.
2. **The fetch now pays for itself, at BOTH prediction points**: -2.12 ms (pre)
   and -2.51 ms (post). This is still the first configuration where the bytes
   have come out ahead -- the conclusion was right, the attribution was not.
3. **The post state is slightly better**, 0.32 ms of step rate and 0.39 ms of
   fetch effect, consistent in direction with its +5 whole-layer coverage points
   but far smaller than that suggested. `KT_PRED_POINT=post` is worth taking and
   is not worth a headline.
4. **Still not a win**: `66.96 + 5.53 - 2.51 = 69.98`, +3.02 ms over baseline.
   The predictor's ~5.5 ms is the entire remaining problem, exactly as before.

## Stage H7 -- where the latency actually is, and why the lookahead cannot win here

One boot, GPU_EXPERTS=100, MEM_FRACTION=0.94, safe2, MTP=1, no prefetch. The
tier is a per-request field, so all three rows come from the SAME server and the
forward shape is identical across them (same 4-token verify batch) -- ms/step is
therefore directly comparable. tok/s is NOT: tier 0 degenerates into repetitive
text that the draft model predicts easily, inflating its accept length to 3.70.

| tier | ms/step | CPU expert path | share of step |
|---|---:|---:|---:|
| top8 -- honour all 8 routed experts | 150.13 | 97.07 ms | 65% |
| top2 -- the shipped default | 66.56 | 13.50 ms | **20%** |
| top0 -- zero CPU experts | 53.06 | -- | the floor |

**At the shipped tier the CPU expert path is 20% of the step.** The remaining
53.06 ms is GPU-side forward work: 78 layers of attention, 75 layers of GPU MoE,
the shared experts, the dense layers, three draft passes and sampling. That
floor is NOT yet decomposed.

### Why the prefetch loses, in one line

The prefetch exists to remove CPU expert work. There is 13.50 ms of it. The
predictor costs 5.47 ms before a byte moves, and the fetch returns 2.1-2.5 ms.
Net +3 ms, and the step gets slower.

**The tier substitution already took the win.** Keeping two true experts and
substituting the rest with resident ones cut the CPU path from 97.07 ms to
13.50 ms -- it removed 86% of precisely the quantity the prefetch was built to
eliminate. Every prefetch row this project has produced was measured in the
regime that lever left behind.

### Why the predictor costs 5.47 ms, which is 73 us per layer

| half | ms/step | measured by |
|---|---:|---|
| lookahead router | 2.25 | PSPLIT-router-only - PSPLIT-baseline (same boot) |
| selection ops | 3.21 | the remainder |

Neither half is arithmetic. The router half is one gate GEMM on a 4-token batch
-- 4x6144 against 6144x256, about 6 MFLOP, microseconds on an H100 -- plus its
scoring tail. The selection half is ~28 kernels each touching a few hundred
numbers. Together that is roughly 36 kernel launches per layer, 75 layers, about
2,700 extra graph nodes per decode step at ~2 us of dispatch each.

Direct evidence that it is launches and not math: `bench/pred_select_kernel.py`
does the identical selection arithmetic in ONE launch and measures 6.9x faster
standalone (4.53 -> 0.66 ms). Nothing about the computation changed.

This is the same pathology measured three separate times today: the walk's
substitution search (20 kernels/layer, 87 us), the grouped-MoE setup for four
tokens through two experts (88 us), and this. **Decode on this box is priced by
launch count**, because every tensor in flight is four tokens wide and no kernel
runs long enough to hide its own dispatch.

Caveat on the split: the 3.21 ms selection figure subtracts two within-run
deltas taken in different boots. The 5.47 ms total is a single same-boot
subtraction and is the firmer number. The standalone harness reads 4.53 ms for
the same ops against the in-server 3.21 -- unexplained.

### The structural cap on scaling it up

`KT_PRED_P` defaults to 2, so the predictor names at most two experts per token
NO MATTER WHAT THE TIER IS -- measured 2.05 wanted/call averaged across all
three tiers. At top8 a layer routes eight experts per token, most non-resident,
so a two-expert prediction cannot cover the layer and the slot rule skips what
it does not cover. That is why raising the pool from 13.50 ms to 97.07 ms bought
only 1.71 ms: the fetch was capped, not badly aimed. Widening it
(`KT_PRED_P=8`, 16 slots) is the open test; the first attempt died on VRAM.

## Stage H8 -- the predictor, folded into the network

Stage H7 established that the predictor's 5.47 ms/step is launch count, not
arithmetic: ~36 kernels per layer on four-token tensors, ~2,700 graph nodes per
decode step. Stage H8 removes them.

### 1. The lookahead GEMM becomes free

At `KT_PRED_POINT=pre`, layer L's own gate and the lookahead for layer L+1 are
applied to the SAME hidden state. So `_kt_fuse_gates` (deepseek_v2.py) re-lays
the 75 gate weights into one contiguous `[75*256, 6144]` tensor after weights
load and before any forward -- the only window, since the first forward may
already be a capture -- and rebinds each layer's `gate.weight` to its own
256-row slice. Row-major slices of rows are contiguous, so every other consumer
sees exactly what it saw before; only the storage moved.

Layer L then holds a **512-row** view spanning itself and its successor, and one
`F.linear` yields both routers. No extra launch, no extra memory. The cost is
reading 3.1 MB more weight per layer, ~0.08 ms/step of HBM at 3 TB/s.

Excluded automatically: the NEXTN draft layer (`is_nextn`, which is set before
the registration runs) and the last MoE layer, both of which keep the ordinary
gate call. The NEXTN exclusion also re-closes the capture bug from Stage G --
a draft layer must never issue a fetch, because the rejoining wait lives in a
target-model layer that never runs inside the draft's graph.

### 2. Everything after the GEMM becomes one launch

`bench/pred_fused_kernel.py` does the scoring, the router's top-8, the top-P
among those, the residency intersect, the demand histogram, the slot pick and
the publication of sel/landed/index/stats in a single kernel. It takes the
lookahead half as a COLUMN SLICE with a row stride, so not even the contiguity
copy costs a launch.

Config specifics that make this tractable, read from config.json rather than
assumed: `scoring_func=sigmoid`, `topk_method=noaux_tc` (so choice = sigmoid +
correction bias), and `n_group=topk_group=1`, which makes the group stage a
no-op. `norm_topk_prob=True` renormalises, but that is a positive per-token
scalar and therefore order-preserving, so the top-P-by-weight is the
top-P-by-raw-score and the renormalise can be skipped entirely.

### The kernel took three shapes, and the first two were wrong

| shape | us/launch | ms/step | why |
|---|---:|---:|---|
| serial scan on thread 0 | 74 | 5.54 | 4 tokens x 8 rounds x 256 comparisons in sequence -- WORSE than the 36 launches it deletes |
| block-wide parallel argmax | 33 | 2.47 | 36 sequential rounds, each paying two block-wide syncs |
| **one warp per token** | **7.6** | **0.572** | a warp holds a whole 256-expert row in registers, 8 per lane; selection is pure shuffles with NO `__syncthreads` in the hot path |

Against the shipped path transcribed in place (7.06 ms/step for the same work),
that is **12.3x**. `--verify` checks it against that transcription across 150
random trials x 6 flag combinations and passes.

Note the standalone harness reads 7.06 ms for work the in-server split prices at
3.21 ms -- the same ~2x overstatement seen with the earlier selection kernel,
still unexplained. The RATIO is the meaningful part; the in-server ladder is
what decides the ms.

### What this predicts, before the ladder returns

Predictor fixed cost should fall from 5.47 ms to roughly 0.6 ms. The fetch
effect was measured at -2.12 ms (pre point, Stage H6). If both hold, the full
path lands near `base + 0.6 - 2.1`, i.e. **~1.5 ms/step BELOW baseline** -- the
first net win this project has produced. Written down before the measurement so
it can be wrong.

## Stage H9 -- the accept-length regression is a race, not rounding

The fused predictor works: 5.47 ms/step down to 1.51, and with the fetch
returning -3.62 ms the whole path is finally net-negative on step time, +3.3%
against baseline. But tokens/second went the wrong way, -4.1%, because MTP
accept length fell 2.857 -> 2.667.

The obvious reading was numerics. A landed expert is computed by the cutlass
W4AFP8 kernel instead of the CPU packed-int4 W4A8 one; the activations are fp8
on one side and int8 on the other, so the two do not agree bit for bit, and a
perturbed target logit rejects draft tokens that were marginal.

That reading is wrong, and the data said so before any code was read. The three
fused-on runs took 73, 77 and 75 forward steps for the same prompt at
temperature 0. Different-but-fixed numerics are still a function: same prompt,
greedy sampling, same answer, same step count, every time. The baseline does
exactly that -- [70, 70, 70], and on the production server five repeats of one
prompt give a single completion hash and agree to 0.1 s of wall time. Three
different answers to the same question is not rounding. Something is racing.

### The mechanism, from the code

`pf_landed` / `pf_landed_cpu` / `pf_index` are written only by the predictor,
and the predictor only fires on a decode-shaped batch (`T <= KT_PRED_TMAX`, 8).
Nothing else ever clears them. But both consumers run unconditionally:

  - `_submit_with_staged_input` applies `pf_landed_cpu[topk_ids] -> -1`, so the
    CPU skips whatever the mask says
  - `apply` folds `pf_landed` / `pf_index` into `pf_eff_mask` / `pf_eff_index`,
    so the GPU routes whatever the mask says to a landing slot

A prefill is not decode-shaped. This benchmark's prompt is ~60 tokens, so the
predictor does not fire, and the masks still hold the LAST DECODE STEP OF THE
PREVIOUS REQUEST. Prefill therefore routes experts to landing slots that hold
some other expert's weights, and tells the CPU to skip experts nothing computed.
The KV cache for the whole prompt is built wrong, and it is built wrong
DIFFERENTLY each time, because what is stale depends on where the previous
generation happened to stop.

In steady-state decode the masks are always fresh -- layer L's are written
during layer L-1's forward inside the same graph replay -- which is why the
mechanism hid: the part of the run that was being timed is the part that is
correct.

### Pre-registered predictions for `bench/determ_ladder.sh`

Written before the rows came back, so they can falsify this rather than
decorate it:

  - `D-base`   1 distinct hash
  - `D-gather` 1 distinct hash, and the SAME hash as D-base. Bytes move but both
    consumers are off, so the masks publish all-false and are a no-op. If this
    row diverges, the transfer is clobbering live weights and the staleness
    story is not the issue -- everything below it is void
  - `D-route`  multiple distinct hashes
  - `D-full`   multiple distinct hashes, and specifically run 0 clean -- the
    masks are zeroed at construction, so the first request after boot has
    nothing stale to inherit -- with runs 1..4 differing from it and from each
    other

If instead `D-route` is deterministic and only `D-full` is not, the staleness
story is wrong for the GPU half and the CPU-skip submit path carries it, which
would point at the D2H/host-callback ordering rather than at mask lifetime.

### The fix this implies

Not zeroing the masks after use: that is two `zero_()` per layer per step, 150
launches, and launch count is the thing this whole stage exists to reduce. The
flag is a HOST-side one, so it costs nothing on the device -- the predictor sets
`pf_fresh`, both consumers require it, `apply` clears it. Under capture the
decode graph records the fresh branch (the predictor is inside the graph, so it
fires on every replay) and the eager extend path records the plain branch, which
is exactly the split that is wanted.

### Results, and a correction to the predictions above

`bench/determ_ladder.sh`, then `bench/determ_fix.sh` after the gate was added.

| row | consumers | distinct / 5 | ms/step | accept | steps | tok/s |
|---|---|---|---|---|---|---|
| D-base / F-base | none | 1 | 66.17 | 2.857 | 70, 70, 70 | 43.18 |
| D-gather | none (bytes only) | 1, SAME hash | -- | -- | -- | -- |
| D-route | GPU | 5 | -- | -- | -- | -- |
| D-full / F-nofix | GPU + CPU-skip | 5 | 64.72 | 2.778 | 67, 72, 83 | 42.92 |
| F-full (gate on) | GPU + CPU-skip | **1** | 66.11 | 2.273 | 88, 88, 88 | 34.38 |

Three of the four predictions held. D-gather matched D-base's hash exactly, so
the transfer is provably innocent and the rows below it are interpretable.
D-route and D-full were 5-for-5 non-deterministic, D-route diverging at char 3.

The prediction that FAILED was "run 0 clean, runs 1..4 differing". Run 0 diverges
too, because CUDA-graph capture runs the predictor during warmup and leaves the
masks set before the first request ever arrives. The mechanism is right; the
claim that boot leaves nothing stale was wrong.

The independent mechanism test did not run. `KT_PRED_TMAX=4096` is incompatible
with `KT_PRED_FUSED=1` -- the fused kernel is one warp per token with 8 warps per
block and raises `one warp per token, 8 warps in the block` on a 2048-token
prefill batch. It needs the unfused predictor, and has not been redone.

Something better arrived instead. F-nofix reproduced D-full's five completion
hashes EXACTLY and in order, across a reboot, on patched code with the gate
disabled. So this was never a hardware race: a read-before-write race on the
landing slots would produce different garbage each boot. It is deterministic
state carry-over -- run N's output is a reproducible function of run N-1's
leftover masks -- which is the staleness mechanism and nothing else. That also
confirms `KT_PREFETCH_FRESH=0` faithfully reproduces the pre-patch code.

The fix works and is not a no-op: F-full is 1-for-5 deterministic AND its hash
differs from F-base, with fetch counters non-zero. Had the gate been applied too
broadly the masks would never be read, and F-full would have collapsed onto
F-base's exact hash.

### What this costs the stage

Every prefetch number ever measured with ROUTE or CPUSKIP on was measured on
corrupted prefill and is void. That includes Stage H8's headline. Corrected:

  - step rate: 66.11 vs 66.17 -- **a wash**, not the +3.3% reported. The -3.62 ms
    "fetch effect" was an artefact; against the fused predictor's +1.51 ms fixed
    cost the real fetch return is about -1.5 ms and the two cancel
  - accept: 2.273 vs 2.857, and now DETERMINISTIC at 88 steps every run. The
    regression is real, was never resolvable while the runs were corrupt, and is
    20% of throughput

So the fused predictor did what it was asked to do -- 5.47 ms/step down to 1.51 --
and it still is not enough, because the fetch it enables returns about what the
predictor costs. `bench/why_accept.sh` chases the remaining question: whether the
accept drop is the CPU/GPU kernel disagreeing (rounding, which would let the
completions track each other for many tokens before drifting) or a residual
correctness defect (which would show in the first few). Its MTP=0 pair removes
the draft entirely, so tok/s is 1000/ms_per_step and the text comparison isolates
what prefetch does to the target model alone.

## Stage H10 -- the gathered expert weights are WRONG

The accept regression survived the staleness fix and got worse: 2.273 vs 2.857,
deterministic at 88 steps every run, with step rate a wash (66.30 vs 65.70). The
standing explanation was numerics -- cutlass W4AFP8 rounding differently from the
CPU packed-int4 W4A8 kernel. That explanation is now refuted.

`bench/gather_truth.sh` asks the model directly, using only env vars, because
ROUTE and CPUSKIP decompose a landed expert into three arithmetic outcomes:

| row | the landed expert is | ms/step | accept | tok/s | diverges from base |
|---|---|---|---|---|---|
| X-base | computed on CPU, as normal | 65.70 | 2.857 | 43.49 | -- |
| X-drop | OMITTED entirely | 64.14 | 2.532 | 39.47 | char 21 |
| X-full | computed on GPU from gathered weights | 66.30 | 2.273 | 34.28 | char 21 |
| X-dbl | CPU AND the gathered GPU copy | 75.60 | 2.532 | 33.49 | char 105 |

All four rows are deterministic (3/3 identical hashes), so these are measurements
rather than impressions, and X-full reproduces the earlier F-full boot exactly
(66.30 vs 66.11 ms/step, accept 2.273 in both).

**X-drop beats X-full.** Losing the expert's contribution outright costs 0.33
accept; computing it from the gathered weights costs 0.58. Computing something
can only be worse than computing nothing if what you computed is wrong. No
rounding difference can produce that ordering -- precision error lands BETWEEN
baseline and drop, never below drop.

Two details agree. X-full diverges from baseline at char 21, about the fifth
token; int8-vs-fp8 activation rounding drifts apart over hundreds of tokens, it
does not flip a token that early. And X-dbl -- which KEEPS the correct CPU
contribution and merely adds the gathered copy on top -- diverges only at char
105, the closest of the three to baseline. The arm that REPLACES the correct
contribution with the gathered one is the one that breaks immediately, which is
the signature of a wrong value of modest magnitude rather than a wild one.

### Why this survived so long

`gather_kernel_probe.py --verify` checked the gather byte-identical against the
OLD gather at 8/64/256 blocks for n_tp 1 and 2. Two implementations of the same
misunderstanding agree perfectly. Nothing ever compared the gather against an
independent reader of the same store -- and one exists: `_prepare_weight_w4afp8`
goes through kt's own C++ `write_weights_to_buffer` and is bit-coherent in
production GPU prefill.

So every prefetch number in this document that had ROUTE on was measured on
experts computing wrong values. The stage never once ran correctly.

### Not yet attributed

WHICH part of the gather is wrong is NOT established. The scale interleave
matches `interleave_scales` on paper -- `dst[g, n*4+a] = src[n, g*4+a]`, and the
alignment really is 4 because G13 = 6144/128 = 48 -- so suspicion falls on the
weight layout or on kt's store packing differing from the checkpoint's. That is a
guess, and this stage has already burned two of those.

The definitive test is one boot, `pf_verify_against_kt`: materialise the same
NON-resident expert twice -- once through the CUDA gather into a landing slot,
once through kt's proven staging into a full-context layer -- and diff the four
tensors. A resident expert cannot be used: `build_pointer_table` gives GPU-tier
experts a NULL entry by design ("sel must never name one"), which is why the
first attempt at this took an illegal memory access. That fault was the harness's,
not the gather's, and is not evidence either way.

## Stage H11 -- ROOT CAUSE: the landing-slot index was relative, not absolute

Stage H10 concluded the gathered weights were wrong. That conclusion was WRONG,
and the checkpoint said so.

`bench/gather_vs_checkpoint.py` reconstructs a single expert's cutlass tensors
straight from the safetensors -- TP shard, gate|up concatenation, and the
`interleave_scales` permutation -- and diffs them against a one-boot dump. The
dump carries a RESIDENT expert alongside the gathered one precisely so the
offline reader validates itself before it is allowed to judge anything: if the
resident expert does not match, the reader is wrong and the verdict is withheld.

Result, layers 3 and 40, both ranks, resident AND gathered:

    w13_weight            OK  0/6291456 bytes differ
    w2_weight             OK  0/3145728 bytes differ
    w13_weight_scale_inv  OK  0/98304 differ  max|d|=0
    w2_weight_scale_inv   OK  0/49152 differ  max|d|=0

**The gather is byte-perfect.** It always was.

### What was actually wrong

`apply()` builds the effective routing table as

    pf_eff_index = where(pf_landed, pf_index, logical_to_gpu_index_cuda)

`logical_to_gpu_index` holds ABSOLUTE indices into the layer's cutlass tensors,
which `create_weights` allocated with `num_gpu_experts + slots` entries -- the
landing slots are the trailing four. The unfused `pf_issue` respects that:
`pf_slot_ids = arange(_pf_slot_base, _pf_slot_base + slots)` = 100..103.

The fused kernel wrote `index[pick] = k` -- the RELATIVE slot number, 0..3.

So every prefetched expert was routed to GPU expert 0, 1, 2 or 3: four real,
resident, entirely unrelated experts. Not garbage bytes -- a different expert's
perfectly valid weights, which is why the output stayed fluent while quality
fell, and why it was deterministic.

It also explains the result that made no sense. X-drop beat X-full (accept 2.532
vs 2.273) because **a wrong expert is worse than no expert**, and X-dbl -- which
keeps the correct CPU contribution and merely adds the wrong GPU one -- diverged
latest of the three, at char 105 against char 21.

### Why it survived 150 passing trials

`pred_fused_kernel.py --verify` asserted `int(kindex[e]) != k` -- the relative
convention. The kernel and its test encoded the same misunderstanding, so the
test could never fail. This is the SECOND instance of that failure mode in this
stage: `gather_kernel_probe.py --verify` checked the gather byte-identical
against an OLDER GATHER, which is why nobody noticed the gather was fine and
looked elsewhere.

The lesson both times: a test written from the same mental model as the code
under test proves only that the model is self-consistent. Validate against an
INDEPENDENT source -- the checkpoint, another implementation, the model's own
output -- or do not claim validation.

Fixed: `slot_base` is now a kernel parameter, `index[pick] = slot_base + k`, and
the assertion checks `slot_base + k`. `--slot-base` defaults to 100, not 0, so a
relative index cannot pass silently again.

### What this voids

Every fused-predictor measurement in Stages H8-H10 routed to the wrong experts.
The whole "accept regression" investigation was chasing an artefact of this one
line. Stage H9's staleness defect was real and independently worth fixing, but it
was never the cause of the accept drop.

Scope: the fused predictor is behind `KT_PRED_FUSED=1`, and the prefetch path
needs `KT_PREFETCH_SLOTS > 0` which defaults to 0. Production never ran it.

## Stage H12 -- the shipping comparison, against what production actually runs

Every prefetch number to this point compared `GPU_EXPERTS=100 + 4 slots` against
`GPU_EXPERTS=100` with those slots allocated but idle. Equal VRAM, which is the
right control for "does the machinery pay for itself" -- but NOT the shipping
question, because production holds 104 resident and no slots, and a landing slot
is bought with a resident expert (~9.73 MiB/card/layer).

`bench/fair_baseline.sh`, one ladder, all rows deterministic:

| row | ms/step | accept | tok/s |
|---|---|---|---|
| F104-plain -- production | 65.27 | 2.778 | 42.56 |
| F100-idle -- the handicapped baseline | 65.63 | 2.857 | 43.53 |
| F100-fetch -- prefetch, index fixed | **64.05** | 2.857 | **44.61** |

**The four extra resident experts are worth 0.36 ms/step.** That is the whole
handicap, and it is small -- consistent with the placement work finding that
residency coverage barely moves speed once the tier substitution has run.

**Prefetch beats production by 1.22 ms/step = +1.9%**, down from the +2.1%
claimed against the handicapped baseline. The correction is real but does not
change the verdict: it ships, marginally.

### Do not bank the tok/s gap

44.61 vs 42.56 is +4.8%, and most of that is not the prefetch. The two residency
configs produce DIFFERENT TEXT (different substitution -> different completion ->
different draft agreement), so accept differs: 2.778 at 104, 2.857 at 100.
Holding accept fixed:

  - prefetch's own step-rate contribution: **+1.9%**
  - the 104->100 change yielding a better-accepting completion here: +2.8%

The second is one-prompt content luck. **ms/step is the only clean cross-config
metric**; tok/s across configs mixes step rate with which completion the config
happened to produce. Same trap as the accept-quantisation warning in
TODO item 8, arriving from a different direction.

### Cross-day stability, which licenses the comparison

Every config reproduced its exact completion hash across a full shutdown and cold
reboot: F104-plain = `9e49a7738c` (production's hash from the previous day),
F100-idle = `5093e29512` (every 100-expert baseline), F100-fetch = `45c4fbb027`
(yesterday's Y-full). The rig is stable across days; the rule about subtracting
only within a run still stands, but the baselines are not drifting.
