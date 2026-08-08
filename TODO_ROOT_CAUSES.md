# Tomorrow: root-cause each idea before judging it

Standing rule for this list (see memory `root-cause-before-conclusions`): a
number is not a cause. Every item below is a MEASURED quantity whose cause has
NOT been established. Take each apart until the parts sum to the whole; only
then write a verdict. Precedent: the gather looked "structurally expensive" for
half a day and turned out to be a byte-granular copy kernel with an emulated
64-bit divide per byte -- 2.6x once fixed.

Everything referenced here is written up in `EXPERT_PREFETCH_PLAN.md`,
Stages F, G, G2, H.

---

## 1. ~~The walk costs 18.2 ms/step per walked layer. WHY?~~ **CLOSED 2026-08-07**

Decomposed with `bench/walk_decomp.sh`. Parts sum to the whole exactly
(17.45 ms), noise floor 0.08 ms, accept length identical in every row.

| component | ms/step | us/layer | share |
|---|---:|---:|---:|
| resident substitution + remap + slot mask | 6.55 | 87.3 | 37.5% |
| GPU expert kernel (at only 2 slots, 4 tokens) | 6.61 | 88.1 | 37.9% |
| shared expert | 3.34 | 44.5 | 19.1% |
| walked layer's gate + topk | 0.57 | 7.6 | 3.3% |
| TP all-reduce | 0.38 | 5.1 | 2.2% |

**The prime suspect was WRONG.** I predicted the 75 extra NCCL all-reduces would
carry 2-4 ms; they carry 0.38 ms. A 4-token all-reduce on NVLink is cheap.

**Cause: two fixed per-invocation costs that do not scale with the useful work.**
~20 tiny kernels in `_resident_ids` (topk over all 256 experts, two argsorts,
gathers, scatters) at ~4.4 us each, and the grouped-MoE kernel's setup (token
sort, problem-size array, grouped GEMM launch) for four tokens through two
experts. The latter also explains why 2 -> 8 slots only cost 3.4 ms: the fixed
part dominates and widening moved the small part.

Follow-up **DONE** the same day (Stage H3, `bench/walk_fix.sh`):
`KT_CHAIN_PF_SUB=drop` cut the substitution 6.55 -> 1.55 ms (76%), but the whole
walk only 17.45 -> 14.85 ms (15%) -- dropping the top-2 cap leaves ~3 live slots
instead of 2 and hands ~2.40 ms back to the grouped-MoE setup term. Shippable
rows: SEARCH-real 88.56 -> DROP-real 85.55 ms/step, 31.70 -> 33.00 tok/s,
accept identical (2.817). Same-boot baseline FIX-base 66.78.

Still does not change the shipped choice: depth 2 predicts WORSE than depth 1
(72.8% vs 81.9% coverage), so a zero-cost walk would not win here. The cost work
matters only for a machine that NEEDS two layers of lead time.

**NOT BANKABLE -- measured 2026-08-07.** `bench/drop_acc.sh`, 608 steps, all
arms in one boot: drop reads 72.3 / 64.1 / 60.6 / 60.9 whole-layer coverage at
d=1..4 against search's 78.0 / 70.1 / 67.1 / 67.3. That is ~6 points down, and
it is BELOW `chain_shared` (no routed experts at all) and below `direct` (no
walk at all). The 3.01 ms buys a walk worse than not walking. Search stays the
default; both modes remain under `KT_CHAIN_PF_SUB`.

**New open question (see item 9).** Why is dropping worse than omitting?

## 2. The direct predictor's 5.03 ms/step. Split, but not explained.

Stage F split it: lookahead router 2.25 ms, selection ops 2.45 ms. Neither is
attributed. 2.25 ms for 75 gate GEMMs on 4 tokens is ~30 us per layer for what
should be microseconds of arithmetic -- so it is dispatch, not math, but that is
an inference, not a measurement. `bench/pred_select_kernel.py` exists (a fused
single-launch replacement for the ~28 selection ops) and is UNVERIFIED on GPU.
Verify it, measure it, and find out whether the router half is dispatch or the
topk scoring path.

Reviewed it 2026-08-07 before spending GPU time on it, and **found a defect in
the harness, not the kernel**: `--bench` timed the incumbent by calling
`python_reference`, which ALLOCATES its five outputs every call. `pf_issue` does
not -- it writes into preallocated buffers in place. Charging the incumbent for
75 allocations it never does would have manufactured exactly the speedup the
file exists to test. Replaced with a literal in-place transcription of
`pf_issue`, so the measurement is launch count against launch count.

Also flagged: demand is a count over 4 tokens x 2 predictions, so ties at demand
1 are the COMMON case. The kernel breaks ties to the lower expert id; CUDA topk
does not document its tie order. A `sel`-only mismatch in `--verify` is a tie,
not a bug -- but `landed` / `index` / `stats` mismatches are real.

## 3. Chain-d1 beats direct-d1 by 3.1 ms/step. **RETRACTED — the gap is not the prediction point**

> **READ THIS FIRST (2026-08-07, later the same day).** The closure written
> below is WRONG and Stage H6 supersedes it. `bench/predpoint.sh` tested the
> claim on the shipped path, same day, same box:
>
> | row | ms/step | overhead | fixed | fetch effect |
> |---|---:|---:|---:|---:|
> | PP-base | 66.96 | -- | -- | -- |
> | PP-pre | 70.31 | +3.34 | 5.47 | -2.12 |
> | PP-post | 69.98 | +3.02 | 5.53 | -2.51 |
>
> **The prediction point is worth 0.32 ms, not 3.1.** The fetch pays for itself
> at BOTH points. The error: I computed the direct predictor's fetch effect as
> yesterday's `DEEP-d1 - DEEP-baseline` minus today's `I3-direct-off - I3-base`,
> after checking only that the BASELINES agreed (67.08 vs 66.79). They did --
> and the prefetch rows did not. DEEP-d1 read 73.54 where today's identical
> configuration reads 70.31.
>
> **Rule: a baseline agreeing across boots licenses nothing about the rows
> measured beside it. Subtract only within a run.**
>
> Still true, all same-day: both predictors cost the same fixed amount (5.47 vs
> 5.53); the fetch is now a 2.1-2.5 ms SAVING, the first time it has ever come
> out ahead; the post state is slightly better and worth taking; and it is still
> not a net win, because the predictor's ~5.5 ms remains.
>
> **New open question: why did DEEP-d1 read 3.2 ms slow?** Only MEM_FRACTION
> differs (0.95 there, 0.94 here) -- see item 7 -- and if that is it, every
> prefetch number measured at 0.95 is suspect.

### Superseded closure, kept for the record

`bench/item3_attrib.sh`, three boots, prediction computed but nothing moved
(`GATHER=0 ROUTE=0 CPUSKIP=0`):

| row | ms/step | predictor's fixed cost |
|---|---:|---:|
| I3-base (no predictor) | 66.79 | -- |
| I3-direct-off | 72.25 | +5.47 |
| I3-chain-off (depth 1) | 72.32 | +5.53 |

**The two predictors cost the same** -- 0.07 ms apart, at the 0.08 ms noise
floor. So NONE of the 3.1 ms is predictor overhead, which was my leading
explanation. Subtracting the fixed cost from each row's total overhead leaves
the fetch effect:

| | total overhead | fixed cost | **fetch effect** |
|---|---:|---:|---:|
| direct predictor (DEEP-d1) | +6.46 | 5.47 | **+0.99 ms (a COST)** |
| post-state predictor (CPF-d1) | +3.35 | 5.53 | **-2.18 ms (a SAVING)** |

Difference 3.17 ms against the observed 3.11 ms gap -- **0.06 ms apart, inside
the noise floor. The decomposition closes.**

### What it means

This is the first configuration on this box where the prefetch has ever PAID
FOR ITSELF: the bytes it moves buy 2.18 ms/step more CPU-expert skipping than
the gather costs. The direct predictor at the same fetch settings loses 0.99 ms.

The cause is prediction quality and the accuracy table says so independently:
the shipped predictor feeds the PRE-MoE hidden state to the next layer's router
(`direct`, 74.3% whole-layer coverage at d=1), while `chain_prefetch.py` at
depth 1 feeds the POST-layer residual stream (`post`, 79.5%). +5.2 points, and
the post state is already computed at the hook, so it costs nothing extra --
it costs LEAD TIME, since the fetch now starts after layer L's MoE instead of
before it. That trade is evidently worth taking here.

**Actionable: move the shipped predictor's input from the pre-MoE hidden state
to the post-layer residual.** Free in compute, better in coverage, and it flips
the fetch from a cost into a saving.

### What it does NOT mean

It is still not a win overall: `66.79 + 5.53 - 2.18 = 70.14`, and the predictor's
own 5.47 ms is what keeps it above baseline. To beat 66.79 the predictor must
come in under ~2.2 ms. Stage F split it into 2.25 ms of lookahead router and
2.45 ms of selection ops; the fused kernel (item 2) targets the selection half
only, so the best it can do alone is ~2.35 ms total -- `66.79 + 2.35 - 2.18 =
66.96`, a wash. **The router half has to come down too, or this ends in a tie.**
That is arithmetic on measured parts, not a projection.

---

Original entry, kept because the counter reading in it is still worth knowing:

70.43 (CPF-d1) vs 73.54 (DEEP-d1). I read the counters as coverage evidence.
Read the code: they are not.

`kt_ep_wrapper.pf_issue` increments `pf_stats[1] += n_want`, and `n_want` is
`(distinct non-resident experts in the PREDICTED set)` -- a property of the
PREDICTION, not of what the layer genuinely needed. So "1.83 vs 2.02 wanted per
layer-call" says only that the chain path named a SMALLER non-resident set, and
`fetched/call` 1.65 vs 1.71 says it moved fewer bytes. A predictor that asks for
less is cheaper whether or not it is more accurate. The server has no counter
for residual CPU demand at all; whole-layer coverage exists only in the offline
instrument.

Second confound: the two rows are not the same mechanism. DEEP-d1 is the shipped
direct predictor (its own ~5.03 ms of selection ops, item 2); CPF-d1 is
`bench/chain_prefetch.py` at depth 1, which does no walk and reuses the exact
post-layer state. Different code, different fixed cost.

To attribute it: boot both at `GATHER=0 ROUTE=0 CPUSKIP=0` (prediction computed,
nothing moved). That difference is pure predictor overhead; the remainder of the
3.1 ms is the fetch effect. Until then this is **3.1 ms, cause not attributed**,
and it must not be quoted as "the chain predicts better".

## 4. The gather is 61% hidden. What is the other 39%?

Stage F got exposed gather from 22.46 -> 8.70 ms/step. 8.70 ms is still exposed
and the reason is not established. Is it head-of-line blocking (Stage E measured
+31.7% decode penalty from that, with queue depth 1 as the fix), insufficient
lead time, or the landing-slot skip rate (18.6% of layer-calls want more than 4
slots)?

## 5. ~~`prevlayer` reads ~1% recall. Real, or an instrument bug?~~ **CLOSED 2026-08-07**

Checked independently against `bench/profile_out/routing/phase2D.pt` (2740
dumped (layer, topk_ids) records, reconstructed into 42 forward passes). The
instrument is right, and the result is regime-invariant:

| regime in the dump | prevlayer d=1, P=2 | chance floor |
|---|---:|---:|
| decode (T<=6) | 1.1% | 0.78% |
| batched decode (T=37) | 0.7% | 0.78% |
| prefill (T>100) | 1.0% | 0.78% |

Also at chance for d=2,3,4 (0.6-0.8%). **Expert identity carries no signal
across layer depth** -- an adjacent MoE layer is no better than drawing two of
256 at random. So the "replace some experts with the previous layer's choices"
idea is dead on this model, and the reason the walk helps at all is that it
carries the HIDDEN STATE forward, never the expert ids.

NOT cross-checked here: persistence. This dump reads 46.8% (decode) / 91.5%
(batched) against the live instrument's 22.8%, but its T=4 rows are MTP draft
tokens and its T=37 rows are replicated identical prompts, so row i at step t is
not the same sequence position as row i at step t+1. The dump cannot measure
persistence; it can measure prevlayer, which is a within-pass comparison.

## 6. ~~KNOWN DEFECT: the `chain_post` arm double-applies layer L.~~ **CLOSED 2026-08-07**

Fix: the walk now skips its `d == 1` iteration for `chain_post` only (that
iteration's source layer IS L, whose real update the arm already carries).

**Validated by the fix's own prediction.** At d=1 `chain_post` must degenerate
to `post`, because with layer L real and nothing after it approximated the two
arms are the same computation. It now reads **79.5 against post's 79.5**. Before
the fix it read 67.8 against 76.7.

Corrected numbers (whole-layer coverage, P=2, 608 steps): 79.5 / 71.0 / 67.6 /
67.6 at d=1..4 -- the best arm in the table at every depth, ahead of `chain`
(78.0 / 70.1 / 67.1 / 67.3), which is what starting from the layer's REAL output
ought to do. This is also exactly what `bench/chain_prefetch.py` implements, so
the shipped mechanism was never affected. Original text below.


In `bench/chain_predict.py` the walk loop starts at `src = lid + d - 1`, so for
the `chain_post` arm -- which already starts from `r_next`, i.e. layer L's real
output -- depth 1 adds layer L's approximate update ON TOP of its real one. Its
numbers (67.8% coverage at d=1 vs `post`'s 76.7%) are meaningless. Fix: the
`chain_post` walk must start at `src = lid + d`. **The capture-safe MECHANISM in
`chain_prefetch.py` does this correctly** (`for s in range(1, _DEPTH)`), so only
the instrument arm is affected.

## 7. Why does the box OOM at GPU_EXPERTS=100 + landing slots? **Bounded, not explained**

New data 2026-08-07. The cliff is now bracketed by two boots at the same
GPU_EXPERTS=100:

| MEM_FRACTION | landing slots | buffer | result |
|---|---:|---:|---|
| 0.95 | 4 | 38 MiB | OOM (two boots, 562 MiB and 2 MiB free) |
| 0.94 | 4 | 38 MiB | boots |
| 0.94 | 16 | 153 MiB | **OOM, 16 MiB free, needed 28 MiB** |
| 0.92 | 16 | 153 MiB | (running) |

The 0.94/16-slot failure is in `deepseek_weight_loader.post_load_weights` ->
`block_quant_dequant`, i.e. a TRANSIENT dequant buffer during weight loading,
with the process already holding 93.06 of the card's 93.09 GiB.

**ROOT CAUSE FOUND.** Re-running the 16-slot boot at 0.92 failed with the
byte-for-byte identical message: **16.12 MiB free, 28 MiB requested, 93.06 of
93.09 GiB in use** -- the same free figure as at 0.94, and both logs confirm
mem_fraction_static really was 0.92 and 0.94.

**MEM_FRACTION has no effect at the moment of failure.** The OOM is inside
`deepseek_weight_loader.post_load_weights` -> `block_quant_dequant`, i.e. during
WEIGHT LOADING, and the KV pool that mem-fraction sizes is not allocated until
after that. What fills the card in that phase is model weights + kt's resident
GPU experts + the prefetch landing buffer, none of which mem-fraction touches.

So `MEM_FRACTION=0.94` never bought headroom for the landing slots -- it only
looked that way because the two earlier OOMs at 0.95 had a different cause. The
only lever on this phase is **GPU_EXPERTS**, at ~9.56 MiB/expert/card.

Budget, from the measured 16.12 MiB free at 16 slots:
  4 slots  = 38 MiB   -> ~131 MiB free   (boots)
  8 slots  = 76 MiB   -> ~93 MiB free    (should boot, untested)
  16 slots = 153 MiB  -> 16 MiB free     (OOM)
To run 16 slots, drop GPU_EXPERTS by ~12 (100 -> 88) and re-measure the
no-prefetch baseline there too, since resident count changes the CPU pool.

Still true regardless: comparisons across mem-fraction values are invalid
(Stage H6), even though this particular failure was not caused by it.

**More evidence 2026-08-07, from the fused-predictor boots.** Adding an
instrumented free-memory read to `_kt_fuse_gates` shows the picture is not a
simple shortage:

  * right after the 225 MiB gate restack + `empty_cache()`: **7356 MiB free**
  * a little later, in the NEXTN weight load: **2 MiB free**, 28 MiB requested,
    and **706 MiB "reserved but unallocated"**

So 7.3 GB is consumed between those two points -- kt staging its 100 resident
GPU experts, which is inside `load_model` and therefore BEFORE the KV pool that
mem-fraction sizes exists. That is why mem-fraction does not move this failure.

And the proximate cause is **fragmentation, not exhaustion**: 706 MiB is
reserved and unusable for a 28 MiB request. The fused path is net-zero memory
(one 225 MiB block replacing 75 separate 3 MiB tensors) yet only the fused rows
die, which is exactly what a changed allocation pattern does to a fragmented
heap. Next lever tried: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
which the allocator itself recommends for this signature. If that is not enough,
GPU_EXPERTS is the only remaining knob on this phase.

### Original entry

Two boots died there today (TUNE-P3-b8-s6 with 562 MiB free, CPF-d3-k2 with 2
MiB). `MEM_FRACTION=0.94` is a workaround, not an explanation. Find what is
actually consuming the last GB so the ladder can run at a fixed, safe operating
point instead of on a cliff edge.

## 8. Accept length moves with CPUSKIP. **CLOSED 2026-08-07: the fused kernel routed to the WRONG EXPERTS**

Final answer, after two wrong ones of mine. The fused predictor wrote the
RELATIVE landing-slot number into `pf_index` (`index[pick] = k`) where `apply()`
merges it with `logical_to_gpu_index`, which is ABSOLUTE. Landing slots are the
trailing experts [num_gpu_experts, +slots) of the same cutlass tensor, so every
prefetched expert was computed as GPU expert 0..3 -- four real, resident,
unrelated experts. Fluent output, quality quietly down, perfectly deterministic.

Proof it was routing and not the weights: `bench/gather_vs_checkpoint.py` diffs a
gathered expert against the checkpoint itself and reports 0 differing bytes on
all four tensors, both ranks, layers 3 and 40 -- with a resident expert in the
same dump validating the offline reader first.

Proof it was harmful rather than merely different: `bench/gather_truth.sh`
decomposes ROUTE/CPUSKIP into three arithmetic outcomes. OMITTING the expert
(accept 2.532) beat COMPUTING it (2.273) against a 2.857 baseline. A wrong expert
is worse than no expert; a rounding difference could never land below drop.

It passed 150 verify trials because the test asserted the same relative
convention the kernel used. Second time in this stage a test shared the code's
mental model -- see also `gather_kernel_probe.py --verify` checking the gather
against an older gather.

Fixed: `slot_base` threaded into the kernel, `index[pick] = slot_base + k`,
assertion updated, `--slot-base` defaults to 100 so relative cannot pass.

### Superseded: the stale-mask finding (real, but not this)



The accept drop is not an accept phenomenon at all, and not numerics. It is a
correctness bug, and the giveaway was reproducibility rather than any timing.

`bench/determinism.py`, 5 greedy repeats of one prompt, one boot each
(GPU_EXPERTS=100, MEM_FRACTION=0.94, safe2, MTP=1):

| row | consumers enabled | distinct completions / 5 | vs base |
|---|---|---|---|
| D-base | none | 1 | -- |
| D-gather | none (bytes move only) | 1 | **same hash** |
| D-route | GPU (`pf_landed`) | **5** | differs, first at char 3 |
| D-full | GPU + CPU-skip | **5** | differs, first at char 209 |

Production is deterministic too: 5/5 identical, agreeing to 0.1 s of wall time.

A fixed numerics difference -- cutlass W4AFP8 rounding differently from the CPU
packed-int4 W4A8 kernel -- is still a FUNCTION. Same prompt, greedy sampling,
same answer every run. Five different answers cannot be rounding. And D-gather
proves the transfer itself is innocent: the bytes move and the hash does not
budge, so it is the CONSUMPTION of the prediction that is unsound.

Mechanism, from the code. `pf_landed` / `pf_landed_cpu` / `pf_index` are written
only by the predictor, and the predictor only fires on a decode-shaped batch
(`T <= KT_PRED_TMAX`, default 8). Nothing else ever clears them. Both consumers,
however, run unconditionally on every `apply()`:

  - `_submit_with_staged_input` -> `pf_landed_cpu[topk_ids] -> -1`, CPU skips
  - `apply` -> `pf_eff_mask` / `pf_eff_index`, GPU routes to a landing slot

A prefill is not decode-shaped (this prompt is ~60 tokens), so the predictor does
not fire and the masks still hold **the last decode step of the previous
request**. Prefill routes experts to landing slots holding some other expert's
weights and tells the CPU to skip experts nothing computed. The KV cache for the
whole prompt is built wrong -- differently each time, because what is stale
depends on where the previous generation stopped. Hence divergence at char 3.

Confirmed structurally: the full-GPU prefill early return needs
`gpu_prefill_token_threshold > 0`, and every one of these boots sets
`KT_GPU_PREFILL_THRESHOLD=0`, so prefill does fall through to both consumers.

Scope: `KT_PREFETCH_SLOTS` defaults to 0, so the whole path is off unless a
benchmark enables it. **Production has never run this code.** Every prefetch
number measured with ROUTE or CPUSKIP on is measured on corrupted prefill and is
void -- including Stage H9's "+3.3% step rate", which was read off runs whose
output text was wrong.

Fix: `pf_issued` already records "a prediction was published for this forward"
(it exists so `apply` only joins a fork that exists). It simply is not consulted
where the masks are read, and is reset only inside the wait branch. Gate both
consumers on it and clear it on every exit path -- a host-side flag, so zero
device launches, which matters because launch count is what this whole stage is
trying to reduce.

`bench/determ_fix.sh` tests the fix AND an independent check that does not
depend on the fix being right: `KT_PRED_TMAX=4096` makes the predictor fire on
the prefill batch too, giving fresh masks by a different route. If that row goes
deterministic, stale masks are the cause and the remaining suspect -- a
read-before-write race on the landing slots -- is dead, because such a race would
survive it.

### Original entry (the quantisation point, still true and still worth heeding)

`bench/prefetch_rate.py` computes `accept = n_tok / len(stamps)` and takes the
MEDIAN of 3 runs. At `--tokens 200` that is ~200/70, so **one extra forward step
moves accept by ~0.04** -- and the observed values are exactly those rungs:
across 60 recorded rows only nine distinct accept values appear, and 2.857
(200/70), 2.817 (200/71) and 2.778 (200/72) account for 52 of them.

So: differences of 0.04 are ONE step and carry no information. The full observed
spread (2.70-2.90) is ~5 steps, which is larger than the quantum, so something
real is moving too -- but the instrument cannot resolve it. **Fix the instrument
before re-opening the question**: report `steps` directly (it is already in the
per-run dict, just not in the summary) and raise `--tokens` so the quantum
shrinks.

Practical consequence, already used: the Stage H3 comparison is safe because
SEARCH-real and DROP-real returned the SAME accept (2.817), so their 3.01 ms
gap is entirely step rate. Any future comparison that straddles an accept rung
must be treated as unresolved.

## 9. ~~Why is DROP worse than OMITTING the routed experts entirely?~~ **CLOSED 2026-08-07**

**It was the renormalise, exactly as predicted.** `chain_drop_raw` (drop, do NOT
rescale) reads 75.0 / 67.4 / 65.0 / 66.4 at d=1..4 against `chain_drop`'s
71.4 / 62.3 / 59.7 / 61.7 -- one division, isolated by a single-variable change,
worth 3.6-5.3 coverage points. It clears `chain_shared` (74.8 at d=1) as the
first prediction below required, so the renormalise is the culprit and the cheap
substitution is viable.

Mechanism: rescaling gives the two or three survivors the FULL routed magnitude,
so the update is about the right size in the wrong direction and compounds
through the walk; leaving the sum short says correctly that the missing experts
contributed nothing. `search` still needs the division -- there the top-K mask
zeroes real slots whose weight does belong to the survivors -- so the division is
right where it started and wrong where it was copied to.

Shipped as `KT_CHAIN_PF_SUB=drop_raw`: ~2 coverage points behind the search for
5.0 ms less. Does not change the verdict on this box. Original entry below.


`chain_drop` (keep resident slots, zero the rest, renormalise) reads 72.3%
whole-layer coverage at d=1; `chain_shared` (run NO routed experts, only the
shared one) reads 75.1%. Adding a wrong routed update is worse than adding none.
That is not obvious and it is not explained.

Hypothesis, untested: the renormalisation. It rescales the two or three
surviving experts up to carry the full routed weight, so the residual gets an
update of roughly the right MAGNITUDE pointing in the wrong DIRECTION, while
shared-only gets a small one. A wrong-magnitude error compounds through the
walk; a missing one does not.

Discriminating measurement, already built: the `chain_drop_raw` arm zeroes the
non-resident slots and does NOT renormalise -- it changes exactly the one term
in question. Predictions worth committing to before the run:
  - if renormalise is the culprit, `chain_drop_raw` should land at or above
    `chain_shared` (>= 75.1 at d=1);
  - if it lands with `chain_drop` (~72.3), the renormalise is innocent and the
    damage is from routing tokens through SUBSTITUTE-free resident experts,
    which would mean the search's stand-ins are doing real work rather than
    merely filling slots.

This matters beyond the walk: the same "drop vs substitute" choice is what
`RUNGLM_TOPK_MODE` safe-tiering makes on the REAL forward path.


---

## Unrun measurements from today

- `chain_acc2` walk-K=2 accuracy row (the K=8 row completed at 240 steps;
  treat it as provisional, the other sweeps used 592).
- Direct predictor at depths 3 and 4 (`scratchpad/deep_redo.sh`, ready to run;
  both earlier attempts were voided by the NEXTN draft-capture bug, now fixed).
- The whole walk decomposition in item 1.

## 10. DEFECT in the shipped `pf_issue`: expert 0 can never be prefetched

Found 2026-08-07 while verifying the fused kernel. `kt_ep_wrapper.pf_issue`
publishes the landing map with

    ids0 = self.pf_sel.clamp_min(0)
    self.pf_landed.scatter_(0, ids0, valid)
    self.pf_index.scatter_(0, ids0, self.pf_slot_ids)

`clamp_min(0)` maps EVERY empty slot (-1) onto index 0, so with duplicate
indices the last write wins. Reproduced on CPU, no GPU needed:

    pf_sel = [0, -1, -1, -1]          # expert 0 fetched into slot 0
    -> pf_landed[0] == False          # cleared by the three empty slots
    -> pf_index[0]  == 3              # garbage, but never read

**Consequence: a wasted transfer and a missed skip, NOT corruption.** The bytes
for expert 0 are streamed, then `pf_landed[0]` is false so the layer routes it
to the CPU anyway. Because the landed flag is cleared, `pf_index[0]`'s garbage
is never read, so no wrong weights reach cutlass. Triggers whenever expert id 0
is in the wanted set AND at least one slot is empty -- ~0.3% of configurations
in a synthetic sweep, and slots are usually partly empty in production (1.65
fetched per call into 4 slots).

Fix, branch-free and static-shape: size `pf_landed` and `pf_index` at n_exp + 1
and send empty slots to the pad with `torch.where(sel >= 0, sel, n_exp)` instead
of `clamp_min(0)`; readers take `[:n_exp]`. NOT applied yet -- it touches the
shipped path and the effect is sub-1%, so it wants its own measured boot rather
than riding along with a prediction experiment.

The fused kernel does NOT have this bug (it writes only for `pick >= 0`), which
is how it surfaced.

