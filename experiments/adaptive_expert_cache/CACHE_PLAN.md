# Plan: make the GPU-hit probability drastically high for practical workloads

Branch: `experiment/adaptive-expert-cache`
Date: 2026-07-07
Goal: on real applications, keep the token's active experts GPU-resident with high
probability, so the CPU expert path (the exposed decode pole) rarely fires.

---

## 0. What we now know (measured), and why it changes the target

Four facts from this session's profiling anchor everything below:

1. **Each expert served from CPU instead of GPU costs ~3.0 ms/token** — dead-linear,
   0→8 experts (68.25 ms at 8 vs 44.14 ms at 0). So raising the GPU-hit rate has a
   direct, large, quantified payoff: ~3 ms/token recovered per expert kept resident.
2. **During decode the GPU is idle waiting for the CPU** (cpu_wait pole). We are not
   compute-bound — we're CPU-expert-throughput bound. More resident experts = strictly
   less GPU idle. Spending VRAM on residency is "free" on the compute side.
3. **The cache only learns from PREFILL.** `_update_gpu_experts_from_batch` is called
   solely inside the `num_tokens >= gpu_prefill_token_threshold` branch
   (`kt_ep_wrapper.py` ~L3486). At decode (`num_tokens==1`) the resident set is NEVER
   updated. The 96 residents are frozen to the *prompt's* routing; every decode token
   that routes elsewhere hits the CPU for the rest of the generation. **This is the
   root cause of the "cache floors at +8%" result** — it was measured on diverse
   prompts AND fed only prefill stats, so it never saw the generation's own working set.
4. **The prior "placement floors" conclusion was regime-specific.** It used 4 unrelated
   prompts + random decode (worst case for a cache). Practical apps are the opposite:
   strong temporal locality (consecutive decode tokens reuse experts) and domain
   locality (a conversation/task activates a stable expert subset). A cache that learns
   from the decode stream can actually cover that working set.

=> The target is not "cover any token" (impossible at 96/256 — the oracle floor). It is
**"cover the current generation's working set,"** which for practical, locality-rich
workloads is small enough to fit.

---

## 1. The hard architectural constraint (shapes the whole design)

Decode runs inside a **captured CUDA graph**. You cannot copy expert weights or mutate
shapes during graph replay. Therefore:

- **Stats collection** happens *inside* `apply()` (cheap, append-only), captured-graph-safe.
- **Re-selection + weight swaps** must happen **between decode steps**, in an eager
  scheduler-level hook — NOT inside `apply()`. The swap writes new weights and flips the
  **in-place `gpu_experts_mask_cuda` / `logical_to_gpu_index_cuda` buffers**, which are
  already designed to be CUDA-graph-stable (in-place `.copy_()`, L3815). The next graph
  replay then transparently uses the new residency. This is the mechanism the plan
  builds on; it already exists for the prefill path — we extend it to decode.

Swaps cost PCIe bandwidth (`copy_experts_weights_int4`). Updates must be throttled and
benefit-gated (Section 4), never per-token.

---

## 2. North-star metric first (Phase 0)

We have never measured the thing that matters: **decode-time GPU-hit rate**. Prefill
coverage read 1.0 and was misleading.

- Add a cheap per-decode counter (rank-0, no graph break): of the token's genuine top-k
  (k=2 and k=8), how many were GPU-resident. Maintain an EWMA; log every N tokens.
  (`KT_DUMP_TOPK` already dumps routing traces for offline analysis.)
- Measure it on three trace classes:
  - **Synthetic diverse** (current bench) — worst case.
  - **Long single-topic generation** — temporal locality.
  - **Multi-turn agentic / coding session** — domain locality.
- Confirm the hypothesis: decode hit-rate is low and *flat* (frozen cache), while the
  token stream's expert autocorrelation is high (so a decode-fed cache will move it).

Deliverable: hit-rate is the success metric for every phase below.

---

## 3. Core fix — learn from the decode stream (Phase 1)

The single highest-leverage change. Close the loop the prefill path already has, for decode:

- **Accumulate** per-layer routing stats over a sliding window of decode tokens (the
  generation's working set) — reuse `ExpertCacheState` / `_update_expert_cache_state_from_batch`.
- **Re-select** resident experts every N decode tokens (N ≈ 32–128, tuned) via a
  between-steps scheduler hook, flipping the in-place masks (Section 1).
- **Score the top-2 working set** (P1's `SGLANG_KT_EXPERT_CACHE_RANK_CREDIT="1.2,1.0,0…"`
  + disabled tail protection) — now correct, because it is finally fed *decode* stats
  instead of prefill. Tail experts don't need residency under `safe2` (they're
  substituted), so the whole budget goes to the genuine top-2 working set.
- This directly exploits temporal locality: the hot set over the last N tokens predicts
  the next token's experts well when the stream is locality-rich.

Expected: on locality-rich traces, decode top-2 hit-rate climbs from "frozen/low" toward
90%+, and `safe2` decode moves from ~20 toward the 0-CPU ceiling (~22.6 tok/s) — with the
genuine top-2 preserved (accuracy-safe).

---

## 4. Locality- and cost-aware policy (Phase 2)

- **Decay tuned to a working set, not a lifetime**: shorten the score half-life so the
  resident set tracks the *recent* window; long generations drift with the topic.
- **Benefit-gated swaps** using the measured cost model: swap an expert in only if
  `Δhit × 3 ms/token × est_remaining_tokens  >  swap_cost`. We finally have the 3 ms/expert
  constant to make this a real inequality, not a heuristic.
- **Hysteresis / grace**: keep the existing top-2 / strong-tail grace windows so a hot
  expert isn't evicted then immediately re-fetched (PCIe thrash).
- **Warm-up schedule**: update aggressively for the first ~M decode tokens (cheap, big
  gains as the working set reveals itself), then settle to periodic.

---

## 5. Persist the working set across turns (Phase 3) — the "practical apps" lever

Real deployments are sessions, not one-shot prompts.

- **Per-conversation resident snapshot**: on a new turn in the same session, seed the
  residents from the previous turn's converged set (the domain persists) instead of
  rebuilding from the new prompt's prefill. Eliminates the per-turn cold start.
- Optional: a small **library of precomputed resident masks** for common domains
  (code / math / chat), selected by a cheap prompt classifier or the prefill footprint,
  as the warm-start before the decode-fed cache refines it.

---

## 6. Enlarge the budget where it's free (Phase 4)

The GPU is idle during decode → not compute-bound, only VRAM-bound on expert weights.

- **Raise `GPU_EXPERTS` toward the VRAM ceiling.** Measure MB/expert (W4AFP8) and free
  VRAM at the target context length; every +expert resident is strictly fewer CPU
  experts (~3 ms/token each). 96/256 = 37.5%; pushing to 128+ raises coverage
  mechanically, independent of the policy.
- **Asymmetric residency**: give more slots to the layers whose distributions are
  peakiest / most reused (spend budget where it buys the most hit-rate); fewer to flat
  layers. Requires the per-layer stats from Phase 0/1.

---

## 7. (Research) prefetch into the idle GPU window (Phase 5)

The GPU idles ~0.11 ms/layer waiting on the CPU. Using next-token expert correlation,
opportunistically stream a *few* high-probability experts CPU→GPU at layer boundaries /
into that idle window. Bounded by PCIe (per-token *weight* streaming is a known dead end),
so restrict to high-confidence, high-reuse experts amortized over many tokens. Gate
strictly on the Phase-0 locality data; treat as speculative.

---

## 8. Success criteria & honest guardrails

- **North-star**: decode top-2 GPU-hit ≥ 90% on locality-rich practical traces; tok/s
  recovers toward the 0-CPU ceiling (22.6) and beyond as the budget grows.
- **Accuracy-safe by construction**: run on top of `safe2` routing (never drop genuine
  top-2). High hit-rate ⇒ near-top0 speed *with* full accuracy; low hit-rate degrades
  gracefully to CPU compute (slower), never to worse quality. The cache becomes a pure
  speed optimization with no accuracy downside.
- **Stated ceiling**: for adversarially diverse token streams, 96/256 cannot cover
  8-active — that is the oracle floor (~22 tok/s). This plan explicitly targets the
  practical / locality regime and *measures* it, rather than pretending diverse decode is
  cacheable.
- **Orthogonal to MTP**: MTP remains the lever for the fixed 35.5 ms attention base; this
  plan attacks the CPU-expert pole. They compose.

---

## 9. Suggested order
1. **Phase 0** (metric) — 1 day. Without it we're blind; it also validates the premise.
2. **Phase 1** (decode-fed cache) — the core fix; biggest expected gain.
3. **Phase 4** (raise `GPU_EXPERTS`) — cheap, mechanical, parallelizable with Phase 1.
4. **Phase 2** (cost-aware policy) — refine once Phase 1 shows swap pressure.
5. **Phase 3** (session persistence) — productionization for real deployments.
6. **Phase 5** (prefetch) — only if Phase 0 shows the locality headroom is there.

## 10. Harnesses
- Hit-rate + decode rate: extend `bench/intelligence_tier/cache_convergence_bench.py`
  with the Phase-0 counter; add realistic multi-turn / long-generation traces.
- Overlap re-check: `SGLANG_KT_HYBRID_TIMING=1` (+`_DEEP=1`, eager) to confirm cpu_wait
  shrinks as hit-rate rises.
- Cost model input: the measured ~3.0 ms/token/expert curve (this session).
