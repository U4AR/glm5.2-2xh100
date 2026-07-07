# Reducing hybrid CPU-expert cost while always keeping the genuine top-2

Branch: `experiment/adaptive-expert-cache`
Date: 2026-07-07
Constraint: **the genuine top-2 experts are never dropped** (accuracy-safe). We do
NOT force routing into the resident set; we only substitute the *tail* (ranks 3-8)
and, at worst, let a non-resident genuine top-2 expert take a CPU round-trip.

---

## 1. Why submit + sync fires every MoE layer, every token

The CPU experts' **weights live in system RAM** and are computed by the AVX512
kernel. The GPU cannot touch them. So to include a CPU-resident expert's
contribution to a token you must hand its work to the CPU and wait. Per MoE layer
(`kt_ep_wrapper.py` ~L3530 submit / ~L3631 sync):

1. `staging_buffer.copy_(x)` — stage the token's hidden state.
2. `submit_forward(staging, topk_ids, topk_weights, cuda_stream)` — pybind11 call,
   records a CUDA event and enqueues the CPU MoE job (non-blocking).
3. GPU computes its resident experts concurrently on the main stream.
4. `sync_forward(...)` — block until the CPU job signals done, then
   `output = output + cpu_output`.

Two structural reasons it can't be amortized away:

- **Sequential layer dependency.** Layer N+1's input *is* layer N's output. You
  cannot pre-compute or batch layers ahead, so all 75 MoE layers each do their own
  submit+sync in order → 75 handshakes per token.
- **Per-token decode.** Decode is batch-size-1; each generated token traverses all
  75 layers → 75 handshakes on every token.

The **fixed** part is the *handshake floor*: pybind11 crossing + CUDA event
record/wait + CPU-thread wake latency. It is ~constant whether the CPU computes 1
expert or 8 — it does not shrink with less work.

---

## 2. The measurement reframes how big "fixed" actually is

Cache-informed top2 forces *zero* CPU experts, so the only thing left is the pure
handshake. Two runs isolate it (GPU_EXPERTS=96, RAWINT4, MEM_FRACTION=0.90):

| config | tok/s | note |
|---|---|---|
| top8 placement-only            | 14.6 | baseline |
| top2, all-resident, **no skip** | 22.3 | handshake fires, CPU computes nothing |
| top2, all-resident, **skip**    | 23.5 | handshake removed |

- **CPU compute** (8 experts → ~2) bought **+52%** (14.6 → 22.3).
- **Fixed handshake** bought only **+5%** (22.3 → 23.5 = 1.2 tok/s).

=> The fixed submit/sync cost is real but small. The expensive thing is CPU
*compute* of non-resident experts. The accuracy-safe plan therefore attacks the big
cost (compute), not the small one (handshake).

---

## 3. Plan (ranked by payoff x safety; genuine top-2 always kept)

### P1 — Turn non-resident genuine top-2 into ~free GPU compute  [BIGGEST, SAFE]
The real cost is when a genuine top-2 expert isn't resident → it runs on CPU. Fix
*placement*, not routing: make the adaptive cache **target the top-2 working set**
(score experts by top-2 hits, not top-8), so the 96 GPU slots hold what the genuine
top-2 actually needs. When top-2 is resident, its compute is on-GPU and nearly free
and the CPU has nothing to compute → only the ~5% handshake remains.
- Change: cache scorer weights top-2 selections much higher than tail selections
  (`_update_gpu_experts_from_batch` / `ExpertCacheState` scoring; the
  `last_top2_event_cpu` vs `last_tail_event_cpu` tracking already exists ~L145-165).
- Success metric: top2_hit high on **diverse** prompts (not just repeated), and
  decode → ~22 with genuine top-2 preserved.

### P2 — Conditional per-layer skip  [removes the ~5% handshake, SAFE]
When a layer's routed experts are all GPU-resident this token, skip submit+sync for
that layer. Safe (skipping adds nothing when the CPU has no experts). Obstacle:
the handshake is baked into the captured CUDA graph (`_KT_SKIP_CPU` is import-time,
L82), so a per-token data branch is not normally capturable.
- Route (a): **CUDA graph conditional nodes** (CUDA 12.4+ `cudaGraphConditionalNode`)
  — a runtime-evaluated branch inside the graph. Correct modern fix, higher effort.
- Route (b): **cheap C++ early-out** in the CPU kernel when the token has zero CPU
  experts — removes the compute path but leaves the GPU-side event handshake
  (partial win only).

### P3 — Shave the handshake constant itself  [CHEAP, INCREMENTAL]
Independent of P2.
- Collapse the extra CPU stream onto the main stream:
  `SGLANG_KT_HYBRID_NO_CPU_STREAM=1` already exists (L3522) — drops the per-layer
  fork/wait event pair.
- Keep CPU worker threads spinning so wake latency ~ 0.
- A/B with `SGLANG_KT_HYBRID_TIMING=1` to read per-layer cpu_wait_ms and confirm the
  ~5% ceiling / how much is actually recoverable.

### Dead ends (documented so we don't chase them)
- **Per-token weight streaming** of the missing expert to GPU — PCIe is ~6x slower
  than GH200 C2C; cannot stream per token in decode. (See memory
  `glm52-bottleneck-and-gh200-blog`.)
- **Global `/tmp/kt_skip_cpu`** — fast but drops genuine top-2 on novel questions
  (repetition looping observed). Keep ONLY as an opt-in "repeated-workload" mode,
  not the general default.

---

## 4. Suggested order
1. **P3** (~1h): confirm the handshake budget, bank whatever is free.
2. **P1** (the real lever): retarget the cache scorer to the top-2 working set;
   measure top2_hit + decode on diverse prompts.
3. **P2** only if P1+P3 leave meaningful handshake on the table.

## 5. Harnesses
- Speed / convergence: `bench/intelligence_tier/cache_convergence_bench.py`
  (streaming decode-rate, N passes; excludes prefill).
- Per-layer CPU timing: launch with `SGLANG_KT_HYBRID_TIMING=1`.
- Boot (accuracy-safe genuine top-2, NO skip): `KT_METHOD=RAWINT4`
  `KT_RAWINT4_BACKEND=avx512_packed` `GPU_EXPERTS=96` `MEM_FRACTION=0.90`
  `DYN_UPDATE=1`; `/tmp/kt_topk_mode=safe2`; **do not** create `/tmp/kt_skip_cpu`.

---

## 6. EXECUTION RESULTS (2026-07-07)

All at GPU_EXPERTS=96, RAWINT4, MEM_FRACTION=0.90, cache_convergence_bench (4 fixed
prompts, 3 passes, gen=150, streaming decode rate).

### New code shipped this pass
- `deepseek_v2.py`: **`safe2` routing mode** — ACCURACY-SAFE. Keeps the genuine
  top-K ALWAYS (a non-resident top-K expert takes a CPU round-trip); substitutes
  ONLY the tail (ranks K..7) with resident experts. `sub2` (residency-forcing,
  drops non-resident top-K) is retained for repeated-workload speed.
- `kt_ep_wrapper.py`: `_KT_EXPERT_CACHE_RANK_CREDIT` is now env-overridable via
  `SGLANG_KT_EXPERT_CACHE_RANK_CREDIT` (P1 top-2 retarget), plus a
  `torch.nan_to_num` sanitize on router weights before cache scoring (prevents a
  NaN/inf from poisoning `score_cpu` via `index_add_`).

### The ladder (measured)
| config | tok/s | cost |
|---|---|---|
| residency-forcing `sub2` + global skip | 23.5 | drops non-resident top-2 (accuracy hole) |
| residency-forcing `sub2`, NO skip (P3-A) | 22.4 | zero CPU compute, handshake fires |
| residency-forcing `sub2`, `NO_CPU_STREAM=1` (P3-B) | 19.1 | REGRESSION — lost CPU/GPU overlap |
| **`safe2` accuracy-safe, default scorer** | **20.2** | non-resident top-2 → CPU compute + handshake |
| `safe2` + P1 top-2 retarget (credit `1.2,1.0,0..` + tail grace 0) | 20.0 | NULL — placement can't raise residency enough |

### Conclusions
- **P3 = dead end for a free win.** `NO_CPU_STREAM=1` collapses the CPU stream onto
  the main stream and *loses* the CPU/GPU overlap (22.4 → 19.1). The separate CPU
  stream is worth far more than the fork/wait event pair it costs. The ~5%
  handshake (22.4 → 23.5) is ONLY reclaimable via P2 conditional skip, never via
  stream collapse.
- **P1 = null.** Retargeting the adaptive cache to the top-2 working set (top-2-only
  rank credit + disabled tail protection) did NOT improve decode (20.0 vs 20.2).
  Consistent with `glm52-adaptive-cache-floor` / `glm52-oracle-placement-ceiling`:
  at 96/256 residency the top-2 SET per layer across diverse tokens exceeds 96
  slots, so some genuine top-2 is always non-resident → CPU compute placement
  cannot remove.
- **Accuracy-safe floor ≈ 20 tok/s and it is ARCHITECTURAL.** safe2's 2.2 tok/s gap
  to 22.4 is CPU *compute* of non-resident top-2 (removable only by dropping them =
  accuracy-unsafe, or making them resident = P1, which failed). The remaining
  1.1 tok/s to 23.5 is the handshake (P2 territory).
- **`odesk`-style token glitch is NOT a safe2 artifact** — the exact-baseline
  `-top8` tier produces it too, so it is a base-model/decoding effect, independent
  of substitution.

### P2 assessment (conditional per-layer skip) — LOW ROI, not implemented
Upside is bounded by the handshake (≤5%) AND only realizable on layer-tokens where
BOTH genuine top-2 are resident. On safe2 (top-2 residency imperfect → 20.2 < 22.4)
the realistic ceiling is ~20.2 → ~21 (+3%). It requires CUDA-graph **conditional
nodes** (CUDA 12.4+) because the handshake is baked into the captured decode graph
(`_KT_SKIP_CPU_PATH`, import-time) — days of work for ~3%. The cheap C++ early-out
(route b) leaves the GPU event handshake, so it wins ~nothing. **Not worth it.**

### The real accuracy-safe speed lever: MTP / spec-decode (already shipped)
MTP is accuracy-PRESERVING (draft tokens are verified against the true model, so no
top-2 is ever dropped) AND amortizes the per-token CPU handshake across multiple
accepted tokens — the one mechanism that beats the per-layer/per-token floor without
trading accuracy. `MTP=1 ./run_fast.sh` → ~34 tok/s (see `glm52-mtp-cudagraph-fix`).
Recommendation: for accuracy-safe speed, use **safe2 routing + MTP**, not placement.
