# Adaptive Expert Cache — Plan & Options

Status: design. Branch `experiment/adaptive-expert-cache`. Author aid: Claude.
Date: 2026-07-07.

---

## 0. TL;DR / recommendation

The current in-tree attempt does not work because the swap logic is wired **only**
into the full-GPU prefill fallback (`_update_gpu_experts_from_batch`, called at
[kt_ep_wrapper.py:3484]), so it fires at most once per long prompt and never during
decode. See §1.

Re-wiring it to run continuously is **feasible** — the CUDA graph tolerates in-place
weight/mask swaps (§3), PCIe is a non-issue (measured 44–56 GB/s, §2), and the CPU
side already holds all 256 experts so swap-out is free (§2). **But the expected
upside is small and capped**: our own oracle-placement experiment
([[glm52-oracle-placement-ceiling]]) measured that *perfect* static placement only
buys **1.15–1.35×** and floors at ~22–23 tok/s, because the CPU path cost is
dominated by fixed submit/sync + `max(cpu,gpu)` overlap, **not** by how many experts
sit on CPU. A dynamic cache can only *approach* that static-oracle ceiling, never
beat it, and it pays a warm-up penalty on top.

**Recommended path:** do the cheap thing first (Option B, between-requests
re-placement — reuses ~70% existing machinery, works on every RAM tier, ~2 days) and
**gate** any investment in the expensive per-step pool (Option C) on a measurement
that the hit-rate gains actually move tok/s past the oracle floor. Current evidence
says they mostly won't; the real speed lever remains CPU **elimination**
(substitution / `adaptN`, already shipped, 1.5–2×), not better CPU placement.

---

## 1. Root cause of the current failure

Two features are tangled on this branch:

* **`adaptN` intelligence tier** (deepseek_v2.py) — per-token K substitution, works,
  CUDA-graph-safe, shipped. Not the subject here.
* **Adaptive expert cache** (`ExpertCacheState`, `select_top_experts_weighted`,
  `_update_gpu_experts_from_batch` in kt_ep_wrapper.py) — the thing that "isn't
  working."

The cache's only call site is inside the full-GPU prefill branch
([kt_ep_wrapper.py:3461-3484]), gated on
`gpu_prefill_token_threshold>0 && num_tokens>=2048 && kt_enable_dynamic_expert_update`
(the flag is OFF by default: `DYN_UPDATE=1` → `--kt-enable-dynamic-expert-update`).
Consequences:

1. For normal chat prompts (<2048 tok) it **never fires**.
2. Even for long prompts it fires **once**, at prefill, then the resident set is
   frozen for the whole decode.
3. The scoring state machine (grace windows of 208 "events", half-life 96, rank EMA)
   is written for **per-decode-step** invocation but receives one giant prefill batch
   (`event_counter` jumps by thousands) → decay/grace logic is meaningless.
4. Empirically: **zero** `"KT adaptive expert cache"` log lines in any run under
   `experiments/adaptive_router_topn/runs/` — it has never executed in anger.

So it is not a math bug; it is attached to the wrong place in the pipeline.

---

## 2. Measured facts (uncertainties resolved 2026-07-07)

Box was upgraded since the memory notes (was 373 GB). Server was down for tests.

| Fact | Value | How measured |
|---|---|---|
| Host RAM | **629 GB total, 620 free** | `free -g` |
| VRAM | 2× H100 **95 GB each** | `nvidia-smi` |
| PCIe | **Gen5 ×16** | `nvidia-smi -q` |
| Per expert (W4AFP8 GPU fmt) | **19.46 MB** | computed = 373 GB/(256×75), agree |
| MoE layers | 75 (78 − 3 dense) | config.json |
| Experts / layer | 256 routed, 8 active/token | config.json |
| **Pinned H2D** | **56 GB/s**, 0.34 ms/expert | torch microbench |
| **Pageable H2D** | **44 GB/s**, 0.44 ms/expert | torch microbench |
| Experts hideable / 70 ms token | **~158 (pageable) / ~200 (pinned)** | derived |
| Large pin cost | 4 GB=1.75 s, 16 GB=7 s (~2.3 GB/s) | torch microbench |
| CPU expert store | **ALL 256 experts** (identity map load_weights) | [kt_ep_wrapper.py:3202-3205] |

**Implications**

* **Don't pin the pool.** Pageable H2D is 44 GB/s — plenty. Pinning 300+ GB would
  take minutes and lock the box; use pageable, optionally a small (~2–4 GB) pinned
  staging ring only if profiling demands it.
* **PCIe is never the bottleneck.** We can refresh a large fraction of the resident
  set every token on a side stream and still hide it behind the ~70 ms/token compute.
* **Swap-out is free.** CPU holds every expert already; demoting a GPU expert means
  "stop marking it −1" (a mask flip via `update_kt_wrapper_masks`, already exists).
  Only swap-*in* needs data movement (H2D of GPU-format weights).
* **RAM is the real constraint, not bandwidth.** The GPU-format pool must coexist
  with the existing ~248 GB CPU-format set. See §4 accounting.

---

## 3. Why the CUDA graph is NOT the blocker (corrected)

A CUDA graph captures **memory addresses, not values**. A captured MoE kernel holds a
baked pointer to `layer.w13_weight` and dereferences it at replay time. Overwriting
that buffer's *contents* in place between replays is legal (Rule 1: data mutation) and
is exactly how sglang already feeds fresh `input_ids` each step. The existing code was
written for this — [kt_ep_wrapper.py:3811] comment: *"Use .copy_() for CUDA tensors to
maintain same buffer for CUDA graph compatibility."*

What the graph forbids is changing the **set/shape/addresses** of kernels — i.e.
recapture-only changes (e.g. changing `num_gpu_experts`). A **fixed-count content
swap** needs none of that.

The real constraints are therefore logistical, not graph-legal:

1. The decide-and-copy must run **outside** the replay (between decode steps, in the
   scheduler loop) — today it lives inside `apply()`, which is captured.
2. Swap-in needs a **host source in post-`process_weights_after_loading` GPU format**
   so it is a pure memcpy (the CPU-format copies are a different quant layout).
3. Adaptation granularity is **per-N-steps**, not per-layer (all 75 layers are inside
   one replay). Fine for decode: token-to-token expert reuse is high.

---

## 4. The three options

### Option A — Status quo (static uniform placement)
* **What:** current behaviour. Fixed per-layer GPU experts (`uniform`/`oracle`).
* **RAM:** baseline (~248 GB CPU set + resident VRAM).
* **Gain:** 0 (this is the baseline: 14 trad / 22 top2 / 34 MTP tok/s).
* **Risk:** none.
* **Machine target:** all.
* **Use when:** the cheaper options don't clear the oracle ceiling (likely).

### Option B — Rebuild between requests (low-RAM friendly) ★ recommended first
* **What:** accumulate router stats during decode (cheap, in-graph counter). At a
  **request boundary** (or every M≫1 tokens when idle), re-derive placement (MRS
  score) and re-stream the needed experts. **No persistent GPU-format pool** — stream
  from the on-disk checkpoint (NVMe) or reuse the existing `_build_full_context`
  scratch-layer path that already stages all experts and copies the selected subset.
* **RAM:** **+0** persistent (transient scratch only). Works on 256–384 GB boxes.
* **Cost per rebuild:** stream ~R experts. From NVMe (~3–6 GB/s): 100×19 MB ≈ 1.9 GB
  ≈ 0.3–0.6 s. From the existing full-context path: similar. Amortised over a whole
  generation → negligible; can be async / off the decode critical path.
* **Gain (expected):** ≤ oracle ceiling, **1.15–1.35×**, realistically less because
  placement is chosen from the *previous* request's stats (cross-request locality is
  weaker than intra-generation). Best case: workloads with stable domains (long coding
  session hitting the same experts) — matches the "domain-aware prefetch" literature.
* **Risk:** low. Reuses `_build_full_context`, `copy_experts_weights_w4afp8`,
  `update_gpu_expert_mappings`, `update_kt_wrapper_masks` — all present.
* **Machine target:** all, especially low-RAM.

### Option C — Per-step adaptive cache with persistent host pool
* **What:** the "real" adaptive cache. In-graph score counter; a between-step
  controller (every N≈16–32 tokens) reads the counter, computes desired resident set
  with hysteresis, and swaps a bounded number of experts via H2D from a **persistent
  pinned/pageable host pool of GPU-format experts**, updating mask/mapping in place.
* **RAM:** pool must coexist with the 248 GB CPU set. Accounting (this box, R≈100
  resident/layer):
  * Full pool (all 256 × 75, GPU fmt) = **374 GB** → 248+374 = 622 GB ≈ all of 629 GB
    → **infeasible** (no headroom).
  * Pool for non-resident only (156/256 × 374) ≈ **228 GB** → 248+228 = **476 GB** →
    fits on 629 GB, ~150 GB headroom. **Feasible here.**
  * **Bounded swap-zone** (only Z swap-eligible experts/layer keep a GPU-fmt host
    copy): Z=64 → 91 GB; Z=32 → 46 GB → total ~294 GB → fits even on a 384 GB box.
    This is the tunable knob that lets Option C scale down.
* **Cost per swap:** 0.44 ms/expert pageable, hidden on a side stream. Swap ≤ a few
  experts/layer per interval → invisible.
* **Gain (expected):** still capped at the oracle ceiling **1.15–1.35×**, minus a
  cold-start warm-up ("start slow, pick up speed"). The advantage over Option B is
  *intra-generation* adaptation (converges within one response). Whether that beats B
  is the open question — and per the oracle data the delta is small.
* **Risk:** medium. New host pool + load-time GPU-format processing of all experts +
  between-step controller + in-graph counter. Pinning avoided (§2).
* **Machine target:** ≥512 GB (non-resident pool) or any box using a bounded swap-zone.

---

## 5. The hard truth about upside (read before building)

[[glm52-oracle-placement-ceiling]]: we built the *actual* static oracle
(`--kt-expert-placement-strategy oracle`, per-question mask). Result:

* Real oracle top8 = **1.15–1.35×** (global 22.5 vs 19.5).
* It **floors at ~22–23 tok/s regardless of coverage** (69%→92% barely moves it).
* Only CPU **elimination** (top0 / substitution) gets the 3.7× — better *placement*
  does not, because the CPU critical path is fixed submit+sync + `max(cpu,gpu)`
  overlap, which rebalancing merely shuffles.

Adaptive caching is an online approximation of that static oracle → its ceiling is the
oracle's ceiling. So **the entire adaptive-cache direction is a ≤1.35× play at best**,
while the already-shipped substitution path (`adaptN`, top2) is a 1.5–2× play. Invest
accordingly.

---

## 6. Phased plan (gated)

**Phase 0 — Cheap validation gate (½ day). DO THIS BEFORE ANY BUILD.**
Prove the premise that better *dynamic* placement moves tok/s at all, using zero new
infrastructure:
1. Capture a router trace for a representative workload (KT_DUMP_TOPK already exists).
2. Compute the ideal per-layer resident set offline; write an oracle mask.
3. Launch with `--kt-expert-placement-strategy oracle` on that workload and measure
   decode tok/s vs `uniform`.
4. **Gate:** if the warm oracle gain is <1.15× on our target workloads, **stop** —
   dynamic caching cannot exceed it. Redirect effort to substitution/MTP.

**Phase 1 — Option B (2 days), if Phase 0 passes.**
1. Add an in-graph per-layer expert score counter (scatter-add of routing weights;
   captured, no CPU sync). Read-out only between requests.
2. Add a between-request hook in the scheduler event loop
   ([scheduler.py:2387 run_batch] boundary) that, when a request finishes / queue
   idle, computes MRS placement and calls the existing update path
   (`_build_full_context` + `copy_experts_weights_w4afp8` +
   `update_gpu_expert_mappings` + in-place mask `.copy_()`).
3. Reuse `select_top_experts_weighted` scoring (already MRS-shaped) but feed it the
   accumulated counter, not a single batch.
4. Validate: `bench/intelligence_tier/validate_tiers.py` for coherence; decode tok/s
   vs baseline on a stable-domain workload (coding session) and a diverse one.

**Phase 2 — Option C (4–6 days), only if B shows intra-generation adaptation helps.**
1. At load, run `process_weights_after_loading` for **all** experts; park
   non-resident (or swap-zone) GPU-format tensors in a pageable host pool
   (`ExpertHostPool`, keyed by (layer, expert)). Bounded swap-zone size = env knob.
2. Move decide-and-copy out of `apply()` into a between-step controller invoked every
   N tokens from the scheduler loop; run copies on a dedicated side stream; sync
   before the next replay reads swapped slots (or double-buffer).
3. In-place update `gpu_experts_mask_cuda` / `logical_to_gpu_index_cuda` +
   `update_kt_wrapper_masks`. Hysteresis/min-swap to prevent thrash (logic exists).
4. Validate coherence + tok/s + swap-rate telemetry; A/B vs Option B.

---

## 7. Risks & open questions

* **R1 (upside):** oracle ceiling says ≤1.35×. Phase 0 gate exists to kill early. HIGH.
* **R2 (RAM on this box):** full pool infeasible (622 GB); must use non-resident-only
  (476 GB) or bounded swap-zone. Accounted, MED.
* **R3 (GPU-format host copy):** all experts must be processed to cutlass layout at
  load and kept host-side — extra load time + the pool RAM. MED.
* **R4 (thrash):** dynamic re-placement can oscillate; MRS + min-swap + grace windows
  mitigate (present but untuned). MED.
* **R5 (cross-request vs intra-generation locality):** Option B bets on cross-request
  stability; unproven for chat. Phase 1 measures it. MED.
* **R6 (side-stream correctness under CUDA graph):** swaps must complete before the
  slot is read by the next replay; needs an event/stream barrier. LOW (well-understood).
* **Resolved:** CUDA-graph legality (§3), PCIe bandwidth (§2), pinning (skip it),
  CPU-holds-all-experts (§2).

---

## 8. Appendix — measurement scripts

* PCIe / H2D microbench: inline (see §2 numbers; reproduce with a torch `copy_` loop
  at 19.46 MB granularity, pinned vs pageable).
* Router trace: `KT_DUMP_TOPK=1` (deepseek_v2.py `_kt_dump_topk`).
* Oracle mask build + warm bench: `oracle_ceiling_bench.py`, `build_oracle_masks.py`
  ([[glm52-oracle-placement-ceiling]]).
* Coherence/tok-s: `bench/intelligence_tier/validate_tiers.py`,
  `bench/intelligence_tier/adaptive_bench.py`.
