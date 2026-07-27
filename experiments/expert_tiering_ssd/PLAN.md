# Three-tier expert store: GPU VRAM / host RAM / NVMe SSD

Branch `experiment/expert-tiering-ssd` (snapshot of the live tree at `9e007c2`,
also tagged `backup/adaptive-decode-cache-20260727`).

Goal: run GLM-5.2 on **any** split of VRAM + RAM + SSD, not just boxes with
~400 GB of RAM. Experts live in three tiers; a routed expert that is only on SSD
is **never fetched on the critical path** — it is substituted with its nearest
equivalent from RAM/GPU, and its demand is recorded so the cache promotes it on
the next tick.

---

## 1. What already exists (this is mostly a third tier bolted onto shipped code)

| Piece | Where | Status |
|---|---|---|
| GPU tier + live restage | `kt_ep_wrapper._apply_expert_selection` | shipped, CUDA-graph-safe |
| Decode-time demand counters | `deepseek_v2._kt_topk_experiment` → `quant_method._kt_decode_counts` | shipped, in-graph `index_add_` |
| Between-step adaptive tick | `kt_ep_wrapper.kt_adaptive_on_decode_step` | shipped, TP-lockstep |
| Substitution ("nearest equivalent") | `_kt_topk_experiment` `sub`/`safe` modes | shipped |
| Per-expert CPU-store staging skip | C++ `GeneralMOEConfig::should_skip_expert` | shipped |

The substitution machinery the user asked for is *already* the top-2 transfer
kernel: it replaces a slot with the best-scoring **resident** expert from the
router's own logits. The only change is what counts as "resident".

### Measured constants (2×H100 box, W4AFP8 int4, 78 layers / 75 routed / 256 experts)

- CPU store: 18 MiB packed int4 + ~1.4 MiB fp32 scales & int16 sums = **~19.4 MiB
  per expert per layer** → **~1.42 GiB of host RAM per expert-slot across the model**.
- All 256 staged = ~364 GiB (observed RSS 393 GB on the live server). ✓
- GPU: ~9.45 MiB/card/expert + 17.5 GiB/card dense base (from the low-VRAM ladder).

So the RAM ladder is steep and worth having: at `GPU=96`, dropping the RAM tier
from 160 → 32 experts/layer takes host RSS from ~390 GB to **~65 GB**.

### Key structural findings from reading the C++

1. `AMX_MOE_BASE` **always `aligned_alloc`s all 256 per-expert `BufferB`s**;
   `should_skip_expert` only skips *filling* them. Untouched pages are never
   faulted in, so the address space is free — a RAM slot is simply "filled or not".
   → eviction = `madvise(MADV_DONTNEED)`; promotion = fill it. No slot indirection
   needed, no kernel change.
2. In the inference path `gpu_experts_mask` is consumed **only** by
   `should_skip_expert` (`num_gpu_experts` is used exclusively by `sft_moe.hpp`,
   which we don't run). So that one mask can be repurposed as **"not CPU-resident"
   = GPU-tier ∪ SSD-tier** with no C++ change, and it doubles as a safety net:
   a stray SSD-tier id contributes zero instead of garbage.
3. Packed int4 weights in the CPU store are **byte-identical to the on-disk
   W4AFP8 safetensors** (`from_raw_mat` memcpys the rows); only the bf16→fp32
   scale conversion and the int16 `weight_sums` are derived. → **the SSD tier is
   the existing model directory**. No new file format, no 373 GB pre-write.

**Consequence: Phase 0 is Python-only.** C++ is needed only for runtime
promote/evict (Phase 1), and it is ~2 small methods.

---

## 2. Design

### Tiers and the single ranking

Per layer, one decayed demand vector `counts[256]` (already exists) is cut twice:

```
rank experts by counts  ->  [ 0 .. N_gpu )      GPU-resident   (fast path)
                            [ N_gpu .. N_gpu+N_ram )  RAM-resident (CPU kernel)
                            [ N_gpu+N_ram .. 256 )    SSD only    (substituted)
```

Both boundaries get the existing hysteresis (`KT_ADAPTIVE_MARGIN` ratio gate +
`KT_ADAPTIVE_MIN_GAIN` absolute benefit gate — the second one is what made the
GPU tier quiesce and it is equally required here).

Promotion is therefore transitive and automatic: an SSD expert that fires
repeatedly climbs into RAM; if it keeps firing it climbs into GPU; whatever it
displaced falls one tier. That is exactly "move to ram/gpu depending on usage
and move others to ssd".

### Masks

| Mask | Lives | Used by |
|---|---|---|
| `gpu_experts_mask_cuda` | per rank, GPU | `_kt_topk_experiment` fill pool, `mask_and_remap_expert_ids` |
| `ram_experts_mask_cuda` **(new)** | per rank, GPU | `_kt_topk_experiment` keep filter |
| kt wrapper `gpu_experts_mask` (pinned) | rank 0 only | C++ staging + forward skip; set to `~(gpu ∪ ram)` i.e. **not-CPU-resident** |

All three flip **in place** (`.copy_()`) — the decode CUDA graph captured their
addresses. Tier selection runs on rank 0 and is broadcast, exactly like the
shipped GPU selection, so ranks stay in lockstep (a rank-divergent early return
deadlocks NCCL — the existing code comments on this and we keep the discipline).

### Routing change (`_kt_topk_experiment`, `safe` mode)

Today `safe` keeps the genuine top-K unconditionally. With an SSD tier that is
unsafe — a top-K expert with no weights anywhere would be silently skipped by
the CPU kernel and lose its mass. One line changes it to keep-if-resident-anywhere:

```python
resident_any = gpu_vec | ram_vec              # (n_experts,)
keep_mask &= resident_any[ids.long()]         # SSD-tier slots fall through to fill
```

The fill pool (the "nearest equivalent") is selected by router score among a
configurable set:

- `KT_TIER_FILL_POOL=resident` (default) — GPU ∪ RAM. Best accuracy; an SSD
  expert is replaced by the best RAM expert, which still costs a CPU round-trip.
  This is the literal reading of "use the nearest equivalent from RAM".
- `KT_TIER_FILL_POOL=gpu` — GPU only. Faster (SSD misses become free), less
  faithful. Worth measuring as the speed variant.

Everything stays branch-free and shape-static, so it captures into the one
decode graph and coexists with the per-request `-topN` tiers.

### The two cache-update modes (the user's explicit ask)

The counter is fed from **pre-substitution** ids, so experts we never computed
are still counted — that is what lets an SSD expert be discovered at all.

```python
cnt.index_add_(0, ids.long().reshape(-1), (rank < COUNT_K).float().reshape(-1))
```

- `KT_TIER_COUNT_MODE=top8` → `COUNT_K = 8`: every routed expert votes, weighted
  by nothing (or by `topk_weights`, see below). Richer signal, sees demand for
  experts that would never survive substitution.
- `KT_TIER_COUNT_MODE=top2` → `COUNT_K = 2`: only the genuine top-2 vote. This is
  today's shipped behaviour and matches what actually costs a CPU round-trip.

Optional third variant worth a row in the table: `top8w`, weighting each vote by
its router weight (a rank-8 expert contributing 2% of mass shouldn't buy a slot
as cheaply as a rank-1 expert). One extra `torch.where`.

### Promotion / demotion mechanics (Phase 1)

On the adaptive tick, per layer (already round-robin, 2 layers per 32 steps):

1. rank 0 ranks `counts`, cuts at `N_gpu` / `N_gpu+N_ram`, applies hysteresis,
   caps churn at `KT_TIER_MAX_PROMOTE` per visit.
2. GPU set changes → existing `_apply_expert_selection` (unchanged).
3. RAM set changes → for each demoted expert `wrapper.release_expert(e)`; for
   each promoted expert load its 6 tensors from the model safetensors and call
   `wrapper.load_expert(e, ptrs...)`.
4. Flip the pinned not-CPU-resident mask, broadcast the new masks, done.

Cost estimate per promoted expert: ~18 MiB read from NVMe (3–7 GB/s → 3–6 ms)
plus the `weight_sums` nibble walk, parallelised over the CPU pool. Budget
~10–20 ms/expert; with `MAX_PROMOTE=4/visit` that is well under the 140–200 ms
the GPU restage already costs, so the tick stays off the critical path.

---

## 3. Phasing

### Phase 0 — static three-tier, Python only, no rebuild *(fast, de-risks everything)*

1. `kt_ep_wrapper`: add `KT_RAM_EXPERTS` (per-layer RAM tier size). Build the
   initial RAM set from the existing `hot_core_ranking.pt` prior (ranks below the
   GPU cut), pass `~(gpu ∪ ram)` as the **load-time** wrapper mask so the CPU
   store stages exactly `N_ram` experts, and keep that same mask live for forward.
2. `amx.py` / `loader.py`: skip `load_tensor(...).contiguous()` for experts that
   won't be staged. (Also fixes the known ~25 min boot at low N — today it
   materialises all 256 regardless.)
3. `deepseek_v2`: add `ram_experts_mask_cuda`, the keep-if-resident-anywhere
   filter, `KT_TIER_FILL_POOL`, and `KT_TIER_COUNT_MODE`.
4. Ship a `run_tiered.sh` taking `GPU_EXPERTS` / `RAM_EXPERTS`.

**Deliverable:** tok/s + quality + RSS across the RAM ladder with a frozen tier
split. This alone answers "can it run on a small-RAM box, and what does it cost?"

### Phase 1 — dynamic promotion/demotion (C++)

5. `moe_base.hpp`: record `(ptr, bytes)` per `BufferB` alloc; bump alignment
   64 → 4096 so the range is page-aligned; add `release_expert(e)` →
   `madvise(MADV_DONTNEED)` over each TP part's three buffers.
6. `rawint4_packed_avx512vnni-moe.hpp`: add `load_expert(e, src ptrs)` — the body
   of `TP_MOE::load_weights` restricted to one expert (incl. the down-proj column
   gather), then `from_raw_mat` for that expert only.
7. `loader.py`: `CompressedSafeTensorLoader.load_expert(base_key, exp_id)`
   (per-expert instead of the whole layer); keep the mmap handles open for the
   life of the server rather than `close_all_handles()` after load.
8. `ext_bindings.cpp` + `experts.py`: expose both.
9. `kt_ep_wrapper`: two-cut selection + the promote/evict step in the tick.

Rebuild discipline (from the packed-int4 handoff): `install.sh build` with
`CPUINFER_USE_CUDA=1` and `CMAKE_PREFIX_PATH`/`PKG_CONFIG_PATH` pointed at the
venv — a manual `cmake` produces a CPU-only `.so` that drops
`submit_with_cuda_stream` and silently kills CUDA-graph overlap.

### Phase 2 — measurement (below)

---

## 4. Measurement matrix

Fixed: `GPU_EXPERTS=96`, `MEM_FRACTION=0.85`, MTP depth-3, `safe` mode, the
canonical `bench/decode_bench.sh` + held-out `conv_driver.py`.

| Axis | Values |
|---|---|
| `RAM_EXPERTS` | 160 (=all, today's baseline), 96, 64, 32, 16, 8, 0 |
| `KT_TIER_COUNT_MODE` | `top2`, `top8` (and `top8w` if time) |
| `KT_TIER_FILL_POOL` | `resident`, `gpu` |
| tiering | static (P0) vs adaptive (P1) |

Metrics per cell:

1. **tok/s** — `usage.completion_tokens` only; **bench twice**, the cache adapts
   to the bench prompt during the first run (this bit us before).
2. **Host RSS** — the whole point; confirm the RAM budget is actually honoured.
3. **Genuine top-2 coverage** by tier: `P(top-2 ∈ GPU)`, `P(∈ RAM)`, `P(∈ SSD)`.
4. **Quality** — held-out coherence (`conv_diverse.py`) + a KL-vs-top8-baseline
   on a fixed prompt set. Coverage alone is not quality; the honest metric is
   divergence from the untiered model.
5. **Cache dynamics** (P1) — promotions/min, time-to-converge, whether it
   quiesces. Ratio hysteresis alone never quiesced on the GPU tier; expect the
   same and use the absolute benefit gate.

Expected shape, stated up front so a surprise is informative: tok/s should be
**roughly flat** as RAM shrinks (CPU compute per token is unchanged; SSD misses
are substituted, not fetched) and may even *rise* with `FILL_POOL=gpu` as more
routing collapses onto the GPU tier — while **quality** degrades. If tok/s falls
instead, the tick's promote traffic is on the critical path and `MAX_PROMOTE`
needs lowering.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| `MADV_DONTNEED` doesn't free (buffer under `brk`, not mmap) | 19 MiB ≫ mmap threshold, so it is mmap-backed; verify with RSS before/after a forced evict sweep |
| Promotion races the CPU kernel | Runs only in the between-step tick, after the CPU sync, same place the GPU restage already runs |
| Rank divergence → NCCL deadlock | Unconditional broadcast of the tier masks; guards evaluated on rank-identical state (existing pattern) |
| Non-resident id reaches the CPU kernel | `should_skip_expert` already covers it via the not-CPU-resident mask: contributes zero, not garbage |
| Boot time at small `N_ram` | Phase 0 step 2 skips loading unstaged tensors — should *improve* on today's ~25 min |
| `N_ram = 0` | The GPU tier must be ≥1 (empty resident set crashes the substitution `max()`); `N_ram=0` is fine as long as `N_gpu ≥ 1` |

## 6. Not disturbed

The running server (PID 42591, `experiment/adaptive-decode-cache` config) is
untouched: the backup was a branch pointer + commit, no file contents changed.
All new work lands on `experiment/expert-tiering-ssd`.
