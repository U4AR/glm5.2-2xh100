# Plan: Adaptive Routed-Expert Cache

Objective: build a runtime expert-cache mechanism that chooses which routed
expert weights stay in the fastest memory tier while the model evaluates. The
policy should react to router output, promote likely-to-be-reused experts, and
depopulate cold experts as the request/batch evolves.

The cache key is always:

```text
(layer_id, expert_id)
```

Do not cache by `expert_id` alone. Each layer owns different expert tensors.

## Motivation From The Experiment

Important measurements:

- Full routed expert body: **337.5 GiB**.
- Top-2 usage-p90 across five tasks: **134.1 GiB** covers 90% of top-2
  selections.
- Top-8 usage-p90 across five tasks: **220.2 GiB** covers 90% of top-8
  selections.
- Top-2 unique touched across five tasks: **267.0 GiB**.
- Top-8 unique touched across five tasks: **334.7 GiB**.
- Top-2 rare/task-specific experts repeat after first use **63.5%** of the time.
- Top-2 rare experts shared by at most two tasks repeat **64.8%** of the time.
- Median gap to second top-2 rare hit:
  - task-specific: **104 expert-selection events**
  - shared by at most two tasks: **67 expert-selection events**
- Top-8 tail-to-top2 transition:
  - rank 3 later enters top-2: **54.3%**
  - rank 4 later enters top-2: **46.8%**
  - rank 5 later enters top-2: **41.0%**
  - rank 6 later enters top-2: **35.8%**
  - rank 7 later enters top-2: **31.4%**
  - rank 8 later enters top-2: **27.1%**

Implication:

Top-2 is the main cache signal. Tail ranks 3 and 4 are useful weak predictors.
Ranks 7 and 8 should barely affect residency.

## Existing Code To Build On

The KT wrapper already has a dynamic update skeleton:

- `select_top_experts_from_batch(...)` in
  `.venv/lib/python3.12/site-packages/sglang/srt/layers/moe/kt_ep_wrapper.py`
  currently selects by raw batch frequency.
- `_update_gpu_experts_from_batch(...)` copies selected weights into the GPU
  expert slots and updates mapping tables.
- `update_gpu_expert_mappings(...)` and `update_kt_wrapper_masks(...)` update
  masks/mappings. CUDA graph compatibility depends on in-place tensor updates.
- `--kt-enable-dynamic-expert-update` already exists as a server flag.

The first implementation should modify the selection policy, not invent a
separate relocation system.

## Memory Tiers

Tier 0: GPU VRAM / HBM

- Holds currently resident expert weight blocks.
- Fastest tier.
- Capacity is limited by `kt_num_gpu_experts` per layer or by a global budget.

Tier 1: pinned CPU RAM / kt CPU expert store

- Source of truth for non-resident experts.
- Used directly by current KT hybrid CPU path.
- Promotion to GPU copies from this tier into a GPU expert slot.

Tier 2: regular CPU RAM / mmap / disk-backed checkpoint

- Cold source for reload/rebuild paths.
- Avoid touching in steady-state decode.

Demotion from GPU does not need to copy back. The CPU copy already exists.
Demotion means:

```text
remove block from GPU-resident mask, then overwrite that GPU slot with a promoted block
```

## Cache Score

Maintain per-layer arrays of length 256:

```text
score[layer, expert]
last_seen_event[layer, expert]
last_top2_event[layer, expert]
last_tail_event[layer, expert]
hit_count_window[layer, expert]
rank_ema[layer, expert]
resident_state[layer, expert]
```

On every routed batch, update scores from sorted router rows.

Suggested event weights:

| router rank | score credit |
|---:|---:|
| 1 | 1.20 |
| 2 | 1.00 |
| 3 | 0.35 |
| 4 | 0.25 |
| 5 | 0.15 |
| 6 | 0.08 |
| 7 | 0.04 |
| 8 | 0.02 |

Rationale:

- Rank 1 and 2 are actual top-2 compute targets.
- Rank 3 and 4 are the only tail positions with strong transition probability
  into later top-2.
- Rank 7 and 8 should have tiny influence because they are less likely than
  baseline to become top-2 later.

If router weights are available, multiply rank credit by normalized router
weight:

```text
credit = rank_credit[rank] * (router_weight / sum_top8_router_weight)
```

Score update:

```text
score[e] = score[e] * decay(delta_events) + credit
```

Use an exponential decay with a half-life tied to the measured repeat gap:

```text
half_life_top2 = 96 expert-selection events
half_life_tail = 192 expert-selection events
```

These numbers are intentionally close to the measured median gaps:

- top2 task-specific second-hit median: 104
- top2 rare/shared second-hit median: 67
- top8 rare/shared second-hit median: 199

The first production sweep should tune these values.

## Promotion Policy

For each layer, maintain `K_gpu` resident experts. Current production default is
104 GPU experts per layer.

At an update point:

1. Compute `candidate_score[e]` for all 256 experts.
2. Add hysteresis:

   ```text
   if resident[e]:
       effective_score[e] = score[e] + resident_bonus
   else:
       effective_score[e] = score[e] - promotion_cost
   ```

3. Select top `K_gpu` experts by `effective_score`.
4. Compare selected set to current resident set.
5. Only update if churn is worthwhile:

   ```text
   changed_slots >= min_swap_count
   and expected_saved_cpu_hits > copy_cost_threshold
   ```

Suggested starting values:

```text
resident_bonus = 0.25
promotion_cost = 0.10
min_swap_count = 2 experts per layer
max_swap_count = 8 experts per layer per update
```

Limit swaps to prevent thrashing and PCIe copy spikes.

## Depopulation Policy

An expert can be demoted from GPU if:

```text
not in top K_gpu by effective score
and no top2 hit for eviction_grace_events
and score below layer eviction threshold
```

Suggested grace windows:

| signal type | grace |
|---|---:|
| top2 hit | 2 * 104 = 208 expert-selection events |
| rank 3-4 tail hit | 2 * 199 = 398 expert-selection events |
| rank 5-8 tail hit | 104 expert-selection events |

Interpretation:

- A top-2 hit means the block has a strong chance of repeating soon. Keep it at
  least about two median-repeat windows.
- A rank 3-4 hit is a meaningful near-miss. Keep it warm longer than weak tail
  hits if there is room.
- Rank 7-8 should not protect a block from eviction for long.

For real code, express grace in per-layer token rows or global expert-selection
events. Keep one consistent unit and log it.

## Update Timing

Do not update on every decode token.

Recommended schedule:

1. **Prefill update**
   - After prefill, compute per-layer scores from the prompt.
   - Promote the top expected experts before decode.
   - This is the highest-leverage update point.

2. **Decode rolling update**
   - Every `N` generated tokens, e.g. 16 or 32.
   - Only update layers with enough score drift.
   - Cap total swaps per update cycle.

3. **Request-local cooling**
   - At request end, decay request-local boosts.
   - Preserve global EMA so common experts remain hot across requests.

Use separate score components:

```text
score = global_score + request_score + batch_score
```

Suggested:

```text
global_score: slow EMA across many requests
request_score: fast EMA within current request
batch_score: immediate prefill/decode window
```

## Multi-Request Batching

When multiple requests share a batch, avoid letting one request thrash the cache
for everyone.

Use:

```text
score_update = sum over requests (request_weight * per_request_credit)
```

Where:

```text
request_weight = 1 / max(1, number_of_active_requests)
```

Optionally keep per-request score sketches and merge them at update boundaries.

## Initial Offline Simulator

Before changing runtime code, build an offline simulator over the existing traces.

Inputs:

- `runs/exp3_*.pt`
- GPU capacity per layer: test 32, 64, 96, 104, 128, 160, 192.
- Scoring coefficients.
- Decay half-life.
- Update interval.
- Swap cap.

Metrics:

- Top-2 GPU hit rate.
- Top-8 GPU hit rate.
- CPU expert calls per token.
- Number of swaps.
- Estimated copy volume.
- Estimated copy time.
- Hit-rate improvement vs static uniform placement.
- Hit-rate improvement vs current batch-frequency-only dynamic update.

This simulator should answer:

```text
How many GPU-resident experts per layer are needed to cover 90%, 95%, 99% of
future top-2 calls under a realistic promotion/demotion policy?
```

## Runtime Implementation Plan

### Phase 1: Instrumentation and Offline Policy

1. Add `simulate_cache.py` in this experiment directory.
2. Implement policy classes:
   - `LRUPolicy`
   - `FrequencyPolicy`
   - `WeightedRouterPolicy`
   - `WeightedRouterWithTailPolicy`
3. Replay `runs/exp3_*.pt` token rows in order.
4. Emit tables and SVGs:
   - hit rate vs GPU experts/layer
   - swaps vs update interval
   - copy volume vs hit rate
   - top2 miss rate over time

Exit criteria:

- Identify a default score/rank/decay policy that beats frequency-only.
- Pick safe starting values for runtime update interval and swap cap.

### Phase 2: Replace Batch-Frequency Selector

Modify:

```text
select_top_experts_from_batch(...)
```

or add a new selector:

```text
select_top_experts_weighted(...)
```

The new selector should consume both:

```text
topk_ids
topk_weights
```

and score ranks with the policy above.

Current selector only counts raw frequency over all top-k slots. That treats rank
8 the same as rank 1, which the experiment shows is wrong.

### Phase 3: Persistent Per-Layer Cache State

Add a cache state object to `KTEPWrapperMethod`:

```text
ExpertCacheState(
    score_cpu: torch.Tensor[256],
    last_seen_cpu: torch.Tensor[256],
    last_top2_cpu: torch.Tensor[256],
    resident_mask_cpu: torch.Tensor[256],
    event_counter: int,
)
```

Keep this state on CPU for easy updates. Only masks/mappings need to be copied to
GPU in-place.

### Phase 4: Safe Runtime Promotion/Demotion

Use `_update_gpu_experts_from_batch(...)` as the update mechanism:

1. Select new resident experts.
2. Copy selected expert weights into existing GPU slots.
3. Update `gpu_experts_mask_cuda` and `logical_to_gpu_index_cuda` in-place.
4. Update KT wrapper's pinned CPU mask in-place.
5. Log:
   - layer id
   - old residents
   - new residents
   - promoted/demoted experts
   - estimated copy MiB
   - top2 hit rate before/after

Do not replace tensors captured by CUDA graphs. Existing code already notes this
constraint.

### Phase 5: Async Copy and Churn Control

Once correctness is stable:

1. Move weight copies onto a dedicated copy stream.
2. Double-buffer GPU expert slots if needed.
3. Never evict a resident expert while a forward using it is in flight.
4. Add a layer-level update mutex or epoch counter.
5. Limit update work per decode step.

## Proposed Scoring Formula

Initial formula:

```text
score[e] =
    global_ema[e]
  + request_ema[e]
  + immediate_window[e]
  + resident_bonus * is_resident[e]
  - promotion_cost * is_not_resident[e]
```

Per event:

```text
rank_credit = {1:1.20, 2:1.00, 3:0.35, 4:0.25, 5:0.15, 6:0.08, 7:0.04, 8:0.02}
event_credit = rank_credit[rank] * normalized_router_weight
```

Decay:

```text
request_ema[e] *= 0.5 ** (delta_events / half_life)
request_ema[e] += event_credit
```

Use:

```text
half_life = 96 for top2 hits
half_life = 192 for rank 3-4 tail hits
half_life = 64 for rank 5-8 tail hits
```

This makes the cache responsive to current task-local structure without letting
weak tail hits dominate.

## Validation Experiments

Run these before declaring the cache useful:

1. Offline trace replay:
   - static uniform
   - static hot from first task
   - LRU
   - raw frequency
   - weighted router score
   - weighted router score with tail predictor

2. Live prefill-only:
   - launch with dynamic update enabled
   - capture GPU/CPU expert hit ratios
   - compare TTFT and total request time

3. Live decode:
   - CUDA graphs on
   - update every 16/32/64 tokens
   - verify no graph pointer invalidation
   - compare tok/s and coherence

4. Stress:
   - mixed short and long prompts
   - repeated similar tasks
   - unrelated tasks
   - concurrent requests

## Risks

- Copying 18 MiB expert blocks too often can erase the benefit.
- Top-8 already touches almost everything; top-8-based caching may saturate VRAM
  and thrash.
- Batch-level selection can be unstable for concurrent unrelated requests.
- CUDA graph safety requires in-place mask/mapping updates only.
- Current dynamic update path may be prefill-oriented; decode updates need
  careful synchronization.
- For W4AFP8, ensure the copy helper uses the correct resident weight layout.

## Success Criteria

A useful cache should show:

- Higher top-2 GPU hit rate than static uniform placement.
- Lower CPU expert calls per decode token.
- Net tok/s or latency improvement after copy overhead.
- Bounded churn: no runaway swap volume.
- Stable output quality compared with baseline top-8 or existing top2/sub2 mode.

Minimum target:

```text
At 104 GPU experts/layer, weighted adaptive placement should beat uniform
placement on top2 GPU-hit rate by at least 10 percentage points without hurting
decode throughput.
```

## Suggested Next File To Write

Create:

```text
experiments/expert_footprint_top2_vs_top8/simulate_cache.py
```

Start offline. A simulator will let us tune rank weights, half-lives, update
intervals, and swap caps without repeatedly rebooting the full server.

---

## Verification Appendix (2026-07-03) — required objects, verified against code + traces

This appendix was added after verifying every code reference, the trace schema,
and the numbers above against the live checkout. Read it before writing
`simulate_cache.py` or touching runtime code. It resolves ambiguities that would
otherwise cost the next agent a day.

### A. VERIFIED — everything the plan claims exists, exists

- `select_top_experts_from_batch(topk_ids, num_experts, num_gpu_experts)` —
  [kt_ep_wrapper.py:2245](../../.venv/lib/python3.12/site-packages/sglang/srt/layers/moe/kt_ep_wrapper.py#L2245).
  Confirmed: counts **raw frequency over all 8 slots**, ignores router weight and
  rank — exactly the weakness the plan calls out. It takes `topk_ids` only; it
  does **not** take `topk_weights`. Phase 2's `select_top_experts_weighted` must
  add a `topk_weights` parameter and thread it from the caller at
  [kt_ep_wrapper.py:3267](../../.venv/lib/python3.12/site-packages/sglang/srt/layers/moe/kt_ep_wrapper.py#L3267)
  (`dispatch_output.topk_output.topk_ids` → also grab `.topk_weights`).
- `_update_gpu_experts_from_batch` @ line 3247, `update_gpu_expert_mappings` @
  2432, `update_kt_wrapper_masks` @ 2469 — all present, in-place `.copy_()` on
  the `_cuda` tensors (lines 3325–3329) already respects CUDA-graph buffer reuse.
- Server flag `--kt-enable-dynamic-expert-update` — present
  ([server_args.py:4765](../../.venv/lib/python3.12/site-packages/sglang/srt/server_args.py#L4765),
  default `False`). Gated call at
  [kt_ep_wrapper.py:3044](../../.venv/lib/python3.12/site-packages/sglang/srt/layers/moe/kt_ep_wrapper.py#L3044).
- Traces `runs/exp3_*.pt` — present (5 tasks + exp1 + exp2). Motivation numbers
  trace to `runs/*.summary.json`, `runs/tail_to_top2_transition.json`,
  `runs/within_task_repeat_after_first_call.json`,
  `runs/usage_coverage_and_growth.json`.

### B. TRACE DATA CONTRACT (the simulator's actual input — was unspecified)

Each `.pt` is a Python `list` of per-forward records. Each record is a 3-tuple:

```text
(layer_idx: int, ids: Tensor[num_tokens, 8] int32, weights: Tensor[num_tokens, 8] float32)
```

Verified on `exp3_fix-git.pt`: 150 records, 75 distinct layers (ids 3..77), 46
tokens/row, 3525 total token-layer rows, `weights` present on all records.

Two non-obvious facts the simulator MUST honor:

1. **`ids` are NOT sorted by router weight.** Rank is defined by `weights`
   descending. Do not assume `ids[:2]` are the top-2 — sort first. Reuse the
   exact, already-tested logic in `analyze.py::iter_sorted_rows`
   ([analyze.py:39](analyze.py#L39)), which sorts each row by weight, gathers
   ids, and yields `(layer_idx, sorted_row)`. Import it; do not reimplement.
2. **Import constants from `_consts.py`** ([_consts.py](_consts.py)) — do not
   hardcode. `N_ROUTED_EXPERTS=256`, `NUM_EXPERTS_PER_TOKEN=8`, `TOP2_K=2`,
   `ROUTED_LAYER_IDS=range(3,78)` (75 layers), `PER_EXPERT_MIB≈18.0`. The plan's
   "arrays of length 256" and "18 MiB blocks" are these constants.

**CAVEAT — traces are PREFILL-ONLY** (`capture_manifest.json`: `max_tokens=1`).
There is no real decode trace. The plan's temporal unit ("expert-selection
events", median repeat gaps 67/104/199, decode rolling update) is derived by
treating token-rows as a sequence *within a prefill batch* — where all tokens are
actually processed in one parallel forward, so "event ordering" is a modeling
convention, not real decode time. This is fine for tuning relative policy, but
**half-life / grace numbers cannot be trusted as absolute decode-token counts
until a decode trace exists.** `capture.py` already supports this:
`./capture.py --with-decode --decode-tokens 128 --exp exp3`. Capture at least one
decode trace before Phase 2 sign-off.

### C. W4AFP8 Runtime Copy Path — fixed on `experiment/adaptive-expert-cache`

The production model is **GLM-5.2-W4AFP8**. The original
`_update_gpu_experts_from_batch` dispatch copied by quant type like this:

```text
is_fp8_quant          -> copy_experts_weights_fp8
is_fp8_channel_quant  -> copy_experts_weights_fp8_channel
is_bf16_quant         -> copy_experts_weights_bf16
else                  -> copy_experts_weights_int4   # Marlin GPTQ layout
```

That was unsafe for W4AFP8 because `SharedFullContext` sets
`ctx.is_w4afp8_quant = True`, but the old dispatch did not check it, so W4AFP8
fell through to Marlin int4 tensor names.

This branch adds:

```text
copy_experts_weights_w4afp8(src_layer, dst_layer, selected_experts)
  + an  elif ctx.is_w4afp8_quant:  branch in _update_gpu_experts_from_batch
```

The helper copies the resident-format W4AFP8 tensors after the full scratch
context has already run `_prepare_weight_w4afp8(...)` and
`W4AFp8MoEMethod.process_weights_after_loading(...)`:

```text
w13_weight
w13_weight_scale_inv
w2_weight
w2_weight_scale_inv
```

The cutlass bookkeeping tensors are handled by the existing full-context W4AFP8
prepare path. Dynamic promotion now copies from that processed scratch layer
into the runtime resident slots.

### D. CUDA-graph decode safety — stronger than "verify no pointer invalidation"

The mask/mapping updates are in-place `.copy_()` (safe). But the **weight copies
themselves** (`copy_experts_weights_*`) do per-expert `.copy_()` into
`dst_weight[dst_idx]`, i.e. into the resident-expert weight buffers that the
decode CUDA graph captured. Mutating captured weight buffers **during** a graph
replay races the kernels reading them. Therefore decode-time promotion must be
scheduled **between** graph replays (at the scheduler step boundary), never from
inside `apply()` mid-replay. The current call site runs eagerly with a
`torch.cuda.synchronize()` in the prefill-fallback path — that synchronize is why
it is safe today, and why Phase 5's copy-stream/double-buffer work is mandatory
(not optional) before decode updates go live. Add this to Phase 4 exit criteria.

### E. simulate_cache.py — concrete object list (so it's implementable directly)

```text
# reuse, don't reinvent:
from analyze import iter_sorted_rows           # weight-sorted (layer, row) stream
from _consts import N_ROUTED_EXPERTS, TOP2_K, NUM_EXPERTS_PER_TOKEN, ROUTED_LAYER_IDS, PER_EXPERT_MIB

class CachePolicy(Protocol):
    def on_token_row(self, layer:int, ranked_ids:list[int], ranked_wts:list[float]) -> None: ...
    def resident_set(self, layer:int) -> set[int]: ...      # size == k_gpu
    def stats(self) -> dict: ...                            # swaps, copy_MiB

# implement: LRUPolicy, FrequencyPolicy (= current runtime baseline),
#            WeightedRouterPolicy, WeightedRouterWithTailPolicy
# NOTE: iter_sorted_rows currently yields (layer, ids) and drops weights —
#       extend it (or add iter_sorted_rows_with_weights) to also yield the
#       gathered weights, since the weighted policies need normalized_router_weight.

def replay(trace_paths, policy_factory, k_gpu) -> Metrics:
    # per (layer,expert) key; k_gpu is PER-LAYER (matches runtime num_gpu_experts,
    # e.g. production 104). Metrics: top2_hit_rate, top8_hit_rate,
    # cpu_calls_per_token, swaps, copy_MiB (= swaps * PER_EXPERT_MIB), copy_time_est.
```

Sweep `k_gpu ∈ {32,64,96,104,128,160,192}` × policies × decay half-lives.
Baseline to beat = `FrequencyPolicy` (the current runtime selector) and static
uniform. Success bar (from plan): +10pp top-2 GPU-hit at k_gpu=104 vs uniform.

## Implementation Status — `experiment/adaptive-expert-cache`

Implemented in
`.venv/lib/python3.12/site-packages/sglang/srt/layers/moe/kt_ep_wrapper.py`:

- `ExpertCacheState`: per-layer CPU state keyed by `(layer_id, expert_id)` via
  one state object per `KTEPWrapperMethod`.
- `select_top_experts_weighted(...)`: sorts ids by router weight, updates
  persistent score arrays, applies top-2 and tail rank credits, adds resident
  hysteresis, and limits churn with `min_swap_count` / `max_swap_count`.
- Dynamic update dispatch now reads both `topk_ids` and `topk_weights`; the old
  frequency selector is still available with
  `SGLANG_KT_EXPERT_CACHE_POLICY=frequency`.
- No-change updates now return before copying expert weights, so a keep decision
  does not recopy all resident experts.
- W4AFP8 no longer falls through to the Marlin int4 copy helper. A dedicated
  `copy_experts_weights_w4afp8(...)` copies the already-interleaved cutlass W4A8
  tensors from the full scratch context into the runtime resident slots.

Implemented in this experiment directory:

- `simulate_cache.py`: offline trace replay with `static_uniform`, `lru`,
  `frequency`, `weighted_top2`, and `weighted_with_tail` policies.
- Outputs:
  - `runs/cache_simulation.json`
  - `runs/cache_simulation.md`
  - `runs/figs/cache_top2_hit_rate_vs_capacity.svg`

Validated first replay:

```bash
.venv/bin/python experiments/expert_footprint_top2_vs_top8/simulate_cache.py \
  --capacity 104 --update-interval 32
```

At 104 GPU experts/layer, update interval 32:

| policy | top2 hit | top8 hit | swaps | copy GiB |
|---|---:|---:|---:|---:|
| static_uniform | 41.15% | 41.07% | 0 | 0.00 |
| frequency | 72.25% | 66.72% | 12,817 | 225.30 |
| weighted_top2 | 81.14% | 58.98% | 8,578 | 150.79 |
| weighted_with_tail | 81.98% | 61.81% | 9,818 | 172.58 |

Remaining before production decode updates:

- Capture real decode traces (`capture.py --with-decode`) and retune half-life /
  grace windows against decode order, not just prefill-order replay.
- Schedule promotions only between decode CUDA graph replays, or add
  double-buffered expert slots plus a copy stream and epoch guard.
- Add live latency counters for top-2 GPU hit rate, CPU expert calls/token,
  copy volume, and update latency.

## Live Validation — 2026-07-03

Two live launches were tested.

Attempt 1 used the normal fast serving profile plus dynamic update:

```text
DYN_UPDATE=1
KT_GPU_PREFILL_THRESHOLD=512
GPU_EXPERTS=104
MEM_FRACTION=0.95
MTP=1
```

It booted and served a short request, but a long request OOM'd while building the
full W4AFP8 scratch context:

```text
torch.OutOfMemoryError: Tried to allocate 768.00 MiB ... ~38 MiB free
```

This proves production-fast settings leave too little VRAM headroom for
full-GPU prefill plus dynamic promotion.

Attempt 2 used a validation-safe profile:

```text
DYN_UPDATE=1
SGLANG_KT_EXPERT_CACHE_POLICY=weighted
SGLANG_KT_W4AFP8_VALIDATE=1
KT_GPU_PREFILL_THRESHOLD=512
GPU_EXPERTS=96
MEM_FRACTION=0.92
MAX_TOTAL_TOKENS=32768
MTP=0
./run_fast.sh
```

It passed:

- `/v1/models` ready on port `8000`.
- Short chat request returned HTTP 200.
- Long request: 2,492 prompt tokens, 96 completion tokens, 18.23 seconds, HTTP
  200.
- Adaptive updates ran on routed layers with bounded swaps.
- W4AFP8 validation showed exact sampled resident-copy matches:
  `max=0 mean=0 mismatch_frac=0` for all four copied tensors.
- Decode continued after promotion under CUDA graphs and returned HTTP 200.

Representative update:

```text
KT adaptive expert cache: layer=46 swaps=8 top2_hit 0.889->0.989 top8_hit 0.841->0.927
```

Operational rule from the live test:

```text
Dynamic W4AFP8 promotion requires explicit scratch headroom.
Do not enable it with the max-speed 104-expert + MTP + mem_fraction=0.95 profile.
```
