# Expert Footprint: Top-2 vs Top-8

Date: 2026-07-03

This experiment measures how much GLM-5.2 routed-expert weight is touched by a
prompt or task, comparing:

- **top-2**: only the two highest-router-weight experts from each token's top-8.
- **top-8**: all eight routed experts selected by the model.

The useful memory unit is an **expert weight block**:

```text
(layer_id, expert_id)
```

Expert 17 in layer 3 and expert 17 in layer 40 are different weight blocks,
because they are different tensors and occupy separate memory. All totals below
treat every layer's expert separately.

## Model Constants

GLM-5.2-W4AFP8 has:

- 78 hidden layers.
- First 3 dense layers.
- 75 routed MoE layers, layer ids 3..77.
- 256 routed experts per MoE layer.
- 8 routed experts per token.
- One routed expert weight block is approximately **18.0 MiB** in W4AFP8/int4.

The full routed expert body is:

```text
75 layers * 256 experts/layer * 18.0 MiB = 345,600 MiB = 337.5 GiB
```

## What Was Captured

The server was run once in plain top-8 routing mode with `KT_DUMP_TOPK=1`.
For each request, the hook captured every routed MoE layer's `topk_ids` and
`topk_weights`. The top-2 view was derived offline by sorting each top-8 row by
router weight and taking the first two experts.

Captured prompts:

- `exp1_single`: short hand-written prompt.
- `exp2_llm-inference-batching-scheduler`: one Terminal-Bench task.
- `exp3_*`: five Terminal-Bench tasks:
  - `llm-inference-batching-scheduler`
  - `largest-eigenval`
  - `fix-git`
  - `compile-compcert`
  - `git-multibranch`

The repeated `exp3_llm-inference-batching-scheduler` request hit prefix cache,
so its trace was replaced with the uncached `exp2` trace for the same task before
suite-level analysis.

## Artifact Map

- Capture driver: `capture.py`
- Analysis driver: `analyze.py`
- SVG chart driver: `make_charts.py`
- Raw traces: `runs/*.pt`
- Per-trace summaries: `runs/*.summary.json`
- Per-task table: `runs/results.md`
- Usage coverage table: `runs/usage_coverage_and_growth.md`
- Cross-task overlap: `runs/cross_task_common_experts.json`
- Within-task repeat stats: `runs/within_task_repeat_after_first_call.json`
- Tail-to-top2 transition stats: `runs/tail_to_top2_transition.json`
- Figures: `runs/figs/*.svg`

Important figures:

- `runs/figs/summary_total_footprint.svg`
- `runs/figs/combined_usage_distribution.svg`
- `runs/figs/cumulative_unique_experts_by_task.svg`

## Per-Task Footprint

This table reports unique expert weight blocks touched by each trace.

| trace | top2 total MiB | top8 total MiB | top8/top2 | top2 p90 per-layer MiB | top8 p90 per-layer MiB |
|---|---:|---:|---:|---:|---:|
| exp1_single | 46,962 | 144,288 | 3.07x | 774 | 2,372 |
| exp2_llm-inference-batching-scheduler | 248,004 | 338,292 | 1.36x | 3,708 | 4,608 |
| exp3_compile-compcert | 78,102 | 216,630 | 2.77x | 1,386 | 3,406 |
| exp3_fix-git | 57,420 | 171,774 | 2.99x | 972 | 2,754 |
| exp3_git-multibranch | 134,946 | 293,400 | 2.17x | 2,178 | 4,306 |
| exp3_largest-eigenval | 132,858 | 282,402 | 2.13x | 2,117 | 4,291 |
| exp3_llm-inference-batching-scheduler | 248,004 | 338,292 | 1.36x | 3,708 | 4,608 |

The `p90 per-layer` columns are **not** usage-p90. They answer:

```text
Across the 75 routed layers, how much expert weight does a heavy layer touch?
```

For example, top-8 p90 of 4,608 MiB means the 90th-percentile layer touches all
256 experts:

```text
256 * 18 MiB = 4,608 MiB
```

## Whole 5-Task Suite: Unique Weight Touched

Across all five Terminal-Bench tasks:

| view | unique blocks touched | weight MiB | weight GiB | share of full routed experts |
|---|---:|---:|---:|---:|
| top2 | 15,191 | 273,438 | 267.0 | 79.1% |
| top8 | 19,042 | 342,756 | 334.7 | 99.2% |

Top-8 nearly saturates the routed expert model across only five varied tasks.
Top-2 is much more selective, but still touches most of the routed expert body.

## Usage Coverage: Correct Usage-P90

The useful cache question is:

```text
If expert blocks are sorted by how often they are selected, how much weight is
needed to cover X% of all routing selections?
```

Top-2:

| coverage | blocks needed | weight GiB | share of full routed experts |
|---:|---:|---:|---:|
| 50% | 1,735 | 30.5 | 9.0% |
| 75% | 4,372 | 76.9 | 22.8% |
| 90% | 7,631 | 134.1 | 39.7% |
| 95% | 9,697 | 170.5 | 50.5% |
| 99% | 12,971 | 228.0 | 67.6% |
| 100% | 15,191 | 267.0 | 79.1% |

Top-8:

| coverage | blocks needed | weight GiB | share of full routed experts |
|---:|---:|---:|---:|
| 50% | 4,013 | 70.5 | 20.9% |
| 75% | 8,409 | 147.8 | 43.8% |
| 90% | 12,527 | 220.2 | 65.2% |
| 95% | 14,582 | 256.3 | 75.9% |
| 99% | 17,213 | 302.6 | 89.7% |
| 100% | 19,042 | 334.7 | 99.2% |

So the corrected usage-p90 is:

- **top2**: 134.1 GiB covers 90% of top-2 selections.
- **top8**: 220.2 GiB covers 90% of top-8 selections.

## Unique Expert Growth Across Tasks

The first long task touches most of the suite's expert set. Later tasks mostly
reuse already-seen expert blocks.

Top-2 cumulative:

| task | task unique blocks | new blocks added | cumulative GiB |
|---|---:|---:|---:|
| llm-inference-batching-scheduler | 13,778 | 13,778 | 242.2 |
| largest-eigenval | 7,381 | 543 | 251.7 |
| fix-git | 3,190 | 236 | 255.9 |
| compile-compcert | 4,339 | 181 | 259.1 |
| git-multibranch | 7,497 | 453 | 267.0 |

Top-8 cumulative:

| task | task unique blocks | new blocks added | cumulative GiB |
|---|---:|---:|---:|
| llm-inference-batching-scheduler | 18,794 | 18,794 | 330.4 |
| largest-eigenval | 15,689 | 123 | 332.5 |
| fix-git | 9,543 | 47 | 333.4 |
| compile-compcert | 12,035 | 17 | 333.7 |
| git-multibranch | 16,300 | 61 | 334.7 |

## Cross-Task Commonality

Top-2:

| task count | blocks | percent of touched | weight GiB |
|---:|---:|---:|---:|
| exactly 1 task | 5,027 | 33.1% | 88.4 |
| exactly 2 tasks | 4,140 | 27.3% | 72.8 |
| exactly 3 tasks | 2,812 | 18.5% | 49.4 |
| exactly 4 tasks | 1,618 | 10.7% | 28.4 |
| all 5 tasks | 1,594 | 10.5% | 28.0 |

Top-8:

| task count | blocks | percent of touched | weight GiB |
|---:|---:|---:|---:|
| exactly 1 task | 867 | 4.6% | 15.2 |
| exactly 2 tasks | 2,286 | 12.0% | 40.2 |
| exactly 3 tasks | 3,786 | 19.9% | 66.6 |
| exactly 4 tasks | 4,951 | 26.0% | 87.0 |
| all 5 tasks | 7,152 | 37.6% | 125.7 |

Top-8 has a huge shared core: 95.5% of touched top-8 blocks are used by at least
two tasks. Top-2 is more task-selective: 66.9% are used by at least two tasks.

## Within-Task Repeat

Question: if a rare expert block is called once in a task, is it likely to be
called again in that same task?

Task-specific blocks:

| view | cases | repeated in same task | median calls | p90 calls |
|---|---:|---:|---:|---:|
| top2 | 5,027 | 63.5% | 2 | 12 |
| top8 | 867 | 82.6% | 6 | 31 |

Rare/shared at most two tasks:

| view | cases | repeated in same task | median calls | p90 calls |
|---|---:|---:|---:|---:|
| top2 | 13,307 | 64.8% | 2 | 14 |
| top8 | 5,439 | 80.1% | 5 | 34 |

Median gap until second call:

| population | median gap |
|---|---:|
| top2 task-specific | 104 expert-selection events |
| top2 shared by at most 2 tasks | 67 expert-selection events |
| top8 task-specific | 446 expert-selection events |
| top8 shared by at most 2 tasks | 199 expert-selection events |

This supports request-local caching: even rare experts are often reused once they
appear.

## Tail-to-Top2 Transition

Question: if an expert appears in top-8 but not top-2, does that make it more
likely to enter top-2 later?

Across all tasks:

| condition | probability |
|---|---:|
| any expert block appears in top-2 during a task | 37.7% |
| appeared in top-8 tail first, later appears in top-2 | 38.5% |

The overall lift is small, but rank matters:

| first tail rank | later appears in top-2 |
|---:|---:|
| 3 | 54.3% |
| 4 | 46.8% |
| 5 | 41.0% |
| 6 | 35.8% |
| 7 | 31.4% |
| 8 | 27.1% |

Rank 3 and 4 are useful predictive signals. Rank 7 and 8 are weak signals and
should receive much smaller cache-score credit.

## Takeaways

1. Top-8 quickly touches almost the whole routed expert model.
2. Top-2 is much more concentrated but still broad over long diverse prompts.
3. Usage is heavy-tailed: top-2 needs 134.1 GiB for 90% of selections, not the
   full 267.0 GiB touched by the suite.
4. Rare experts often repeat within the same task after first use.
5. Top-8 tail rank is predictive only near the boundary: rank 3 and 4 deserve
   meaningful warm-cache credit; rank 7 and 8 do not.
6. Cache decisions should be per `(layer, expert_id)`, not expert id alone.

## Adaptive Cache Implementation

Implementation branch: `experiment/adaptive-expert-cache`.

Runtime changes are in
`.venv/lib/python3.12/site-packages/sglang/srt/layers/moe/kt_ep_wrapper.py`.
When `--kt-enable-dynamic-expert-update` is enabled, the dynamic selector now
defaults to a persistent weighted router policy instead of raw top-k frequency.
It sorts router ids by router weight first, gives top-2 strong credit, gives
ranks 3-8 weaker tail credit, applies resident hysteresis, and skips the weight
copy entirely when the selected resident set does not change.

The W4AFP8 runtime path now has an explicit copy helper instead of falling
through to the Marlin int4 helper. The helper copies the already-processed
cutlass W4A8 resident-format tensors:

```text
w13_weight
w13_weight_scale_inv
w2_weight
w2_weight_scale_inv
```

Tunable environment variables:

```text
SGLANG_KT_EXPERT_CACHE_POLICY=weighted|frequency
SGLANG_KT_EXPERT_CACHE_HALF_LIFE=96
SGLANG_KT_EXPERT_CACHE_RESIDENT_BONUS=0.25
SGLANG_KT_EXPERT_CACHE_PROMOTION_COST=0.10
SGLANG_KT_EXPERT_CACHE_MIN_SWAP=2
SGLANG_KT_EXPERT_CACHE_MAX_SWAP=8
SGLANG_KT_EXPERT_CACHE_TOP2_GRACE=208
SGLANG_KT_EXPERT_CACHE_STRONG_TAIL_GRACE=398
SGLANG_KT_EXPERT_CACHE_WEAK_TAIL_GRACE=104
```

Offline simulator:

```bash
.venv/bin/python experiments/expert_footprint_top2_vs_top8/simulate_cache.py --capacity 104 --update-interval 32
```

First replay result at 104 GPU experts/layer and update interval 32:

| policy | top2 hit | top8 hit | swaps | copy GiB |
|---|---:|---:|---:|---:|
| static_uniform | 41.15% | 41.07% | 0 | 0.00 |
| frequency | 72.25% | 66.72% | 12,817 | 225.30 |
| weighted_top2 | 81.14% | 58.98% | 8,578 | 150.79 |
| weighted_with_tail | 81.98% | 61.81% | 9,818 | 172.58 |

## Live Validation

Validated on the live OpenAI-compatible server on 2026-07-03.

The first attempted launch used the normal fast profile:

```text
DYN_UPDATE=1 KT_GPU_PREFILL_THRESHOLD=512 GPU_EXPERTS=104 MEM_FRACTION=0.95 MTP=1
```

That booted and handled a short request, but the first long request crashed with
CUDA OOM while allocating the W4AFP8 full scratch context. The failure happened
before expert update; only about 2 GiB was free after CUDA graph and MTP setup.

The passing validation profile was:

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

Results:

- Server booted on `:8000` with `--kt-enable-dynamic-expert-update`.
- Short chat request returned HTTP 200.
- Long chat request with 2,492 prompt tokens and 96 output tokens returned HTTP
  200 in 18.23 seconds.
- The long request triggered layerwise full-GPU prefill and adaptive expert
  updates on routed layers.
- W4AFP8 validation reported exact matches for sampled resident experts:
  `max=0 mean=0 mismatch_frac=0` for `w13_weight`, `w13_weight_scale_inv`,
  `w2_weight`, and `w2_weight_scale_inv`.
- Example live update line:

```text
KT adaptive expert cache: layer=46 swaps=8 top2_hit 0.889->0.989 top8_hit 0.841->0.927
```

Takeaway: the implementation works live when enough VRAM is reserved for the
full W4AFP8 scratch context. The default fast profile with 104 GPU experts, MTP,
and `MEM_FRACTION=0.95` does not leave enough headroom for dynamic updates.

## Live Speed A/B

### Corrected MTP-On Run

Benchmark profile for both sides:

```text
KT_GPU_PREFILL_THRESHOLD=512
GPU_EXPERTS=96
MEM_FRACTION=0.92
MAX_TOTAL_TOKENS=32768
MTP=1
max_tokens=128
```

Only `DYN_UPDATE` changed.

| task | prompt tok | baseline s | adaptive s | delta s | speedup |
|---|---:|---:|---:|---:|---:|
| llm-inference-batching-scheduler | 1,169 | 14.048 | 14.203 | +0.155 | 0.989x |
| largest-eigenval | 179 | 3.955 | 3.641 | -0.314 | 1.086x |
| fix-git | 46 | 3.763 | 3.746 | -0.017 | 1.005x |
| compile-compcert | 84 | 3.615 | 3.658 | +0.043 | 0.988x |
| git-multibranch | 253 | 3.930 | 4.081 | +0.151 | 0.963x |

Suite total:

| mode | total s | completion tok/s | total tok/s |
|---|---:|---:|---:|
| baseline (`DYN_UPDATE=0`) | 29.311 | 21.835 | 80.891 |
| adaptive (`DYN_UPDATE=1`) | 29.329 | 21.821 | 80.841 |

Net result: **0.999x** total speedup. Server-side steady decode throughput was
healthy with MTP on: baseline 34.553 tok/s mean, adaptive 34.845 tok/s mean.

The adaptive run performed 75 layer updates and 452 total expert swaps, improving
the logged top2 resident hit rate from 76.9% to 86.3%. This did not translate
into end-to-end speed because:

- the updater currently learns from post-substitution top-k ids instead of the
  original router top-8;
- W4AFP8 full-GPU prefill still streams all 256 experts per layer before the
  cache update, so prepare-weight time was unchanged;
- the one-pass benchmark changes tasks after the first cache update;
- MTP already amortizes much of the top-2 CPU-expert bottleneck.

Detailed root cause: `ROOT_CAUSE_MTP_ADAPTIVE.md`.

Artifacts:

- `runs/live_bench/baseline_mtp_dyn0.json`
- `runs/live_bench/adaptive_mtp_dyn1.json`
- `runs/live_bench/compare_mtp_dyn0_vs_dyn1.md`

### Invalid No-MTP Run

The earlier live A/B below is kept for history only. It is **not** the correct
fast-path comparison because it used `MTP=0`.

Benchmark profile for both sides:

```text
KT_GPU_PREFILL_THRESHOLD=512
GPU_EXPERTS=96
MEM_FRACTION=0.92
MAX_TOTAL_TOKENS=32768
MTP=0
max_tokens=128
```

Only `DYN_UPDATE` changed.

| task | prompt tok | baseline s | adaptive s | delta s | speedup |
|---|---:|---:|---:|---:|---:|
| llm-inference-batching-scheduler | 1,169 | 17.410 | 17.733 | +0.323 | 0.982x |
| largest-eigenval | 179 | 7.705 | 7.394 | -0.311 | 1.042x |
| fix-git | 46 | 7.014 | 6.966 | -0.048 | 1.007x |
| compile-compcert | 84 | 7.179 | 7.008 | -0.171 | 1.024x |
| git-multibranch | 253 | 7.836 | 7.564 | -0.272 | 1.036x |

Suite total:

| mode | total s | completion tok/s | total tok/s |
|---|---:|---:|---:|
| baseline (`DYN_UPDATE=0`) | 47.144 | 13.575 | 50.293 |
| adaptive (`DYN_UPDATE=1`) | 46.665 | 13.715 | 50.809 |

Net result: **1.010x** total speedup on this one-pass five-prompt suite. The
first long prompt is slightly slower because it pays the promotion cost. The
following shorter prompts are consistently faster.

Adaptive update logs during the first long prompt:

| metric | value |
|---|---:|
| layer updates | 75 |
| total expert-slot swaps | 461 |
| mean top2 hit before -> after | 76.9% -> 86.6% |
| mean top8 hit before -> after | 84.5% -> 88.9% |

Artifacts:

- `runs/live_bench/baseline_dyn0.json`
- `runs/live_bench/adaptive_dyn1.json`
- `runs/live_bench/compare_dyn0_vs_dyn1.md`
- `runs/live_bench/compare_dyn0_vs_dyn1.json`

## Rerun Notes

Capture server used:

```bash
rm -f /tmp/kt_topk_mode /tmp/kt_skip_cpu
echo default > /tmp/kt_topk_tag

KT_DUMP_TOPK=1 \
KT_DUMP_TOPK_DIR=/data/models/RunGLM/experiments/expert_footprint_top2_vs_top8/runs \
KT_DUMP_TOPK_TAG=/tmp/kt_topk_tag \
MODE=off MTP=0 DISABLE_CUDA_GRAPH=1 KT_GPU_PREFILL_THRESHOLD=0 \
TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
./run_fast.sh > experiments/expert_footprint_top2_vs_top8/server.log 2>&1
```

Then:

```bash
.venv/bin/python experiments/expert_footprint_top2_vs_top8/capture.py --exp all
.venv/bin/python experiments/expert_footprint_top2_vs_top8/analyze.py
.venv/bin/python experiments/expert_footprint_top2_vs_top8/make_charts.py
```

The normal serving mode was restored afterward with default `run_fast.sh`
(`sub2`, MTP on, CUDA graphs on).
