# Root Cause: MTP Benchmark and Adaptive Cache

Date: 2026-07-03

## Corrected Benchmark Result

The earlier A/B was invalid as a final speed comparison because it launched both
runs with `MTP=0`. The corrected A/B used:

```text
MTP=1
GPU_EXPERTS=96
MEM_FRACTION=0.92
MAX_TOTAL_TOKENS=32768
KT_GPU_PREFILL_THRESHOLD=512
max_tokens=128
```

Results:

```text
baseline DYN_UPDATE=0: 29.311s total, 21.835 completion tok/s
adaptive DYN_UPDATE=1: 29.329s total, 21.821 completion tok/s
speedup: 0.999x
```

The short-request decode logs were healthy with MTP:

```text
baseline steady decode: 34.553 tok/s mean, 34.690 tok/s median
adaptive steady decode: 34.845 tok/s mean, 34.965 tok/s median
```

So the low earlier numbers were caused by the wrong launch profile plus using
end-to-end request time as "completion tok/s". That includes prefill and
first-token latency, not only decode.

## Why The Adaptive Cache Did Not Improve Speed

### 1. The updater learns from already-substituted top-k ids

The top-2 substitution path mutates `topk_output` in place in
`deepseek_v2.py` before dispatching to the KT expert wrapper:

```text
topk_output = self.topk(...)
_kt_topk_experiment(topk_output, router_logits, ...)
final_hidden_states = self.experts(hidden_states, topk_output)
```

The adaptive cache updater then reads:

```text
topk_output = dispatch_output.topk_output
```

inside `_update_gpu_experts_from_batch`.

That means the cache is not learning from the model's original top-8 router
choices. It is learning from the post-substitution ids: true top-2 plus resident
GPU fill experts. This creates a feedback loop where the cache partly reinforces
the current resident set and the reported top8 hit rate is inflated by substituted
resident experts.

Fix direction: preserve the original router top-k ids/weights before
`_kt_topk_experiment`, pass them through `dispatch_output` or side metadata, and
use those original values for cache scoring.

### 2. W4AFP8 prefill still streams all 256 experts per layer

The W4AFP8 full-GPU prefill loader calls the FP8 loader with:

```text
gpu_experts_mask=None
```

so it streams all experts for the scratch full layer, then overlays resident GPU
experts afterward. This makes the long-prompt prefill cost almost identical
before and after adaptive caching:

```text
baseline prepare_weight sum: 10.076s
adaptive prepare_weight sum: 10.073s
adaptive update overhead:   0.532s
```

The cache update occurs after the layer's full scratch weights have already been
loaded and computed, so it cannot reduce that request's heavy prefill cost.

Fix direction: make W4AFP8 prefill resident-aware, or decouple cache promotion
from full-layer scratch prefill so promotions do not require streaming all 256
experts first.

### 3. The current benchmark is one-pass and cross-task

The first long Terminal-Bench task updates the cache. The following tasks are
different prompts, so some resident changes help and some hurt:

```text
largest-eigenval: +8.6%
git-multibranch: -3.7%
overall:          -0.1%
```

The adaptive policy needs either warm repeated prompts from the same distribution
or a global multi-task policy trained from original router traces. One-shot
adaptation from the previous task is not enough.

### 4. MTP amortizes the exact bottleneck the cache tries to improve

With MTP on, accept length was already about 3.2 tokens per target verification.
The cache improved measured top2 residency:

```text
top2 hit: 0.7689 -> 0.8628
```

but this did not materially move decode speed because the remaining cost is
dominated by fixed KT dispatch/sync and speculative verification overhead, not
only by the number of CPU-routed expert slots.

## Next Implementation Fixes

1. Preserve original router top-k before substitution and use it for adaptive
   scoring.
2. Report two hit rates separately:
   - original-router hit rate
   - post-substitution compute hit rate
3. Move cache updates out of the full scratch prefill critical path, or make
   W4AFP8 scratch loading resident-aware.
4. Benchmark warm repeated tasks as well as cross-task one-pass suites.
5. Measure decode-only TPS separately from end-to-end request TPS.
