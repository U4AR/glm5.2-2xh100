# Per-request intelligence-tier harnesses

Scripts behind [BLOG_INTELLIGENCE_TIER.md](../../BLOG_INTELLIGENCE_TIER.md) — the
`GLM5.2-topN` per-request expert-tier knob.

| script | what it does |
|---|---|
| `test_keepk_kernel.py` | Offline CPU unit test of the per-token routing kernel: `K==8` is the exact baseline identity, fixed-K matches the legacy path, mixed per-token K rows are each correct. No server needed: `python bench/intelligence_tier/test_keepk_kernel.py` |
| `validate_tiers.py` | Live checks against `:8000`: per-tier coherence + speed, `bare==-top8` greedy parity, and a mixed-tier concurrent batch. `python bench/intelligence_tier/validate_tiers.py` |
| `batch_bench.py` | Batch-scaling throughput: fires B concurrent decode requests, reports aggregate + per-stream tok/s. `python bench/intelligence_tier/batch_bench.py GLM5.2-top2 256 1,2,4,8` |

Batched numbers in the blog/README were taken with the throughput config:
`GPU_EXPERTS=96 MAX_RUNNING=8 CUDA_GRAPH_MAX_BS=8 MAX_TOTAL_TOKENS=16384 ./run_fast.sh`
(add `MTP=0` for the no-MTP column).
