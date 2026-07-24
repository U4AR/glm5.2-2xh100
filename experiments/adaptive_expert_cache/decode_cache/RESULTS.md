# Decode-time adaptive expert cache — results (2026-07-24)

On-the-fly GPU expert promotion/eviction driven by the DECODE stream, under
safe2 routing (genuine top-2 always kept) + MTP depth-3 + CUDA graphs.
Server: `boot_adaptive_mtp.sh` (uniform cold start, GPU_EXPERTS=96,
MEM_FRACTION=0.85, KT_ADAPTIVE_DECODE=1).

## Speed ladder (canonical bench/decode_bench.sh, coherent)

| config | tok/s |
|---|---|
| safe2 + uniform placement, frozen | 28.2 |
| **safe2 + adaptive cache, converged** | **38–40** (warm 40.3 / measured 38.1) |
| safe2 + static oracle mask (104 experts, needs offline profiling run) | 41.2 |
| top0 fair-equivalent (per-step, but output degenerate) | ~52 |

Converges from uniform in ~2–3 min of traffic (3 sweeps, MAX_SWAP=24/visit);
mean converged genuine-top2 coverage 0.897. Coherence 4/4 before and after.

## Phase test (same prompt ×2 → similar domain → different domain)

| phase | median tok/s | resident-set overlap vs prev (random = 0.375) |
|---|---|---|
| A1 LLM essay, cold | 28.3 → 34.7 within phase | — |
| A2 same prompt | 35.7 | 0.78 |
| B GPU essay (similar) | 34.0 | 0.89 |
| C immunology (different) | 38.1 | 0.86 vs A2, 0.96 vs B |

Hot expert core is largely domain-independent (Zipf head); only a ~10–15%
tail adapts per domain — so cross-domain switches don't pay a full cold start.

## The three fixes that made it work (previous attempts failed)

1. **Stage ALL 256 experts in the kt CPU store at load** (pass all-False mask
   to KTMoEWrapper, restore the real mask post-load). The shipped
   `--kt-enable-dynamic-expert-update` evicts experts that have NO CPU
   weights (load skips GPU-resident ones) -> silent garbage. Cost: +150 GB
   host RSS (~380 GB total).
2. **Learn from decode, in-graph**: genuine-top2 counters accumulated by a
   capture-safe `index_add_` in `deepseek_v2._kt_topk_experiment`; buffer
   pre-allocated outside capture. The old updater only fired on GPU bulk
   prefill (+1–2%).
3. **Benefit-gated swaps**: ratio hysteresis alone never quiesces (tail counts
   5-vs-2 cross any margin every sweep; stuck at 24 swaps/tick ≈ +10 ms/step
   churn). Adding an absolute gain gate (KT_ADAPTIVE_MIN_GAIN=0.001 of layer
   mass) drops steady-state churn to 1–3 swaps/tick.

Mechanics: tick every KT_ADAPTIVE_PERIOD=32 verify steps on the forward
thread of every TP rank (lockstep; rank0 selects, TP-broadcasts
unconditionally), 2 layers round-robin per tick, full-layer restage via the
SharedFullContext prefill pipeline (~140–200 ms/layer), residency flipped via
the in-place CUDA-graph-stable buffers.
