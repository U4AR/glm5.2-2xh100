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

## Hot core size & VRAM ladder simulation (2026-07-24)

Per-expert VRAM = 9.45 MiB/card (int4 W4AFP8, TP2-sharded). Full resident set
(96/layer × 78 layers) = 66.4 GiB/card of experts (+~14 GiB base). The
domain-stable **hot core is ~85–90 experts/layer ≈ ~60 GiB/card (~120 GiB
across both cards)**; the domain-adaptive tail is only ~2–7 GiB/card.

Simulation from real routing captures (441k genuine-top-2 events, 7 tasks,
`simulate_vram_ladder.py`; capture cov@96=0.884 vs live 0.897 — model
calibrated on same-boot uniform 28.2 @cov0.50 and adaptive 39 @cov0.897;
1/tps = t0 + t_miss·(1−cov), principled since only top-2 can miss):

| N/layer | coverage | VRAM/card | predicted tok/s |
|---|---|---|---|
| 16 | 0.42 | 25.5 GiB | 26.7 |
| 24 | 0.51 | 31.3 GiB | 28.5 |
| 32 | 0.59 | 37.0 GiB | 30.0 |
| 48 | 0.70 | 48.6 GiB | 32.6 |
| 64 | 0.78 | 60.1 GiB | 34.9 |
| 96 | 0.88 | 83.1 GiB | **38.5 (meas. 38–40)** |
| 128 | 0.95 | 106.1 GiB | 40.9 |
| 160 | 0.98 | 129.2 GiB | 42.4 |
| 256 | 1.00 | 198.3 GiB | 43.3 (all-GPU; top0+skip meas. ~52) |

Concentration: 50% of top-2 traffic needs mean 24 experts/layer; 80% → 69;
90% → 102; 95% → 129; 99% → 173. So the head is steep but the tail is long —
the LAST 10% of coverage costs as much VRAM as the first 80%.

Figures: `figs/fig1_usage_distribution.png` (Zipf per-layer usage),
`fig2_coverage_vs_N.png`, `fig3_tps_vs_vram.png`, `fig4_tps_vs_N.png`.

Takeaway: adaptive cache degrades GRACEFULLY on smaller-VRAM boxes — a 48
GiB-expert budget (e.g. 2×A6000-class, N=48) still predicts ~33 tok/s (85% of
the 2×H100 result), because the adaptive cache always holds the hottest-N.
Uniform placement at the SAME VRAM gets only ~28 → the cache's advantage GROWS
as VRAM shrinks.

### Live validation of the VRAM ladder (2026-07-24, same protocol per N)

Rebooted the real server at several GPU_EXPERTS values (uniform cold start,
12–14 convergence passes on one topic, then 3-pass measure + coherence test):

| N/layer | expert VRAM/card | conv. coverage | measured tok/s (cold → converged) | model pred |
|---|---|---|---|---|
| 24 | 17.3 GiB | 0.59 | 27 → **31–34** | 28.5 |
| 48 | 34.6 GiB | 0.79 | 25 → **34–39** | 32.6 |
| 64 | 46.1 GiB | 0.85 | 28 → **40–41** | 34.9 |
| 96 | 69.1 GiB | 0.90 | 28 → **38–40** | 38.5 |

All coherent (N=24 showed longer reasoning chains — mild quality cost of
heavier substitution). Measured points sit ON or ABOVE the model curve
because a converged single-topic cache achieves higher coverage than the
diverse-task capture distribution assumes: **N=64 (46 GiB of experts/card)
matches N=96** for focused workloads. The simulation curve is the safe
multi-domain lower bound. Practical floor: even N=24 (~17 GiB experts/card,
would fit a single 48 GiB card + CPU) holds >30 tok/s once converged.
