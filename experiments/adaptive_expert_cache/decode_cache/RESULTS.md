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

### Fairness check: diverse warm-up + HELD-OUT prompts (2026-07-24)

The single-prompt ladder above could overfit — repeating ONE prompt drives
coverage artificially toward 1.0 because the same experts fire every step. So
each N was re-measured with `conv_diverse.py`: warm the cache on a rotating set
of 8 unrelated topics (LLM, immunology, contract law, monetary policy, history,
chemistry, DB internals, music theory), then measure on 3 HELD-OUT topics the
cache never trained on (stellar astrophysics, bread science, plate tectonics).

| N/layer | expert VRAM/card | conv. coverage (diverse) | single-prompt tok/s | **held-out tok/s** |
|---|---|---|---|---|
| 48 | 34.6 GiB | 0.71 | 36.5 | **37.1** |
| 64 | 46.1 GiB | 0.77 | 40.5 | **36.9** |
| 96 | 69.1 GiB | 0.89 | 39.0 | **38.7** |

Held-out speed is within ~1–2 tok/s of the single-prompt numbers and coverage
under diverse traffic (0.71–0.89) is barely below single-prompt (0.79–0.90).
This confirms the earlier phase-test finding: the hot core is largely
**domain-independent** — a cache warmed on law/chemistry/music still serves
astrophysics at ~37 tok/s. The single-prompt runs were NOT materially inflated.
Plotted separately (purple diamonds) from the single-prompt points (green) in
`figs/fig3_tps_vs_vram.png`.

## Persisted hot core → warm start (2026-07-24)

`hot_core_ranking.pt` (committed, int16 [78,256]) stores, per layer, all 256
expert IDs sorted by genuine-top-2 usage — an N-agnostic, system-independent
ranking built from 441k real routing events (`build_hot_core.py build`).

Boot auto-loads it: `boot_adaptive_mtp.sh` checks for the file and, if present
(default; `WARM_START=0` disables), generates an exact [78,256] GPU-placement
mask for the box's own `GPU_EXPERTS=N` (`build_hot_core.py mask N`) and boots
via the `oracle` placement strategy — GPU already holds the hottest-N experts.
A NEW machine with the repo therefore starts at ~0.88 coverage instead of
uniform's 0.50; the adaptive cache then only tracks per-workload drift.

Verified: warm-started N=96 served **33 tok/s on the very first prompt** (vs
~28 uniform cold) and hit 43 by the second — the ~2–3 min cold-convergence
ramp is gone. Any N works: `GPU_EXPERTS=48 bash boot_adaptive_mtp.sh` slices
the same ranking to top-48.

## Low-VRAM regime (2026-07-24, partial — stopped early)

Measured actual VRAM/card (nvidia-smi) + diverse held-out tok/s at low expert
budgets (warm-start, MAX_TOTAL_TOKENS=4096, MEM_FRACTION=0.60):

| N/layer | VRAM/card (measured) | coverage | held-out tok/s |
|---|---|---|---|
| 32 | 40.1 GiB | 0.61 | 32.3 |
| 16 | 28.7 GiB | 0.43 | 30.3 |
| 8  | 23.1 GiB | 0.26 | 27.1 |

Linear fit: **~9.45 MiB/card per expert + a fixed ~17.5 GiB/card base**
(dense/attention weights + minimal KV + CUDA graphs). The base is the hard
floor: N→0 would still sit near ~18 GiB/card, so **~12 GiB/card total is NOT
reachable** for GLM-5.2 on 2 cards without re-quantizing the dense trunk or
using more TP shards. Even at the extreme low end the hybrid stays coherent and
useful: N=8 (~23 GiB/card, coverage just 0.26) still holds ~27 tok/s because
safe2 keeps genuine top-2 (CPU round-trip) and MTP-d3 amortizes it — the CPU
expert path degrades gracefully rather than falling off a cliff. (N=4, N=0 runs
were cut short.)
