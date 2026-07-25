# 2×L40 — measured findings, root causes, and the real bottleneck

Everything here was measured on **this machine** on 2026‑07‑25 and is specific to
it unless marked otherwise. Host: 2× L40 46 GB (sm89, no FP8 cutlass, no
flashmla) + dual EPYC 7773X, **AVX2 only — no AVX‑512**, ~40 usable cores,
DDR4, 500 GB cgroup cap. Model: GLM‑5.2 754B W4AFP8, TP2, 78 layers / 75 MoE,
256 experts, top‑8, NEXTN/MTP depth‑3.

**Headline: 12.94 → ~20–21 tok/s** on a mixed 9‑prompt suite, accept 2.2 → 3.1.

---

## 1. The three root causes that were actually fixed

### 1.1 `kv_cache_dtype=fp8_e4m3` was silently destroying MTP acceptance

The largest single win, and the one that presented as a mystery ("MTP accept
length collapsed on this box").

fp8 KV is only validated on the **flashmla** attention path. Non‑Hopper cards
fall back to Triton, where an fp8 KV cache measurably degrades the draft head's
agreement with the target model. Switching to bf16 (`KV_CACHE_DTYPE=auto`):

| workload | accept fp8 → bf16 |
|---|---|
| technical prose | 2.2 → 2.63 |
| structured list | → 3.21 |
| code generation | → 3.27 |
| reasoning | → 3.36 |

Throughput 12.9 → 16.5 tok/s. **Nearly free:** MLA's `kv_lora_rank=512` means
bf16 KV at 8192 tokens costs under 1 GB (measured 0.01 GB). The H100 never hit
this because it runs flashmla. `hardware_profile.py` now forces `auto` on any
non‑flashmla backend.

### 1.2 `CPUINFER` was set from core count, not memory bandwidth

The CPU expert path is DRAM‑bandwidth‑bound, not FLOP‑bound. Measured
streaming‑read sweep on this host:

| threads | 8 | 16 | 28 | **56** | 112 |
|---|---:|---:|---:|---:|---:|
| GB/s | 163 | 248 | 296 | **356** | 271 |

Bandwidth peaks near 56 and **regresses** past it. The old default of 28 assumed
a 32‑vCPU pod. Going 28 → 56 cut decode step time 170 → 152 ms. Re‑measure per
host; more threads is not monotonically better.

### 1.3 `uniform` expert placement ignored routing entirely

Despite the name, `generate_uniform_masks` spreads the budget evenly *across
layers* and then fills each layer with experts **0..N−1 by index**. Expert 0 was
resident in all 75 layers, expert 200 in none. Top‑2 coverage was just `N/256`.

Routing is strongly concentrated **but the hot set differs per layer** — the best
single *global* set of 30 covers only 17.2%. Per‑layer hottest‑N is the only
placement that helps. New `hotcore` strategy; now the default.

**Measured gain is small: +2.4%** (19.64 → 20.11, both arms in one interleaved
sweep). An earlier +7.6% was cross‑boot drift — see §4.

---

## 2. The real bottleneck: a fixed per‑layer CPU dispatch toll

This is the important finding and it redirects all future work.

`top0` ceiling probe (`KEEP=0` + `/tmp/kt_skip_cpu`; all 8 slots from resident
experts, CPU path skipped entirely; **output incoherent by construction** — this
is a measurement, not a config):

| config | tok/s | accept | ms/step |
|---|---:|---:|---:|
| `safe2` hotcore N=30 | 20.11 | 3.10 | 154.2 |
| `safe2` uniform N=30 | 19.64 | 3.07 | 156.3 |
| **`top0` — no CPU path** | **42.77** | 3.47 | **81.1** |

Removing the CPU path is worth **2.13× tok/s / 1.90× step time**: it costs
**73 ms of a 154 ms step (47%), ≈0.97 ms per MoE layer.**

But throughput is nearly **flat in how much work the CPU does**:

| config | coverage | CPU trips/token | tok/s | ms/token |
|---|---:|---:|---:|---:|
| hotcore N=8 | 29.1% | 1.42 | 18.19 | 54.98 |
| hotcore N=16 | 42.1% | 1.16 | 20.33 | 49.19 |
| hotcore N=30 | 56.8% | 0.86 | 20.11 | 49.73 |
| uniform N=30 | 12.7% | 1.75 | 19.64 | 50.92 |

A 2× range in CPU trips/token moves ms/token only 49.7 → 50.9. (N=8 is
confounded — a poorer substitution pool changes the generated text.)

**Decomposition:** ~2.4 ms per trip/token scales with expert count (≈2 ms/step at
0.86 trips); the remaining **~71 ms/step — 97% — is fixed**, paid per layer
whether the CPU handles two experts or none.

**Bandwidth cross‑check:** 0.86 trips × 4 draft tokens × 19.46 MB = 67 MB per
layer‑step; at the measured 356 GB/s that is 0.19 ms/layer = **14 ms/step of real
DRAM traffic** against 73 ms measured. Expert streaming is already hidden behind
GPU compute. What is exposed is the per‑layer `submit_forward` + cross‑stream
`sync` round trip.

> **So: a 4.5× coverage improvement bought 2% because you were never paying for
> expert traffic. You are paying a ~1 ms toll, 75 times per step.**

Likely mechanism, not yet isolated: waking a 56‑thread, 2‑NUMA‑pool worker and
joining it **75 times per step** is ~5,600 barrier round trips/step. At 10–20 µs
each that is 56–112 ms — the right order of magnitude for the 71 ms observed.

---

## 3. Feasibility: the all‑resident CPU‑skip gate, and caching

If the toll is fixed and per‑layer, the obvious fix is to **not make the call**
when every genuine top‑2 expert of every draft token in that layer is already
resident. That is a **joint** probability over `2 × B = 8` slots (B = 4 draft
tokens), not a coverage number.

Measured on real per‑token routing captures (5 varied agent tasks),
**leave‑one‑task‑out** so the ranking never sees the task it is scored on:

| N/layer | coverage | P(token both resident) | **P(whole layer‑step clean)** | ms saved | tok/s |
|---:|---:|---:|---:|---:|---:|
| 30 (this box) | 41.0% | 18.2% | **1.3%** | 1.0 | 20.23 (+0.6%) |
| 48 | — | — | 3.9% | 2.7 | 20.47 (+1.8%) |
| 104 (H100) | 78.3% | 62.6% | **25.4%** | 18.0 | 22.76 (+13.2%) |

**Verdict for this machine: not viable.** At 30 experts/layer the gate fires
1.3% of the time and is worth +0.6%. It is **capacity‑gated, not policy‑gated** —
the skip rate is roughly `p^8`, so it only switches on near ~90% coverage, which
needs ~104 experts/layer ≈ 74 GB/card. An L40 has ~22 GB spare after non‑expert
weights, the draft head and KV.

Token‑level correlation does help — the real joint rate is ~3.7× the independent
prediction (adjacent draft tokens route alike) — but 3.7 × 1.2% is still 4.5%.

### A relaxed variant is the only version with any value here

Allow up to `k` of the 8 slots to miss and **drop just those** for that
layer‑step (the token keeps its other expert). This is *not* `sub2`: nothing ever
replaces a resident genuine pick, and the drop happens only in the tail where
almost everything was already resident.

| N | k | skip rate | tok/s | gain | genuine top‑2 slots dropped |
|---:|---:|---:|---:|---:|---:|
| 30 | 0 | 1.3% | 20.23 | +0.6% | 0.00% |
| 30 | 1 | 5.0% | 20.57 | +2.3% | 0.45% |
| 30 | 2 | 13.1% | 21.39 | +6.4% | 2.49% |
| 104 | 0 | 25.4% | 22.76 | +13.2% | 0.00% |
| 104 | 1 | 50.5% | 26.19 | +30.3% | 3.14% |

On this box `k=2` buys +6.4% for dropping 2.5% of genuine top‑2 slots — a real
but unattractive trade, and it needs a quality benchmark before anyone ships it.
On a 104‑expert host `k=0` is **lossless and worth +13%**, which is the version
worth building.

### Caching

The decode‑time adaptive cache is **net‑negative on mixed traffic here**:
18.2 / 18.2 / 19.0 tok/s over three passes vs **19.6** without it. Live
`top2_cov` plateaus ~0.51 while 122 layer swaps × 395 ms burn 48 s (~16% of wall
time). Its earlier "+13%" was measured against a slower 170 ms step *and* on
`decbench`'s single repeated prompt — the best case for a cache.

Given §2, this is now over‑determined: **caching only changes coverage, and
coverage is worth ~2.4 ms per trip/token.** Even a perfect cache is bounded by
that. The one scenario where caching becomes valuable again is **in combination
with the skip gate on a high‑capacity host**, because there the payoff is
super‑linear (`p^8`): pushing coverage 78% → 90% moves the gate from 25% to ~50%.
On a 30‑expert box there is no coverage worth buying.

---

## 4. Measurement methodology — read before benchmarking this box

* **Never A/B across separate boots.** Repeated *identical* `hotcore` boots span
  **20.11–21.26 tok/s (5.7%)**, larger than most effects being measured. Both
  arms must be booted back‑to‑back inside one sweep script. The +7.6% originally
  claimed for `hotcore` was almost entirely this artifact.
* **Wait for VRAM, not for PIDs.** `nvidia-smi --query-compute-apps` clears
  *before* the memory is released; the next launch then OOMs. Poll
  `memory.used < 2000`.
* **`[kt-time]` cannot attribute steady‑state decode.** The hooks live in the
  Python `apply()` that CUDA‑graph replay bypasses, so only capture/prefill
  samples appear. Use config‑ablation deltas (`top0` vs `safe2`) instead.
* **`decbench.py` understates this box by ~30%.** Its technical‑essay prompt is
  the worst case for MTP acceptance (2.63 vs 3.58 on structured output). Always
  cross‑check with a code or JSON prompt.
* **The hot‑core ranking was built from these same captures.** In‑sample coverage
  at N=30 is 56.8%; honest **leave‑one‑task‑out is 41.0%**. Quote the latter.
* `/tmp/kt_topk_mode` is read at **module import**, so mode changes need a
  restart.

---

## 5. Ruled out — do not re‑derive

* **Not swapping.** cgroup v1 sits at 99.94% of a 500 GB cap, but the weights are
  anonymous (Pss_Anon 392 GB vs Pss_File 185 MB) and swap is off, so they are
  unevictable. Over 45 s of decode: **0 major faults, 0 `pgsteal_direct`, 5 MB of
  disk I/O**. The large lifetime `pgmajfault` counters are all from model load.
* **Not AVX‑512, and not the older GPU's kernels.** Real GPU compute is ~34 ms of
  a 154 ms step; the CPU path is bandwidth‑bound, not FLOP‑bound, and the AVX2
  packed kernel matches the reference at cos 0.99992.
* **Not the draft/target substitution asymmetry.** The NextN draft holds all 256
  experts and is not kt‑managed, so `_kt_topk_experiment` hits its
  `mask is None → return` guard and substitution never applies to it, while
  target layers do get it. Real, but worth ~0.2 accept (2.2 → 2.4 with
  substitution fully off) at 2.2× the cost.
* **`--triton-attention-reduce-in-fp32` is a no‑op here.** It only gates
  `double_sparsity_attention.py`, which this config never touches.
* **Raising `GPU_EXPERTS` is exhausted.** `32 @ mem_fraction 0.91` OOMs during
  CUDA‑graph capture (fills 44.29 of 44.31 GiB). `30 @ 0.88` is the validated
  maximum, ~7.7 GB/card free.
* **`sub2` is incoherent on this box** at 24–30/256 residency. It is coherent at
  96–104 on 2×H100. `hardware_profile.py` now picks `safe2` automatically below
  96 experts/layer. Do not re‑attempt `sub2` here.

---

## 6. What to try next, in order

1. **Attack the per‑layer dispatch toll** (~71 ms/step, 97% of the CPU path's
   cost). Profile the `submit_forward` + cross‑stream `sync` round trip and the
   threadpool wake/join. This is the only lever on this box with >10% in it.
2. **Re‑sweep `CPUINFER` for decode specifically.** 56 was chosen to maximise
   *streaming bandwidth*, but if the toll is threadpool barrier latency then
   fewer threads may decode faster despite lower bandwidth. Cheap to test, and
   the two objectives are in direct conflict.
3. **Coalesce or pipeline the CPU submit across layers** so the toll is paid
   fewer than 75 times per step.
4. Only on a ≥96‑expert/layer host: build the **lossless `k=0` skip gate**
   (+13%), and revisit caching to feed it.

Not worth pursuing here: better placement masks, more coverage, the adaptive
cache, `sub2`.
