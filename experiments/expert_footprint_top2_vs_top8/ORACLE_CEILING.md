# Oracle Expert-Placement Ceiling — measured 2026-07-03

Goal: the plan's adaptive cache showed ~0.999x (no speedup). Measure the **upper
bound** of what moving the right experts to GPU can ever achieve, by assuming
perfect (oracle) knowledge of which experts a request needs.

## Key idea — the ceiling is measurable without building the oracle

On the live server the per-request tier `GLM5.2-topN` sets how many TRUE experts
each token keeps; the remaining slots are substituted with the best
**GPU-resident** experts:

- `top8` — keep all 8 true; non-resident trues run on the **CPU** expert path.
  This is today's placement with correct output. **Baseline.**
- `top0` — keep 0 true; all 8 slots are GPU-resident experts → **zero CPU
  calls**. Compute profile = "8 experts, all resident."

A perfect oracle that keeps the *true* top-8 but makes them all GPU-resident has
the **identical compute cost as top0** (8 resident experts/token, no CPU) — it
differs only in *which* experts, not in cost. So **top0 decode speed = the oracle
placement speed ceiling.**

## Result — live, MTP depth-3, 96 GPU experts/layer, pure decode tok/s

Server: `--kt-num-gpu-experts 96 --kt-enable-dynamic-expert-update`, NEXTN
depth-3, mem-fraction 0.92. 5 Terminal-Bench prompts, 200 tokens each, TTFT
excluded. Harness: `oracle_ceiling_bench.py`, data `runs/oracle_ceiling.json`.

| tier | what computes | decode tok/s | vs top8 |
|---|---|---:|---:|
| top8 | 8 true, non-resident → CPU path (correct) | 19.50 | 1.00x |
| top4 | 4 true + 4 resident | 31.48 | 1.61x |
| top2 | 2 true + 6 resident | 42.96 | 2.20x |
| top0 | 8 resident, zero CPU (**ceiling**) | 72.67 | **3.73x** |

**The CPU expert path costs 3.73x at decode.** That is the hard ceiling on any
placement policy, oracle included. Removing CPU-routed experts is the single
biggest decode lever in this system.

### Calibration: decode cost is ~linear in CPU-experts-per-token

Per-token decode time (1000 / tok_s) vs estimated CPU experts touched:

| tier | est. CPU experts/token | ms/token |
|---|---:|---:|
| top0 | 0.0 | 13.8 |
| top2 | ~1.2 | 23.3 |
| top4 | ~2.5 | 31.8 |
| top8 | ~4.9 | 51.3 |

Slope ≈ 7–8 ms per CPU expert per token. Decode speed is governed almost
entirely by how many of a token's experts miss GPU.

## But the ceiling is NOT free — two hard limits

### 1. Capacity: the true working set does not fit at 96/256

top0 hits all-resident by *substituting* any resident experts — it never needs
the *right* ones. A real oracle must make the **true** experts resident. From the
footprint study (`README.md`), covering 90% of **top-8** selections needs ~167
experts/layer; we have **96**. A single heavy task touches ~200 distinct experts
per layer in top-2 alone. So a real oracle at 96/layer cannot reach 100% GPU hit;
it is bounded to whatever fraction of the true working set fits.

Coverage math + the slope above put a **realistic per-question static oracle at
~75–90% GPU hit → ~2.0–2.6x** decode with *correct* output (roughly top2 speed
but without the quality loss). The full 3.73x requires more VRAM/capacity or
accepting substitution (top2/adaptive-K).

### 2. The live adaptive cache captures ~none of it (warm-repeat is FLAT)

`oracle_warm_repeat.py` runs the same question 6x at top8 with a fresh prefill
each pass, so the running `--kt-enable-dynamic-expert-update` cache re-learns the
true top-8 every time. Decode does **not** climb:

```
top8 warm-repeat (largest-eigenval): 18.7, 19.3, 16.8, 18.5, 22.2, 19.0 tok/s  -> FLAT
top2 control:                        43.7, 38.7, 40.3, 46.9 tok/s              -> already near its level
```

Running the same question "previously" does not move top8 toward the ceiling.
This reproduces codex's 0.999x and pinpoints why: the adaptive updater (a) fires
on the prefill path, which streams all 256 experts anyway, and (b) at 96/layer
cannot fit the per-token top-8 working set, so learning cannot raise residency
enough to matter.

## Bottom line

- **YES, the ceiling is dramatic: 3.73x decode** (19.5 → 72.7 tok/s). The CPU
  expert path is unambiguously the decode bottleneck.
- **NO, a real oracle at today's 96-expert capacity cannot reach it** — the true
  per-token working set does not fit on GPU. Realistic correct-output oracle is
  ~2–2.6x, essentially top2 speed with top8 quality.
- To actually *bank* the ceiling you need one of: (a) more GPU capacity so the
  working set fits, (b) substitution (top2 / adaptive-K, already shipped, 2.2x
  today at a quality cost), or (c) a decode-time resident-aware promotion path —
  NOT the current prefill-time all-256-streaming updater.

## Static true-expert oracle — BUILT AND MEASURED LIVE

We did not stop at the model. We built the actual oracle: a new
`--kt-expert-placement-strategy oracle` that loads a precomputed boolean mask
(exactly 96 True per routed layer) chosen offline as each question's most-used
experts, and froze it (`--kt-enable-dynamic-expert-update` OFF). Then re-ran the
same questions at top8 (correct output). Masks + coverage: `build_oracle_masks.py`,
`runs/oracle_masks/`. Code: `oracle` branch in `kt_ep_wrapper.py` +
`server_args.py`. Launch: `PLACEMENT=oracle KT_ORACLE_MASK_PT=<mask.pt>
DYN_UPDATE=0 ... ./run_fast.sh`.

Coverage the oracle achieves (fraction of top-8 selections that are GPU-resident),
prefill-trace based:

| oracle | prefill coverage | uniform=37.5% |
|---|---:|---|
| global (hottest across all 5) | 69.0% | |
| per-question largest-eigenval | 80.3% | |
| per-question fix-git (best) | 91.6% | |

Measured live decode tok/s at top8 (correct output), vs uniform baseline and the
per-question top0 ceiling:

| run | oracle top8 | uniform top8 | **oracle speedup** | top0 ceiling |
|---|---:|---:|---:|---:|
| global (all 5 tasks) | 22.5 | 19.5 | **1.15x** | 72.7 (3.73x) |
| eigenval per-question | 23.0 | 18.9 | **1.22x** | 58 (2.5x) |
| fix-git per-question (best) | 22.3 | 16.5 | **1.35x** | 70.0 (3.14x) |

**The realistic oracle buys only 1.15–1.35x — not the 3.1–3.7x ceiling.**

### Why: there is a hard floor, and coverage barely moves it

Oracle top8 lands at **~22–23 tok/s in all three runs**, even as prefill coverage
climbs 69% → 91.6%. Two compounding reasons, both structural:

1. **Prefill != decode working set.** The mask is built from prefill traces, but
   decode selects partly different experts, so real *decode* GPU-hit is well below
   the 69–92% prefill coverage. (A decode trace would tighten this, but see #2.)
2. **The CPU expert path has a large FIXED per-token cost (submit + sync), and
   CPU/GPU compute is overlapped: per-layer time = max(cpu_time, gpu_time).**
   Moving experts CPU→GPU lowers `cpu_time` but raises `gpu_time` — it just
   rebalances the two poles toward their meeting point (~22–23 tok/s) and stops.
   As long as *any* expert misses GPU, the CPU path is invoked and you pay its
   fixed dispatch/sync. Only top0 removes the CPU pole entirely (zero CPU experts
   + `/tmp/kt_skip_cpu` armed) — which is why substitution reaches 3.1–3.7x but
   oracle placement cannot.

So the 3.73x ceiling is real but reachable **only by eliminating CPU compute**
(substitution = top2/adaptive-K, already shipped at 2.2x with a quality cost), or
by fitting *all* of a token's true experts on GPU (impossible at 96/256 for these
working sets). A better expert *placement* at fixed 96-capacity is worth ~1.2–1.35x
at most. This is the deep reason codex's adaptive cache measured 0.999x.

## "Fit the top-2 experts on GPU" — the whole-suite footprint and a per-question win

Question: how big is the set of ALL experts used at top-2 across the 5 questions,
and what if we make them all GPU-resident?

**Total top-2 union across the 5 questions = 15,191 expert blocks = 267.0 GiB**
(79% of the full 337.5 GiB routed body). Per routed layer that is ~203 of 256
experts on average (p90 224, max 229) — even at top-2, five varied tasks touch
almost every expert in every layer. 2xH100 = 180 GiB VRAM total, of which ~140
GiB is usable for experts after attention/KV/MTP/graphs (~106 experts/layer). So
**the 267 GiB suite top-2 union does NOT fit** — there is no compact hot core to
cache. That, fundamentally, is why placement/caching can't win on the suite.

Per SINGLE question it fits, though (top-2 union, max experts/layer):

| task | top-2 union GiB | max experts/layer | fits @96? |
|---|---:|---:|:--:|
| fix-git | 56.1 | 64 | YES |
| compile-compcert | 76.3 | 83 | YES |
| largest-eigenval | 129.7 | 126 | @128 |
| git-multibranch | 131.8 | 127 | @128 |
| llm-inference-batching-scheduler | 242.2 | 219 | no |

So for fix-git we placed its **complete top-2 set** on GPU (`fixgit_top2.pt`,
100% top-2 coverage at 96/layer) and ran at **top-2** (both true kept experts
guaranteed resident). Measured decode tok/s:

| fix-git config | decode tok/s | vs top8 |
|---|---:|---:|
| top8 baseline | 19.2 | 1.00x |
| generic top-2 (uniform placement) | ~36 | 1.9x |
| top-2, full top-2 set resident (no skip) | 40.3 | 2.10x |
| **top-2, resident + CPU-skip armed** | **51.1** | **2.66x** |
| top0 ceiling | 78.3 | 4.07x |

Two lessons:

1. **Placement alone barely helped** (36 → 40): making the kept-2 resident does
   NOT reach the ceiling, because the kt worker fires a per-layer CPU
   `submit_forward + sync` on every token regardless of whether any expert needs
   CPU. That fixed cost is the floor.
2. **Arming the CPU-skip** (`/tmp/kt_skip_cpu`, safe here because 100% of the
   routing is resident so nothing is dropped) removed that fixed cost: 40 → 51,
   and the output stayed coherent. So the user's idea works **per question**:
   fit the question's top-2 experts on GPU AND skip the empty CPU path → **2.66x
   at correct top-2 quality**. The remaining gap to 78 is MTP/substitution
   acceptance mechanics, not expert placement.

Caveats: (a) requires the question's top-2 union to fit VRAM (true for 4/5 here,
false for the suite union and for llm-inference); (b) the skip drops any decode
expert outside the prefill-built mask, so coverage must stay high; (c) today the
skip is import-time + KEEP=0-gated — realizing this in production needs a
per-token "all-resident → skip CPU" gate.

## Harnesses

- `oracle_ceiling_bench.py` — tier sweep, `runs/oracle_ceiling.json`
- `oracle_warm_repeat.py` — adaptive convergence probe
- `build_oracle_masks.py` — build per-question / global 96-per-layer masks
- `oracle` placement strategy (`kt_ep_wrapper.py`) — loads `KT_ORACLE_MASK_PT`
- Data: `runs/oracle_global_top8.json`, `runs/oracle_eigenval.json`,
  `runs/oracle_fixgit.json`, `runs/oracle_masks/coverage.json`
