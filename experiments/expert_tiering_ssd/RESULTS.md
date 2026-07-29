# Three-tier expert store — results

Branch `experiment/expert-tiering-ssd`. 2×H100, GLM-5.2 W4AFP8 int4, 78 layers /
75 routed / 256 experts, `GPU_EXPERTS=96`, `MEM_FRACTION=0.85`, safe2 + MTP
depth-3, per-request tier `top2`.

Every number below carries its provenance, because two separate measurement
faults were found during this work and both produced plausible-looking results.

## 0. Two measurement faults, and what they invalidated

**Fault A — the fill pool changed the shipped contract.** `KT_TIER_FILL_POOL`
was defaulted to `resident`, meaning a substituted slot could be filled from
GPU **or RAM**. With the RAM tier full, "resident" is all 256 experts, so the
six substituted tail slots of a top-2 request were filled with the best-scoring
expert anywhere — usually a CPU expert. A tail that cost zero CPU round-trips
started costing up to six, on the path that is already the decode bottleneck.

A/B, identical residency (RAM=160 ⇒ nothing on SSD ⇒ functionally the shipped
two-tier build), 3 sequential passes each:

| fill pool | tok/s | server accept len |
|---|---|---|
| `gpu` (shipped contract, now the default) | 31.16 / 31.38 / **32.81** | 2.579 |
| `resident` | 18.96 / 19.08 / **18.31** | 2.488 |

It also manufactured a false finding: decode appeared to get *faster* as the RAM
tier shrank (24 → 40.6 tok/s), because a smaller RAM tier makes that pool more
GPU-only, walking the bug back toward the shipped behaviour. **The whole speed
column of the first ladder is void.**

**Fault B — concurrent benchmarking.** The server runs `CUDA_GRAPH_MAX_BS=1`. A
second in-flight request pushes the batch to 2, outside the captured graph, and
both requests fall to eager mode. Running `decbench.py` against a server that
was already being benchmarked recorded 9 tok/s for the interloper and turned a
43.25 tok/s pass into 16.86. Neither number announced itself as wrong. Fixed by
construction: `bench_lock.py` makes benchmarking mutually exclusive and a second
attempt exits with an error rather than queueing.

**Harness note.** Two harnesses disagree by design and it is documented in the
repo's own history (commit `ea0f485`): `bench/perf_probe/decbench.py` (raw
`/generate`) produced the ~40.5 tok/s headline, while the chat-streaming path
"always reads a few tok/s lower" (~33 for the same build). `tier_bench.py` is a
chat-streaming harness, so **its numbers are not comparable to the 40.5
headline** — only to each other.

**Fault C — the accuracy set was too small to rank anything.** 16 questions
means one item is worth 0.0625 and the 95% (Wilson) interval on 12/16 is
±0.20 — wider than the entire quality range being ranked. Every interesting
comparison ("0.75 vs 0.75", "0.9375 vs 0.875") was a one- or two-item
difference reported as a result. The set is now 66 items (±0.10), each accuracy
prints its own interval, and the original 16 are scored separately so earlier
numbers stay comparable.

Extending it exposed a real bug in `accuracy_eval.py compare`: reference and
result were paired **by list position**. A 16-item reference against a 66-item
run would have scored the first 16 pairs, divided by the reference length, and
reported what looked like full agreement across a quarter of the set. Now
paired by question text, with the overlap size printed.

**Fault D — "blocked % of wall time" had the wrong denominator.** It summed the
`took=` field and divided by the span between the first and last `[kt-tier]`
log line. That span includes prefill, idle and the gaps between benchmarks,
none of which can contain a tick, so the denominator was too large and the
number too small. This is the likely source of the "4.5% blocked should give
~38.8 tok/s, measured 33.45" gap that was previously recorded as unexplained —
the 4.5% was an underestimate, not a mystery. Replaced by
`token_latency.py`, which times token arrivals so the denominator is decode and
nothing else, and reports the tail (a 300 ms stall is a visible hitch that a
median hides).

**Fault E — there was no noise floor.** Repeat boots of one config in §2 span
31.16–38.67, which is wider than most differences being called results, and
nobody had separated boot-to-boot from within-boot variation. Nine consecutive
median-of-12 blocks on one unchanged server: mean 33.86, **sd 0.57**. So within
a boot, differences above ~1.2 tok/s are real and smaller ones are not; the
large spread is boot-to-boot, and any comparison across boots must be repeated
in both orders or not made. The same nine blocks double as a drift test —
throughput was flat (33.87 → 33.79) across ~20 minutes of continuous expert
movement, which rules out progressive degradation from accumulated NUMA/page
churn.

## 1. Memory footprint and boot time — unaffected by either fault

| RAM/layer | SSD/layer | host RSS | boot |
|---|---|---|---|
| 160 | 0 | 243.7 GB | 176 s |
| 64 | 96 | 111.2 GB | 131 s |
| 32 | 128 | 67.3 GB | 132 s |
| 16 | 144 | 45.0 GB | 126 s |
| 8 | 152 | **33.6 GB** | 127 s |

Linear at **~1.38 GB of host RAM per expert-slot** across the model, on a ~22 GB
base. **244 GB → 33.6 GB (7.3×)**, and boot drops from ~25 min to ~2 min because
the loader now reads only the experts it will stage instead of the whole 373 GB
checkpoint.

⚠️ With `KT_TIER_KEEP_LOADER=1` (required for runtime promotion) the safetensors
mmaps stay open and their page-cache pages count in RSS: the RAM=32 energy run
reported 123.8 GB rather than ~67 GB. Those pages are reclaimable, not anonymous,
but any RAM-budget claim must say which mode it was measured in.

## 2. Speed on the corrected default (`fill=gpu`)

| config | tok/s (chat harness) | note |
|---|---|---|
| RAM=160, SSD=0 | 31.16 / 31.38 / **32.81** | clean; = the shipped build |
| RAM=32, SSD=128, static | 34.77 / **43.25** / ~~16.86~~ | pass 2 hit by Fault B |
| RAM=32, SSD=128, energy | 34.27 / **35.58** / 33.28 | clean, 3 passes |

RAM=64, 16 and 8 have **no valid speed number** — they were only ever measured
under Fault A.

Two readings so far, both needing confirmation from the sequential re-run:
tiering appears to be genuinely *faster* than full coverage (43.25 vs 32.81),
which is the expected direction once substitution is GPU-only — an SSD miss
removes a CPU round-trip. And energy-driven movement costs ~8 tok/s against the
static split (35.6 vs 43.3), i.e. the promotion traffic is on the critical path.

## 3. Accuracy

Measured with `fill=resident` (Fault A configuration) — these describe *that*
configuration and do **not** transfer to the default:

| RAM/layer | QA acc | loop rate | mean reasoning |
|---|---|---|---|
| 160 | 100 % | 0 % | 278 ch |
| 64 | 81.2 % | 18.8 % | 1462 ch |
| 32 | 62.5 % | 31.2 % | 1823 ch |
| 16 | 62.5 % | 31.2 % | 2158 ch |
| 8 | 56.2 % | 43.8 % | 3083 ch |

Under the corrected default, RAM=32 static scores **56.2 %** — worse than the
62.5 % above, as expected: a GPU-only stand-in is less faithful than the
router's true next-best.

The failure mode is specific and worth keeping: not vaguer answers but
**reasoning loops**. On "what is the chemical symbol for gold" the model burns
its whole 1024-token budget in its reasoning trace and emits no answer. The
full-coverage control loops on 0 of 16, so substitution causes it.

**Accept length is an inverse quality signal over this range.** It rises with the
loop rate — 0 %→2.98, 18.8 %→3.32, 31.2 %→3.41, 43.8 %→3.40 — because looping
text is trivially predictable and the MTP draft head accepts nearly every token.
The 3.22 recorded in `run_thresh0.log` is the same effect: that run used `sub2`,
which substitutes even non-resident top-K experts. So a *drop* in accept length
from 3.4 to 2.58 accompanied the model getting better, not worse.

## 4. What movement actually does

**It never converges.** Every configuration moves exactly its budget on every
visit, from the first quarter of a run to the last:

| config | budget | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| RAM=72 dynamic | gpu 2 | 2.00 | 2.00 | 2.00 | 2.00 |
| RAM=32 nogpu | ram 4 | 4.00 | 4.00 | 3.94 | 3.91 |
| RAM=32 fast | ram 1 | 1.00 | 1.00 | 1.00 | 1.00 |

An adaptive store is supposed to find the working set and settle. This one runs
at maximum rate indefinitely, which is why the per-visit cost never amortises.

**It is not a bad signal, and not a gate that fails to bind.** That was the
obvious hypothesis and it is wrong, on two independent counts. `tier_gate_sim.py`
drives the real selector under Poisson-sampled demand at a realistic vote rate:
the shipped gate largely quiesces (tail 0.66 of a budget of 4) and does not
reproduce the saturation at all. And the server's own ROI accounting finds
**87–89% of promoted experts are called** before the next decision, with only
0.2–1.1% demoted again — noise churn would promote experts nothing asks for.
(`test_tier_quiesce.py` separately confirms the pair rule is correct: on exact
counts from a warm start it moves nothing, as it should.)

What is left is capacity: the working set is larger than the RAM tier, so there
is always another genuinely-warm expert outside it and the selector never runs
out of legitimate work. No gate can fix a capacity limit — and it explains the
otherwise puzzling result that quadrupling the tick rate (promotion latency
84 s → 21 s at RAM=72, 79 s → 20 s at RAM=32) changed accuracy by nothing.

**Coverage bought, per run:** unreachable (SSD-tier) demand falls 3.47% → 3.02%
at RAM=72 over 326 visits, and 9.90% → 8.97% at RAM=32. Under a percentage
point, for a throughput cost of ~25%.

**A GPU swap is never just a GPU copy.** Each demoted expert enters the RAM tier
and must be staged into the kt CPU store — a disk read plus a NUMA repack. The
coupled config therefore performs 5.96 RAM promotions per visit rather than 4,
and the extra ~2 are dragged in by the GPU cycle.

**Accept length: a confound, not a clean signal.** Frozen-vs-dynamic accept
length differs (RAM=72: 2.912 → 2.680; RAM=32: 3.173 → 2.811), and since
tok/s = accept_len / step_time it is tempting to read that as a throughput cost
invisible to `took=` accounting. For RAM=32 that reading is **wrong**: §3 above
established that accept length is an *inverse* quality signal here, rising with
the loop rate because looping text is trivially predictable, and frozen RAM=32
loops on 43.75% of questions against dynamic's 18.75%. Its higher accept length
is largely that artifact. The RAM=72 pair is not explained away this way — both
sides loop at 6.25% — so an ~8% drop there remains a live question. Do not cite
the RAM=32 figure as a movement cost.

## 5. Where the cost actually is (decomposition ladder)

Each rung adds exactly one mechanism to the rung below it, so the drop between
adjacent rungs is that mechanism's price. RAM=72 / SSD=80 / GPU=104, safe2 +
MTP-d3, three median-of-12 blocks after warm-up, run twice in opposite order.

| rung | adds | pass 1 | pass 2 |
|---|---|---|---|
| A frozen | nothing | 40.53 | 40.4x |
| B count | per-step in-graph demand counters | 40.59 | — |
| C decide | full selection every 32 steps, zero moves | 40.56 | — |
| D ram | RAM<->SSD movement | 34.01 | 33.x |
| D0 | D with prefetch lookahead OFF | 32.87 | — |
| **E gpu-incr** | **GPU cycle, stable-slot swap** | **35.58** | **35.15** |
| F gpu-full | GPU cycle, full restage | 28.73 | 29.59 |

**Counting is free. Deciding is free. Only moving costs anything.** The entire
price of an "adaptive" store is the physical movement — nothing in the
observation or decision machinery is worth optimising.

**Movement costs tokens per step, not time per step.** `tok/s = accept_len x
steps/s`, and splitting the two is the single most important line in this
section:

| rung | tok/s | accept | steps/s | Δ steps/s | Δ accept |
|---|---|---|---|---|---|
| A frozen | 40.53 | 2.693 | 15.05 | 0.0% | 0.0% |
| B count | 40.59 | 2.693 | 15.07 | +0.2% | −0.0% |
| C decide | 40.42 | 2.693 | 15.01 | −0.3% | −0.0% |
| **D ram** | 34.01 | 2.318 | 14.67 | **−2.5%** | **−13.9%** |
| D0 nopf | 32.87 | 2.317 | 14.18 | −5.8% | −14.0% |
| **E gpu-incr** | 35.58 | 2.386 | 14.91 | **−0.9%** | −11.4% |
| **F gpu-full** | 29.02 | 2.351 | 12.34 | **−18.0%** | −12.7% |

So the CPU-contention reading of rung D is **wrong as a mechanism**: movement
barely slows the steps (−2.5%), it makes each step deliver fewer tokens
(−13.9%). About 85% of D's loss is degraded MTP draft acceptance. Only rung F
loses real time (−18% step rate, the restage), and the stable-slot swap removes
essentially all of it (E: −0.9%).

**The accept-length cost is not removed by making swaps cheap** — every moving
rung pays 11-14% of it. Two candidate explanations, not yet separated:
1. movement genuinely disrupts draft/target agreement; or
2. movement slightly *improves* quality and accept length is an inverse quality
   signal here (§3), so part of the "loss" is an artifact of the metric.
Distinguishing them needs the 66-item eval run on these rungs. Until then, do
not claim movement costs 16% of throughput *through blocking* — it does not.

**Elapsed time in a phase does not predict its throughput cost.** Per-visit
phase means (rung D): `d2h`=29.6 ms, `promote`=16.4 ms, `evict`=2.3 ms,
everything else under 0.5 ms. The 29.6 ms `d2h` — a device sync to copy a few
hundred bytes of counters — costs **zero** throughput, because the Python
thread was waiting on the GPU anyway. The ~19 ms of promote+evict costs 16%,
because `load_expert`'s NUMA repack runs on the *same CPUInfer thread pool as
the MoE forward*, and CPU is the decode pole. What matters is which resource a
phase consumes, not how long it takes. This invalidates every "blocked % of
wall time" figure in this document as a cost predictor — rung D reports
`blocked 4.5%` against a true cost of 16.1%.

**The prefetch lookahead (7cc7ea0) is dead.** `promote` costs 16.4 ms with the
prefetch running at a 94-98% hit rate and 15.7 ms with it disabled entirely.
Identical. The disk read was never the cost; the repack is, and it stayed on
the critical pool. Throughput A/B is a wash (34.01 vs 32.87, inside noise).
Remove it — it buys nothing and adds a background thread plus GPU->CPU syncs.

**The GPU cycle changes sign once a swap costs what it moves.** This corrects
the earlier finding that "the GPU cycle costs 250 ms and buys nothing", which
measured an implementation artifact rather than the cycle:

| | full restage (F) | stable-slot swap (E) |
|---|---|---|
| GPU apply | 144.0 ms | **2.9 ms** |
| RAM promote | 103.9 ms | **21.3 ms** |
| total visit | 280.8 ms | **56.8 ms** |
| blocked | 22.0% | 5.3% |
| throughput | 29.02 | **35.58** |

The GPU apply drops ~50x, as designed. The *promotions* also drop 5x for the
same work — the full restage streams all 152 non-resident experts through the
CPU store and everything after it in the same visit runs slower (plausibly
cache/bandwidth, not established). So the restage costs far more than its own
144 ms. Net: the GPU cycle goes from **-5.28 tok/s** (a liability, which is why
freezing it looked like a win) to **+1.57 tok/s** — promoting hot experts to
VRAM removes CPU expert work, which is the bottleneck. Do **not** ship
`KT_TIER_MAX_PROMOTE=0`; that was a workaround for a bug that is now fixed.

Validation of the swap itself: coherent on real W4AFP8 weights, **0 declines
across 336 swaps**, narrow build `stream=1.4 ms` / `post+overlay=0.4 ms` — so
the whole-layer post-processing does not dominate, which was the open question
that would have sunk the approach.

## 6. The capacity crossover — and why "movement is a tax" was the wrong frame

Measured PAIRED: both arms inside ONE boot, frozen first (warm-start residency,
which only exists before any drift) then the freeze released via
`KT_TIER_FREEZE_FILE`. Boot-to-boot spread is 31–38 tok/s while within-boot sd
is 0.57, so running the arms as separate boots buried the signal under the
larger noise term. Every frozen arm asserts `tier_visits=0`; a row where the
freeze failed to take is discarded, not read. `move` = the shipped config, RAM
cycle *and* GPU cycle, the latter via the stable-slot swap.

| config | host RSS | tok/s | accuracy (66 items) | loops |
|---|---|---|---|---|
| RAM=152 reference, full coverage | ~230 GB | 30.98 | **1.0000** ±0.028 | 0 % |
| **RAM=48 + move** | **107 GB** | **35.19** | **0.9697** ±0.048 | 1.5 % |
| RAM=48 frozen | 107 GB | 38.89 | 0.7121 ±0.107 | 28.8 % |
| **RAM=32 + move** | **86 GB** | 36.84 | 0.9394 ±0.061 | 6.1 % |
| RAM=32 frozen | 86 GB | 42.55 | 0.5606 ±0.116 | 42.4 % |

**RAM=48 + move beats full coverage on both axes at once**: +13.6 % throughput
at quality whose interval overlaps the reference's, on 2.1× less host RAM.

This corrects the "movement is insurance, not speed" reading in §4. That came
from comparing frozen against move **at fixed RAM**, where movement always looks
like a 10–16 % tax. Compared at fixed **quality** the sign flips: full coverage
is slow *because* it is complete — every expert kept in RAM is one the router
can reach and pay a CPU round-trip for. Movement buys back memory and
throughput together.

There is no crossover in this range. Even at RAM=72, frozen's 0.9375 is beaten
by RAM=48+move's 0.9697 on less memory. Pareto frontier: RAM=72 frozen
(40.53 / 0.9375 / 141 GB) — RAM=32 move (36.84 / 0.9394 / 86 GB) — RAM=48 move
(35.19 / 0.9697 / 107 GB) — RAM=152 (30.98 / 1.0000 / ~230 GB). Every frozen
config below RAM=72 is off the frontier: 42.55 tok/s is worthless at 0.56.

**The GPU cycle's real value is accuracy, not speed.** Every earlier RAM=32
"dynamic" row froze it (`KT_TIER_MAX_PROMOTE=0`) to dodge the 250 ms restage,
and repaired quality only to 0.75. With the stable-slot swap making that cycle
affordable it repairs to **0.9394**. The "+1.57 tok/s" in §5 understated its
worth by measuring the wrong axis.

Frozen quality degrades smoothly with RAM rather than falling off a cliff —
0.5606 (32), 0.7121 (48), 0.9375 (72), 1.0000 (152) — and the failure mode is
reasoning loops, not wrong answers.

### Accept length measures FIDELITY, and the sign was misread twice

Measured on one fixed workload (the `decbench.py` essay prompt) so the configs
are comparable at all:

| config | accuracy | accept |
|---|---|---|
| RAM=152 full coverage | 1.0000 | **2.181** |
| RAM=48 + move | 0.9697 | 2.249 |
| RAM=32 + move | 0.9394 | 2.461 |
| RAM=48 frozen | 0.7121 | 2.604 |
| RAM=32 frozen | 0.5606 | 2.762 |

**Spearman −1.000, Pearson −0.944.** Accept length ranks these configs by
accuracy perfectly, and full coverage — the best model available on this box —
has the LOWEST accept length of anything measured.

So the accept-length drop under movement is not damage. **Movement pushes accept
DOWN TOWARD the full-coverage reference** (RAM=72: 2.68 frozen → 2.22 moving,
landing within 0.04 of the true model's 2.18). Frozen substitution pushes it UP,
above the reference. The moving configs are converging on the true model, not
diverging from it.

The mechanism: **speculative decoding pays for predictability, not correctness.**
A substituted model is confined to fewer experts, so its output is more
stereotyped, so the draft head guesses it easily and each step yields more
tokens. Restore expressiveness — by full coverage or by moving the right experts
in — and the model becomes harder to draft for. This is why full coverage is the
best model and the slowest (30.98 tok/s), while RAM=32 frozen is the worst and
the fastest (42.55): one axis, not a coincidence.

⚠️ **Two corrections, recorded because both were committed as findings.**
(1) §5 frames the accept drop as movement's *cost*, "degraded MTP draft
acceptance". The throughput arithmetic there is right and still stands; the
word "degraded" is wrong. It is the price of fidelity, and it cannot be
optimised away without making the model worse.
(2) An earlier reading claimed the moving configs sat BELOW the reference and
therefore could not be a quality gain. That compared the reference's *QA-eval*
accept (2.585, short factual answers draft easily) against the ladder's *essay*
accept. Different workloads. Accept length varies more with workload than with
config — 2.18 essay vs 3.1 QA on the very same server — so it is only ever
comparable within one fixed prompt set.

## 7. Open

- Re-measure RAM=64/16/8 under `fill=gpu` for both speed and accuracy.
- Confirm RAM=160 `fill=gpu` reproduces the ~40.5 headline on `decbench.py`
  (running: `run_baseline_check.sh`, both harnesses, strictly sequential).
- Next-token / 4-token residency instrumentation (`kt_energy_report`) — see §5
  once the energy sweep completes.
