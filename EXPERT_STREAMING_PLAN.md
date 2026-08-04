# Streaming + CPU hybrid expert execution — plan

Branch: `experiment/expert-streaming-hybrid`

Every layer's routed experts get split two ways at decode time: some are computed
on the CPU where they already live, the rest have their weights streamed over
PCIe and computed on the otherwise-idle GPU. A calibration file fixes the machine
constants; a formula turns them into a per-layer split.

---

## 1. Verified starting point (2026-08-03, live server)

Config: `GPU_EXPERTS=60`, `RAWINT4`, NSA, MTP depth-3, `safe8`, TP2.

`bench/decode_bench.sh` reproduces this morning's profile: **14.23 tok/s** e2e
(≈15.3 decode-only vs 16.0 recorded). Within the profiler's own 8% settle band,
so the box has not drifted and prior numbers stand.

### 1.1 The CPU cost curve, measured

`bench/cpu_fixed_cost.py` sweeps the live per-request tier (`GLM5.2-topN`, no
restart, one CUDA graph) and reads ms/**step** — one SSE chunk is one forward
step, so the metric does not move when accept length does.

| tier | ms/step | Δ vs prev, per expert | steps/s | accept | tok/s |
|-----:|--------:|----------------------:|--------:|-------:|------:|
| 0 | 61.98 | — | 16.11 | 3.85 | 61.94 |
| 1 | 73.77 | 11.8 | 13.49 | 2.44 | 32.90 |
| 2 | 86.04 | 12.3 | 11.53 | 2.63 | 30.33 |
| 4 | 114.94 | 14.4 | 8.63 | 2.74 | 23.65 |
| 6 | 150.33 | 17.7 | 6.64 | 3.23 | 21.41 |
| 8 | 190.07 | 19.9 | 5.24 | 3.28 | 17.18 |

Run-to-run spread ≤2.3%, mostly ≤0.7%.

**The CPU expert path is marginal, not fixed.** 128 ms of the 190 ms step (67%)
scales with expert count, and the slope is *convex*: the 8th expert costs 19.9
ms/step, the 2nd costs 12.3. Removing CPU-side experts pays proportionally, and
pays more when the CPU is loaded.

Fit over tiers>0: `ms/step = 53.3 + 16.6·K`. Per **CPU-side** expert (coverage
23.4%, so 0.77 of each kept expert lands on the CPU): **21.6 ms/step**, and over
the 4-token MTP verify batch, **5.4 ms per CPU-side expert per token** —
0.072 ms per expert per layer per token.

### 1.2 This overturns the "fixed submit cost" reading of the oracle result

[[glm52-oracle-placement-ceiling]] recorded a ~22-23 tok/s floor and I had read
it as a fixed ~1.3 ms/layer submit+sync. That reading was wrong. The oracle moved
coverage 69%→92%, which removes ~1.86 CPU-side experts; at the measured 5.4
ms/expert/token that predicts ~10 ms/token, and the oracle measured 19.5→22.5
tok/s = 7 ms/token. Consistent with a purely linear marginal cost.

There are two unit traps behind the old number. First,
`results.json: profiler_overhead_x = 1.315` means the profiler made the measured
step **1.315× slower**; it does not mean 1.315 ms. Second, a real 1.3 ms cost at
each of 75 MoE layers would total **97.5 ms/step**, so it would not be a rounding
error. Neither interpretation supports assigning 1.3 ms/layer to submit+sync.

The oracle floored because **placement can only ever remove ~1.9 experts** — it is
bounded by VRAM capacity. Streaming is bounded by PCIe bytes instead, which is a
different and much larger budget. That is why this experiment is not the oracle
experiment repeated.

The true fixed cost is bounded but not yet isolated: the top0 step is 62.0 ms, of
which GPU kernels account for ~32 ms (37.3 ms at top8 minus the 5.4 ms of GPU MoE
that top0 does not do). So submit/sync fixed cost is somewhere in **0–30 ms/step
(0–0.40 ms/layer)**, not the 98 ms/step I had claimed. An independent estimate
from the same oracle memory narrows it further: placing fix-git's full top-2 set
on GPU gave 40 tok/s, and arming `/tmp/kt_skip_cpu` on top of it — removing only
the submit, since residency was already 100% — gave 51, i.e. **~13 ms/step
(~0.17 ms/layer)**. Real, worth taking at the corners, but 10× smaller than the
128 ms of marginal cost. Stage 0A measures it directly before either component
is optimized.

### 1.3 The other side of the balance, measured

PCIe H2D, measured on this box just now: **53.4 GB/s per card pinned** (41.9
pageable), Gen5 x16, and each GPU is on its own NUMA node — so ~107 GB/s
aggregate. Expert weights are TP-sharded on the intermediate dim, so a card needs
**9.7 MB** of the 19.46 MB per expert per layer → **0.182 ms per distinct expert
per layer per step**.

Critically, the CPU path reuses each expert's weights across all 4 tokens of the
verify batch (that is why its RAM traffic works out at ~138 GB/s, matching the
profile's 143). **Streaming amortises the same way and better**: transfer cost is
per *distinct* expert per step, while CPU compute cost is per expert-token.

### 1.4 What the split is worth

Two independent estimates, both at `safe8` / 23% coverage:

*Per-layer balance.* Expert path is 1.71 ms/layer all-CPU. Moving one distinct
expert to the stream saves ~0.096 ms of CPU and adds 0.182 ms of PCIe; they
overlap, so solve `1.71 − 0.096x = 0.182x` → x≈6.2, layer time 1.12 ms → **1.53x
on the expert path**.

*Bandwidth budget.* CPU path delivers weights at ~138 GB/s; PCIe adds up to 107
GB/s of otherwise-idle delivery. Total RAM demand ~245 GB/s against ~300-379
GB/s achievable → **~1.78x**, minus contention.

Both land in the same place: expert path 128 → ~84 ms/step, step 190 → ~146 ms,
**≈22 tok/s at full top-8 fidelity, up from 17.2** (~1.3x end-to-end). Higher GPU
coverage (`GPU_EXPERTS=104` + adaptive cache) shifts both terms favourably.

### 1.5 One consequence of the chosen fidelity target

Composing with top-2 substitution caps the win. At tier 2 the CPU expert path is
only 24 ms of an 86 ms step (28%), so the same formula yields ~+6-10% there. The
larger prize is the other direction: **streaming makes full top-8 affordable**
(≈22 tok/s vs top-2's 30.3), buying quality back instead of buying speed by
dropping experts. **`safe2` is the headline benchmark, never `sub2`** (2026-08-04
direction, and [[glm52-sub2-never-default]]): `sub` drops a genuine top-K expert
whenever it is not GPU-resident, so its speed comes from not doing what the router
asked for. Every phase reports safe8 alongside safe2, because that is where the
mechanism actually pays.

---

## 2. Phases

### Phase 0 — optimize the existing CPU expert path first

Streaming design does not start until Stages 0A–0F finish. Each optimization is
independently selectable so a failed stage can remain disabled while later work
is measured against the last passing stack.

#### Execution checkpoints — 2026-08-03 to 2026-08-04

The first implementation/measurement session is intentionally stopped at Stage
0C. These are five-run, unprofiled screening measurements at `GPU_EXPERTS=60`,
safe8, MTP depth 3, packed RAWINT4, TP=2. They are not the final interleaved tier
sweep required by Stage 0F.

| Stage | Runtime selection | Exact real-kernel parity | Median decode rate | Change vs 0A | Decision |
|---|---|---:|---:|---:|---|
| 0A legacy | `none` | saved reference | 15.49 tok/s | baseline | retained |
| 0B guided scheduler | `scheduler` | pass | 14.87 tok/s | -4.0% | reject; keep disabled |
| 0C shared prequant | `prequant` | pass | 15.70 tok/s | +1.36% | reject; misses +2% gate |

The per-run results are stored in
`bench/profile_out/rates/cpu-stage0a-legacy-e60-safe8.json`,
`bench/profile_out/rates/cpu-stage0b-scheduler-e60-safe8.json`, and
`bench/profile_out/rates/cpu-stage0c-prequant-e60-safe8.json`. The legacy packed
RAWINT4 output reference is
`bench/profile_out/cpu_stage0a_rawint4.pt`.

The implementation is present behind `KT_CPU_EXPERT_OPTS` and defaults to
`none`. Real packed-RAWINT4 layer output is bitwise equal for `none`, each
implemented option, and the combined option set in the current single-layer
harness case. That is an implementation sanity check, not completion of the full
validation matrix below. In particular, Stage 0A instrumentation/tier sweep,
Stage 0D end-to-end timing and soak, Stage 0E top0/no-CPU comparison, deterministic
server prompts, DOC_QA, LiveBench, sanitizers, and Stage 0F remain open.

Stage 0A–0F execution completed on 2026-08-04. No optimization passed all of its
gates, so the selected Stage 0F stack is still `none`; the experimental code
remains runtime-gated and disabled by default.

##### Stage 0A measured cost and attribution

The matched normal/no-CPU sweep gives the actual CPU-path contribution:

| K | Normal ms/step | No-CPU ms/step | CPU-path ms/step |
|---:|---:|---:|---:|
| 0 | 52.51 | 51.11 | 1.40 |
| 1 | 64.80 | 50.37 | 14.43 |
| 2 | 79.42 | 49.86 | 29.56 |
| 4 | 105.40 | 50.05 | 55.35 |
| 6 | 143.26 | 47.76 | 95.51 |
| 8 | 180.91 | 47.20 | 133.70 |

The fixed K=0 component is only **1.40 ms/step**, or 1.0% of the K=8 CPU
path. The marginal cost is **16.54 ms per kept expert per step** (0.220 ms per
expert per MoE layer). Thus the earlier 1.315 value was neither milliseconds
nor evidence of a per-layer fixed tax.

An attribution-only `KT_CPU_EXPERT_PROFILE=1` run slowed safe8 from about 179
to 225 ms/step, so its absolute timing is not a performance result. Across
steady 1,000-job windows, its medians were approximately 7 microseconds/job in
the submit callback, 12 microseconds/job of queue delay, 2.33 milliseconds/job
of worker execution, and 2.31 milliseconds/job in the sync wait. The sync wait
therefore tracks the worker almost one-for-one: it is waiting for expert work,
not spending milliseconds in synchronization machinery. Activation
quantization was about 11.5% of summed GEMM thread time; those two counters sum
parallel worker time and must not be read as wall time.

##### Stage root causes and decisions

| Stage | Measured result | Root cause | Decision |
|---|---|---|---|
| 0B scheduler | 14.87 vs 15.49 tok/s (-4.0%); distinct routes exact, repeated routes differ | guided blocks reduce fine-grained balancing across unequal expert/NUMA work; decode also schedules duplicate expert IDs onto the same buffers, so changing ownership exposes a legacy write race | reject |
| 0C prequant | 15.70 vs 15.49 tok/s (+1.36%); distinct routes exact, high-magnitude repeated routes differ | the kernel remains weight-bandwidth bound; prequant still writes and rereads a cache, so removing redundant activation scans is not on the critical path. Repeated decode routes also expose the same shared-buffer race | reject |
| 0D queue | K=0 52.34 vs 52.50 ms (-0.31%); K=8 178.61 vs 178.26 ms (+0.20%); all 80 matrix cases exact | raw descriptors remove a small allocation/dispatch cost, but the two CUDA callbacks remain and the pool adds node-pool mutex traffic. The whole fixed submit+queue budget is only about 19 microseconds/job | reject |
| 0E empty fast path | K=0 53.05 vs 52.50 ms (+1.03%); K=8 178.82 vs 178.26 ms (+0.32%); all 80 matrix cases exact | the early return is inside the queued CPU job, after staging, callback, and queue costs. The legacy zero-route job already runs no GEMMs, so replacing its small zero-output path cannot approach the callback-free graph | reject |

The expanded packed-RAWINT4 harness covers M={1,2,4,8}, CPU route counts
{0,1,2,4,8}, zero/high/random inputs, two seeds, and repeated/distinct routes:
80 cases total. `none`, `queue`, and `empty_fastpath` match the saved legacy
artifact in all 80 cases. Every option matches all 40 distinct-route cases.
Scheduler/prequant combinations fail the deliberately repeated-route cases due
to the pre-existing decode duplicate-ID race described above. A proposed
deduplication produces the mathematically intended value but not the original
legacy bits, so it was reverted rather than weakening the parity contract.

Stage 0D's 10,000-step soak and Stage 0C/combined-0E DOC_QA and LiveBench runs
were not promoted after their prerequisite performance/parity gates failed.
They cannot turn a rejected implementation into a passing one, and no default
model path changed.

##### Stage 0F final unprofiled rebaseline (`none`)

Five 400-token rounds, interleaved across tiers:

| K | Median ms/step | Acceptance length | tok/s | Spread |
|---:|---:|---:|---:|---:|
| 0 | 52.38 | 2.78 | 52.94 | 0.7% |
| 1 | 63.00 | 2.33 | 36.46 | 0.9% |
| 2 | 77.08 | 2.65 | 34.09 | 0.4% |
| 4 | 105.47 | 2.92 | 27.53 | 0.7% |
| 6 | 140.65 | 3.10 | 22.04 | 0.5% |
| 8 | 179.09 | 2.88 | 16.07 | 0.4% |

The final slope is 16.51 ms per kept expert per step. Safe8 remains below the
approximately 22 tok/s streaming target, so streaming remains justified and
may proceed to Phase 1 using only this rebaseline. The artifact is
`bench/profile_out/stage0f_none_final.json`.

#### Stage 0A — direct baseline and instrumentation

- Add low-overhead timers for callback entry/exit, queue delay, worker execution,
  activation quantization, GEMM, and synchronization wait. Absolute performance
  always comes from unprofiled runs; CUPTI/torch profiles are attribution only.
- Run `bench/cpu_fixed_cost.py` at K={0,1,2,4,6,8}, both normally and with the
  CPU path completely removed. `step_ms(K) − step_ms_noCPU(K)` is the CPU path's
  contribution; the K=0 difference directly isolates its fixed component.
- Save legacy RAWINT4 kernel outputs and deterministic server generations as the
  parity references used by every later stage.

#### Stage 0B — scheduler chunking

- Remove the forced `block = 1` override in the work-stealing loop and use its
  already-computed guided chunk size.
- Keep task outputs disjoint and leave every arithmetic and reduction order
  unchanged.
- Gate: bitwise kernel parity, at least 5% lower scheduler/GEMM time, and at
  least 2% lower safe8 ms/step.

#### Stage 0C — shared activation pre-quantization

- Quantize each BF16 activation group once into the **same** biased uint8 bytes
  and float scale currently regenerated by every output shard.
- Gate and up share one NUMA-local quantized activation cache because they have
  the same BF16 input. Down owns a separate cache populated only after the
  existing BF16 SwiGLU boundary.
- Do not change the quantizer, rounding mode, scale type, BF16 boundaries, group
  order, or GEMM accumulation order. The cache is fixed-size/NUMA-local and does
  not allocate in the decode path.
- Gate: `torch.equal` against the legacy kernel for every test case. Any differing
  bit rejects the stage; model-level quality tolerance is not a substitute.
  Performance acceptance also requires at least 2% lower safe8 ms/step.

This refactor should not affect model accuracy: today each output shard computes
the same deterministic quantized representation from the same BF16 values. The
optimization shares that representation instead of recomputing it. Requiring
bitwise equality proves that statement rather than assuming it.

#### Stage 0D — callback and task-queue path

- Preserve submit before GPU work and completion after GPU work. Collapsing both
  into a blocking submit callback would serialize CPU/GPU execution and is
  explicitly forbidden.
- Remove per-layer heap allocation and generic lambda/`std::function` queue
  indirection; use preallocated job descriptors and completion counters.
- Initially retain both graph positions. Replace the completion callback with a
  graph-safe host completion-word wait only if a standalone spike proves that it
  preserves overlap and reduces fixed time.
- Gate: bitwise output parity, at least 20% lower fixed K=0 CPU-path time, no more
  than 1% safe8 regression, and a 10,000-step graph-replay soak with no hangs,
  pending jobs, shutdown races, or memory growth.

#### Stage 0E — empty CPU-route fast path

- Promote the existing `KT_SKIP_CPU_FILE` experiment into a supported graph-wide
  no-CPU variant.
- Short-circuit an individual CPU job when its routed CPU expert set is empty;
  never skip mixed or non-empty routes.
- Gate: top0 lands within 2% of the explicit no-CPU graph and no K>0 tier regresses
  by more than 1%.

#### Stage 0F — rebaseline before streaming

- Rerun the complete tier sweep and replace every fixed and marginal CPU constant.
  The measurements in Section 1 remain historical baseline numbers, not inputs
  to the streaming policy.
- If optimized safe8 reaches or exceeds the current ~22 tok/s streaming target,
  record the result and park streaming unless it still has a separately justified
  target. Otherwise continue to Phase 1 using only the new cost curve.

#### Runtime selection and rollback

Add temporary `KT_CPU_EXPERT_OPTS`: `none`, or a comma-separated set of
`scheduler`, `prequant`, `queue`, and `empty_fastpath`. One build must support
isolated A/B tests and combined-stage tests. All options default off during
validation; only the stages that pass Stage 0F become the default, while `none`
remains the rollback path.

#### Validation required after every Stage 0 change

- Extend the real RAWINT4 harness to compare legacy and selected implementations
  for M={1,2,4,8}, CPU route counts {0,1,2,4,8}, zero and high-magnitude inputs,
  repeated and distinct routes, and multiple deterministic seeds.
- Require bitwise equality, finite outputs, unchanged dequantized-reference error,
  and clean race/thread-sanitizer results where supported.
- Run a fixed deterministic safe8 prompt set with MTP enabled and disabled. Token
  IDs must exactly match the saved baseline.
- Run at least five interleaved performance trials and report median ms/step,
  acceptance length, and tok/s together.
- Run full DOC_QA and LiveBench after Stage 0C and after the combined Stage 0E
  stack. Scores and deterministic generations must not regress.
- A failed stage stays disabled. The next stage is evaluated on the last passing
  stack, never on an unverified combination.

Persistent-worker/semaphore redesign is out of scope for Phase 0. Reconsider it
only if Stage 0F still attributes material time to worker wake-up. No Phase 0
stage may change routing, expert weights, model precision, or expert-count policy.

### Phase 1 — finish the streaming baseline

- **1a. Re-verify the headline configs** so "exceeded" means something:
  `GPU_EXPERTS=104` at safe2 (recorded 30.5) and safe8 (recorded ~17.5), via
  `bench/decode_rate.py --runs 5`, results committed to `bench/profile_out/rates/`.
  Anything that fails to reproduce gets chased before design work continues.
- **1b. Measure contention directly.** Run the H2D benchmark *while* the
  optimized CPU expert path is decoding. The headline assumes DMA and CPU expert
  compute can share RAM; this is the single biggest way the estimate can fail.

Gate: if 1b shows PCIe collapses under CPU load, the whole design is capped and
we say so before building it.

#### Phase 1 execution — 2026-08-04

Two boots of `GPU_EXPERTS=104`, RAWINT4 packed, TP2, MTP depth-3, dense MLA,
`KT_CPU_EXPERT_OPTS=none`. Per-request tiers (`GLM5.2-topN`) select routing live
inside one CUDA graph, so every row within a boot is directly comparable.

The measurement boot is **`safe`** (sentinel `safe2`). An earlier `sub2` boot was
taken first and is retained only as a rejected control: per direction and
[[glm52-sub2-never-default]], `sub` drops a genuine top-K expert whenever it is
not GPU-resident, so it buys speed by not doing what the router asked. It also
destroys the experiment — under `sub`, every tier below 8 routes **zero** experts
to the CPU (top2 53.96 vs top0 54.17 ms/step), so the CPU cost curve a
CPU/GPU-split experiment exists to measure does not exist in that boot.

##### 1a — the headline configs, re-verified

`bench/decode_rate.py --runs 5 --tokens 400`, decode-only:

| config | recorded | measured | spread | artifact |
|---|---:|---:|---:|---|
| e104 **safe2** (`-top2`) | 30.5 | **42.04** | 1.0% | `rates/phase1a-e104-safe2.json` |
| e104 **safe8** (`-top8`) | ~17.5 | **20.71** | 0.4% | `rates/phase1a-e104-safe8-safemode.json` |
| e104 safe8, `sub` boot | — | 20.60 | 0.4% | `rates/phase1a-e104-safe8.json` |
| e104 sub2 — rejected control | 45.05 | 43.61 | 0.4% | `rates/phase1a-e104-sub2.json` |

Neither headline reproduces, both in the *good* direction, and the two cross-check
each other: safe8 measured 20.71 in the safe boot and 20.60 in the sub boot (0.5%
apart), exactly as the code implies — at `K == E` the residency filter is skipped,
so `sub8` and `safe8` are the same path.

- **safe2 is 38% above its recorded 30.5** (2026-07-28, GPU=104). That figure
  predates the MTP CUDA-graph fix and the current placement/adaptive defaults.
- **safe8 is 18% above the `~17.5`** it was quoted at, which was never measured
  at 104 experts — it is `run_fast.sh`'s header row, and Stage 0F measured 16.07
  at `GPU_EXPERTS=60`. The whole Section 1 curve is an e60 measurement.
- **sub2 is worth only +3.7% over safe2** (43.61 vs 42.04). The unsafe routing
  contract buys less than 4%, which settles the question independently of quality.

##### 1a — the real per-mode cost curve (safe boot)

`bench/cpu_fixed_cost.py --tiers 0,1,2,4,6,8 --runs 3`, interleaved. This is the
curve the streaming policy needs, and it only exists in a `safe` boot:

| mode | ms/step | accept | tok/s | vs safe8 | CPU path (− top0) | CPU share | CPU ms/layer |
|---|---:|---:|---:|---:|---:|---:|---:|
| top0 (no experts) | 53.57 | 3.57 | 67.84 | 3.20x | — | — | — |
| safe1 | 58.46 | 2.17 | 36.78 | 1.74x | 4.89 | 8.4% | 0.065 |
| **safe2** (default) | 65.81 | 2.78 | 41.89 | 1.98x | 12.24 | 18.6% | 0.163 |
| safe4 | 87.44 | 3.17 | 35.99 | 1.70x | 33.87 | 38.7% | 0.452 |
| safe6 | 113.10 | 3.17 | 27.84 | 1.31x | 59.53 | 52.6% | 0.794 |
| safe8 (full fidelity) | 147.78 | 3.12 | 21.20 | 1.00x | 94.21 | 63.8% | 1.256 |

Fit over tiers>0: `ms/step = 41.27 + 12.68·K`, i.e. **12.68 ms per kept expert per
step** (0.169 ms per expert per layer), or 21.3 ms per *CPU-side* expert at
104/256 residency. Artifact `phase1_e104_safemode_tiers.json`.

Read the ms/step column, not tok/s: safe1 is *slower* than safe2 in tok/s
(36.78 vs 41.89) purely because MTP acceptance drops to 2.17, the failure mode
[[glm52-tier-movement-synchronous]] warned about. Steps got cheaper; tokens per
step got cheaper faster.

##### 1b — contention, measured in both directions

`bench/pcie_contention.py`. One numactl-bound worker per card (gpu0→node0,
gpu1→node1), 9.7 MB pinned H2D chunks issued *during* a live decode, with the
decode's own ms/step read over the same window.

| case | mode | H2D GB/s | decode alone | under DMA | Δ | ms decode per GB |
|---|---|---:|---:|---:|---:|---:|
| idle server | — | 112.3 (56.2/card) | — | — | — | — |
| queued 64 MB | safe8 | 108.4 | 146.79 | 193.30 | +31.7% | 2.22 |
| queued 64 MB | safe2 | 107.9 | 65.67 | 94.29 | +43.6% | 2.81 |
| **queue depth 1** | safe8 | 104.2 | 146.48 | **165.11** | **+12.7%** | **1.08** |
| **queue depth 1** | safe2 | 104.0 | 65.71 | **76.36** | **+16.2%** | **1.34** |

Mechanism controls (taken in the `sub` boot; they probe the transport, not the
routing, so the boot does not matter — and the safe-boot safe8 penalty, +31.7%,
reproduces the sub boot's +31.8% exactly):

| control | result | rules out |
|---|---:|---|
| spin-only, same cores, zero bytes | −0.9% | CPU core theft |
| tier-0 load (no expert traffic) | +71.5% / +38.7 ms — same *absolute* penalty as tier 8 | DRAM contention with expert weights |
| single card (gpu0 only) | +29.4% of the +31.8% both cards cost | per-card additive bandwidth |

**The gate passes: PCIe does not collapse.** A live safe8 decode still leaves
108.4 of 112.3 GB/s (96.5%), and a 9.7 MB expert costs 0.18 ms per card under
load, as designed.

**What the decode pays is head-of-line blocking, not bandwidth.** kt ships
activations host-ward and back once per MoE layer on the same link; those small
latency-critical transfers queue behind whatever bulk copies are already
submitted — hence ~0.5 ms x 75 layers, hence the tier-independence, hence gpu0
(where the kt path lives) carrying almost all of it.

**That is fixable by submission shape, and it is the improvement Phase 1 actually
delivers to every mode:** submitting one chunk at a time instead of 64 MB of
queued copies keeps 96% of the bandwidth and cuts the concurrent-DMA penalty
**2.5x at safe8 (+31.7% → +12.7%) and 2.7x at safe2 (+43.6% → +16.2%)**, an
exchange rate of 1.08-1.34 ms of decode per GB streamed. Queue depth, not
bandwidth, is the thing to design around; Phase 3 inherits it as a constraint on
both mechanism candidates.

##### What this does to the expected payoff, per mode

Per layer per step: CPU pole `C` from the table above, `U` expert-token units
(`K x 0.594 x 4` verify tokens), and each streamed distinct expert costs 0.185 ms
of PCIe plus 0.021 ms of contention = **0.206 ms**, while removing `U/D` units,
where `D` is the distinct non-resident experts a layer routes across the verify
batch (between `U/4` and `U`; unmeasured). Optimum `x* = C/(0.185 + C/D)`.

| mode | C (ms/layer) | streamed x* | predicted tok/s | gain |
|---|---:|---:|---:|---:|
| safe2 | 0.163 | 0.7-0.7 | 42.4-43.0 | **+1-3%** |
| safe4 | 0.452 | 1.7-1.9 | 37.7-39.2 | **+5-9%** |
| safe6 | 0.794 | 2.9-3.3 | 30.1-32.0 | **+8-15%** |
| safe8 | 1.256 | 4.3-5.0 | 23.9-26.0 | **+13-22%** |

(ranges span `D` = no reuse across the verify batch to `D` = U/1.6 reuse.)

**The decisive number is safe2's.** Its entire CPU pole is 0.163 ms/layer, which
is *less than the 0.206 ms cost of streaming a single expert in that layer* — the
first unit of transfer already costs more than all the CPU work it could remove.
Streaming cannot pay at the shipped default. Break-even is `C > ~0.21 ms/layer`,
i.e. tier 3 and above.

So the mechanism's honest positioning changed: it does not speed up the default,
it **makes fidelity cheaper**. It narrows the safe2→safe8 gap from today's 1.98x
to about 1.6-1.75x, buying back routing quality rather than tok/s — which is the
same direction Section 1.5 argued for, now with the safe-mode numbers to support
it and with sub2 out of the comparison entirely.

`D` is the pivotal unknown and is cheap to read off the existing decode counters.
**Phase 2 measures `D` first**, and it is go/no-go: at `D` = U (no reuse) even
safe8 returns only +13%, which is close enough to Phase 5's "under ~10% gets
parked" that building Phases 3-4 would be hard to justify for a tier that is not
the default.

### Phase 2 — offline calibration file

`bench/calibrate_hybrid.py` → `bench/profile_out/hybrid_profile.<hostname>.json`,
read by the server at boot, never measured in-process (chosen: offline only).

Constants:
- `pcie_gbs[gpu]` — pinned H2D per card, and the same figure under CPU load (1b:
  56.2 idle, 52-54 under decode, ~unchanged at queue depth 1)
- `distinct_experts_per_layer` — `D`, the union of non-resident routed experts
  over the verify batch. **Measure this first**; 1b's payoff table turns on it
  and it decides whether Phases 3-4 are worth building at all.
- `contention_ms_per_gb` — decode time the step pays per GB streamed: **1.08
  (safe8) / 1.34 (safe2) at queue depth 1**, 2.2-2.8 with 64 MB queued (1b)
- `cpu_ms_per_layer[mode]` — the safe-boot curve: 0.163 (safe2), 0.452 (safe4),
  0.794 (safe6), 1.256 (safe8). Supersedes the Stage 0F e60 curve for policy use.
  Break-even for streaming is ~0.21 ms/layer, so the policy must be allowed to
  return "stream nothing" for the whole model at low tiers.
- `cpu_fixed_ms_per_layer` — from the Stage 0F no-CPU comparison
- `gpu_ms_per_expert` — GPU MoE marginal cost per streamed expert
- `bytes_per_expert_per_card` — from the model config, not assumed
- `ram_gbs_peak`, `ram_gbs_cpu_path` — the shared budget both poles draw from

A missing or stale-hashed file falls back to the optimized all-CPU behaviour
rather than guessing.

### Phase 3 — transfer mechanism spike (both, then pick)

Standalone harness, no serving path, measuring achieved GB/s **under CUDA graph
replay with the optimized CPU expert path running concurrently**.

Two problems to solve, and the second is the real one:

**(i) Graph-safe dynamic addressing.** Captured memcpy nodes have fixed
addresses, so per-step expert choice needs either:
- **A. UVA gather kernel** — `cudaHostRegister` a window of host expert memory; a
  captured kernel reads it via device-computed indices into stable VRAM slots.
  Fastest and graph-safe; pinned pages cannot swap and free RAM is currently 7 GB,
  so the window must be sized deliberately.
- **B. Host-node staging + fixed H2D** — a host node in the graph (like kt's
  existing `submit`) memcpys chosen experts into a fixed pinned staging buffer; a
  captured memcpy with fixed addresses moves it. Simple, but doubles RAM traffic
  on exactly the resource that is already the shared constraint.

**(ii) Layout.** kt's CPU store holds RAWINT4 in the *CPU kernel's* packed layout;
the GPU wants cutlass W4A8 with interleaved scales, and today's whole-layer
interleave costs ~136 ms — unusable per step. Three candidates:
- host-side second copy of a streamable window already in GPU layout (~1.46 GB per
  expert-slot across the model; a 32-slot window ≈ 47 GB, affordable against ~300
  GB available)
- transfer raw and **repack in a GPU kernel** (the GPU is 80% idle, so this is
  plausibly free)
- stream from `weights/GLM-5.2-W4AFP8` via mmap + page cache, which is *already*
  in GPU layout (373 GB file set vs 301 GB page cache — needs a residency check)

Deliverable: one measured recommendation with numbers, not a preference.

### Phase 4 — the split policy and plumbing

- Per-layer decision computed in-graph and branch-free, in the style of the
  existing `keep_k` kernel in [deepseek_v2.py](.venv/lib/python3.12/site-packages/sglang/srt/models/deepseek_v2.py) `_kt_topk_experiment`.
- Solve `min over k of max(a + b·(m−k), c + k·bytes/BW + gpu(k))` per layer, where
  `m` is the layer's non-resident routed count. Because `a > 0`, the optimum is a
  **corner** whenever `m` is small — the formula must be allowed to return k=m and
  drop the CPU submit for that layer, not just interior splits.
- CPU set keeps the existing kt path (streamed experts masked to −1, which the
  C++ `should_skip_expert` already handles safely).
- Stream set lands in stable VRAM slots and joins the existing GPU MoE.
- Model-wide corner: if every layer returns k=m, use the no-CPU graph variant so
  the CPU leaves the loop entirely.

VRAM: landing slots are cheap — 8 double-buffered slots × 9.7 MB ≈ 78 MB/card
against 1.5 GB currently free. Not a blocker at `GPU_EXPERTS=60`; re-check at 104.

### Phase 5 — verification

The point of the experiment is to **exceed** the Stage-0F numbers, so:

- `bench/decode_rate.py` and `bench/cpu_fixed_cost.py` rerun at safe2 (headline)
  and safe8 (where the mechanism pays), same prompts, ≥5 runs, interleaved. No
  phase measures `sub2`.
- Report ms/step *and* accept length together — [[glm52-tier-movement-synchronous]]
  showed ~85% of a movement scheme's cost can hide in accept length while step
  rate looks fine.
- Quality: `bench/livebench` + `DOC_QA_RUNBOOK.md` at safe8, which must be
  bit-comparable to the optimized all-CPU path since streaming changes *where*
  an expert is computed, not *which*.
- Failure is a real outcome: if the measured gain is under ~10% end-to-end the
  branch gets written up and parked, not tuned indefinitely.

---

### Phases 2-3 execution — 2026-08-04

#### Phase 2a — `D`, the go/no-go constant

Measured from a `KT_DUMP_TOPK=1` routing dump (prefill, so not graph-captured;
consecutive real tokens as the closest available proxy for a verify batch),
`bench/measure_distinct_experts.py`:

| tier | U (expert-token units/layer) | D (distinct/layer) | reuse U/D |
|---:|---:|---:|---:|
| 2 | 3.2-3.6 | 2.3-2.5 | 1.39-1.44 |
| 4 | 7.3-8.0 | 5.3-5.6 | 1.39-1.43 |
| 6 | 11.9-12.9 | 8.5-9.0 | 1.40-1.44 |
| 8 | 16.9-18.1 | 11.9-12.4 | 1.42-1.46 |

Ranges span two dumps: a clean 2600-token prefill and a later one contaminated
by short benchmark prefills, which shifts D about 4%. Not material.

**Reuse is real: ~1.44.** One transfer removes 1.44 tokens' worth of CPU work,
so the transfer bill is 1.44x cheaper than the naive per-token count. That put
this box at the top of Phase 1's band, not the bottom -- a GO for the high tiers.

#### Phase 2b — the calibration file

`bench/calibrate_hybrid.py` writes `hybrid_profile.<hostname>.json`;
`bench/hybrid_policy.py` holds the decision and is shared by the calibrator, the
tests and (eventually) the serving path, so those three cannot disagree. Per
layer it minimises `max(cpu_ms(k), link_ms(k)) + extra_ms(k)` over k, which
reaches the interior optimum and both corners including k=D (CPU leaves the
layer entirely).

#### Phase 3 — mechanism, measured

| mechanism | per expert | achieved | verdict |
|---|---:|---:|---|
| **A. UVA gather kernel in-graph** | **0.197 ms** | 51.7 GB/s | **chosen** |
| B. host staging + captured H2D | 0.629 ms | 16.2 GB/s | 3.2x worse, doubles host traffic |
| GPU repack of one expert | 0.037 ms | — | upper bound (random permutation) |

A picks *different experts after capture* -- verified bitwise against the source
window. That is the entire requirement for a decode graph and it holds.

**The pinned-memory blocker dissolved.** The gather can only read page-locked
memory, and kt's non-resident store is ~222 GB, far past what a copy could pin.
A pinned *window* was the fallback, and the routing dump prices it honestly:

| window | pinned RAM | demand covered |
|---:|---:|---:|
| 8/layer | 11.4 GB | 40.8% |
| 16/layer | 22.7 GB | 55.5% |
| 32/layer | 45.5 GB | 72.6% |
| 64/layer | 90.9 GB | 89.7% |

But `cudaHostRegister` page-locks pageable memory **in place**: 4 GB in 0.50 s
(8 GB/s), and a gather from it is correct and full-speed (0.198 ms/expert). So
kt's store gets locked where it already lives -- ~28 s of boot for 222 GB, no
extra RAM, and **100% coverage instead of 55%**. No window, no background
refill, no coverage discount.

#### What the measured mechanism does to the payoff

Folding in the real costs (0.197 gather + 0.037 repack, not an idealised 0.185
DMA) lowers the prediction:

| mode | CPU ms/layer | stream k | step ms | -> | speedup |
|---|---:|---:|---:|---:|---:|
| safe2 (default) | 0.177 | **0** | 66.82 | 66.82 | **1.000x** |
| safe4 | 0.474 | 1 | 89.11 | 86.47 | 1.031x |
| safe6 | 0.811 | 3 | 114.39 | 105.88 | 1.080x |
| safe8 | 1.263 | 4 | 148.33 | 132.44 | **1.120x** |

safe2 needs 0.232 ms/layer of CPU work before one transfer pays and has 0.177.

#### Portability — the reason to keep the code

`bench/test_hybrid_policy.py`. Same code, constants from a different machine:

| machine | k | speedup |
|---|---:|---:|
| this box, safe8 | 4 | 1.12x |
| this box, safe2 | 0 | 1.00x |
| GH200-class link (900 GB/s C2C), safe8 | 11 | ~1.54x |
| GH200-class link, at the tier this box refuses | 2 | 1.07x |
| PCIe Gen3 x8, safe8 | 0 | 1.00x |
| small-VRAM box (bigger CPU pole) | 10 | ~1.47x |

k is monotonic in link bandwidth (0,1,2,4,7,10,11 across 6→900 GB/s) and in the
CPU pole; a zero-bandwidth or missing profile falls back to all-CPU.

### Phase 4-5 status — built to the gate, not through it

Done and committed: the policy, the calibration file, the measured mechanism,
the in-place page-locking result, and the portability tests. **Not built: the
serving-path implementation.** Two pieces remain, and neither is blocked by
anything unknown any more:

1. **A real cutlass W4A8 repack kernel.** The 0.037 ms is a random-permutation
   upper bound, not the actual interleave. The kernel has to reproduce
   `process_weights_after_loading`'s output bitwise for one expert.
2. **Plumbing.** In-graph per-layer k and index computation, marking streamed
   experts as -1 so kt's `should_skip_expert` drops them, landing slots feeding
   the existing GPU MoE, and the model-wide corner that swaps to the no-CPU
   graph when every layer returns k=m.

**The gate says stop here on this machine.** Phase 5's own rule is "under ~10%
end-to-end gets written up and parked". The measured prediction is 1.120x at
safe8 -- and **1.000x at safe2, the shipped default**. Building both remaining
pieces buys nothing at the tier that actually serves traffic, and 12% at a tier
that is 2x slower to begin with.

That verdict is machine-specific, which is exactly why the calibration exists.
On a GH200-class host the same profile says stream 11 of 12.4 experts for ~1.54x
at safe8 *and* a win at the low tier, so the remaining two pieces are worth
building the moment this runs somewhere with a fatter host-to-device link. Run
`bench/calibrate_hybrid.py` there and it will say so in its own numbers.

## 3. Risks

| risk | why it matters | first check |
|---|---|---|
| Forced one-task scheduling | every tiny output shard contends on the same atomic | Stage 0B guided-chunk A/B |
| Shared activation cache crosses NUMA | remote reads can erase the saved quantization work | Stage 0C per-node timing |
| Pre-quantization changes a BF16 or rounding boundary | a speedup is invalid if any CPU output bit changes | Stage 0C bitwise matrix |
| Callback optimization serializes CPU and GPU | merging the two graph positions destroys their overlap | Stage 0D timeline and soak |
| Empty-route fast path skips real work | mixed routing must never select the no-CPU path | Stage 0E alternating-route tests |
| DMA and CPU experts contend for RAM | the entire streaming headline assumes they add | ✅ Phase 1b: they do add — PCIe keeps 96.7% under load. The real cost is head-of-line blocking of kt's per-layer round-trips, priced at 1.07 ms/GB at queue depth 1 |
| Bulk copies delay kt's own per-layer transfers | 64 MB of queued copies costs +31.8% decode for the same bandwidth | ✅ Phase 1b: submit shallow (queue depth 1) → +12.6%; carry into Phase 3 |
| Layout conversion cost per streamed expert | 136 ms whole-layer interleave is fatal per step | ⚠️ Phase 3: a GPU-side repack is bounded at 0.037 ms/expert, but the real cutlass kernel is still unwritten — the one remaining unknown |
| Pinned window vs 7 GB free RAM | pinned pages cannot swap; the box already OOM-cratered once | ✅ Phase 3: no window needed — cudaHostRegister locks kt's store in place (8 GB/s, no extra RAM, 100% coverage vs a 22.7 GB window's 55%) |
| Streaming on the critical path | layer L+1's routing needs layer L's output, so there is no prefetch distance | measure serial cost in Phase 3 |
| Convex CPU slope | the split point moves with load; a static formula may sit off-optimum | Stage 0F calibration captures the curve |
