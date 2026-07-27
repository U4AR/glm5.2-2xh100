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

## 4. Open

- Re-measure RAM=64/16/8 under `fill=gpu` for both speed and accuracy.
- Confirm RAM=160 `fill=gpu` reproduces the ~40.5 headline on `decbench.py`
  (running: `run_baseline_check.sh`, both harnesses, strictly sequential).
- Next-token / 4-token residency instrumentation (`kt_energy_report`) — see §5
  once the energy sweep completes.
