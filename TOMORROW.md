# Tomorrow — start here (written 2026-08-07, end of day)

Box is **shut down clean**: no servers, both GPUs at 4 MiB, `/dev/shm` clear.
Branch `experiment/expert-streaming-hybrid`. Nothing is committed — see
"State of the tree" below before doing anything destructive.

Bring the server back with:

```bash
cd /data/models/RunGLM
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache ./run_fast.sh
```

Shut it down again with `bash bench/_shutdown.sh` (kill patterns live inside that
file on purpose — see the warning at the bottom of this doc).

---

## What actually happened today

Two real defects were found and fixed. The headline changed three times, and the
first two headlines were mine and wrong; both are recorded so they are not
rediscovered.

### 1. FIXED — the fused predictor routed to the WRONG EXPERTS

`pf_index` is merged with `logical_to_gpu_index` in `apply()`, and that table
holds ABSOLUTE indices into the layer's `num_gpu_experts + slots` cutlass tensor.
The landing slots are its trailing entries (100..103). The fused kernel wrote the
RELATIVE slot number (`index[pick] = k`, i.e. 0..3), so **every prefetched expert
was computed as GPU expert 0..3** — four real, resident, unrelated experts.

Not corrupted memory: a valid expert, the wrong one. Output stayed fluent,
quality quietly dropped, deterministic every run.

Fix: `slot_base` threaded into `k_pred_fused`, `index[pick] = slot_base + k`.
`bench/pred_fused_kernel.py` line ~251, and the caller passes
`self._pf_slot_base` in `pf_issue_fused`.

**Why 150 verify trials passed anyway:** the test asserted `kindex[e] == k` — the
same relative convention the kernel used. Kernel and test shared one
misunderstanding. The assertion now checks `slot_base + k` and `--slot-base`
defaults to 100, not 0, so a relative index cannot pass silently.

### 2. FIXED — stale prefetch masks corrupted PREFILL

`pf_landed` / `pf_landed_cpu` / `pf_index` are written only by the predictor,
which only fires on a decode-shaped batch (`T <= KT_PRED_TMAX`, 8). Nothing
cleared them, but both consumers ran on every forward. A ~60-token prefill is not
decode-shaped, so it read **the previous request's** masks: experts routed to
slots holding another expert's weights, CPU told to skip experts nothing
computed. Every prompt's KV cache built wrong, differently each time.

Fix: `pf_issued` is latched and cleared at the top of `apply()` before any early
return, and both consumers require it (`_KT_PREFETCH_FRESH`, default on; set to
`0` to reproduce the defect).

Real bug, worth fixing, but it was **not** the cause of the accept-length drop —
that was #1.

### 3. REFUTED (twice, by me)

- "The accept drop is CPU-vs-GPU numerics." No: numerics are deterministic, and
  the prefetch gave five different completions for one greedy prompt.
- "The gathered weights are wrong." No: `bench/gather_vs_checkpoint.py` diffs a
  gathered expert against the checkpoint and reports **0 differing bytes** on all
  four tensors, both ranks, layers 3 and 40 — with a resident expert in the same
  dump validating the offline reader first. The gather was always correct.

---

## Where the numbers stand

One boot, `GPU_EXPERTS=100`, `MEM_FRACTION=0.94`, safe2, MTP depth-3, tier top2.

| | ms/step | accept | tok/s |
|---|---|---|---|
| baseline | 65.68 | 2.857 | 43.50 |
| prefetch, fixed | **64.35** | 2.857 | **44.40** |

**+2.1%.** First time the prefetch has ever been ahead.

Anatomy (`bench/step_anatomy.sh`, sums exactly):

```
baseline  66.07 = 52.68 GPU floor + 13.39 CPU experts exposed
prefetch  63.89 = 52.68 GPU floor + 1.76 predictor + 9.24 gather + 0.21 residual
```

- Tier sweep: top8 149.48 / top2 66.07 / top0 52.68 (prefetch off).
- **Prefetch overhead measured cleanly at top0** (where it can save nothing):
  **+11.00 ms**. It spends 11.0 to buy back 13.2.
- Gather = **24.20 ms/step of GPU kernel time**, ~62% hidden. `k_w13_weights`
  alone is 13.25 ms/step — the largest single kernel in the trace.
- ~1.3 GB moved per rank per step at ~55 GB/s. That is PCIe link speed; the
  gather is not badly written, it is bandwidth-bound.
- Report: `bench/profile_out/step_anatomy.html` (self-contained, open directly).

---

## Pick up here

### A. The comparison that decides shipping — NOT YET RUN
`bench/fair_baseline.sh` is written and ready, never launched.

Everything above compares `GPU_EXPERTS=100 + 4 slots` against `GPU_EXPERTS=100`
with those slots **allocated but idle**. Equal VRAM, correct for "does the
machinery pay" — but production runs **104 resident and no slots**, and a landing
slot costs a resident expert (~9.73 MiB/card/layer). So the honest question is
whether four experts of prefetching beats four experts of residency.

```bash
bash bench/fair_baseline.sh     # F104-plain / F100-idle / F100-fetch
```

Expectation: it eats into the +2.1% but probably not all of it (placement work
found residency coverage 50→67% bought only ~8%). **That is an inference from a
different experiment, not a measurement** — two such inferences were wrong today.

### B. Prediction accuracy at depth 1, 2, ... — INTERRUPTED
`bench/chain_predict_run.sh` was mid-run at shutdown and produced nothing.

```bash
GPU_EXPERTS=100 KT_CHAIN_DEPTH=4 KT_CHAIN_P=1,2,3,4 bash bench/chain_predict_run.sh
```

Runs eager (the walk is real Python per layer, uncapturable) so it is slow; it
measures accuracy, which is a property of the trajectory not of launches.
`GPU_EXPERTS` is now overridable (was hardcoded 104) — 100 matches everything
else measured today, but the harness's built-in cross-check (the `direct` arm
landing near its recorded ~77% top-2 recall at d=1) was calibrated at 104, so
expect a shift and do not read it as a harness failure.

Known operational numbers meanwhile: fires on **74 of 75** MoE layers, **2.18**
distinct non-resident experts wanted per layer-call, **1.78** fetched, **18.8%**
of layer-calls fetch nothing. That last counter lumps "wanted more than 4 slots"
together with "needed nothing non-resident" — the log label overstates congestion.

### C. The 52.68 ms floor is the real target
82% of the step and untouched by any of this. `norm / rope / elemwise` alone is
12.29 ms/step across thousands of tiny launches — more than attention and dense
GEMM combined. Consistent with the standing finding that this decode is priced by
**launch count**, not FLOPs. Nothing here has attacked it.

### D. Smaller open items
- The prefetch is all-or-nothing per layer; 18.8% of calls fetch nothing. Whether
  partial coverage plus a shortened CPU round-trip beats nothing is unmeasured.
- `KT_PRED_POINT=post` measured worth 0.32 ms — marginal, not a lever.
- Item 10 in `TODO_ROOT_CAUSES.md` (expert 0 can never be prefetched in the
  UNFUSED `pf_issue`, via `clamp_min(0)` + last-write-wins scatter) is still
  unfixed. It does not affect the fused path.

---

## State of the tree — NOTHING IS COMMITTED

Modified, tracked:

- `.venv/.../sglang/srt/layers/moe/kt_ep_wrapper.py` — the staleness fix
  (`_KT_PREFETCH_FRESH`, latch in `apply`), `slot_base` passed to the fused
  kernel, plus `pf_dump_for_truth` / `pf_verify_against_resident` diagnostics
  behind `KT_PF_DUMP` / `KT_PREFETCH_VERIFY`.
- `EXPERT_PREFETCH_PLAN.md` — Stages H9, H10, H11 (H10's conclusion is retracted
  inside H11; read them in order).
- `TODO_ROOT_CAUSES.md` — item 8 closed with the real cause.

New, untracked (~116 files under `bench/`). The ones that matter:

| file | what it is |
|---|---|
| `bench/gather_vs_checkpoint.py` | diffs a gathered expert against the checkpoint; **validates its own reader** against a resident expert before judging |
| `bench/determinism.py` | N greedy repeats, distinct completion hashes, first divergence point |
| `bench/gather_truth.sh` | ROUTE/CPUSKIP decomposed into computed-once / omitted / double-counted |
| `bench/step_anatomy.sh` | the tier sweep + torch profiler trace behind the numbers above |
| `bench/fair_baseline.sh` | **ready, unrun** — see A |
| `bench/_shutdown.sh`, `bench/_kill_servers.sh` | safe process control |

The venv is tracked here, so the two fixes are preserved by the working tree
alone — but they are one `git checkout` away from being lost. Commit before any
branch surgery.

---

## Two traps that cost real time today

**Never put a kill pattern on the calling command line.** `pkill -f
sglang.launch_server` typed inline matches the invoking shell's own argv and
kills the caller (exit 144) — that happened four times. A queue wrapper whose
`pgrep -f "bench/determ_fix.sh"` matched its own `bash -c` string deadlocked
forever waiting for itself. Patterns live inside `bench/_shutdown.sh` and
`bench/_kill_servers.sh`; call those.

**A test written from the same mental model as the code proves nothing.** Twice
today: `gather_kernel_probe.py --verify` checked the gather byte-identical
against an *older gather*, and `pred_fused_kernel.py --verify` asserted the same
relative slot convention the kernel used. Both passed while the thing under test
was wrong. Validate against an INDEPENDENT source — the checkpoint, another
implementation, the model's own output — or do not claim validation.
