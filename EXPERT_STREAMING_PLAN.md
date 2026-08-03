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
128 ms of marginal cost. Phase 0a confirms it directly.

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
dropping experts. sub2 stays the headline benchmark as chosen, but every phase
reports safe8 alongside it, because that is where the mechanism actually pays.

---

## 2. Phases

### Phase 0 — finish the baseline (no new code paths)

- **0a. Isolate the fixed submit/sync cost.** One restart with `/tmp/kt_skip_cpu`
  armed, rerun the tier sweep. `step_ms(K) − step_ms_noCPU(K)` is the CPU path's
  true contribution at each K; the K=0 difference is the fixed cost. This is the
  `a` in the formula and it is currently only bounded to 0–30 ms/step.
- **0b. Re-verify the headline configs** so "exceeded" means something:
  `GPU_EXPERTS=104` at sub2 (recorded 45.05) and safe8 (recorded ~17.5), via
  `bench/decode_rate.py --runs 5`, results committed to `bench/profile_out/rates/`.
  Anything that fails to reproduce gets chased before design work continues.
- **0c. Measure contention directly.** Run the H2D benchmark *while* the decode
  loop is running. The plan's headline number assumes DMA and CPU expert compute
  can share RAM; that assumption is currently untested and is the single biggest
  way the estimate could be wrong.

Gate: if 0c shows PCIe collapses under CPU load, the whole design is capped and
we say so before building it.

### Phase 1 — offline calibration file

`bench/calibrate_hybrid.py` → `bench/profile_out/hybrid_profile.<hostname>.json`,
read by the server at boot, never measured in-process (chosen: offline only).

Constants:
- `pcie_gbs[gpu]` — pinned H2D per card, and the same figure under CPU load (0c)
- `cpu_ms_per_expert_token` — from the tier sweep slope, per layer
- `cpu_fixed_ms_per_layer` — from 0a
- `gpu_ms_per_expert` — GPU MoE marginal cost per streamed expert
- `bytes_per_expert_per_card` — from the model config, not assumed
- `ram_gbs_peak`, `ram_gbs_cpu_path` — the shared budget both poles draw from

A missing or stale-hashed file falls back to today's all-CPU behaviour rather
than guessing.

### Phase 2 — transfer mechanism spike (both, then pick)

Standalone harness, no serving path, measuring achieved GB/s **under CUDA graph
replay with the CPU expert path running concurrently**.

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

### Phase 3 — the split policy and plumbing

- Per-layer decision computed in-graph and branch-free, in the style of the
  existing `keep_k` kernel in [deepseek_v2.py](.venv/lib/python3.12/site-packages/sglang/srt/models/deepseek_v2.py) `_kt_topk_experiment`.
- Solve `min over k of max(a + b·(m−k), c + k·bytes/BW + gpu(k))` per layer, where
  `m` is the layer's non-resident routed count. Because `a > 0`, the optimum is a
  **corner** whenever `m` is small — the formula must be allowed to return k=m and
  drop the CPU submit for that layer, not just interior splits.
- CPU set keeps the existing kt path (streamed experts masked to −1, which the
  C++ `should_skip_expert` already handles safely).
- Stream set lands in stable VRAM slots and joins the existing GPU MoE.
- Model-wide corner: if every layer returns k=m, use the `KT_SKIP_CPU_PATH` graph
  variant so the CPU leaves the loop entirely.

VRAM: landing slots are cheap — 8 double-buffered slots × 9.7 MB ≈ 78 MB/card
against 1.5 GB currently free. Not a blocker at `GPU_EXPERTS=60`; re-check at 104.

### Phase 4 — verification

The point of the experiment is to **exceed** the Phase-0 numbers, so:

- `bench/decode_rate.py` and `bench/cpu_fixed_cost.py` rerun at sub2 (headline)
  and safe8 (where the mechanism pays), same prompts, ≥5 runs, interleaved.
- Report ms/step *and* accept length together — [[glm52-tier-movement-synchronous]]
  showed ~85% of a movement scheme's cost can hide in accept length while step
  rate looks fine.
- Quality: `bench/livebench` + `DOC_QA_RUNBOOK.md` at safe8, which must be
  bit-comparable to the all-CPU path since streaming changes *where* an expert is
  computed, not *which*.
- Failure is a real outcome: if the measured gain is under ~10% end-to-end the
  branch gets written up and parked, not tuned indefinitely.

---

## 3. Risks

| risk | why it matters | first check |
|---|---|---|
| DMA and CPU experts contend for RAM | the entire headline assumes they add | Phase 0c, before any building |
| Layout conversion cost per streamed expert | 136 ms whole-layer interleave is fatal per step | Phase 2(ii) |
| Pinned window vs 7 GB free RAM | pinned pages cannot swap; the box already OOM-cratered once | size window in Phase 1 |
| Streaming on the critical path | layer L+1's routing needs layer L's output, so there is no prefetch distance | measure serial cost in Phase 2 |
| Convex CPU slope | the split point moves with load; a static formula may sit off-optimum | calibration captures the curve, not one slope |
