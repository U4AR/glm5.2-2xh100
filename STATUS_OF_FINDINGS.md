# STATUS OF FINDINGS — what is solid, what is provisional, what was refuted

This archive contains ~30 write-ups accumulated over ~7 weeks of experiments. They
were written *as the work happened*, so a document's claim is often superseded by
a later document, and sometimes by a later paragraph of the same document. This
file is the index of record. **Where this file and a `.md` write-up disagree, this
file wins.**

Three grades:

- **[SOLID]** — reproduced, and the mechanism is understood.
- **[PROVISIONAL]** — measured, but the number is fragile: one boot, an unvalidated
  instrument, an upper bound, or a mechanism that is described but not proven.
- **[REFUTED]** — believed and written down, then shown wrong. Left here on purpose.

Speeds are single-stream decode on 2×H100 + Zen4 host unless stated.

---

## [SOLID] — the things that actually made it fast

| Finding | Number | Where |
|---|---|---|
| The decode bottleneck was per-step launch/serialisation, not compute or NVMe swap. CUDA graphs fixed it. | 3.3 → 8.7 tok/s | `PERF_CUDA_GRAPHS.md` |
| Packed RAWINT4 W4A8 CPU MoE kernel beats the FP8 CPU path. | 10.9 → 13.4 tok/s | `PACKED_INT4_KERNEL_HANDOFF.md` |
| int4 GPU experts are coherent once `topk_ids == -1` is remapped at `ep_size=1`. | 10.1–10.6 tok/s | `INT4_PROJECT.md` |
| W4AFP8 GPU bulk prefill. Decode unchanged, bit-coherent. | ~1.9× TTFT | `BLOG.md` |
| **Top-K expert substitution** — keep the true top-K, substitute the tail with the best *resident* experts. The single largest lever in the project. | keep=4 → 1.25×, keep=2 → 1.5× | `BLOG_TOP2_EXPERTS.md` |
| MTP/NEXTN works under CUDA graphs once the verify-batch size is registered for capture. | keep=2 + depth-3 → **34 tok/s** | `BLOG_MTP_CUDAGRAPH.md` |
| Per-request intelligence tier: `<base>-topN` in the OpenAI `model` field, mixed-batch, one CUDA graph, no recapture. | top8 parity / top2 39.8 | `BLOG_INTELLIGENCE_TIER.md` |
| NSA sparse attention works; GLM-5.2 shares one indexer per 4 layers (IndexShare). | −4.1 GB/card | commit `edd4e1e` |
| MTP long-context corruption was NSA multistep drafting giving every spec step step-0 metadata. | correct to 533k ctx | `TODO_ROOT_CAUSES.md` |
| One GPU is a supported target (TP1, 24 GPU experts). | 23.74 tok/s, 66/66 acc | commit `3b71355` |
| Decode-time adaptive expert cache, staging all 256 experts + benefit-gated swaps. | 28.2 → 38–40 tok/s | `experiments/adaptive_expert_cache/` |
| GPU-only resident experts, prefetch off. | 53.48 ms/step, coherent | `experiments/` |
| **The decode is priced by launch count, not by FLOPs or bandwidth.** CPU cores 94% busy but int8 at 2% and DRAM at 38% of 379 GB/s. | — | `glm52-occupancy-profile` |
| At the shipped tier the CPU expert path is only ~20% of the step (top8 150.1 / top2 66.6 / top0 53.1 ms). Substitution had already removed 86% of it. **This is why prefetch cannot win.** | — | `TODO_ROOT_CAUSES.md` |

## [PROVISIONAL] — measured, but do not quote without the caveat

- **LiveBench reasoning 73.3% (n=60)** at hybrid top8 — but **97.7% of the 44 that
  finished**. 16 hit the 8000-token cap, so the headline is a *finish rate*, not a
  reasoning score. The accuracy ladder across tiers was never completed.
- **Low-VRAM ladder** (N=4…96, 19.9→54 GiB): footprint numbers are real, tok/s is an
  **upper bound** — measured on full H100 compute. The 2×24 GiB projection (~10–12 tok/s
  on 2×4090) is a bandwidth model, not a measurement. N=0 is unsupported (empty-resident crash).
- **Static oracle placement ceiling ~1.15–1.35×.** Built and measured, but it *floors*
  around 22–23 tok/s no matter the coverage (69%→92% barely moves it), which is
  explained but not proven: fixed CPU submit+sync cost plus `max(cpu, gpu)` overlap.
- **Adaptive placement cache floors at +8%** (14.2→15.3). Same explanation, same caveat.
- **Prefetch is now a −2.1 to −2.5 ms *saving*** at both prediction points — the first
  time it came out ahead. Still not a net win, because the predictor costs 5.5 ms.
  ⚠️ **Every prefetch number taken at `MEM_FRACTION=0.95` is suspect.**
- **`KT_PRED_LAYER_STRIDE=4` buys 2.12 ms**, cache+prefetch both live at 54.51 ms /
  50.86 tok/s against a 53.48 ms floor. Cross-boot noise here is ~0.7 ms, so this is
  ~3σ, not comfortable. Cache and prefetch **fight**: reuse needs stable residency.
- **Prediction accuracy 80.6% one layer ahead / 70.1% two.** Whole-layer coverage 75.0%
  *over the layers that need anything* — 53.8% of calls need nothing and were being
  scored as failures. Prediction is not the bottleneck; the link is.
- **Expert streaming over PCIe** (`EXPERT_STREAMING_PLAN.md`): the gate passes (96.5%),
  UVA in-graph gather beats staging 3.2×, `cudaHostRegister` pins kt's 222 GB store in
  place. Payoff is 1.12× at safe8 and 1.00× at safe2 → **parked, not shipped**. A
  GH200-class link would give ~1.54×; PCIe is ~6× too slow.
- **Gather and cutlass GEMMs compete for SMs** — a U-shape at identical bytes, marginal
  cost 2.4→8.6 ms/expert. Mechanism is a slope fit on one sweep.
- **`walk` = 17.45 ms/step** decomposes exactly into substitution search 6.55 ms +
  grouped-MoE setup 6.61 ms. Solid arithmetic; the *fix* was never built.

## [REFUTED] — believed, written down, then disproved

Left in deliberately. Several of these cost days.

- ~~"CPU/GPU overlap needs to be built."~~ It was **already implemented** in
  `kt_ep_wrapper.apply()` (async submit + concurrent GPU + sync). Per-layer cost is
  already `max(cpu, gpu)`.
- ~~"GPU imbalance is lever #1."~~ Red herring. CPU experts run only on `tp_rank0`, so
  GPU1's 100% is NCCL spin-wait, not work.
- ~~"MTP is a wash (~8.8 ≈ baseline)."~~ Superseded — MTP works under CUDA graphs and
  gives +55%. The wash was the missing capture registration.
- ~~"A buggy fused NSA metadata-copy kernel is the problem."~~ The kernel is **never
  called** at `--speculative-num-steps 1`. Wrong root cause; the flag was a no-op.
- ~~"NSA is broken because the topk kernels assert 2048."~~ Wrong. The real cause was
  IndexShare. `DISABLE_NSA=1` was a workaround for a misdiagnosis, and is superseded.
- ~~"Chain prediction beats the direct predictor by 3.1 ms at d=1."~~ **Retracted** — the
  gap came from subtracting across two different days' boots. At d=1 the walk is worth
  nothing; `post` is free and ties it.
- ~~`chain_drop`~~ (saves 3.01 ms but 6 coverage points worse — below doing nothing) and
  ~~`prevlayer`~~ (chance floor in every regime).
- ~~The SSD expert-tier ladder.~~ **VOID.** One CPU/CUDA mask defect silently killed
  tiering, the adaptive cache, and all tier movement — in four separate places. Any
  pairing of static tiers with adaptive results from that period is invalid.
- ~~"Prefetch is worth 1.23×."~~ Became 1.10× when the bytes that actually move were
  counted, then dropped further when the superset was priced against the link.
- ~~"`sub2` should be a default."~~ Removed twice, most recently in `9a5f81b`. It buys
  3.7%. The famous 40.5 / 42 tok/s headlines were **safe** numbers, not sub2 numbers.
- ~~"Nucleus routing gives per-layer adaptive N."~~ The GLM router is flat (p=0.9 → ~7 of
  8 experts), so nucleus cannot beat top-2. The begin/mid/end hypothesis is not supported.
- ~~"Expert 0 can be prefetched."~~ Defect: prefetched experts were routed to expert 0
  rather than their landing slot (`7d47156`).

## Methodology traps that produced false results here

Recorded because each one faked a believable number:

1. **Silently inert flags.** `KT_PRED_FUSED_STACK` and the layer stride were implemented
   only on a code path that is off by default. The test reproduced the *unstrided* number
   to 0.26 ms and read as a clean null result.
2. **Cross-boot subtraction.** Baselines agreeing across two boots licenses **nothing**
   about the rows beside them. Subtract only within a run.
3. **A runaway process.** A stray `ugrep -r /` at 1154% CPU for 5.8 h cratered decode
   (14 → 0.77 tok/s) and produced a confident "too slow" verdict. Check `ps --sort=-pcpu`
   whenever performance drops *globally*.
4. **A multi-GB allocation while profiling** on a box with ~7 GB free (TTFT 1.3 → 29.6 s).
   Probe, then let the machine settle, then measure.
5. **Coherence detection.** Use 1200 tokens, not 200, and score at the **character** level
   — `</think>` × 400 contains two spaces and fooled a word-level detector twice. The gate:
   `top0` must score degenerate on the same boot, or the run proves nothing.
6. **Accept length is not a quality proxy** — and its sign was backwards for a while.
   Also: never use tok/s at low reachability; use ms/step, because a degenerate `top0`
   inflates accept length.
7. **Tests that share the code's mental model** prove self-consistency only. A gather
   verified against an older gather, and a test asserting the kernel's own wrong
   convention, both passed while broken.
8. Harness landmines: NEXTN draft-capture, `curl -s` exits 0 on HTTP 503, a multi-line
   curl JSON body returns 400, and an inline `pkill` kills its own shell.

## Open, at the point of archiving

From `TODO_ROOT_CAUSES.md` — 7 items were still open:

1. The direct predictor's 5.03 ms/step: split, but not explained.
2. The gather is 61% hidden — what is the other 39%?
3. Why the box OOMs at `GPU_EXPERTS=100` + landing slots: bounded, not explained.
4. The 52.68 ms floor is the real target; the shipped config sits at 54.51.
5. The comparison that decides whether prefetch ships was **never run**.
6. Prediction accuracy at depth 1, 2, … — interrupted mid-sweep.
7. The accuracy ladder across tiers — only the top8 reference row exists.
