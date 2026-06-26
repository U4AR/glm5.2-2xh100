# Serving a 754B model on two GPUs: how we got GLM-5.2 from 3.4 to 13.4 tok/s

*An engineering log of running GLM‑5.2 (754B MoE) on a 2×H100 box with a CPU that
has no AMX — and the custom 4‑bit kernel that ended up beating everything else.*

---

## The setup

We wanted to serve **GLM‑5.2**, a 754B‑parameter Mixture‑of‑Experts model
(`GlmMoeDsaForCausalLM`: 78 layers, 256 routed experts/layer, 8 active per token,
MLA + Native Sparse Attention), on a single modest box:

- **2× H100 NVL** (96 GB each, 192 GB total VRAM), TP2.
- **AMD EPYC 9V84** (Zen4 Genoa), 80 cores, 2 NUMA nodes, **629 GB RAM**.
  Crucially: AVX‑512 (F/BW/VNNI/BF16/VBMI) but **no Intel AMX**, and no 256‑bit
  `avx_vnni` either.
- NVMe scratch disks.

The serving stack is **SGLang + KTransformers (kt‑kernel)**. The trick that makes a
754B model fit at all is *heterogeneous MoE*: a slice of each layer's experts lives
on the GPUs (fast), and the rest live in CPU DRAM and are computed by hand‑written
AVX‑512 kernels. Per decode step, the per‑layer latency is roughly
`max(CPU‑expert time, GPU‑expert + attention time)` — the two run concurrently and
get merged.

This is the story of four wins that took decode throughput from a barely‑usable
3.4 tok/s to **13.4 tok/s** — a **~3.9×** improvement — with no change to the model
weights' quality beyond quantization.

---

## Win #1 — Make it run at all: NVMe swap + memory fraction

The first wall was simply *fitting*. In FP8 the routed experts are ~684 GB; the box
has 629 GB of RAM. With no swap, the CPU‑expert build OOM‑killed the rank‑0 worker
around layer 68 of 78.

The fix was unglamorous but decisive:

1. **384 GB of NVMe swap** (256 GB + 128 GB across two NVMe drives) with
   `vm.swappiness=10`. The hot working set stays resident in RAM; cold LRU experts
   spill to NVMe. Steady‑state swap‑out stays near zero — the kernel drops page
   cache instead of thrashing.
2. **`--mem-fraction-static 0.94–0.95`**. After placing the GPU experts, the KV pool
   had no room at 0.90 and the server refused to start ("Not enough memory"). Nudging
   the GPU memory budget up gave the KV cache its headroom.

Result: GLM‑5.2‑FP8 served end‑to‑end for the first time — at **~3.4 tok/s**. Slow,
but alive. Everything after this is about speed.

---

## Win #2 — CUDA graphs: 3.4 → 8.7 tok/s (~2.6×)

The obvious hypothesis for "3.4 tok/s is slow" was *the model is swapping* or
*the CPU experts are compute‑bound*. Both were wrong.

Profiling showed the CPU worker threads spending ~72% of their time in
`clock_gettime` — i.e. **spin‑waiting**, not computing. The real bottleneck was
**per‑step launch and serialization overhead**: at batch size 1, decode fires
thousands of tiny kernel launches per token, and the Python/host overhead between
them dominated.

The fix is the standard one for low‑batch LLM decode — **CUDA graphs** — but it
took three coupled changes to capture cleanly on this stack:

- `--mem-fraction-static 0.94` so the graph capture reserve fits inside the budget,
- DeepGEMM enabled (the NSA indexer's dual‑stream path needs it under graphs),
- `--disable-custom-all-reduce` (the custom all‑reduce doesn't capture here).

Enabling graphs collapsed the per‑layer launch storm into a single replay and
**~2.6×'d decode to ~8.7 tok/s**, with no model change. This became the baseline
everything else built on.

> **Lesson:** at batch size 1, decode is almost always launch‑bound before it is
> compute‑bound. Measure where the time actually goes before optimizing the math.

---

## Win #3 — 4‑bit experts on the GPU: 8.7 → 10.6 tok/s

Decode was now bound by the **CPU‑expert path** (the GPUs sat ~50% idle waiting for
it). The lever: move *more* experts onto the GPU. But FP8 experts are big (36 MB
each), so only ~48/layer fit per card alongside the KV cache and CUDA graphs.

**INT4 experts are half the size** → roughly twice as many fit on the GPU → fewer
experts fall to the slow CPU path. SGLang already ships a GPU INT4 expert kernel —
the **W4AFP8 cutlass** path (4‑bit weights, FP8 activations) that Phala runs on 8‑GPU
deployments. In our heterogeneous CPU+GPU integration, though, it produced pure
**garbage**.

The debugging took a while, but the root cause was a one‑liner. kt‑kernel marks
CPU‑resident experts with `-1` in `topk_ids` (so the GPU kernel skips them). The
cutlass kernel *does* handle that sentinel — but only remaps `-1 → num_local_experts`
when the expert‑parallel world size is `> 1`. We run `ep_size = 1`, so the `-1` fell
straight through, indexed out of bounds, and corrupted the GPU MoE into NaNs.

The fix, in `W4AFp8MoEMethod.apply`:

```python
topk_ids = torch.where(topk_ids == -1, num_local_experts, topk_ids)
```

With that (plus an idempotency guard on weight post‑processing), the W4AFP8 GPU
experts produced coherent output. Now ~104 INT4 experts/layer fit per card (88 GB),
more than double the FP8 count. Decode rose to **10.1–10.6 tok/s**.

> **Lesson:** "the kernel is broken" is usually "the kernel is being fed wrong." An
> offline single‑layer harness that compares the real kernel against a reference
> dequant‑matmul (cos‑sim) localized this in minutes instead of 5‑minute server
> reboots.

---

## Win #4 — A custom packed‑INT4 CPU kernel: 10.9 → 13.4 tok/s (+23%)

The GPUs were now well‑fed, and the **CPU‑expert window was again the long pole**.
Could we make the CPU experts themselves faster?

### The roofline said yes

We measured DRAM bandwidth on the live server using AMD's demand‑fill counters, at
**10 ms bins** (this matters — see below). During the CPU‑expert active window,
bandwidth **burst to 140–220 GB/s**, against a measured achievable ceiling of
~170 GB/s on this VM. In other words, **the CPU‑expert path is memory‑bandwidth
bound in its active window**, not compute bound.

> A trap worth flagging: at 100 ms bins (≈ one decode step) the same workload looks
> like a flat ~78 GB/s and fools you into thinking there's bandwidth to spare. That
> average mixes the bandwidth‑saturated compute window with the GPU‑only/spin part of
> the step. **Always measure at fine bins for bursty workloads.**

If the path is memory‑bound, then **fewer weight bytes = proportionally faster**.
INT4 is half the bytes of FP8 — a real ~2× of headroom.

### The catch: the existing INT4 CPU kernel threw the win away

The existing INT4 CPU kernel **pre‑expanded** the packed 4‑bit weights into int8 at
load time. So at inference it read **1 byte/weight — exactly like FP8**. All the
memory advantage of INT4 was gone before the first token. That's why INT4‑on‑CPU was
*losing* to FP8‑on‑CPU (~5.7 vs 8.7 tok/s).

### The fix: keep the weights packed, unpack in‑register

We wrote a new CPU MoE kernel that keeps weights **packed at 4 bits in DRAM**
(0.5 byte/weight) and unpacks the nibbles **in‑register**, immediately before the
`VPDPBUSD` (the u8×s8 dot‑product instruction). The activations stay int8 (W4A8);
all the memory saving is on the weight side, which is the entire footprint at batch
size 1.

Three things made this land — and each was a trap the handoff notes got *wrong*:

1. **The right ISA.** The plan said to use 256‑bit `AVX‑VNNI`
   (`_mm256_dpbusd_avx_epi32`). This Zen4 chip **does not have `avx_vnni`** — that
   instruction `SIGILL`s here. It *does* have AVX‑512 VNNI, so we used the EVEX form
   `_mm256_dpbusd_epi32` (256‑bit, but gated on `avx512vl+avx512vnni`). The packing
   win is independent of SIMD width — it's about bytes moved, not lanes — so 256‑bit
   was plenty and kept the nibble de‑interleave inside 128‑bit lanes (no cross‑lane
   permute, obviously correct).
2. **Two's‑complement, not nibble‑bias.** The 4‑bit values are signed two's‑complement
   (`−8..7`), *not* the `(nibble ^ 8) − 8` biased encoding an earlier note assumed.
   Getting this wrong produces high‑but‑not‑perfect cos‑sim — the tell‑tale of a
   sign/ordering bug. We validated nibble de‑interleave offline (cos‑sim ≥ 0.99997)
   before ever booting the server.
3. **Copy into the TP buffers.** With TP2, a zero‑copy weight handoff silently
   collapsed to TP1 and segfaulted inside the CUDA graph. The FP8 kernel copies into
   per‑TP buffers; the new kernel had to as well.

Build gotcha for posterity: kt‑kernel's `setup.py` auto‑detects CUDA via `nvcc`, but
a bare `cmake` reconfigure builds a **CPU‑only** extension that silently drops
`submit_with_cuda_stream` (the multi‑stream overlap entry point) and breaks the
server. Always build via `install.sh build` with `CPUINFER_USE_CUDA=1`.

Result: coherent output, and decode rose from 10.9 to **13.41 tok/s — +23%** — while
*also halving* the CPU‑side expert RAM. INT4 weights at 4 bits in DRAM means the CPU
experts now fit in ~220 GB of RAM instead of needing the full 629 GB + swap.

---

## Where it landed

| Stage | Config | Decode (tok/s) |
|---|---:|---:|
| FP8, NVMe swap, eager | first time it ran | ~3.4 |
| FP8 + CUDA graphs | the launch‑bound fix | ~8.7 |
| INT4 GPU experts (W4AFP8) + FP8 CPU | more experts on GPU | ~10.6 |
| **Packed INT4 GPU + packed INT4 CPU** | **custom AVX‑512 VNNI kernel** | **13.4** |

A **~3.9×** end‑to‑end speedup, and the model now fits in roughly **260 GB of RAM**
plus 2×H100 instead of needing every byte of a 629 GB box.

## Things that didn't pan out (and why)

Not every lever moved the needle — worth recording so nobody re‑runs them:

- **MTP / speculative decode (NEXTN).** The draft model runs fine pure‑GPU and the
  draft quality is excellent (acceptance ~2.0 in eager mode). But the spec **verify**
  forward only stays coherent in eager mode; under CUDA graphs it corrupts, because
  the kt heterogeneous CPU‑expert multi‑stream submit doesn't capture/replay cleanly
  inside the verify graph. Eager + spec is correct but slower than the 13.4 baseline;
  graphs + spec is fast but garbled. No configuration both beat the baseline *and*
  stayed coherent without deep kt‑kernel surgery, and the box is CPU‑bound anyway, so
  the ceiling on the win was low. Parked.
- **Rebalancing GPU experts to fix a "GPU imbalance."** GPU1 shows 100% utilization,
  but that's NCCL all‑reduce spin‑wait, not real work — the CPU experts only run on
  TP rank 0. Rebalancing GPU‑side experts doesn't help; the rank‑0 CPU path is the
  true critical path.

## Takeaways

1. **Profile before you optimize.** Two of the four wins (CUDA graphs, the packed
   kernel) came from discovering the real bottleneck was *not* what it looked like —
   spin‑wait masquerading as compute, and a coarse‑grained average hiding a
   bandwidth‑saturated burst.
2. **For memory‑bound kernels, bytes are the only currency.** Halving the weight
   width (keeping it packed) was worth more than any SIMD‑width cleverness.
3. **Hardware specifics dominate.** No AMX, no `avx_vnni`, a bandwidth‑limited VM
   slice — every one of these invalidated a "standard" recipe and forced the actual
   solution.
4. **Cheap offline harnesses pay for themselves.** Single‑layer cos‑sim checks turned
   5‑minute reboot debugging loops into seconds and caught sign/ordering bugs before
   they ever reached the server.

*All code, launch scripts, the custom kernel, and exact run instructions are in this
repository.*
