# INT4 CPU-experts project — run GLM-5.2-W4AFP8 via kt for higher decode speed

Status: **SCOPED, not started.** Created 2026-06-24. Goal chosen by user:
"use the int4 experts as a new project … using the downloaded weights try to run
them to get a higher speed."

## Why this should be faster
Decode is bottlenecked by the kt CPU-expert MoE critical path (only 48/256
routed experts fit on GPU; the rest run on the AMD EPYC CPU in **FP8** today,
~8.7–8.8 tok/s). INT4 expert weights = ~½ the bytes and ~½ the CPU compute/
bandwidth of FP8 → a genuine decode speedup (the only lever that actually attacks
the bottleneck, unlike the fused kernel — see FUSED_KERNEL_FINDINGS.md).

## The asset (already downloaded — DO NOT re-download)
`PhalaCloud/GLM-5.2-W4AFP8`, **373 GB**, at
`/cache/nvme0/models/GLM-5.2-W4AFP8` (40 safetensors shards, 48 files).
- ⚠️ **/cache/nvme0 is EPHEMERAL** (reformatted on boot). /data has only ~70 GB
  free (93% full) → CANNOT hold a 373 GB copy. /cache/nvme1 has ~3.4 TB free
  (also ephemeral, different disk). If wiped, re-download from HF. Treat the
  weights as disposable; keep this doc + the repack script as the durable assets.

## Checkpoint layout (verified by safetensors-header inspection)
Same architecture as the FP8 model: `GlmMoeDsaForCausalLM` / model_type
deepseek_v3, 78 layers, 256 routed experts, 8/tok, first_k_dense_replace=3,
1 nextn predict layer. `quantization_config.quant_method = "w4afp8"`.
- **Routed experts:** weights `I8` (INT4, 2-packed into int8), e.g.
  `gate_proj.weight [2048,3072]`, `down_proj.weight [6144,1024]`; scales
  `weight_scale_inv` BF16 group-128 (e.g. `[2048,48]`, `[6144,16]`); plus a
  single BF16 `input_scale [1]` per expert. (counts in shard0: 1009 I8 + 2018 BF16)
- **Non-expert (attention, dense MLP, shared experts):** `F8_E4M3` weights +
  `F32 weight_scale_inv` (block-FP8), and `BF16` for norms/embeddings.
  → these stay GPU-resident exactly like the current FP8 model.

So the ONLY new problem is making kt consume the INT4 routed experts.

## kt INT4 support (the crux / main risk)
`--kt-method` accepts: `AMXINT4, AMXINT8, RAWINT4, FP8, FP8_PERCHANNEL, BF16,
LLAMAFILE`. This CPU is **AMD EPYC 9V84 — AVX512 but NO AMX**, so AMX* are out.
The INT4 path for us is **`RAWINT4`** ("native INT4 weights shared by CPU and
GPU"; needs AVX512F+BW which we have, with software fallbacks for VNNI/BF16).
- ⚠️ kt docs say RAWINT4 "currently supports **Kimi-K2-Thinking** model" → the
  RAWINT4 loader likely expects K2's specific on-disk INT4 layout, which is
  probably NOT identical to this checkpoint's `w4afp8` layout. **Expect to write
  a repack** (w4afp8 int4+bf16-group128-scales+fp8-input-scale → kt RAWINT4
  layout). RAWINT4 docs: kt-kernel "Kimi-K2-Thinking-Native" tutorial + the
  Native-Precision tutorial (kvcache-ai/ktransformers repo).

## Plan (phased)
0. **Recon kt RAWINT4 loader.** Read kt-kernel's RAWINT4 expert loader + the K2
   native tutorial to learn the exact expected tensor names/shapes/scale grouping
   and how CPU/GPU experts are split. Compare to the w4afp8 layout above. Decide:
   load as-is vs repack.
1. **Try the easy path first:** point a launch at the W4AFP8 weights with
   `--kt-method RAWINT4` (and `--model-path`/`--kt-weight-path` =
   /cache/nvme0/models/GLM-5.2-W4AFP8). It will probably fail on layout mismatch
   — capture the exact loader error; it defines the repack spec.
2. **Repack script** (offline, one-time): convert routed-expert tensors to kt
   RAWINT4 format; pass non-expert FP8/BF16 tensors through unchanged. Write
   output to /cache/nvme1 (has space). Validate dequant numerically vs w4afp8.
3. **Launch + benchmark** decode tok/s vs the 8.7–8.8 FP8 baseline; tune
   GPU_EXPERTS (INT4 experts are smaller → can fit MORE on GPU for free).
4. If RAWINT4 proves too model-specific, fallback: check whether sglang's native
   `w4afp8` GPU path (W4AFp8Config, used by deepseek_v2/v4) can be married to kt
   offload, or quantize experts to a kt-native format we control.

## How to launch (adapt run_server.sh)
Current launch hardcodes `--kt-method FP8` (run_server.sh:96) and
`MODEL=/data/models/GLM-5.2-FP8` (line 8). For this project add a `KT_METHOD`
env knob and a `MODEL` override, e.g. `--kt-method RAWINT4` with MODEL pointing
at the (repacked) INT4 weights. Keep TP2, NSA backend, CUDA graphs ON.

## Baseline to beat
FP8 CPU experts: ~8.7 tok/s (no MTP) / ~8.8 (MTP, a wash). Target: meaningfully
above that from halving CPU-expert cost + fitting more experts on GPU.

## Current server
The known-good FP8 server is left running on :8000 as the baseline. Stop it
before launching the INT4 experiment (needs the VRAM).

---

# PHASE 1 RESULTS (2026-06-24) — RAWINT4 BLOCKED by a kt kernel bug

## What worked (keep this — it's correct and reusable)
1. **INT4 convention solved.** Offline cross-check of W4AFP8 expert dequant vs
   the SAME expert in GLM-5.2-FP8 → **cos 0.9915**. The W4AFP8 packing is the
   standard compressed-tensors layout: **lo-nibble-first, signed two's-complement
   int4, 2-packed along K, bf16 group-128 scale (multiply), symmetric/no
   zero-point.** This is byte-identical to what kt's RAWINT4 expects
   (`.so` strings: "RAWINT4 ... KGroup signed INT4 without zero point").
   Scripts: `scratchpad/int4_convention_check.py`.
2. **Loader patch works (no 370GB repack needed).** Patched
   `kt_kernel/utils/loader.py::CompressedSafeTensorLoader.load_experts` to
   auto-detect the W4AFP8 suffix scheme (`.weight`/`.weight_scale_inv`) in
   addition to kt-native (`.weight_packed`/`.weight_scale`). It also gained
   env-gated fixups (KT_INT4_WEIGHT_XOR, KT_INT4_NIBBLE_SWAP, KT_INT4_SCALE_RECIP,
   KT_INT4_SCALE_T) used during debugging — all default OFF / no-op.
3. **Full server boots.** `run_server_int4.sh` (MODEL=W4AFP8, --kt-method
   RAWINT4) loads all 77 MoE layers via the patched loader, captures CUDA graphs,
   and serves. **GPU usage ~48 GB/card vs ~84 GB for FP8** at the same
   GPU_EXPERTS=48 — int4 frees ~36 GB/card (lots of headroom to raise
   GPU_EXPERTS later, IF the kernel worked).

## THE BLOCKER: kt RAWINT4 KGroup kernel mishandles the down-projection
Output is garbage → NaN logits → `torch.multinomial` device-side assert (or
greedy emits a constant "!" token). Root-caused with an offline harness that
drives the REAL kt kernel on one layer's experts vs a reference dequant+SwiGLU
matmul (`scratchpad/kt_int4_kernel_test.py`, `..._test2.py`):
- Reference dequant output is clean (norm ~0.086); **kt kernel output is NaN.**
- NaN is INDEPENDENT of: nibble order (swap), sign (xor 0x88), scale reciprocal,
  scale transpose, AND input magnitude (NaN even at input×1e-8). Zero input → 0.
  So it is NOT a value/convention/overflow problem.
- Harness is VALID: the same harness with `--kt-method FP8` on the FP8 model
  gives finite, correct output (norm ~0.148). So the FP8 NativeMoEWrapper
  per-expert-pointer path works; **only the RAWINT4 kernel is broken.**
- The AVX2 RAWINT4 backend (`KT_RAWINT4_BACKEND=avx2`) surfaces the real cause
  via a thrown error: **"RAWINT4 per-expert pointer mode requires
  kt_threadpool_count=1 (down projection TP column-gather is unsupported with
  direct pointers)."**
  - With threadpool_count=2 (our NUMA setup): AVX2 throws that error; the
    default avx512 ("AMXInt4_KGroup") kernel silently returns **NaN**.
  - With threadpool_count=1: avx512 still **NaN**; AVX2 **segfaults** in the
    down-projection forward.
- There is NO alternative stacked/contiguous weight mode for RAWINT4 in kt
  0.6.2.post3 (only "per-expert pointers"; the stacked `torch.cat` path exists
  only for the AMX-quantized `AMXMoEWrapper`, which needs real AMX we don't have).

**Conclusion:** kt 0.6.2.post3's RAWINT4 down-projection in per-expert-pointer
mode is broken for GLM-5.2's expert shapes on this hardware. It is a compiled
`.so` (no source) → not fixable locally.

## Options to get INT4 CPU experts (pick one, next session)
A. **GPTQ_INT4 path (most promising).** kt's `GPTQ_INT4` uses a DIFFERENT kernel
   family (`AVX2GPTQInt4_MOE` / `AVXVNNI256GPTQInt4_MOE`) that needs NO AMX, and
   it routes through the SAME NativeMoEWrapper per-expert-pointer path that works
   for FP8 — so it likely avoids the RAWINT4 down-proj bug. COST: offline repack
   W4AFP8 → GPTQ layout (`qweight` int32-packed + `qzeros` + `scales`,
   asymmetric: add zero_point=8), output to /cache/nvme1, then `--kt-method
   GPTQ_INT4`. Risk: must match GPTQ packing order exactly; validate with the
   same offline harness BEFORE a 4-min server boot.
B. **Upgrade kt-kernel** to a newer build where RAWINT4 supports more models /
   fixes the down-proj per-expert path (check kvcache-ai/ktransformers releases
   since 0.6.2.post3). Cheapest if a fix exists.
C. **Accept FP8 CPU experts** (current baseline) and instead push GPU_EXPERTS up
   for speed — but FP8 experts don't free VRAM the way int4 would, so limited.
D. Pivot the int4 win to GPU-only via sglang W4AFp8 cutlass for the GPU experts +
   FP8 CPU experts — but kt owns both from one method, so this needs nontrivial
   integration glue.

Recommendation: **A (GPTQ_INT4 repack)**, validated offline via the harness
first. Verify GPTQ kernel correctness on one layer before committing to the
full repack + boot.

## Debug assets (scratchpad/)
- `int4_convention_check.py` — proves the W4AFP8 int4 convention (cos 0.99).
- `kt_int4_kernel_test.py` / `_test2.py` — drive the real kt kernel on one layer
  vs reference; the tool to validate ANY future int4 path (GPTQ) offline.
- Loader patch is in the venv at
  `kt_kernel/utils/loader.py::CompressedSafeTensorLoader` (kept; harmless to FP8).
