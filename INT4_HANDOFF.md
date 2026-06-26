# INT4 experts on GLM-5.2 (2×H100 + kt offload) — HANDOFF

> ## ✅ RESOLVED 2026-06-25 — int4 GPU experts work AND beat the FP8 baseline.
> The old conclusion below ("GPU w4afp8 path garbages, dead end") was WRONG. The
> w4afp8 cutlass kernel was always correct (offline harness cos 0.995). The real
> bug: kt_ep_wrapper marks CPU-resident experts with `-1` in topk_ids, but
> `cutlass_w4a8_moe` only remaps `-1 -> num_local_experts` (its skip sentinel)
> when EP world size > 1; we run ep_size=1, so `-1` corrupted the GPU MoE → NaN →
> garbage. **FIX** in `.venv/.../quantization/w4afp8.py` `W4AFp8MoEMethod.apply`:
> `topk_ids = torch.where(topk_ids == -1, num_local_experts, topk_ids)` (+ an
> idempotency guard on `process_weights_after_loading`). 
>
> **WINNING CONFIG (~10.1-10.6 tok/s decode, vs 8.7 FP8 baseline):** int4 GPU
> experts (MODEL=W4AFP8) + FP8 CPU experts + GPU_EXPERTS=104:
> `MODEL=/cache/nvme0/models/GLM-5.2-W4AFP8 KT_METHOD=FP8 \
>  KT_WEIGHT_PATH=/data/models/GLM-5.2-FP8 GPU_EXPERTS=104 \
>  MAX_TOTAL_TOKENS=4096 bash run_server_int4.sh`
> VRAM 88GB/card (96 expts=82GB safe w/ 8192 KV; 112 OOMs). Decode is CPU-bound on
> the FP8 CPU experts → maxing GPU_EXPERTS (fewer CPU experts) is the win. New
> GPU harness: `int4_scripts/w4afp8_gpu_kernel_test.py`. See memory
> `glm52-w4afp8-not-runnable` for the full debug trail. Everything below predates this.



Written 2026-06-24. Read this top-to-bottom before touching anything. Companion
docs: `INT4_PROJECT.md` (original scoping + RAWINT4 dead-end), `FUSED_KERNEL_FINDINGS.md`
(earlier MTP work). Persistent scripts in `int4_scripts/` (copied from a session
scratchpad — they are the debugging tools you'll reuse).

## TL;DR — where this stands
Goal: make GLM-5.2 **int4 experts** decode FASTER than the FP8 baseline (~8.7–8.8
tok/s) on this 2×H100 box via kt CPU+GPU offload.

- ✅ int4 experts now **RUN correctly** (coherent output). Two changes made it work:
  a kt loader patch + a GPTQ repack of the experts.
- ❌ int4 **CPU** experts are **SLOWER** (~5.7 tok/s) than FP8 (~8.7). Root cause:
  the only *working* int4 CPU kernel here is **AVX2** (256-bit); FP8 uses **AVX512**
  (512-bit). Decode @ bs=1 is compute-bound, so narrower SIMD loses. Dead end on
  this hardware unless a faster int4 CPU kernel becomes available.
- 🔜 **THE REAL LEVER (your task): int4 experts on the GPU.** int4 is half the size
  of FP8, so the GPU can hold ~2× more experts → fewer experts fall to the CPU
  bottleneck → faster. The GPU int4 path is sglang's **w4afp8 cutlass** kernel
  (Phala's proven 8-GPU path), but in our kt offload integration it currently
  produces **garbage**. Fix that and you win.

## Hardware / environment (constraints that shaped everything)
- 2× H100 NVL (96 GB each). TP2. ~95 GB usable/card.
- AMD EPYC 9V84 (Zen4), 80 cores, 2 NUMA nodes (40 cores each), 629 GB RAM.
  **Has AVX512(F/BW/BF16/VBMI/VNNI) but NO AMX and NO `avx_vnni` (256-bit VEX VNNI).**
  This is why fast int4 kernels are unavailable (see below).
- kt-kernel **0.6.2.post3** (compiled .so — cannot edit C++), sglang `0.0.0.dev0`.
- Disks: `/data` persistent but only ~70 GB free. `/cache/nvme0`,`/cache/nvme1`
  are **EPHEMERAL** (wiped on reboot), 3.4 TB each. Big artifacts live there.
- sudo password `<sudo-password>` (single-user box). Harness **BLOCKS foreground `sleep`** —
  never put a bare `sleep` in a Bash call; use it only inside a `for` loop.
- `pgrep -f` self-matches your shell; use specific patterns like `launch_server --model-path`.

## Assets (all present, verified)
- **Weights (experts), original:** `/cache/nvme0/models/GLM-5.2-W4AFP8` (373 GB, 40
  shards). Phala's w4afp8: I8 int4 (2-packed, signed two's-comp, lo-nibble-first),
  bf16 group-128 `weight_scale_inv`, bf16 `input_scale`. EPHEMERAL; re-download from
  HF `PhalaCloud/GLM-5.2-W4AFP8` if wiped.
- **Weights (experts), GPTQ repack:** `/cache/nvme1/GLM-5.2-W4-GPTQ-experts` (353 GB,
  76 shards `experts-layer-NNN.safetensors`, one per MoE layer 3..78). kt GPTQ format:
  `qweight` int32 [in/8,out] (8×4bit lo-first along K, q=signed+8), `scales` fp16
  [groups,out], symmetric group-128. **Validated cos 1.0 vs reference** (see harness).
  EPHEMERAL; regenerate with `int4_scripts/gptq_full_repack.py` (~15 min, 6 workers).
- **FP8 model (baseline + GPU/non-expert source):** `/data/models/GLM-5.2-FP8` (persistent).
- **kt loader patch (IN THE VENV, persists with venv):**
  `.venv/lib/python3.12/site-packages/kt_kernel/utils/loader.py` ::
  `CompressedSafeTensorLoader.load_experts` — auto-detects W4AFP8 names and has
  env-gated int4 fixups (`KT_INT4_WEIGHT_XOR/_NIBBLE_SWAP/_SCALE_RECIP/_SCALE_T`,
  all default OFF). Harmless to FP8. Verify present: `grep -c _SUFFIX_SCHEMES ...loader.py` → ≥1.
- **Launch script:** `run_server_int4.sh` (env: MODEL, KT_WEIGHT_PATH, KT_METHOD,
  GPU_EXPERTS, etc.).
- **Debug scripts (`int4_scripts/`):**
  - `kt_int4_kernel_test.py` — **the key tool.** Drives the REAL kt MoE kernel on one
    layer vs a reference dequant+SwiGLU matmul. Use it to validate ANY kernel/format
    offline in seconds (no 5-min server boot). Edit `method=`/`weight_path=` to test.
  - `int4_convention_check.py` — proved the W4AFP8 int4 convention (cos 0.99 vs FP8).
  - `gptq_repack_validate.py` — repack ONE layer + validate via the kernel (cos 1.0).
  - `gptq_full_repack.py` — full parallel repack → GPTQ dir.

## What was tried, with evidence (don't repeat these)
1. **RAWINT4 (`--kt-method RAWINT4`)**: loads, boots, but the KGroup down-projection
   kernel is broken for GLM in per-expert-pointer mode → NaN (avx512 variant) /
   segfault (avx2 variant). Error string: *"RAWINT4 per-expert pointer mode requires
   kt_threadpool_count=1 (down projection TP column-gather is unsupported with direct
   pointers)"* — and even at tp=1 it still fails. The FAST int4 CPU kernel; unusable.
2. **GPTQ_INT4 (`--kt-method GPTQ_INT4`)**: CPU kernel is **correct** (cos 1.0, both
   tp=1 and tp=2) and the full server is **coherent** with FP8 GPU experts. But it
   only has an **AVX2** backend here (`AVX2_GPTQ_INT4_MOE`; AVXVNNI256 needs `avx_vnni`
   we lack) → ~5.7 tok/s < 8.7 FP8. Correct but slow.
3. **w4afp8 GPU experts** (config quant_method=w4afp8 → sglang `W4AFp8MoEMethod`
   cutlass for the GPU-resident experts): **garbage output.** Isolated by elimination:
   the kt CPU GPTQ path is proven correct (it computes CPU experts and zeros the
   GPU-assigned ones for sglang to fill), so the garbage is the GPU w4afp8 path.
   `GPU_EXPERTS=0` crashes earlier in `w4afp8.py:272` (`.max()` on empty input_scale).

## Architecture you must understand (kt_ep_wrapper)
`sglang/srt/layers/moe/kt_ep_wrapper.py`. Per MoE layer, experts are split:
- **GPU experts** (`--kt-num-gpu-experts N` per layer) run via sglang's `gpu_method`
  = `quant_config.get_quant_method(...)`, chosen from the **model-path config's**
  `quant_method`. For W4AFP8 → `W4AFp8MoEMethod` (cutlass w4a8). For FP8 → `Fp8MoEMethod`.
- **CPU experts** (the rest) run via **kt** (`--kt-method`, weights from `--kt-weight-path`).
- kt computes only the CPU experts and **zeros** the GPU-assigned slots; sglang's
  gpu_method fills the GPU ones; they're summed. (Verified via the harness.)
- So GPU and CPU expert weights come from DIFFERENT places and CAN be different
  formats/quant — that's why the working "hybrid" below is possible.

## The current WORKING (but slow) config — the hybrid
`MODEL=/data/models/GLM-5.2-FP8  KT_WEIGHT_PATH=/cache/nvme1/GLM-5.2-W4-GPTQ-experts
 KT_METHOD=GPTQ_INT4  GPU_EXPERTS=48  bash run_server_int4.sh`
→ GPU experts = FP8 (proven cutlass), non-experts = FP8, CPU experts = int4 GPTQ.
Coherent. ~5.7 tok/s. (This server may still be running on :8000.) It proves int4
CPU works end-to-end; it's just slower than FP8. **Not the goal** — documented as the
correctness proof.

## YOUR TASK: make the GPU run int4 experts (so we can offload MORE to GPU)
The win = more experts on GPU as int4 (half-size → ~2× count vs FP8) → less CPU work.
Two sub-approaches:

### Approach A — fix the sglang w4afp8 cutlass GPU path (highest value)
Phala runs this exact kernel successfully pure-GPU, so the kernel works; the bug is
in our integration / weight handling. Plan:
1. Build an **offline harness for the GPU path** (mirror `kt_int4_kernel_test.py` but
   for sglang's `W4AFp8MoEMethod`: create a `FusedMoE`-like layer, run its
   `create_weights` → load W4AFP8 expert tensors via the w4afp8 weight_loader →
   `process_weights_after_loading` → `apply()` with a synthetic routing) and compare
   to the reference dequant matmul. This tells you if the kernel itself is correct on
   GLM dims or if the integration mis-feeds it. (cutlass_w4a8_moe is in
   `sglang/srt/layers/moe/cutlass_w4a8_moe.py`; method in `.../quantization/w4afp8.py`.)
2. Likely suspects: the `interleave_scales` step, the per-expert `input_scale` max,
   the gate/up (w13) fusion vs W4AFP8's separate gate/up tensors, or expert-index
   mapping in kt_ep_wrapper when GPU/CPU are split.
3. Once correct, set `GPU_EXPERTS` as high as VRAM allows (int4 GPU experts are small
   — estimate ~100–120/layer fits vs 48 for FP8; non-experts ~? GB, KV, graphs).
   MODEL = W4AFP8 (so gpu_method=w4afp8), KT_WEIGHT_PATH = GPTQ dir (CPU side).
   Benchmark vs 8.7.

### Approach B — newer kt-kernel (cheap to check first)
Upgrade kt-kernel past 0.6.2.post3 and re-test `--kt-method RAWINT4` (fast AVX512
int4 CPU kernel) — if the down-proj per-expert bug is fixed, RAWINT4 CPU experts
would be fast AND we already proved the convention/loader. Check
github.com/kvcache-ai/ktransformers releases. Lowest effort if a fix exists.

### Approach C — accept FP8 (fallback)
If A and B fail, FP8 (8.7) is the best on this box; int4 doesn't help here.

## How to run / measure
- **Baseline FP8 (fast, known-good):** `SGLANG_ENABLE_SPEC_V2=True SPEC_DECODE=1
  GPU_EXPERTS=48 bash run_server.sh`  (~8.7–8.8 tok/s). MTP is a wash (see memory).
- **Coherence test:** POST to `http://localhost:8000/v1/chat/completions`, greedy
  (`temperature:0`) so a bad sample can't mask garbage. GLM emits `reasoning_content`;
  check both `content` and `reasoning_content`. Garbage looks like `"!!!!"`.
- **Decode tok/s:** read the server log lines `Decode batch ... gen throughput (token/s)`
  (ground truth), or time a 256-token completion / completion_tokens.
- **Kill a server cleanly:** `pkill -9 -f 'launch_server'` then poll
  `nvidia-smi --query-compute-apps=pid` until empty (a zombie can hold VRAM ~20 s).
- **Boot time** ~5 min (loads ~360 GB experts to RAM + graph capture). Loads layers
  3..78; watch `loader for layer N` then `fired up and ready`.

## Gotchas
- `GPU_EXPERTS=0` is NOT supported (w4afp8 `.max()` on empty). Keep ≥1.
- RAWINT4 needs `kt_threadpool_count=1` AND still fails — don't bother.
- The GPTQ repack OFFSET=8 (q=signed+8) is correct; don't change.
- nvme0/nvme1 are ephemeral — if rebooted, re-download W4AFP8 and re-run the repack.
- Don't move the only FP8 model copy; it's the working baseline.
