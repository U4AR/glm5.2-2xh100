# Handoff #2: Packed-INT4 CPU kernel — DONE (13.41 tok/s), and the next levers

**Author:** implementing Claude session, 2026-06-25.
**Predecessor doc:** `PACKED_INT4_KERNEL_HANDOFF.md` (the original spec + roofline; read its
"⚠️ ISA CORRECTION" block first — several of its assumptions were wrong, see below).
**Status:** ✅ The packed-int4 W4A8 CPU MoE kernel is **built, correct, and beats the baseline.**

---

## 0. TL;DR — current state

- **Result:** `bench/decode_bench.sh 5 256` → **median 13.41 tok/s** (min 13.40 / max 13.44),
  coherent output. Baseline was **10.9** (FP8 CPU experts, same everything else). **+23%**,
  inside the original 13–16 target.
- **What changed vs baseline:** ONLY the CPU-expert kernel (FP8 → packed-int4). Same model,
  same GPU experts (int4 cutlass), same tp=2 + CUDA graph, GPU_EXPERTS=104, mem 0.94, max_total 4096.
- **Why it's faster:** weights stay PACKED at 0.5 byte/weight in DRAM and nibbles are unpacked
  in-register right before VPDPBUSD. The CPU-expert decode window is memory-bandwidth-bound, so
  halving the weight bytes ~halves that window. (Roofline rationale in handoff #1 §1 was correct.)
- **Working launch (this is the validated command):**
  ```bash
  MODEL=/cache/nvme0/models/GLM-5.2-W4AFP8 \
  KT_METHOD=RAWINT4 \
  KT_WEIGHT_PATH=/cache/nvme0/models/GLM-5.2-W4AFP8 \
  KT_RAWINT4_BACKEND=avx512_packed \
  GPU_EXPERTS=104 MAX_TOTAL_TOKENS=4096 MEM_FRACTION=0.94 CPUINFER=72 \
  bash run_server_int4.sh
  ```
- A server is (or was) left running with exactly this config on :8000.

---

## 1. What was built (in detail)

### 1.1 The kernel
New file: `ktransformers/kt-kernel/operators/avx2/rawint4_packed_avx512vnni-moe.hpp`
- namespace `avx512_rawint4_packed`, gemm kernel struct `GemmKernelAVX512RawInt4Packed`,
  TP class `AVX512_RAW_INT4_PACKED_MOE_TP`, free fn `gemm_rawint4_packed_avx512`.
- **W4A8:** activations quantized on-the-fly to group-wise int8 (biased u8 for VPDPBUSD);
  weights stay packed 4-bit. Per (output row n, group g): VPDPBUSD over the group, then
  `dot = hsum - 128*weight_sum[n,g]` (the +128 activation-bias correction), then
  `out += dot * a_scale * w_scale[n,g]`. group_size=128, bf16 scales → fp32 at load.
- The in-register nibble unpack (`unpack_int4_chunk`): load 16 packed bytes → split low/high
  nibbles → `_mm_unpacklo/hi_epi8` (stays within 128-bit lanes) → `set_m128i` → K-contiguous
  32×int8. Correctness only needs element-wise a↔w agreement because VPDPBUSD horizontally
  sums the whole group.

### 1.2 THREE root-cause fixes (handoff #1 missed all three; RAWINT4 had never actually run e2e here)

1. **ISA: this CPU has `avx512_vnni`, NOT `avx_vnni`.** (AMD EPYC 9V84 / Zen4.) The 256-bit
   VEX form `_mm256_dpbusd_avx_epi32` (what handoff #1 said to use, what
   `rawint4_avxvnni-moe.hpp` uses) is an **illegal instruction** here — verified SIGILL. Built
   on **AVX-512 VNNI (EVEX)**: `_mm256_dpbusd_epi32`, target
   `avx512f,avx512bw,avx512vl,avx512vnni`. 256-bit (not 512) on purpose: keeps the nibble
   deinterleave inside 128-bit lanes (no cross-lane permute) → simple + obviously correct, and
   the win is from packing, not SIMD width. ⚠️ `rawint4_avxvnni-moe.hpp` is a latent SIGILL
   landmine on this box; it's never selected only because python `_HOST_HAS_AVX_VNNI` is False.

2. **int4 encoding is TWO'S-COMPLEMENT, not offset-binary.** Every kt RAWINT4 kernel decodes
   `value = nibble - 8` (offset binary). **W4AFP8 stores two's-complement**
   (`value = nibble>=8 ? nibble-16 : nibble`). Ground truth = the coherent GPU cutlass path,
   `int4_scripts/w4afp8_gpu_kernel_test.py:147`. Using the wrong convention produced fluent-
   looking GARBAGE (`"odeskodeskodesk..."`) — the model ran, weights decoded to plausible
   magnitudes, but wrong signs. Fix: `value = (nibble XOR 8) - 8` in the SIMD unpack
   (`_mm256_sub_epi8(_mm256_xor_si256(w, eight), eight)`) AND two's-complement in the
   `weight_sums` precompute (they must match the dpbusd operand exactly).
   ⚠️ This means `rawint4-moe.hpp` (avx2-plain) and `rawint4_avxvnni-moe.hpp` ALSO mis-decode
   W4AFP8 — the whole kt RAWINT4 family is wrong-convention for this model. Only this new
   kernel is correct. (That's why prior sessions used FP8/GPTQ CPU experts, never RAWINT4.)

3. **tp=2 needs COPY-into-TP-buffers, not zero-copy.** SGLang's `NativeMoEWrapper`
   (`.venv/.../kt_kernel/utils/amx.py` load_weights) ALWAYS passes per-expert pointers
   (`config.gate_projs`). The RAWINT4 zero-copy per-expert path (mmap'd weights point directly
   into BufferB) can't do the down-projection's cross-NUMA column-gather, so it throws unless
   `kt_threadpool_count=1`. And the tp=1 path **segfaults during CUDA-graph capture** (avx2-plain
   RAWINT4 crashes identically → pre-existing, NOT our kernel). **CUDA graph is required for the
   10+ tok/s regime** (eager ≈ 3.3), so tp=1/eager are dead ends. Fix: mirror `fp8-moe.hpp` —
   `TP_MOE<...>::load_weights` COPIES per-expert sources into per-NUMA TP-sliced buffers (gate/up
   contiguous N-slice; down per-row column-gather; down-scale block-gather; `should_skip_expert`
   for GPU experts), then the inner `load_weights` runs the normal flat path. Works at any
   threadpool_count, incl. **tp=2 + CUDA graph**. The packing win is on the *inference* weight
   stream; the one-time load copy is irrelevant.

### 1.3 Wiring / other edits (all in the repo, which is the source of truth — `install.sh build`
reinstalls python from repo into the venv)
- `ext_bindings.cpp`: `#include` the new header + `bind_moe_module<...>(moe_module, "AVX512RawInt4Packed_MOE")`.
- `python/utils/amx.py`: added `AVX512RawInt4Packed_MOE` symbol, `_HAS_AVX512_RAW_INT4_PACKED_SUPPORT`,
  `_HOST_HAS_AVX512_VNNI`, and a branch in `_select_rawint4_backend` for
  `KT_RAWINT4_BACKEND=avx512_packed` (aliases `avxvnni_packed`, `packed`).
- `python/utils/loader.py` `CompressedSafeTensorLoader`: **W4AFP8 name shim** — accepts both
  `.weight_packed`/`.weight_scale` (kt native) and `.weight`/`.weight_scale_inv` (W4AFP8). The
  payload is byte-identical; the fp8 `input_scale` tensors are ignored by RAWINT4. Without this
  the RAWINT4 loader can't read the W4AFP8 model at all.
- Dead code left in the header (harmless, unused): `BufferB` scale-only constructor +
  `required_size_scale_only` + `finalize_external` (from the abandoned zero-copy path). A future
  agent could delete them, or REVIVE them to add a true zero-copy per-expert tp=2 path (see §3).

### 1.4 Validation done
- Standalone SIMD logic (`scratchpad/unpack_test.cpp`): nibble-unpack + dpbusd + correction
  cos-sim ≥ 0.99997 vs scalar.
- End-to-end through the REAL kt pipeline with synthetic RAWINT4 weights
  (`scratchpad/packed_e2e_test.py`, monkeypatched loader, all experts on CPU): cos 0.99988 vs a
  two's-complement numpy reference. (avx2-plain reference path = 0.99999; the ~1e-4 gap is the
  expected W4A8 int8-activation quant vs avx2-plain's fp32 accumulate.)
- In-server: coherent generation + `decode_bench.sh` = 13.41 tok/s.

---

## 2. Build / run / validate recipe (the gotchas will bite you)

**Build (ALWAYS clean, ALWAYS with CUDA):**
```bash
cd /data/models/RunGLM/ktransformers/kt-kernel
source /data/models/RunGLM/.venv/bin/activate
export PKG_CONFIG_PATH=/data/models/RunGLM/.venv/lib/pkgconfig \
       CMAKE_PREFIX_PATH=/data/models/RunGLM/.venv \
       CMAKE_LIBRARY_PATH=/data/models/RunGLM/.venv/lib \
       CMAKE_INCLUDE_PATH=/data/models/RunGLM/.venv/include \
       CUDA_HOME=/usr/local/cuda-12.9 CPUINFER_USE_CUDA=1 TMPDIR=/data/tmp
bash install.sh build      # ~4–5 min; recompiles + reinstalls into the venv
```
- ⚠️ **Do NOT hand-reconfigure `build/temp/.../CMakeCache` with bare `cmake`** — it silently
  regenerates a **CPU-only** build (`-DKTRANSFORMERS_CPU_ONLY=1`), which DROPS
  `CPUInfer.submit_with_cuda_stream`, and the server dies at load with
  "TP rank N could finish ... others didn't". The clean `install.sh build` auto-detects nvcc and
  enables CUDA (log must say "CUDA detected / enabling CUDA").
- The hwloc/numa are source-built into the venv; the env vars above are mandatory (see NEXT_AGENT.md).
- Verify the build:
  ```bash
  python -c "from kt_kernel import kt_kernel_ext as m; import kt_kernel.utils.amx as a; \
    print(a.AVX512RawInt4Packed_MOE is not None, hasattr(m.CPUInfer,'submit_with_cuda_stream'))"
  # must print: True True
  strings .venv/lib/python3.12/site-packages/kt_kernel/kt_kernel_ext*.so | grep -c AVX512RawInt4Packed
  ```

**Offline correctness (fast, NO server needed):**
```bash
KT_RAWINT4_BACKEND=avx512_packed python scratchpad/packed_e2e_test.py   # expect cos >= 0.999
```
⚠️ **NEVER run an offline kt test while the server is up.** Two kt instances collide on
SysV/`/dev/shm` shared-expert buffers and **crash the running server** (learned the hard way —
it killed the live baseline). One kt process at a time, period.

**Launch knobs** (`run_server_int4.sh`): `KT_METHOD=RAWINT4`, `KT_RAWINT4_BACKEND=avx512_packed`,
`KT_WEIGHT_PATH` = the W4AFP8 dir (loader shim handles it), `GPU_EXPERTS`, `CPUINFER`,
`MEM_FRACTION`, `MAX_TOTAL_TOKENS`, `SPEC_DECODE`. Boot from nvme0 W4AFP8 ≈ 2.5 min
(int4, half the bytes; much faster than the FP8-from-/data ~50 min). If a boot fails to start
with an empty log, it's a launcher race — clear `/dev/shm/psm_* /dev/shm/sem.mp-*` and relaunch
via `setsid env ... bash run_server_int4.sh > log 2>&1 < /dev/null &`.

---

## 3. Additional KERNEL gains to pursue (decreasing confidence)

The window is memory-bound, so most compute tricks won't help; these are about squeezing the
remaining bandwidth/front-end:

1. **512-bit EVEX variant** (`_mm512_dpbusd_epi32`, 64 int8/op). Zen4 double-pumps AVX-512 so
   peak compute is the same, but it halves instruction count → front-end relief, which matters
   now that the unpack adds ALU ops. The catch handoff #1 warned about: the 256→512 nibble
   deinterleave crosses 128-bit lanes, so you need a cross-lane permute (or VBMI
   `_mm512_multishift_epi64_epi8` / GFNI `_mm512_gf2p8affine` — this box HAS avx512_vbmi + gfni).
   Process the group in 64-k chunks with a 32-k tail. Validate with `unpack_test.cpp` first.
   Expected: small (few %), since memory-bound. Do it ONLY if profiling shows front-end stalls.
2. **Software prefetch + n-tiling** of the packed weight stream. Steal the register-blocking /
   prefetch cadence from `operators/avx2/fp8-moe.hpp` (the kernel that was winning at 10.9). The
   inner loop currently streams one output row's group at a time; tiling N to reuse the activation
   `a_u8` across rows and prefetching the next rows' packed bytes could lift effective BW toward
   the ~170 GB/s ceiling.
3. **Re-confirm the roofline AFTER packing.** Re-run the §7 recipe from handoff #1 (`perf stat -e
   ls_any_fills_from_sys.dram_io_all -a -I 10`) during decode. If the in-window BW is now BELOW
   ~140 GB/s, you've become compute/front-end bound (then #1/#2 help). If still ~140–200, you're
   still BW-bound and the only kernel lever left is fewer bytes (you're already at int4; next
   would be int3/asym, not worth it).
4. **True zero-copy per-expert at tp=2** (RAM saver, not speed): revive the dead `finalize_external`
   + a `row_stride` field on BufferB so down can point directly at the mmap'd weights with the
   full row stride but a TP K-slice (and strided scale copy). Saves the load-time copy + ~the CPU
   RAM for CPU experts. Only worth it if host RAM becomes the constraint.

⚠️ If you touch the unpack or weight_sums, the TWO'S-COMPLEMENT convention (`(n^8)-8`) MUST stay
in both places or you get fluent garbage again.

---

## 4. Lever A (optional) — re-tune GPU_EXPERTS

Per-layer decode latency ≈ `max(CPU-expert time, GPU-expert+attn time)` (they already overlap
in `kt_ep_wrapper.apply()`). The split is set by `GPU_EXPERTS` (of 256 routed experts/layer,
this many go to GPU; rest to the packed-int4 CPU kernel). Two things just shifted the optimum:
the packed CPU kernel is **faster per expert** AND uses **~½ the CPU RAM**, and int4 GPU experts
use less VRAM than FP8.

**How to retune:** boot with `SGLANG_KT_HYBRID_TIMING=1` (logs per-layer submit/gpu/sync/merge/
**cpu_wait** for tp_rank0, layers 0/5/20/35), sweep `GPU_EXPERTS`, and read `cpu_wait`:
- `cpu_wait > 0` (GPU finishes first, waits on CPU) → CPU is the long pole → **raise** GPU_EXPERTS.
- `cpu_wait ≈ 0` and GPU+attn is the long pole → **lower** GPU_EXPERTS to free VRAM with no speed
  loss (use the freed VRAM for KV cache / longer context / MTP draft buffers).
The sweet spot is where neither side waits. Because the CPU side got cheaper, you can likely
**lower** GPU_EXPERTS from 104 and hold ~13 tok/s while reclaiming VRAM — directly enabling
Lever B. Sweep e.g. 64/80/96/104/112 (112 OOM'd for FP8; int4 may fit higher). Bench each with
`decode_bench.sh 5 256`.

---

## 5. Lever B (your idea) — MTP / NEXTN with the draft layer on GPU

**The model supports it:** `config.json` has `num_nextn_predict_layers: 1` (one NEXTN/MTP head),
78 hidden layers (3 dense + 75 MoE). `run_server_int4.sh` already wires it:
`SPEC_DECODE=1` → `--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1
--speculative-num-draft-tokens 2` and sets `SGLANG_ENABLE_SPEC_V2=True` (required — it turns on the
correct spec-v2 verify + overlap scheduler for GlmMoeDsa).
**Prereq:** tilelang must import — `pip install "apache-tvm-ffi==0.1.11"` (0.1.12 is broken). See
memory `glm52-tilelang-fix`.

**Why it was a WASH before (memory `glm52-mtp-specv2-result`): MTP ≈ 8.8 ≈ baseline 8.7.** The
verify step pushes (1 + draft_tokens) tokens through the main model in one forward. For a *dense*
layer that amortizes weight reads (read once, apply to all tokens) → free speedup. But for **MoE,
different draft tokens route to different experts**, so verify reads the *union* of all draft
tokens' top-k experts → MORE weight bytes through the bandwidth-bound CPU window, not amortized.
The extra acceptance didn't pay for the extra CPU-expert traffic.

**Your refinement is the right attack on that:** both devices sit ~50% idle during the
CPU-expert window (memory `glm52-decode-bottleneck-v2`). If the **NEXTN draft layer's experts are
on GPU**, drafting is GPU-only and can run in the otherwise-idle GPU time, and — critically — the
*draft* never touches the slow CPU window. Then verify is the only CPU-expert cost, and if draft
tokens have routing locality (adjacent tokens often share experts), verify's expert-union may be
near the single-token set → little extra CPU traffic for >1 accepted token. Net: extra tokens
"for free" in GPU idle time. This is plausible and worth testing; it was NOT tried before with the
draft layer pinned to GPU.

**How to try it:**
1. Re-tune (Lever A) to free VRAM first — you need headroom for the NEXTN layer + draft KV.
2. Boot with `SPEC_DECODE=1` and confirm coherence + measure `decode_bench.sh`. Compare accepted-
   tokens/step (sglang logs acceptance length) and tok/s vs the 13.41 baseline.
3. **Pin the NEXTN layer's experts to GPU.** Expert placement lives in
   `.venv/.../sglang/srt/layers/moe/kt_ep_wrapper.py` (`build_*_expert_placement`,
   `experts_per_layer = num_gpu_experts // num_moe_layers` ~L1645). The NEXTN/MTP layer (index 78)
   must be forced to all-GPU experts (or at least its hot experts) rather than inheriting the
   per-layer CPU/GPU split. Check whether the MTP layer is even included in the kt placement loop;
   if it falls through to the CPU path, that's the first thing to change. There may be no flag —
   expect a small code change to the placement builder to special-case the NEXTN layer.
4. Measure with `SGLANG_KT_HYBRID_TIMING=1` whether draft now overlaps the CPU window (GPU busy
   during cpu_wait) and whether verify's CPU-expert traffic stayed bounded.

**Open questions to answer empirically:**
- Does sglang's spec-v2 scheduler actually overlap GPU-draft with the main model's CPU-expert
  verify, or is it serial draft→verify? (If serial, the "fill idle GPU time" benefit shrinks to
  just faster drafting.)
- Acceptance length at temp=0 for this model with NEXTN steps=1, draft_tokens=2 (try 2–4).
- Verify's expert-union growth: instrument how many distinct CPU experts verify touches per step
  vs single-token decode (routing locality determines the whole payoff).
- Does pinning the MTP layer to GPU cost enough VRAM to force GPU_EXPERTS down elsewhere (net
  trade)?

**Honest expectation:** if verify's CPU expert-union stays ~flat (good routing locality) and
draft overlaps GPU idle time, this could add meaningful tok/s on top of 13.41. If verify's union
blows up (poor locality), it'll wash again regardless of where the draft layer sits. The packed
kernel already shrank the CPU window, so MTP starts from a better place than the 8.7-era attempt.

---

## 6. Pitfalls / environment facts (save yourself the pain)
- **One kt process at a time.** A second kt instance (offline test) crashes the running server via
  shared-memory collision. Kill the server before any offline kt run.
- **Build = `install.sh build` + `CPUINFER_USE_CUDA=1` only.** Manual cmake → CPU-only .so →
  server dies at load. Verify `submit_with_cuda_stream` exists after every build.
- **CUDA graph is required** for 10+ tok/s (eager ≈ 3.3). Don't "fix" anything by disabling it.
- **Two's-complement int4** for W4AFP8. Offset-binary `nibble-8` → fluent garbage.
- **No `avx_vnni`** on this Zen4 (only `avx512_vnni`). Anything using `_mm256_dpbusd_avx_epi32`
  SIGILLs. Use EVEX `_mm256_dpbusd_epi32`.
- **perf:** `echo 'bel@123' | sudo -S sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'`.
- **Harness blocks foreground `sleep`**; launch servers with `setsid ... &` and poll a logfile.
- Weights are EPHEMERAL on /cache/nvme0 (W4AFP8) + /cache/nvme1 (GPTQ); re-download via
  `int4_scripts/download_w4afp8.py`, repack via `int4_scripts/gptq_full_repack.py`.

## 7. File map
| File | Change |
|---|---|
| `ktransformers/kt-kernel/operators/avx2/rawint4_packed_avx512vnni-moe.hpp` | NEW — the kernel |
| `ktransformers/kt-kernel/ext_bindings.cpp` | include + bind `AVX512RawInt4Packed_MOE` |
| `ktransformers/kt-kernel/python/utils/amx.py` | symbol + `_select_rawint4_backend` (`avx512_packed`) |
| `ktransformers/kt-kernel/python/utils/loader.py` | W4AFP8 name shim in `CompressedSafeTensorLoader` |
| `scratchpad/unpack_test.cpp` | standalone SIMD logic check |
| `scratchpad/packed_e2e_test.py` | e2e correctness (synthetic, real pipeline) |
| `scratchpad/run_int4_tp1.sh` | tp=1 variant (only used for the segfault bisection; not for serving) |
| `bench/decode_bench.sh` | the benchmark (median tok/s) |

Related memory: `glm52-packed-int4-cpu-kernel` (this work), `glm52-w4afp8-not-runnable` (int4 GPU
+ the -1 remap fix), `glm52-decode-bottleneck-v2` (overlap/balance), `glm52-mtp-specv2-result`
(prior MTP wash), `glm52-tilelang-fix` (NEXTN prereq). Baseline beaten: **10.9 → 13.41 tok/s.**
