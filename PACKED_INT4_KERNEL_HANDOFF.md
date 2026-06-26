# Handoff: Packed-INT4 W4A8 CPU MoE kernel for GLM-5.2

**Author:** prior Claude session (2026-06-25). **For:** next agent building the kernel.
**Goal:** Make the int4 CPU experts BEAT the FP8 CPU experts on single-stream decode by
keeping weights PACKED at 4-bit in DRAM (halving weight memory traffic in the
memory-bound window). Target: lift decode from **10.9 tok/s** baseline toward ~13–16.

---

## ⚠️ ISA CORRECTION (2026-06-25, implementing session) — READ BEFORE §4c

The handoff below says "build the 256-bit **AVX-VNNI** variant first" and to mirror
`rawint4_avxvnni-moe.hpp` (target `avx2,avxvnni,fma`, intrinsic `_mm256_dpbusd_avx_epi32`).
**That ISA is NOT available on this CPU.** `/proc/cpuinfo` on the AMD EPYC 9V84 (Zen4)
shows `avx512_vnni` but **NOT `avx_vnni`** — the 256-bit VEX AVX-VNNI form
(`_mm256_dpbusd_avx_epi32`) is an **illegal instruction** here (verified: SIGILL). The
existing `rawint4_avxvnni-moe.hpp` would ALSO SIGILL on this box; it's a latent landmine,
never actually selected (python `_HOST_HAS_AVX_VNNI` is False → falls back to AVX2-plain).

**What was actually built:** the packed kernel uses **AVX-512 VNNI (EVEX)** —
`_mm256_dpbusd_epi32` (256-bit, gated by `avx512vl+avx512vnni`), target
`avx512f,avx512bw,avx512vl,avx512vnni`. 256-bit (not 512) keeps the nibble deinterleave
inside 128-bit lanes (no cross-lane permute) → simple + obviously correct. The packing
(memory) win is independent of SIMD width, exactly as the roofline argues.

**Concrete integration that shipped (supersedes §5 names):**
- Kernel: `operators/avx2/rawint4_packed_avx512vnni-moe.hpp`
  (namespace `avx512_rawint4_packed`, class `AVX512_RAW_INT4_PACKED_MOE_TP`).
- Binding (`ext_bindings.cpp`): `AVX512RawInt4Packed_MOE`.
- Backend select (`python/utils/amx.py` `_select_rawint4_backend`): env value
  **`KT_RAWINT4_BACKEND=avx512_packed`** (aliases also accepted: `avxvnni_packed`, `packed`).
- Launch: `MODEL=W4AFP8 KT_METHOD=GPTQ_INT4 KT_WEIGHT_PATH=<nvme1 GPTQ int4 experts>`
  `KT_RAWINT4_BACKEND=avx512_packed GPU_EXPERTS=N bash run_server_int4.sh`.
- Offline nibble-logic check (no build): `scratchpad/unpack_test.cpp` — cos-sim ≥ 0.99997.
- **Build gotcha:** `setup.py` auto-detects CUDA via `nvcc`; a bare `cmake` reconfigure
  builds **CPU-only** (drops `submit_with_cuda_stream`, breaks the server). Build with
  `bash install.sh build` + `CPUINFER_USE_CUDA=1` and the hwloc/numa env vars
  (see `NEXT_AGENT.md`). Verify symbol: `strings <.so> | grep AVX512RawInt4Packed`.

---

---

## 0. TL;DR of the decision (read this first)

- Serving: GLM-5.2 MoE via SGLang + KTransformers (kt-kernel), heterogeneous CPU+GPU MoE,
  TP2 over 2×H100. Current winning config = **int4 GPU experts (104) + FP8 CPU experts**,
  median **10.9 tok/s** warm. (`run_server_int4.sh`, `bench/decode_bench.sh`.)
- We proved (roofline, below) that the **CPU-expert path is MEMORY-BANDWIDTH-BOUND in its
  active window**, not compute-bound. So fewer weight bytes = faster. int4 = half the bytes
  of FP8 → real ~2× headroom on the CPU-expert segment.
- The EXISTING int4 CPU kernel throws this away: it **pre-expands packed int4 → int8 at
  load**, so at inference it reads 1 byte/weight, same as FP8. That's why int4-CPU currently
  loses to FP8-CPU.
- **The fix = a new kernel that keeps weights PACKED (0.5 byte/weight) and unpacks nibbles
  in-register right before the VPDPBUSD.** ~70% is copy-paste from the existing kernel; only
  the packed inner loop + nibble unpack is new.
- **W4A8 awareness:** weights 4-bit, **activations 8-bit**. Activations STAY int8 (mandatory:
  VPDPBUSD is u8×s8). At bs=1 the activation is one token's hidden vector — negligible memory.
  ALL the memory win is on the weight side. Do not touch the activation path.

---

## 1. The roofline evidence (why this is worth building)

Measured 2026-06-25 on the LIVE FP8-CPU server (non-destructive), AMD core demand-fill
counters `ls_any_fills_from_sys.dram_io_{near,far,all}` (×64B/cacheline):

- 8s steady decode avg: **67–78 GB/s**, near=99.4% (NUMA placement near-perfect, far=0.6%).
- VM achievable BW (OMP STREAM triad, 80 threads): **~170 GB/s** (read-only weight streaming
  ceiling is higher, ~200+, no write-back/RFO). NOTE: theoretical 12ch DDR5-4800 = 460 GB/s,
  but this 80-vCPU Azure slice only reaches ~170 — **use the measured ceiling, not theoretical.**
- **KEY: at -I 10 (10ms bins) the CPU-expert active window BURSTS to 140–220 GB/s** (modal
  140–160, peak 200–220). That is ~80–100% of achievable BW → **MEMORY-BOUND in-window.**
- TRAP: at -I 100 (100ms bins ≈ 1 decode step @ ~95ms) it looks flat at ~78 GB/s and fools you
  — that averages the bound compute-window with the GPU-only/spin part of the step.
  **Always measure at -I 10 for this.**

Conclusion: halving weight bytes ~halves the bound window. The Zen4 "AVX512 is double-pumped,
width gives no compute gain" caveat is MOOT here because the window is MEMORY-bound, not
compute-bound — **PACKING is the win, not SIMD width.** A 256-bit VNNI kernel that keeps
weights packed would already help; 512-bit is a secondary front-end-relief bonus.

Reproduce: see §7.

---

## 2. The exact data format (on-disk packed RAWINT4)

From `ktransformers/kt-kernel/operators/avx2/rawint4_avxvnni-moe.hpp` `BufferB::from_raw_mat`
(verified by reading the code):

- `src_packed`: `[N, K/2]` uint8, row-major. **Low nibble = even k, high nibble = odd k.**
- Weight value = `(nibble - 8)` → **symmetric signed int4, range −8..7.**
- `src_scales`: bf16 `[N, num_groups]`, `num_groups = K / group_size`. Convert bf16→fp32.
- `group_size`: multiple of 32, ≤256, default **128**. K divisible by 8 and group_size.
- Deinterleaving the two nibbles of each byte (low then high) yields **K-contiguous** int8
  `[k0,k1,k2,k3,...]`. This matches the activation order — important for VPDPBUSD lane alignment.

`weight_sums[N, num_groups]` = per-group sum of the unpacked signed int8 weights (int16).
Used for the +128 activation-offset correction (see §3). It can be precomputed at load
DIRECTLY from the packed bytes (no need to materialize the int8 array).

---

## 3. The W4A8 math to REUSE verbatim (from rawint4_avxvnni-moe.hpp)

The arithmetic is correct and validated — copy it. Only the weight storage/read changes.

- **Activation quant** `quantize_activation_group_u8(const ggml_bf16_t* src, int gs, uint8_t* dst)`
  (L55): per-group, finds absmax, quantizes hidden to **unsigned u8 with +128 offset**, returns
  `a_scale`. Keep exactly. This is the "A8" of W4A8.
- **Inner accumulate** `gemm_rawint4_avxvnni256` (L208–245): per (output-row n, group g):
  1. `a_scale = quantize_activation_group_u8(a_row + k_base, group_size, a_u8)`
  2. loop k in group: `_mm256_dpbusd_avx_epi32(acc, a_u8_vec /*u8*/, w_s8_vec /*s8*/)`
  3. `dot = hsum_epi32(acc) - 128 * weight_sums[n*ng + g]`  ← the +128 offset correction
  4. `out += a_scale * scales[n*ng+g] * dot`
- **Correction rationale:** activations were stored as `u8 = s8 + 128` to feed VPDPBUSD's
  unsigned operand; subtracting `128 * sum(weights_in_group)` removes the bias. Keep identical.
- **NUMA split:** `avx2::split_range(n, ith, nth)` over output rows; threadpool drives ith/nth.
  Reuse unchanged (this is what `--kt-threadpool-count 2 --kt-numa-nodes 0 1` exploits).

---

## 4. What is NEW: the packed BufferB + packed inner loop

### 4a. BufferB — keep weights packed (THE fix)
Today (the flaw):
```
qweight_s8 = [N, K] int8   // FULL EXPANSION — reads 1 byte/weight at inference
required_size = k*n*sizeof(int8) + ng*n*sizeof(float) + ng*n*sizeof(int16)
```
New:
```
qweight_packed = [N, K/2] uint8   // STAYS PACKED — 0.5 byte/weight
scales         = [N, ng]  float32
weight_sums    = [N, ng]  int16   // precompute from packed bytes at load (no expansion)
required_size  = (k/2)*n + ng*n*sizeof(float) + ng*n*sizeof(int16)   // ~half the weight term
```
`from_raw_mat`: just `memcpy` the packed `[N,K/2]` rows (or store pointer), convert scales
bf16→fp32, and compute `weight_sums` by iterating the packed nibbles (cheap, load-time only).

### 4b. Inner loop — unpack nibbles in-register, then VPDPBUSD
Per group, per chunk of K:
1. Load packed bytes (16 bytes = 32 nibbles per 128-bit; 32 bytes = 64 per 256-bit lane).
2. **Deinterleave to K-contiguous signed int8:**
   - `lo = x & 0x0F; hi = (x >> 4) & 0x0F;`
   - interleave: `_mm_unpacklo/hi_epi8(lo, hi)` → `[lo0,hi0,lo1,hi1,...]` (= k0,k1,k2,k3…)
   - `w_s8 = result - 8`  (broadcast `_mm*_set1_epi8(8)`, subtract)
   - (Optional fast path: GFNI `_mm*_gf2p8affine` / VBMI `_mm512_multishift_epi64_epi8` — this
     box HAS gfni, vaes, avx512_vbmi. But AND+shift+unpack+sub is simplest and avx2-portable;
     do that FIRST, optimize unpack only if profiling says it matters. Remember: memory-bound,
     so a few extra unpack ALU ops are free as long as they don't stall.)
3. `_mm256_dpbusd_avx_epi32(acc, a_u8, w_s8)` (or `_mm512_dpbusd_epi32` for the 512 variant).
4. Same correction + scale as §3.

### 4c. 256-bit first, 512-bit second
- Build `rawint4_avxvnni256` PACKED variant FIRST (smallest diff from existing, same target
  `avx2,avxvnni,fma`). This ALONE should capture the memory win (it's the packing, not width).
- THEN optionally a 512-bit `rawint4_avx512vnni` (target `avx512f,avx512bw,avx512vnni`) for
  front-end relief: load 32 packed bytes → 64 int8 in `__m512i`, `_mm512_dpbusd_epi32`. Steal
  512-bit register-blocking / n-tiling / prefetch cadence from `operators/avx2/fp8-moe.hpp`
  (and `operators/amx/fp8-moe.hpp`) — the FP8 path is the kernel currently winning at 10.9.

---

## 5. Integration points (exact locations)

- **Kernel file:** new `operators/avx2/rawint4_packed_avxvnni-moe.hpp` (mirror existing
  `rawint4_avxvnni-moe.hpp` structure: BufferA/BufferB/BufferC + Gemm wrapper + from_raw_mat).
- **Binding:** `ktransformers/kt-kernel/ext_bindings.cpp` L827–833 currently binds
  `AVX2RawInt4_MOE`, `AVXVNNI256RawInt4_MOE`, GPTQ variants. Add e.g.
  `AVXVNNI256RawInt4Packed_MOE` next to L833. Mirror the `bind_moe_module<AVXVNNI256_RAW_INT4_MOE_TP<...>>` pattern.
- **Backend select:** `ktransformers/kt-kernel/python/experts.py` — RAWINT4 routes to
  `NativeMoEWrapper` (L338). The concrete kernel is chosen via env `KT_RAWINT4_BACKEND`
  (default tries `AMXInt4_KGroup_MOE` which FAULTS — no AMX on this box; current fallback used
  `avx2` plain or `avxvnni`). Add a value like `avxvnni_packed` selecting the new kernel.
- **Launch:** `MODEL=W4AFP8 KT_METHOD=GPTQ_INT4 KT_WEIGHT_PATH=<nvme1 GPTQ-int4 experts>`
  `KT_RAWINT4_BACKEND=avxvnni_packed GPU_EXPERTS=N bash run_server_int4.sh`. (GPTQ int4 CPU
  experts on nvme1 boot ~3min. The FP8 CPU baseline from /data boots ~50min — see warning §8.)

---

## 6. Validation (DO offline BEFORE any server boot)

Nibble-deinterleave correctness is the #1 risk. Test offline first:
- `int4_scripts/kt_int4_kernel_test.py` — CPU kernel cos-sim vs reference. Adapt to call the new
  backend; require cos-sim ≥ ~0.999 vs the existing (correct) expanded kernel on random inputs.
- (`int4_scripts/w4afp8_gpu_kernel_test.py` is the GPU-side analog — not needed here.)
- Watch the K-ordering: if cos-sim is high but not ~1, suspect the nibble interleave order
  (low=even-k / high=odd-k) or group boundary handling. If it's near-zero, suspect the +128
  correction or scale.
- Rebuild kt-kernel cleanly and CONFIRM the new symbol is in the `.so` (the build can silently
  use a stale object — verify with `nm`/`strings` on the built extension).

Then: bind → boot → `bench/decode_bench.sh 5 256` → compare median vs **10.9**.

---

## 7. Roofline reproduction recipe (non-destructive, on live server)

```bash
echo '<sudo-password>' | sudo -S sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'
# drive a long decode, measure 10ms DRAM bins during steady state:
curl -s http://localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"text":"[gMASK]<sop><|user|>\nWrite a long technical essay.<|assistant|>\n",
       "sampling_params":{"temperature":0.7,"max_new_tokens":1100}}' >/dev/null &
echo '<sudo-password>' | sudo -S bash -c 'sleep 4; perf stat -e ls_any_fills_from_sys.dram_io_all \
  -a -I 10 sleep 6 2>&1 | grep dram_io | \
  awk "{gb=\$2*64/1e9/0.01; b=int(gb/20)*20; print b}" | sort -n | uniq -c'
```
GB/s = `count * 64 / interval_seconds`. Use `timeout`/`sleep` INSIDE sudo bash, never a
foreground `sleep` as a perf-stat target by itself outside (harness blocks foreground sleep;
`perf stat ... sleep N` as the measured command is fine).

VM bandwidth ceiling probe (OMP triad, while server idle): compile the triad in
`scratchpad/bw.c` (3-stream `c=a+3b`, 200M doubles, `-O3 -march=native -fopenmp`,
`OMP_NUM_THREADS=80 OMP_PROC_BIND=spread`) → ~170 GB/s here.

---

## 8. Gotchas / environment facts (save yourself the pain)

- **Hardware:** AMD EPYC 9V84 (Zen4 Genoa), 80 cores 1-thread/core, 2 NUMA nodes
  (node0=cores 0–39 / 322GB, node1=40–79 / 322GB, ~629GB total). AVX512 incl. avx512_vnni,
  avx512_bf16, avx512_vbmi, gfni, vaes — **NO AMX** (the default AMXInt4 backend FAULTS).
  Zen4 double-pumps AVX512 → 512 vs 256 VNNI ≈ same peak COMPUTE (width = front-end relief only).
- **Boot times:** GPTQ-int4 CPU experts from nvme1 = ~3min. FP8 CPU experts from /data = ~50min.
  ⚠️ If you tear down the current FP8-CPU 10.9 baseline server, getting it back costs ~50min.
  So do roofline/measurement on the LIVE server first; only reload when you must test the kernel.
- **Weights are EPHEMERAL** on /cache/nvme0 + /cache/nvme1. Re-download:
  `int4_scripts/download_w4afp8.py`; repack: `int4_scripts/gptq_full_repack.py`.
- **kt worker busy-spin:** `worker_pool.cpp:226` (50ms gate) shows as `clock_gettime` ~48% in
  perf — that's spin, NOT compute. Don't read CPU% as "compute-bound".
- **CPU/GPU overlap ALREADY EXISTS** in `kt_ep_wrapper.py` `apply()` (~L2758–2987): CPU experts
  submit async on `_cpu_stream`, GPU experts run concurrently, then sync+merge. So per-MoE-layer
  latency ≈ max(CPU, GPU+attn)+merge. CPU is the current long-pole (that's why shrinking the
  CPU-expert window via packed-int4 helps). Built-in timer: `SGLANG_KT_HYBRID_TIMING=1`
  (+`_DEEP=1`, tp_rank0, layers 0/5/20/35) logs per-layer submit/mask/gpu/sync/merge/cpu_wait —
  use it to confirm cpu_wait drops after the kernel lands, and to find the new balance point.
- **CPU experts pinned to tp_rank==0** (`kt_ep_wrapper.py` L2784/L2873); GPU experts are
  TP-replicated (`num_local_experts = global_num_experts`, L375) — true EP is a code change, not
  a flag (separate lever, see memory `glm52-decode-bottleneck-v2`).
- perf: `echo '<sudo-password>' | sudo -S sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'`.

---

## 9. Expected outcome & honesty

- Best case: CPU-expert window memory traffic ~halves → CPU segment time drops toward ~½ (to the
  extent it's memory-bound, which the roofline says it largely is) → since CPU is the overlap
  long-pole, decode could rise toward ~13–16 tok/s AND CPU RAM halves. After it lands, re-balance
  GPU_EXPERTS (cpu_wait→0) and re-measure — the bottleneck may shift to GPU+attn.
- Worst case: unpack overhead or layout mismatch eats the win, or after halving CPU the GPU
  segment becomes the bound (then this caps out and you pivot to the EP / hot-placement levers).
- Effort: days. ~70% reused (quant, correction, scales, layout parse, threading, bindings).
  Only the packed BufferB + nibble-unpack inner loop is new.

Related memory: `glm52-decode-bottleneck-v2`, `glm52-w4afp8-not-runnable`,
`glm52-decode-bottleneck`. Bench: `bench/decode_bench.sh`. Baseline to beat: **10.9 tok/s**.
