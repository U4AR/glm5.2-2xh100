# GLM-5.2-FP8 on 2×H100 + KT-Kernel — Handoff

> **UPDATE 2026-06-23 — ✅ DECODE NOW ~8.7 tok/s (was ~3.3), via CUDA graphs.**
> The bottleneck was NOT swap/compute — it was per-step launch/serialization
> overhead (CPU workers spin-wait 72% in `clock_gettime`; eager per-layer launches).
> Enabling CUDA graphs ~2.6×'d decode (3.3 → ~8.5-9.0 tok/s steady), no model change.
> This is now the **default** in `run_server.sh` (just `./run_server.sh`).
> Full write-up + the 3 capture-mode fixes (mem-fraction 0.94, DeepGEMM, disable
> custom-all-reduce) and the **test UI** (`./start_ui.sh`, port 8080):
> see **[PERF_CUDA_GRAPHS.md](PERF_CUDA_GRAPHS.md)**. The §12 swap recipe below still
> applies (the model still needs swap to *load*); CUDA graphs are an orthogonal
> decode-speed win on top of it.

**Status as of 2026-06-22 — ✅ FP8 IS SERVING.**
GLM-5.2-FP8 now serves end-to-end on this box at **~3.4 tok/s** via
**NVMe swap + mem-fraction 0.95**. The earlier "FP8 cannot fit" conclusion was
true *without swap*; swap closes the ~20-80 GB gap. To launch:
`cd /data/models/RunGLM && MEM_FRACTION=0.95 ./run_server.sh`
(swap is auto-enabled on boot by `setup-nvme-caches.service`). Read **§12** for
the working recipe; §11 is the prior (pre-swap) investigation.

TL;DR of the fix: (1) **NVMe swap** = 256 GB on `/cache/nvme0` + 128 GB on
`/cache/nvme1` (384 GB total). The FP8 expert footprint is ~649 GB anon vs 629 GB
RAM, so with no swap the CPU expert build OOM-kills rank 0 at ~layer 68/78. Swap
lets the build finish and lets the cold LRU experts spill to NVMe. `swappiness=10`
keeps hot experts in RAM (swap-out stays ~0; kernel drops page-cache instead).
(2) **`mem-fraction-static` 0.90 → 0.95**: after the 54 GPU experts/card the KV
pool had no room at 0.90 (`RuntimeError: Not enough memory`); 0.95 gives the
~0.47 GB KV pool its headroom (final GPU ~87.4/91 GB per card). (3) Both the swap
files and `vm.swappiness=10` are now baked into
`/usr/local/sbin/setup-nvme-caches.sh` so they persist across reboot.
Perf: cold start ~0.01 tok/s, warms to **~3.4 tok/s** steady-state as the working
set becomes RAM-resident. INT4 (§11) is still the path to *higher* speed (experts
would fit in RAM with no swap) but is not required to run the model anymore.

---

## 1. Hardware / environment (this VM = H100-VM1)
- **GPUs:** 2× NVIDIA H100 NVL, 95.8 GB each (191.6 GB total). Driver 575, CUDA 12.9.
- **CPU:** AMD EPYC 9V84, **80 cores, 2 NUMA nodes** (0:0-39, 1:40-79).
  **No Intel AMX** — kt-kernel uses the AVX-512 path (VNNI+BF16+VBMI all present).
- **RAM:** 629 GB total, no swap. **This is the binding constraint.**
- **Disk:** `/data` (persistent) is **93% full, ~70 GB free**. `/` 28 GB free.
  `/cache/nvme0|1` are root-owned + not mounted as the big NVMe right now — unusable.
  Everything important lives on `/data` (persistent across reboot). Do NOT rely on
  `/dev/shm` or `/mnt` (ephemeral).

## 2. What is installed (all under `/data/models/RunGLM`, persistent)
- **venv:** `/data/models/RunGLM/.venv` (Python 3.12). `source .venv/bin/activate`
  also exports `LD_LIBRARY_PATH=$VENV/lib` — **required** (see §3).
- **sglang-kt** fork + **kt-kernel 0.6.2.post3** (built for this AMD CPU),
  torch 2.9.1+cu128, transformers 5.12.1. Source repo: `./ktransformers/`.
- `kt doctor` passes all checks. Verify anytime: `source .venv/bin/activate && kt doctor`.

## 3. CRITICAL runtime requirement
kt-kernel links **hwloc** and **libnuma**, which we built from source into the venv
prefix (no sudo was available). They are **not** on the system linker path. So:
- `LD_LIBRARY_PATH` must include `/data/models/RunGLM/.venv/lib` for ANY process that
  imports kt_kernel (the server, `kt chat`, etc.). This is already appended to
  `.venv/bin/activate`, so **always `source` the venv** — don't call the binaries directly.

## 4. Model
- Path: `/data/models/GLM-5.2-FP8` (704 GB, 141 shards, FP8 e4m3 block-quant).
- Arch `glm_moe_dsa`: 78 layers, **256 routed experts/layer, 8 active/tok**, MLA + NSA
  (DeepSeek-style sparse attn). Per expert = **36 MB FP8**. 76 MoE layers (ids 3..78).
  - Routed experts total **684 GB**; non-expert weight only **16.9 GB** (MLA is compact).

## 5. The core problem (read this before tuning)
704 GB FP8 model vs **629 GB RAM**. KT puts a few experts on GPU and the rest in CPU RAM:
- GPU fits ~**54 experts/layer** (`gpu_experts=54` → 87 GB/card, validated to load).
- That leaves ~**540 GB** of CPU experts. Plus per-rank torch/CUDA + final-init buffers,
  peak RAM hit **605 GB and OOM-killed rank 0** at the very end of init.
- More GPU experts = less CPU RAM **and** faster, but GPU is already ~full at 54.
  Fewer GPU experts = more CPU RAM (worse). So FP8 is at the ragged edge on this box.

## 6. Current launch config — `run_server.sh`
TP2, `kt-method FP8`, NUMA-sharded (`--kt-threadpool-count 2 --kt-numa-nodes 0 1`,
experts are **split** across nodes, not duplicated), `gpu_experts=54`, `cpuinfer=72`,
`nsa` attention, `fp8_e4m3` KV, `cutlass` FP8 GEMM. The retry added, to fit RAM:
**`--disable-cuda-graph`** (its capture buffers likely caused the OOM, and decode is
CPU-bound here so graphs add little speed) + `max-total-tokens 8192`,
`max-running-requests 2`, `chunked-prefill 2048`. All overridable via env vars.

Expert-build phase takes **~50 min** per launch (AVX-512 FP8 packing, ~37 s/layer).

## 7. Next steps (in priority order)

### A. Resume the trimmed FP8 launch (fastest path; was untested when we stopped)
```bash
cd /data/models/RunGLM
nohup bash run_server.sh > logs/server3.log 2>&1 &
# watch for ready / OOM:
tail -f logs/server3.log        # look for "Uvicorn running" / "Application startup complete"
# resources:  watch -n5 'nvidia-smi --query-gpu=memory.used --format=csv,noheader; free -g | grep Mem'
```
Watch `free -g` near the end of expert-build; if `available` approaches ~0 around the
last layers, it will OOM again → go to B. If it reaches "ready", test with §8.

### B. If it still OOMs at final init — two levers
1. **More aggressive trim, same FP8:** also set `GPU_EXPERTS=56` (moves ~5.5 GB CPU→GPU;
   check it still fits VRAM — 54 was 87 GB/card, 56 ≈ 90 GB/card) and keep graphs off:
   `GPU_EXPERTS=56 bash run_server.sh`. Marginal; may or may not close the gap.
2. **INT4 CPU experts (the proper fix — fits with huge margin, likely FASTER on this
   no-AMX CPU):** CPU experts drop from 540 GB to ~170 GB. **Blocked by disk:** needs a
   `kt quant` step writing ~170 GB and `/data` has only ~70 GB free. **First free ≥150 GB
   on `/data`** (ask the user what's removable), then quantize and switch
   `--kt-method` to an INT4 variant (`MOE_INT4`/`GPTQ_INT4`/`LLAMAFILE`; **not** AMXINT4 —
   that needs Intel AMX). Inspect `kt quant --help` and
   `.venv/.../sglang/srt/layers/moe/kt_ep_wrapper.py` for the exact weight-path workflow.

### C. If it runs but you want more speed later
Once stable with headroom, try re-enabling cuda graphs (`DISABLE_CUDA_GRAPH=0`) and/or
raising `max-total-tokens`, watching RAM/VRAM. Expect CPU-bound throughput regardless
(most of the 8 experts/token are on CPU). `--kt-expert-placement-strategy frequency` +
`--kt-enable-dynamic-expert-update` (already on) improves GPU hit-rate over time.

## 8. Test once "ready"
```bash
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"GLM5.2","messages":[{"role":"user","content":"hi, who are you?"}],
       "temperature":1.0,"top_p":0.95,"stream":false}' | head
# or interactive:  source .venv/bin/activate && kt chat --port 8000 --model GLM5.2
```

## 9. Stop the server
```bash
for pid in $(pgrep -f "python -m sglang|sglang::"); do kill -9 "$pid"; done
# verify: nvidia-smi  (0 MiB) ;  free -g  ;  ls /dev/shm
```

## 10. Gotchas already handled (don't redo)
- A stale **209 GB `/dev/shm/psai_patch_cache/grids.npy`** (+ its `dataset_viewer.py`,
  port 7860) was eating RAM and was removed. Re-check `df -h /dev/shm` and
  `ps aux --sort=-rss | head` before launching — RAM headroom is everything here.
- `transformers` is **5.12.1** (the fork's pin), intentionally NOT the tutorial's 5.3.0.
- Reboot-safe: venv + repo + model are on `/data`; CUDA at `/usr/local/cuda-12.9`. After
  reboot just `source .venv/bin/activate` and launch. (If kt-kernel import fails with
  `libhwloc.so.15 not found`, the activate LD_LIBRARY_PATH line was lost — re-add it.)

---

## 11. 2026-06-22 findings — READ BEFORE TOUCHING INT4 (supersedes §7B)

### 11.1 FP8 is a hard no (confirmed twice)
Fresh-boot re-run of `run_server.sh` (trimmed: cuda-graph off, max-total-tokens 8192,
gpu_experts=54) **OOM-killed rank 0 at ~layer 68/78** of the expert build. RAM `used`
(anonymous) hit ~513 GB by layer 57 and kept climbing ~8 GB per MoE layer; `available`
crossed zero around layer 68. Page cache (mmap'd shards, freed per layer) is reclaimable
but the **anonymous expert RAM alone exceeds 629 GB**. Raising gpu_experts 54→~58 only
moves ~16 GB CPU→GPU (and may not fit VRAM) — nowhere near the ~60–80 GB gap. **Do not
retry FP8.**

### 11.2 NVMe scratch is now available + persistent
- `/cache/nvme0` and `/cache/nvme1` = **two ext4 3.5T local NVMe disks**, owned `bel`.
  They were raw/unmounted; formatted + mounted on 2026-06-22.
- Restored **`/usr/local/sbin/setup-nvme-caches.sh`** (was missing → `nvme-caches.service`
  failed 203/EXEC). Source of truth: `/data/models/RunGLM/setup-nvme-caches.sh`. The
  service is `enabled` + verified `active`; it **mkfs only when unformatted** (won't wipe
  scratch on reboot) then mounts + chowns to `bel`. So the drives auto-mount on boot.
- Disk is no longer the binding constraint. Use `/cache/nvme0` for any int4 weights.
- sudo password for this box: `<sudo-password>` (passwordless sudo is NOT configured).

### 11.3 The INT4 reality on THIS box (AMD EPYC, no AMX) + THIS kt-kernel build
Verified empirically from the installed package:
- `--kt-method` choices: `AMXINT4, AMXINT8, RAWINT4, FP8, BF16, FP8_PERCHANNEL,
  LLAMAFILE, MOE_INT4, MOE_INT8`.
- Compiled-in support on this build (`kt_kernel.utils.amx` / `.moe_kernel`):
  - `_HAS_RAWINT4_SUPPORT=True` (+ AVX2 / AVX-VNNI-256 raw-int4 variants True) ✅
  - `_HAS_INT4_SUPPORT (MOE_INT4) = False` ❌ not compiled
  - LLAMAFILE: **no llamafile/gguf symbols in `kt_kernel_ext`** ❌ effectively not built
  - AMXINT4/AMXINT8: need Intel AMX — this CPU has none ❌
- So **RAWINT4 is the only usable INT4 expert backend here.**
- Backend → on-disk format (`kt-kernel/python/utils/amx.py` `NativeMoEWrapper._create_loader`
  + `utils/loader.py`):
  - RAWINT4 → `CompressedSafeTensorLoader` → keys `...experts.{e}.{up,gate,down}_proj.weight_packed`
    + `weight_scale` = **compressed-tensors W4A16** layout (pre-quantized; the wrapper
    explicitly raises "RAWINT4 expects pre-quantized safetensor weights" — no on-the-fly quant).
  - AMXINT4 → `SafeTensorLoader` → GGUF-style `blk.{L}.ffn_{up,gate,down}_exps.{e}.numa.{n}.weight`.
    This is what `convert_cpu_weights.py --quant-method int4` writes. **Different format from
    RAWINT4 — not interchangeable.**
- `kt quant` CLI only does int4/int8 → AMX format (dir named `-AMXINT4-`). `convert_cpu_weights.py`
  maps int4→AMXINT4, moe_int4→MOE_INT4 (uncompiled). **Neither produces the RAWINT4
  compressed-tensors format.** No local script writes `weight_packed` (only a unit-test
  reference quantizer: `kt-kernel/test/per_commit/test_moe_rawint4_accuracy.py`).
- Serving always needs BOTH: `--model-path` = **safetensors** for GPU experts + non-expert
  (MLA/attn/embed) weights (so the **704 GB FP8 must stay** for the GPU side), and
  `--kt-weight-path` = the CPU-expert weights in the backend's format.

### 11.4 Off-the-shelf GLM-5.2 int4 checkpoints (none load as-is here)
- `cyankiwi/GLM-5.2-AWQ-INT4` (AWQ) — no non-AMX AWQ loader; AWQ converter targets AMX.
- `unsloth/GLM-5.2-GGUF` (GGUF) — would need the LLAMAFILE backend, **not compiled**.
- No compressed-tensors **W4A16** / GPTQ GLM-5.2 found yet (W4A16 exists for GLM-4.6/4.7).

### 11.5 Real paths forward (pick one next session)
1. **Produce compressed-tensors W4A16 locally → serve RAWINT4** (uses what's compiled).
   Quantize GLM-5.2 FP8→W4A16 with **llm-compressor** (not installed; needs calib data +
   a heavy pass over 789B params; dequant FP8→BF16 first). Output (~135–170 GB) to
   `/cache/nvme0`. Then `--kt-method RAWINT4 --kt-weight-path <that dir>`, keep FP8 as
   `--model-path`. Highest confidence it runs on this build, but real quant effort. **Recommended.**
2. **Rebuild kt-kernel with LLAMAFILE (and/or MOE_INT4) enabled.** Then either download
   `unsloth/GLM-5.2-GGUF` → `--kt-method LLAMAFILE`, or `convert_cpu_weights.py --quant-method
   moe_int4` → `--kt-method MOE_INT4`. Avoids a calibration quant but is a from-source rebuild
   (verify build flags expose these backends; `./ktransformers/kt-kernel/install.sh`).
3. **Wait for / find a published GLM-5.2 compressed-tensors W4A16 or GPTQ** → download to
   `/cache/nvme0`, serve `RAWINT4` / `GPTQ_INT4`. Zero local quant; blocked on availability.
4. (Different stack) Serve `unsloth/GLM-5.2-GGUF` with **llama.cpp** directly (offload to the
   2×H100). Abandons sglang-kt; verify llama.cpp GLM-5.2/NSA support first.

---

## 12. ✅ WORKING RECIPE — FP8 via NVMe swap (2026-06-22)

**This supersedes §11's "FP8 is a hard no."** FP8 *does* serve once you add swap.

### Launch
```bash
cd /data/models/RunGLM
MEM_FRACTION=0.95 ./run_server.sh        # default 0.90 fails KV-pool init
```
Swap is auto-enabled at boot by `setup-nvme-caches.service`; to enable manually:
```bash
sudo swapon /cache/nvme0/swapfile /cache/nvme1/swapfile   # 256G + 128G
sudo sysctl vm.swappiness=10
```

### Why it works
- **RAM gap:** FP8 CPU experts ≈ 649 GB anon vs 629 GB RAM → no-swap build OOM-kills
  rank 0 at ~layer 68/78. 384 GB NVMe swap absorbs the overflow; peak swap use during
  build ≈ 289 GB, steady-state serving ≈ 300 GB.
- **swappiness=10:** hot experts stay resident; kernel reclaims page-cache rather than
  swapping anon out (observed swap-out ≈ 0, swap-in 0.7–0.9 GB/s while warming).
- **mem-fraction 0.95:** GPU holds 54 experts/layer/card ≈ 86.9 GB; 0.90 budget (86.2 GB)
  left nothing for the KV pool → `RuntimeError: Not enough memory`. 0.95 budget (91 GB)
  leaves room for the 0.47 GB KV pool. Final GPU ≈ 87.4/95.8 GB per card.

### Observed behavior
- Full load (safetensors + 78-layer CPU expert build) ≈ 50–60 min. Watch for
  `[KT] Recreated NativeMoEWrapper loader for layer N` (build) then
  `KV Cache is allocated` then `Uvicorn running on http://0.0.0.0:8000`.
- **Throughput: cold start ~0.01 tok/s, warms to ~3.4 tok/s** steady-state as the
  per-conversation expert working set becomes RAM-resident. First request after boot
  is slow (faults ~12 GB of experts from NVMe); subsequent ones are faster.
- Verify: `curl -s localhost:8000/v1/models` and a `/v1/chat/completions` call.
  Note GLM-5.2 is a reasoning model — short `max_tokens` returns text in
  `reasoning_content` with `content:null` and `finish_reason:length`; raise `max_tokens`.

### To go FASTER (optional)
Pursue an INT4 CPU-expert path (§11.5) so experts fit in RAM with **no swap**
(removes the NVMe fault penalty → should be well above 3.4 tok/s). Not required to
run the model; FP8 is fully functional today.
