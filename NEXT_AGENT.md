# Continue here — GLM-5.2-FP8 throughput work (handoff for next agent)

**Last worked:** 2026-06-23. **Box:** `H100-VM1` (10.0.0.4), 2× H100 NVL (96GB), AMD EPYC 9V84
(80 cores / 2 NUMA, **no AMX** → AVX-512), 629GB RAM, kt-kernel CPU+GPU heterogeneous MoE.

---

## TL;DR — where things stand

- **Goal:** serve GLM-5.2-**FP8** (704GB) at ~10 tok/s without changing the model. Was stuck at ~3.3 tok/s.
- **Done:** diagnosed the real bottleneck and **enabled CUDA graphs → decode is now ~8.5–9.0 tok/s (~2.6×)**,
  output verified coherent. This is now the **default** in `run_server.sh`.
- **Remaining:** ~1.3 tok/s short of 10. Three stackable levers left (section "Next work").
- **Read first:** [PERF_CUDA_GRAPHS.md](PERF_CUDA_GRAPHS.md) (full write-up + the 3 capture-mode fixes).
  [HANDOFF.md](HANDOFF.md) §12 has the original swap-based load recipe (still needed to *load* the model).

## The verified diagnosis (don't re-derive this)

Decode was **NOT** swap-bound, **NOT** CPU-compute-bound, **NOT** memory-bandwidth-bound. Measured during
steady decode: DRAM BW ~29 GB/s (7% of peak), NUMA-far 0.66%, FLOPs 0.8%, but **`perf record` top symbol =
`clock_gettime` 72.67%** (kt workers spin-wait 50ms in `worker_pool.cpp:223-231`), and **GPU0=17% / GPU1=100%**.
Conclusion: **per-step launch/serialization overhead** was the wall. CUDA graphs (collapsing eager per-layer
launches into one replay) fixed it. The kt CPU-expert MoE survives graph capture/replay.

## How to bring it back up (cold start, ~60 min)

```bash
cd /data/models/RunGLM
./run_server.sh                 # defaults = fast config (CUDA graphs on). Serves on :8000.
# wait ~60 min; ready when log says "The server is fired up and ready to roll!"
./start_ui.sh                   # chat test UI on :8080 (open forwarded port in browser)
```
Verify it's actually fast:
```bash
grep -oE "cuda graph: (True|False)" serve_*.log | sort | uniq -c    # want mostly True
grep "gen throughput" serve_*.log | tail                            # want ~8.5-9.0 tok/s
```
The fast config = graphs on + `mem-fraction 0.94` + `SGLANG_ENABLE_JIT_DEEPGEMM=1` +
`--disable-custom-all-reduce` + `--cuda-graph-max-bs 1`. All wired into `run_server.sh` defaults;
`DISABLE_CUDA_GRAPH=1 ./run_server.sh` reverts to the old slow eager path.

## Next work to reach/exceed 10 tok/s (priority order)

1. **GPU0=17% / GPU1=100% imbalance** — biggest suspected win, and **diagnosable on the RUNNING server (no
   restart)**. One H100 saturates while the other idles during decode. Start: `nvidia-smi dmon -s u` during a
   long generation; check per-rank work split (NSA attention head split? kt CPU-expert combine pinned to rank
   1? `--disable-shared-experts-fusion` is set). If you can balance it, decode could jump significantly.
2. **MTP speculative decoding** — model has `num_nextn_predict_layers=1` (layer-78 nextn weights). Currently
   off (`speculative_algorithm=None`) and the nextn module fails to import: **`No module named 'tilelang'`**.
   Try `pip install tilelang` in `.venv`, then `--speculative-algorithm EAGLE`/NEXTN with the nextn head.
   Since decode is fixed-overhead-dominated, spec-decode is a strong fit (1.5–2.5× on accepted tokens). Risk:
   integration with kt CPU-expert path + cuda graphs is unproven.
3. **Trim the 50ms worker spin** (`ktransformers/kt-kernel/cpu_backend/worker_pool.cpp:226`, the `> 50` ms
   gate). Frees the wasted 72% CPU; modest tps but big efficiency. Needs a kt-kernel rebuild (see below).

## Gotchas that cost time last session

- **Each server restart ≈ 60 min** (loads 571GB FP8 experts; transiently touches ~90GB swap). Budget for it;
  don't iterate blindly. Enabling CUDA graphs was 4 reload cycles of capture-mode whack-a-mole.
- **The harness BLOCKS foreground `sleep`.** Never put `sleep` in a Bash command — it silently rejects the
  WHOLE command (I lost 33 min when a launch line with `sleep 6` never ran). Use `run_in_background: true`
  watchers instead, and a long-interval poll.
- **`pgrep -f "sglang..."` self-matches your own shell command.** Use a specific pattern like
  `pgrep -f "launch_server --model-path"`, or check `ps`.
- **`perf` needs** `echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid` (sudo password: `<sudo-password>`,
  user's single-user box). AMD bandwidth counters: `ls_any_fills_from_sys.dram_io_near/far` (×64B).
- **kt-kernel rebuild** (only if editing C++): the build needs source-built hwloc/numa in `.venv`:
  ```bash
  cd ktransformers/kt-kernel && source .venv/bin/activate   # NOTE: actually use /data/models/RunGLM/.venv
  export PKG_CONFIG_PATH=/data/models/RunGLM/.venv/lib/pkgconfig CMAKE_PREFIX_PATH=/data/models/RunGLM/.venv \
         CMAKE_LIBRARY_PATH=/data/models/RunGLM/.venv/lib CMAKE_INCLUDE_PATH=/data/models/RunGLM/.venv/include
  bash install.sh build          # do a CLEAN build; --no-clean leaves STALE objects (patch won't take)
  ```
  Verify the patch is in the .so: `strings .venv/.../kt_kernel_ext*.so | grep KT-PATCH`.

## What's already patched in the running build ("Option C")

kt-kernel was patched so GPU experts are NOT also stored on the host (saves ~145GB host RAM). Files:
`operators/amx/fp8-moe.hpp` (real FP8 class = AMXFP8_MOE) + `.venv/.../kt_kernel/utils/amx.py` (mask
propagation). Backups exist as `.bak-preC`. There's a debug `fprintf` `[KT-PATCH-DBG]` printing per-layer
`gpu_trues=54` at load — harmless but noisy; remove it (one line in fp8-moe.hpp + rebuild) if you want clean
logs. This patch is orthogonal to CUDA graphs and should stay.

## File map (all under /data/models/RunGLM, persistent)

| File | What |
|---|---|
| `run_server.sh` | launches model on :8000; **defaults to fast CUDA-graph config** |
| `start_ui.sh` + `chat_ui.py` | zero-dep test chat UI on :8080 (proxies to :8000, streams, shows reasoning) |
| `PERF_CUDA_GRAPHS.md` | full perf write-up + fixes + how-to |
| `HANDOFF.md` | original deployment handoff (swap recipe in §12) |
| `serve_fp8_cudagraph4.log` | the good run's log (8.7 tok/s); earlier `cudagraph[1-3].log` = the failed attempts |
| `NEXT_AGENT.md` | this file |

Persistent memory for this project: `/home/bel/.claude/projects/-data-models-RunGLM/memory/`
(`MEMORY.md` index → `glm52-decode-bottleneck.md`, `glm52-kt-deployment.md`).
