# GLM-5.2-FP8 decode speedup: 3.3 → ~8.7 tok/s via CUDA graphs

**Date:** 2026-06-23
**Box:** 2× H100 NVL (96GB), AMD EPYC 9V84 (80 cores, 2 NUMA), 629GB RAM, kt-kernel CPU+GPU MoE
**Result:** FP8 (704GB) decode went from a steady **~3.3 tok/s** to a steady **~8.5–9.0 tok/s** (~2.6×),
**without changing the model**, by enabling CUDA graphs (plus the 3 fixes required to make capture succeed).

---

## 1. The bottleneck was misdiagnosed twice — then measured

Earlier hypotheses (swap-bound, then CPU-compute-bound) were **both wrong**. Measured during steady decode:

| Measurement | Tool | Result | Conclusion |
|---|---|---|---|
| DRAM bandwidth | `perf stat -e ls_any_fills_from_sys.dram_io_near/far` ×64B | **~29 GB/s** of ~400 peak | not memory-bound (7%) |
| NUMA-far fills | same counters | **0.66%** | not NUMA-bound |
| FLOPs | counters vs AVX-512 peak | **~0.8%** | not compute-bound |
| Top CPU symbol | `perf record -a -g` | **`clock_gettime` = 72.67%** | CPU workers **busy-spin** |
| GPU SM util | `nvidia-smi dmon` | GPU0 **17%**, GPU1 **100%** | TP imbalance + serialization |

The CPU "busy" time (`us=92%`) was **72% spin-wait**, not work: kt-kernel workers spin-poll for **50 ms**
(`worker_pool.cpp:223-231`, `if (duration > 50) cv.wait`) after finishing their experts. The expert math is
cheap; the real cost was **per-token forward latency / per-layer launch+serialization overhead**.

To enable `perf` on this box: `echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid`.

## 2. The fix: CUDA graphs

Decode ran **eager** (`cuda graph: False`), launching every kernel of all 78 layers from Python each token.
CUDA graphs collapse that into a single captured replay. Because the bottleneck was launch/serialization
overhead, this ~tripled decode throughput. The kt-kernel CPU-expert MoE **survives capture/replay** and output
stays coherent (verified).

### Three fixes were required for graph capture to succeed
Graph-capture mode activates code paths that the eager path never exercised here. Each failed a full boot
(~60 min reload) until fixed:

1. **`mem-fraction-static = 0.94`** (NOT lower). It is the **weights+KV** budget and the graph-capture reserve
   sits *inside* it. 0.88 starved the KV pool → `RuntimeError: Not enough memory. Please ... increase
   --mem-fraction-static`. GLM weights ≈84GB/card, so ≥0.93 is needed; 0.94 leaves ~5.8GB KV headroom.
2. **`SGLANG_ENABLE_JIT_DEEPGEMM=1`**. Under capture, the NSA indexer turns on a dual-stream path
   (`nsa_indexer.py:1016`, gated on `get_is_capture_mode()`) that calls `deep_gemm.get_num_sms()`. With
   DeepGEMM off (cutlass mode) → `NameError: name 'deep_gemm' is not defined`. DeepGEMM is installed and
   works on H100; it JIT-compiles a few kernels at startup (~minutes, cached after).
3. **`--disable-custom-all-reduce`**. Custom all-reduce registers CUDA-IPC graph buffers via
   `get_graph_buffer_ipc_meta`, which fails capture: `RuntimeError: invalid argument → Capture cuda graph
   failed`. NCCL all-reduce is used instead and is fast on 2× H100 NVLink.

Also: **`--cuda-graph-max-bs 1`** — we only decode single-stream (max-running 2), so capturing just batch-1
minimizes capture time and VRAM (default would capture bs 1..256).

## 3. How to run (now the default)

`run_server.sh` defaults to the fast config. Just:

```bash
./run_server.sh
```

Equivalent explicit form (what the defaults expand to):

```bash
SGLANG_ENABLE_JIT_DEEPGEMM=1 \
DISABLE_CUDA_GRAPH=0 \
MEM_FRACTION=0.94 \
CUDA_GRAPH_MAX_BS=1 \
GPU_EXPERTS=54 CPUINFER=72 \
./run_server.sh
```

Fall back to the old slow eager path with `DISABLE_CUDA_GRAPH=1 ./run_server.sh`.

**Boot takes ~60 min** (loads 571GB of FP8 CPU experts; transiently touches ~90GB swap), then a short
DeepGEMM JIT + graph-capture phase, then `The server is fired up and ready to roll!` on port **8000**.

Verify it's actually using graphs:
```bash
grep -oE "cuda graph: (True|False)" serve_*.log | sort | uniq -c   # expect mostly True
grep "gen throughput" serve_*.log | tail               # expect ~8.5-9.0 tok/s steady
```

## 4. Test UI

```bash
./start_ui.sh         # serves a chat UI on port 8080, proxies to the model on 8000
```
Open the forwarded port 8080 in your browser (VS Code auto-forwards). See `chat_ui.py`.

## 5. Still on the table to reach/exceed 10 tok/s (all stack on top of graphs)

- **GPU0=17% / GPU1=100% imbalance** — one H100 idles while the other saturates; rebalancing could be the
  biggest remaining win. Diagnosable on the running server (no restart).
- **MTP speculative decoding** — model has `num_nextn_predict_layers=1`, but the nextn module needs
  `tilelang` (currently `No module named 'tilelang'`), and `--speculative-algorithm` is unset.
- **Trim the 50 ms worker spin** (`worker_pool.cpp:226`) — frees wasted CPU; modest tps, big efficiency.
