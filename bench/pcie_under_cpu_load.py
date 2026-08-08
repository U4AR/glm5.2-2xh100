#!/usr/bin/env python3
"""Does a saturated CPU slow down the GPU's reads of host memory?

Last untested resource in the prefetcher investigation. Everything else is
eliminated: link capacity, fence placement, block count, CUDA host nodes, and
the byte-granular w2 kernel (a genuine defect, fixed, worth 2.2 ms of the 22).
The gather still comes out ~90% exposed in the server while every idle-GPU
probe says it should hide almost completely.

The one thing an idle-GPU probe cannot reproduce is the state of the host
memory system at the moment the gather runs. The prefetch is deliberately
scheduled into the window where the GPU is waiting on the CPU experts -- which
is exactly the window in which 80 EPYC cores are streaming expert weights out
of DRAM as fast as they can. The gather reads ITS bytes from that same DRAM,
across PCIe, through the IOMMU.

I previously blamed "DRAM contention" on bandwidth grounds and was wrong to:
DRAM sits at 38% of 379 GB/s, and there is no bandwidth shortage. But average
bandwidth is not the same claim as loaded latency or as achievable PCIe read
throughput under load, and those were never measured. This measures them.

    threads = 0    the idle-GPU probe's world, ~48 GiB/s
    threads = 80   what the gather actually runs in

If the achieved rate collapses as cores are loaded, that is the root cause, it
explains why every standalone probe was optimistic, and it is general: the
prefetch window and the memory-pressure window are the same window by
construction, on any machine.

Working set is sized to exceed the 384 MB L3 so the load is real DRAM traffic
and not a cache resident loop. Kept modest in total: this box has ~4 GB free
with the rest in page cache, and evicting the model weights fakes a slowdown
of its own -- see the profiling-reclaim-trap note.
"""
import argparse, json, os, statistics, sys
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <atomic>
#include <thread>
#include <vector>
#include <cstdlib>
#include <cstring>

__global__ void k_pull(const uint4* __restrict__ src, uint4* __restrict__ dst,
                       long n_vec) {
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
       i += (long)gridDim.x * blockDim.x) {
    dst[i] = src[i];
  }
}

void pull(torch::Tensor src_host_ptr, torch::Tensor dst, long n_vec, int blocks) {
  const uint4* s = reinterpret_cast<const uint4*>(src_host_ptr.data_ptr<long>()[0]);
  k_pull<<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      s, reinterpret_cast<uint4*>(dst.data_ptr()), n_vec);
}

// ---- host-side DRAM load ---------------------------------------------------
// One cache line per step so every access is a miss, which is what a CPU expert
// kernel streaming int4 weights out of DRAM looks like from the memory
// controller's point of view.
static std::atomic<bool> g_run{false};
static std::vector<std::thread> g_threads;
static std::vector<void*> g_bufs;

void load_start(int nthreads, long mb_each) {
  g_run.store(true);
  const size_t bytes = (size_t)mb_each * 1024 * 1024;
  for (int t = 0; t < nthreads; ++t) {
    void* b = aligned_alloc(64, bytes);
    if (!b) break;
    memset(b, 1, bytes);
    g_bufs.push_back(b);
    g_threads.emplace_back([b, bytes]() {
      volatile long sink = 0;
      const long* p = (const long*)b;
      const long n = (long)(bytes / sizeof(long));
      while (g_run.load(std::memory_order_relaxed)) {
        long s = 0;
        for (long i = 0; i < n; i += 8) s += p[i];   // 64 B stride
        sink += s;
      }
    });
  }
}

void load_stop() {
  g_run.store(false);
  for (auto& t : g_threads) if (t.joinable()) t.join();
  g_threads.clear();
  for (auto b : g_bufs) free(b);
  g_bufs.clear();
}
"""

DECL = ("void pull(torch::Tensor, torch::Tensor, long, int);\n"
        "void load_start(int, long);\nvoid load_stop();")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--move-mb", type=int, default=16,
                    help="bytes per gather; a layer moves ~16 MiB")
    ap.add_argument("--host-mb", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--threads", default="0,8,20,40,60,80")
    ap.add_argument("--mb-each", type=int, default=48,
                    help="per-thread working set; must exceed L3 share")
    ap.add_argument("--out", default="bench/profile_out/pcie_under_cpu_load.json")
    a = ap.parse_args()

    dev = a.device
    torch.cuda.set_device(dev)
    ext = load_inline(name="pcie_under_cpu_load", cpp_sources=DECL,
                      cuda_sources=SRC,
                      functions=["pull", "load_start", "load_stop"],
                      extra_cuda_cflags=["-O3"], extra_cflags=["-O3"],
                      extra_ldflags=["-lpthread"], verbose=False)

    host = torch.empty(a.host_mb * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    ptr = torch.tensor([host.data_ptr()], dtype=torch.long)
    dst = torch.empty(a.move_mb * 1024 * 1024, dtype=torch.uint8, device=f"cuda:{dev}")
    n_vec = (a.move_mb * 1024 * 1024) // 16

    def measure():
        for _ in range(5):
            ext.pull(ptr, dst, n_vec, a.blocks)
        torch.cuda.synchronize(dev)
        ts = []
        for _ in range(30):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); ext.pull(ptr, dst, n_vec, a.blocks); e1.record()
            torch.cuda.synchronize(dev)
            ts.append(e0.elapsed_time(e1))
        return statistics.median(ts)

    res = {"device": torch.cuda.get_device_name(dev), "move_mb": a.move_mb,
           "blocks": a.blocks, "mb_each": a.mb_each, "rows": []}
    base = None
    print(f"gather {a.move_mb} MiB from pinned host memory, {a.blocks} blocks")
    print(f"load: N threads x {a.mb_each} MiB, one cache line per step\n")
    hdr = f"{'cpu threads':>12}{'ms':>9}{'GiB/s':>9}{'vs idle':>9}"
    print(hdr); print("-" * len(hdr))
    for n in [int(x) for x in a.threads.split(",") if x.strip()]:
        if n:
            ext.load_start(n, a.mb_each)
            import time as _t
            _t.sleep(2.0)          # let the load reach steady state
        t = measure()
        if n:
            ext.load_stop()
        gbs = (a.move_mb / 1024.0) / (t / 1000.0)
        if base is None:
            base = gbs
        res["rows"].append({"threads": n, "ms": t, "gibs": gbs,
                            "frac_of_idle": gbs / base})
        print(f"{n:>12}{t:>9.3f}{gbs:>9.1f}{gbs/base:>8.0%}")

    print()
    print("The gather is scheduled into the CPU-expert window by design, so the")
    print("bottom row is the regime it actually runs in and the top row is the")
    print("one every standalone probe measured.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
