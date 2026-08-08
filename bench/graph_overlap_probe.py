#!/usr/bin/env python3
"""Does a forked host->device gather overlap with compute inside a CUDA graph?

This is the root-cause probe for the expert prefetcher. In the server the
gather costs full price (0.175 ms/expert exposed vs 0.197 ms standalone) even
though it is forked onto its own stream a whole layer ahead of its consumer.
Three explanations survived inspection:

  H1  SM OCCUPANCY. The gather launches grid(blocks, 2*n_tp, n_slot) = 1024
      blocks at the default blocks=64, on a GPU with 132 SMs. It may simply
      seize the machine, so "concurrent" compute queues behind it.
  H2  MEMORY SYSTEM. PCIe reads into pinned host pages interfere with whatever
      else is touching memory, and the two serialize below the hardware limit.
  H3  GRAPH SCHEDULING. Sibling branches of a captured graph do not actually
      run concurrently here, whatever the DAG says.

None of these needs the model, so none of them needs an hour-long boot. The
structure is what matters: fork -> gather -> join, with compute on the main
stream in between, all inside a single capture.

The compute kernel is deliberately PURE ARITHMETIC -- an FMA loop with no
memory traffic at all. That is what separates H1 from H2: if a kernel that
touches no memory still gets slowed by the gather, the memory system is
irrelevant and the contention is for SMs.

    overlap = (t_compute + t_gather - t_both) / min(t_compute, t_gather)
      1.0  perfect -- the shorter one is entirely hidden
      0.0  none    -- they run back to back

Sweeping the gather's block count then says WHY: if overlap improves as blocks
fall, the gather was crowding the GPU out (H1) and throttling it is the fix.
"""
import argparse, json, os, statistics, sys
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// Vectorised host->device pull over PCIe, exactly the shape the real expert
// gather uses: a device-side pointer read through UVA, grid-stride so the
// block count is a free parameter rather than a function of the size.
__global__ void k_pull(const uint4* __restrict__ src, uint4* __restrict__ dst,
                       long n_vec) {
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
       i += (long)gridDim.x * blockDim.x) {
    dst[i] = src[i];
  }
}

// Pure arithmetic. No loads, no stores until the very end, and the result is
// guarded so the compiler cannot delete the loop. This kernel cannot contend
// for memory bandwidth -- it has none -- so any slowdown it suffers when run
// alongside the gather is SM contention and nothing else.
__global__ void k_burn(float* __restrict__ out, long iters, int dummy) {
  float a = threadIdx.x * 1e-3f, b = 1.000001f, c = 0.999999f;
  for (long i = 0; i < iters; ++i) { a = fmaf(a, b, c); a = fmaf(a, c, b); }
  if (dummy && a == 12345.678f) out[blockIdx.x] = a;
}

void pull(torch::Tensor src_host_ptr, torch::Tensor dst, long n_vec, int blocks) {
  const uint4* s = reinterpret_cast<const uint4*>(src_host_ptr.data_ptr<long>()[0]);
  k_pull<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      s, reinterpret_cast<uint4*>(dst.data_ptr()), n_vec);
}

void burn(torch::Tensor out, long iters, int blocks) {
  k_burn<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<float>(), iters, 0);
}
"""


def build():
    return load_inline(
        name="graph_overlap_probe",
        cpp_sources="void pull(torch::Tensor, torch::Tensor, long, int);\n"
                    "void burn(torch::Tensor, long, int);",
        cuda_sources=SRC, functions=["pull", "burn"],
        extra_cuda_cflags=["-O3"], verbose=False,
    )


def timed_graph(fn, dev, iters=20, warmup=5):
    """Capture fn() into a graph and return median replay ms."""
    s = torch.cuda.Stream(device=dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream(dev).wait_stream(s)
    torch.cuda.synchronize(dev)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize(dev)

    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize(dev)
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--host-mb", type=int, default=512)
    ap.add_argument("--move-mb", type=int, default=64,
                    help="bytes the gather pulls per replay")
    ap.add_argument("--blocks", default="8,32,64,128,512",
                    help="gather block counts to sweep")
    ap.add_argument("--burn-blocks", type=int, default=132,
                    help="compute kernel blocks; default = one wave on an H100")
    ap.add_argument("--burn-iters", type=int, default=0,
                    help="0 = auto-tune so compute ~= gather duration")
    ap.add_argument("--out", default="bench/profile_out/graph_overlap.json")
    a = ap.parse_args()

    dev = a.device
    torch.cuda.set_device(dev)
    ext = build()

    # Pinned host source. The real store is cudaHostRegister'd kt memory; what
    # matters for this probe is only that the reads cross PCIe.
    host = torch.empty(a.host_mb * 1024 * 1024, dtype=torch.uint8,
                       pin_memory=True)
    host_ptr = torch.tensor([host.data_ptr()], dtype=torch.long)
    dst = torch.empty(a.move_mb * 1024 * 1024, dtype=torch.uint8, device=f"cuda:{dev}")
    burn_out = torch.zeros(4096, dtype=torch.float32, device=f"cuda:{dev}")
    n_vec = (a.move_mb * 1024 * 1024) // 16

    blocks_list = [int(x) for x in a.blocks.split(",") if x.strip()]

    # Auto-tune the compute kernel to roughly match the gather at the default
    # block count. Equal durations is the regime where an overlap failure is
    # most visible: if one side dominates, near-perfect and near-zero overlap
    # produce nearly the same total and the test says nothing.
    ref_gather = timed_graph(
        lambda: ext.pull(host_ptr, dst, n_vec, 64), dev)
    iters = a.burn_iters
    if iters == 0:
        iters = 20000
        for _ in range(12):
            t = timed_graph(lambda: ext.burn(burn_out, iters, a.burn_blocks), dev)
            if abs(t - ref_gather) / ref_gather < 0.12:
                break
            iters = max(int(iters * (ref_gather / max(t, 1e-6))), 100)
    t_burn = timed_graph(lambda: ext.burn(burn_out, iters, a.burn_blocks), dev)

    res = {
        "device": torch.cuda.get_device_name(dev),
        "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
        "move_mb": a.move_mb, "burn_blocks": a.burn_blocks, "burn_iters": iters,
        "t_compute_ms": t_burn, "rows": [],
    }

    pull_stream = torch.cuda.Stream(device=dev)

    for blocks in blocks_list:
        t_g = timed_graph(lambda b=blocks: ext.pull(host_ptr, dst, n_vec, b), dev)

        # The structure under test: fork the gather onto its own stream, run
        # compute on the main stream, then join. This is exactly what
        # kt_ep_wrapper does -- pf_stream.wait_stream(current), kernel,
        # pf_done.record, and later current.wait_event(pf_done). The join is
        # not optional inside a capture; CUDA rejects an unjoined branch with
        # cudaErrorStreamCaptureUnjoined, which is what killed the WAIT=0 boot.
        def both(b=blocks):
            cur = torch.cuda.current_stream(dev)
            ev = torch.cuda.Event()
            pull_stream.wait_stream(cur)
            with torch.cuda.stream(pull_stream):
                ext.pull(host_ptr, dst, n_vec, b)
                ev.record(pull_stream)
            ext.burn(burn_out, iters, a.burn_blocks)
            cur.wait_event(ev)

        t_both = timed_graph(both, dev)
        serial = t_g + t_burn
        overlap = (serial - t_both) / min(t_g, t_burn)
        res["rows"].append({
            "gather_blocks": blocks,
            "grid_total": blocks,
            "t_gather_ms": t_g,
            "gather_gbs": (a.move_mb / 1024.0) / (t_g / 1000.0),
            "t_both_ms": t_both,
            "t_serial_ms": serial,
            "overlap_frac": overlap,
        })

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)

    print(f"{res['device']}, {res['sms']} SMs")
    print(f"compute kernel alone: {t_burn:.3f} ms "
          f"({a.burn_blocks} blocks, {iters} iters)")
    print(f"moving {a.move_mb} MB per replay from pinned host memory\n")
    hdr = (f"{'blocks':>7}{'gather ms':>11}{'GB/s':>8}{'both ms':>10}"
           f"{'serial ms':>11}{'overlap':>9}")
    print(hdr); print("-" * len(hdr))
    for r in res["rows"]:
        print(f"{r['gather_blocks']:>7}{r['t_gather_ms']:>11.3f}"
              f"{r['gather_gbs']:>8.1f}{r['t_both_ms']:>10.3f}"
              f"{r['t_serial_ms']:>11.3f}{r['overlap_frac']:>8.0%}")
    print()
    print("overlap 100% = gather fully hidden behind compute")
    print("overlap   0% = they ran back to back, no concurrency at all")
    print("compute kernel does ZERO memory traffic, so any slowdown it")
    print("suffers is SM contention, not bandwidth.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
