#!/usr/bin/env python3
"""Do HOST NODES inside a CUDA graph destroy sibling-branch concurrency?

This is attempt five at the prefetcher root cause, and the first one aimed at a
difference I can point to in the source rather than a mechanism I imagined.

The story so far. In the server, a forked host->device expert gather costs full
price -- 0.175 ms/expert exposed, against 0.197 ms/expert measured standalone
on an idle GPU. It is not being slowed down; the step simply grows by the whole
duration of a transfer that was forked a full layer ahead of its consumer. That
is the signature of serialization, not contention. bench/graph_overlap_probe.py
reproduced the same fork/gather/join structure standalone and got 100% overlap.

So the probe disagreed with the server, and the interesting question became:
what does the server's graph contain that the probe's did not?

Answer, from the source rather than from reasoning:

    cpu_backend/cpuinfer.h:94   submit_with_cuda_stream ->
                                cudaLaunchHostFunc(current_stream, func, args)
    cpu_backend/cpuinfer.h:120  sync_with_cuda_stream   ->
                                cudaLaunchHostFunc(current_stream, &sync_, args)

    static void sync_(void*) { ... task_queue_->sync(allow_n_pending); }

Both land on the CURRENT stream, so under capture both become HOST NODES in the
decode graph -- two per MoE layer, ~150 per replay. And sync_ is not a marker:
it blocks the calling thread inside task_queue_->sync() until the CPU experts
have finished, which on this box is the majority of the step.

The hypothesis this file tests is therefore mechanical and general:

    H  A blocking host node on the main stream stalls the WHOLE graph
       execution, not just its own branch. A gather forked onto a sibling
       stream cannot make progress while the driver is parked inside a host
       callback, so it only runs in the gaps -- and ends up fully exposed
       no matter how early it was issued.

The experiment holds everything constant and flips ONE bit: whether the main
stream's per-layer work includes a blocking host node. Same gather, same
compute, same fork/join, same capture.

    HOSTNODE=off   expect high overlap  (this is graph_overlap_probe's result)
    HOSTNODE=on    if overlap collapses, H is confirmed

If H holds it explains every number we have, it is a property of CUDA graphs
rather than of this machine, and it says the fix is structural: the gather must
not share an execution timeline with kt's host nodes.
"""
import argparse, json, os, statistics, sys
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

// Host->device pull through UVA, the shape the real expert gather uses:
// a device-side read of pinned host memory, grid-stride so the block count is
// a free parameter rather than a function of the transfer size.
__global__ void k_pull(const uint4* __restrict__ src, uint4* __restrict__ dst,
                       long n_vec) {
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
       i += (long)gridDim.x * blockDim.x) {
    dst[i] = src[i];
  }
}

// Pure arithmetic, zero memory traffic, result guarded so the loop survives
// the optimiser. Stands in for the GPU experts.
__global__ void k_burn(float* __restrict__ out, long iters, int dummy) {
  float a = threadIdx.x * 1e-3f, b = 1.000001f, c = 0.999999f;
  for (long i = 0; i < iters; ++i) { a = fmaf(a, b, c); a = fmaf(a, c, b); }
  if (dummy && a == 12345.678f) out[blockIdx.x] = a;
}

// Stands in for kt's sync_: a host callback that BLOCKS until the CPU side is
// done. kt spins inside task_queue_->sync(); this spins on a clock. What
// matters is that the driver's host thread is occupied for a real duration.
static void CUDART_CB host_spin(void* p) {
  long us = *(long*)p;
  if (us <= 0) return;
  auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - t0).count() < us) {
  }
}

// Args must outlive the graph, and every captured node needs its own. Leak
// them deliberately; a probe process is short.
void hostnode(long us) {
  long* a = new long(us);
  cudaLaunchHostFunc(at::cuda::getCurrentCUDAStream(),
                     (cudaHostFn_t)&host_spin, (void*)a);
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
        name="graph_hostnode_probe",
        cpp_sources="void pull(torch::Tensor, torch::Tensor, long, int);\n"
                    "void burn(torch::Tensor, long, int);\n"
                    "void hostnode(long);",
        cuda_sources=SRC, functions=["pull", "burn", "hostnode"],
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
    ap.add_argument("--layers", type=int, default=8,
                    help="MoE-like layers per captured graph")
    ap.add_argument("--host-mb", type=int, default=512)
    ap.add_argument("--layer-mb", type=int, default=16,
                    help="bytes the gather pulls per layer; server moves "
                         "~1.71 experts x 9.7 MB = ~16 MB")
    ap.add_argument("--blocks", type=int, default=64,
                    help="gather blocks, matching KT_PREFETCH_BLOCKS default")
    ap.add_argument("--burn-blocks", type=int, default=132)
    ap.add_argument("--gpu-us", type=int, default=350,
                    help="GPU compute per layer; server ~0.35 ms")
    ap.add_argument("--cpu-us", type=int, default=500,
                    help="blocking host-node duration per layer, i.e. how long "
                         "kt's sync_ parks the driver thread")
    ap.add_argument("--out", default="bench/profile_out/graph_hostnode.json")
    a = ap.parse_args()

    dev = a.device
    torch.cuda.set_device(dev)
    ext = build()

    host = torch.empty(a.host_mb * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    host_ptr = torch.tensor([host.data_ptr()], dtype=torch.long)
    dst = torch.empty(a.layer_mb * 1024 * 1024, dtype=torch.uint8,
                      device=f"cuda:{dev}")
    burn_out = torch.zeros(4096, dtype=torch.float32, device=f"cuda:{dev}")
    n_vec = (a.layer_mb * 1024 * 1024) // 16

    # Calibrate the compute kernel to the requested per-layer GPU duration.
    iters = 20000
    for _ in range(14):
        t = timed_graph(lambda: ext.burn(burn_out, iters, a.burn_blocks), dev)
        target = a.gpu_us / 1000.0
        if abs(t - target) / target < 0.08:
            break
        iters = max(int(iters * (target / max(t, 1e-6))), 100)
    t_burn1 = timed_graph(lambda: ext.burn(burn_out, iters, a.burn_blocks), dev)
    t_pull1 = timed_graph(lambda: ext.pull(host_ptr, dst, n_vec, a.blocks), dev)

    pull_stream = torch.cuda.Stream(device=dev)

    def body(hostnode: bool, gather: bool):
        """One captured replay: `layers` repetitions of a MoE-shaped layer.

        With hostnode=True the main stream carries a blocking host callback per
        layer, exactly as kt's sync_with_cuda_stream does. The gather is forked
        one layer AHEAD of where it is joined, mirroring the server's
        pf_issue(L+1) at layer L / wait at layer L+1.
        """
        cur = torch.cuda.current_stream(dev)
        pending = None
        for i in range(a.layers):
            if gather:
                pull_stream.wait_stream(cur)
                with torch.cuda.stream(pull_stream):
                    ext.pull(host_ptr, dst, n_vec, a.blocks)
                    ev = torch.cuda.Event()
                    ev.record(pull_stream)
                nxt = ev
            else:
                nxt = None
            ext.burn(burn_out, iters, a.burn_blocks)      # GPU experts
            if hostnode:
                ext.hostnode(a.cpu_us)                    # kt sync_, blocking
            if pending is not None:
                cur.wait_event(pending)                   # consume last layer's
            pending = nxt
        if pending is not None:
            cur.wait_event(pending)                       # capture must rejoin

    rows = []
    for hn in (False, True):
        t_base = timed_graph(lambda h=hn: body(h, False), dev)
        t_gath = timed_graph(lambda h=hn: body(h, True), dev)
        added = t_gath - t_base
        gather_cost = t_pull1 * a.layers
        hidden = 1.0 - (added / gather_cost) if gather_cost > 0 else 0.0
        rows.append({
            "hostnode": hn, "cpu_us": a.cpu_us if hn else 0,
            "t_base_ms": t_base, "t_gather_ms": t_gath,
            "added_ms": added, "gather_alone_ms": gather_cost,
            "hidden_frac": hidden,
            "per_layer_added_ms": added / a.layers,
        })

    res = {
        "device": torch.cuda.get_device_name(dev),
        "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
        "layers": a.layers, "layer_mb": a.layer_mb, "blocks": a.blocks,
        "t_burn_per_layer_ms": t_burn1, "t_pull_per_layer_ms": t_pull1,
        "pull_gbs": (a.layer_mb / 1024.0) / (t_pull1 / 1000.0),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)

    print(f"{res['device']}, {res['sms']} SMs")
    print(f"{a.layers} layers/graph, {a.layer_mb} MB gathered per layer "
          f"at {a.blocks} blocks")
    print(f"  compute alone {t_burn1:.3f} ms/layer, "
          f"gather alone {t_pull1:.3f} ms/layer ({res['pull_gbs']:.1f} GB/s)\n")
    hdr = (f"{'host node':>10}{'base ms':>10}{'+gather ms':>12}"
           f"{'added':>9}{'per layer':>11}{'hidden':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{('yes' if r['hostnode'] else 'no'):>10}{r['t_base_ms']:>10.3f}"
              f"{r['t_gather_ms']:>12.3f}{r['added_ms']:>9.3f}"
              f"{r['per_layer_added_ms']:>11.3f}{r['hidden_frac']:>8.0%}")
    print()
    print("hidden = fraction of the gather's standalone cost that the graph")
    print("absorbed. 100% = free, 0% = fully exposed (the server's result).")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
