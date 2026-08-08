#!/usr/bin/env python3
"""Where does RAM->VRAM expert prefetching pay, and where does it not?

This box is close to the WORST case for the technique, and that is the whole
reason to write the question down in machine-independent form. Prefetching
trades link time for CPU time: you spend `bytes_per_expert / link_bw` moving an
expert to the GPU so the CPU does not have to compute it, saving
`cpu_ms_per_expert`. Two quantities decide it, and only one of them is a
property of the model:

    R = cpu_ms_per_expert / (bytes_per_expert / link_bw)      unit economics
    F = transfer_ms_per_layer / gpu_idle_ms_per_layer         does it FIT

R > 1 means an expert costs less to ship than to compute, and prefetching pays
outright. R < 1 means it does not -- but that is survivable, because a transfer
that fits inside the window where the GPU is already idle waiting for the CPU
is FREE. That is what F asks. When F < 1 the only term left is the saving, and
even a terrible R turns a profit.

So the technique fails only when BOTH are bad, and it is bought by any machine
whose CPU is slow relative to its link. This host has a dual-socket EPYC with
AVX-512-VNNI and 379 GB/s of DRAM feeding an ordinary PCIe link, which is the
least favourable combination in common circulation: R = 0.38. A single consumer
CPU on dual-channel DDR5 computes experts several times slower over the same
class of link, which moves R proportionally.

Defaults are this machine's MEASURED values, so `--help` doubles as the record
of where they came from. Point it at another machine by measuring the link
(this script does that itself) and passing that machine's CPU numbers.

    ./bench/prefetch_portability.py                 # this box
    ./bench/prefetch_portability.py --cpu-slowdown 4    # a slower CPU host
    ./bench/prefetch_portability.py --link-gbs 24   # PCIe Gen4 x16
"""
import argparse, json, os, statistics, sys
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
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
"""


def measure_link(dev, mb=64, blocks=32):
    """Host->device gather rate through UVA, GiB/s.

    Uses the same access shape as the real expert gather (uint4 grid-stride
    reads of pinned host memory), which after the k_w2_weights fix reaches the
    same 48 GiB/s the plain pull does. Measuring rather than quoting the spec
    matters: a machine's usable rate depends on root-complex topology and IOMMU
    settings, not just on the PCIe generation printed on the box.
    """
    ext = load_inline(name="portability_pull",
                      cpp_sources="void pull(torch::Tensor, torch::Tensor, long, int);",
                      cuda_sources=SRC, functions=["pull"],
                      extra_cuda_cflags=["-O3"], verbose=False)
    host = torch.empty(mb * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    ptr = torch.tensor([host.data_ptr()], dtype=torch.long)
    dst = torch.empty(mb * 1024 * 1024, dtype=torch.uint8, device=f"cuda:{dev}")
    n_vec = (mb * 1024 * 1024) // 16
    for _ in range(5):
        ext.pull(ptr, dst, n_vec, blocks)
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(20):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); ext.pull(ptr, dst, n_vec, blocks); e1.record()
        torch.cuda.synchronize(dev)
        ts.append(e0.elapsed_time(e1))
    return (mb / 1024.0) / (statistics.median(ts) / 1000.0)


def evaluate(bytes_mib, link_gbs, cpu_ms, experts, layers, gpu_layer_ms,
             cpu_fixed_ms, predictor_ms, overlap_eff):
    """One machine's verdict. All times in ms per decode step unless noted.

    Every machine is compared against ITS OWN baseline, not against this box's
    66.64 ms. A host with a 4x slower CPU has a 4x slower baseline too, and
    quoting a speedup against the wrong denominator is how you end up with a
    table claiming 90x. Per layer the CPU and GPU halves overlap already
    (kt submits the CPU experts asynchronously and syncs at the end), so the
    layer costs max(cpu, gpu) and the prefetcher's job is to shorten the CPU
    pole without lengthening the GPU one.
    """
    ship_ms = bytes_mib / 1024.0 / link_gbs * 1000.0        # one expert
    xfer_layer = experts * ship_ms
    # The CPU pole is fixed submit/sync latency PLUS expert compute, and only
    # the second half is what a slower host makes slower. On this box the split
    # is brutal: 0.89 ms/layer of pole, of which 1.71 x 0.074 = 0.13 ms is
    # actual expert arithmetic and the remaining 86% is the cost of handing
    # work to the worker pool and waiting for it. That is why placement and
    # caching both floored around +8-15% here, and it is the single number that
    # decides whether this technique is worth anything on a given machine.
    cpu_compute_ms = experts * cpu_ms
    cpu_layer_ms = cpu_fixed_ms + cpu_compute_ms
    idle_ms = max(0.0, cpu_layer_ms - gpu_layer_ms)          # GPU-idle window
    R = cpu_ms / ship_ms
    F = xfer_layer / idle_ms if idle_ms > 0 else float("inf")

    # What the transfer actually costs. Anything past the idle window is
    # exposed outright; what fits still leaks a little, because concurrent
    # kernels are not free even when there is room for them.
    spill = max(0.0, xfer_layer - idle_ms)
    leak = min(xfer_layer, idle_ms) * (1.0 - overlap_eff)
    exposed_layer = spill + leak
    saved_layer = cpu_compute_ms       # the pole floors at the fixed overhead

    # Exposed transfer time ADDS to the step; it does not get absorbed into the
    # CPU pole even when the GPU is nowhere near being the pole. That is the
    # empirical finding, not an assumption: at overlap_eff 0.10 this form
    # reproduces the server's measured P3 (84.1 predicted vs 84.67 measured)
    # while the max() form predicts a win that does not exist. Whatever the
    # gather is contending with, it is contending with the CPU-expert window
    # itself, which is the very window it was supposed to hide in.
    base_step = layers * max(cpu_layer_ms, gpu_layer_ms)
    pf_step = (layers * (max(cpu_layer_ms - saved_layer, gpu_layer_ms) + exposed_layer)
               + predictor_ms)
    return {
        "R_unit_economics": R, "F_fit": F,
        "transfer_ms_per_layer": xfer_layer,
        "exposed_ms_per_step": exposed_layer * layers,
        "saving_ms_per_step": saved_layer * layers,
        "predictor_ms_per_step": predictor_ms,
        "net_ms_per_step": pf_step - base_step,
        "baseline_ms": base_step, "step_ms": pf_step,
        "cpu_layer_ms": cpu_layer_ms, "cpu_compute_ms": cpu_compute_ms,
        "speedup": base_step / pf_step,
    }


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--link-gbs", type=float, default=None,
                    help="host->device GiB/s; measured on this box if omitted")
    ap.add_argument("--bytes-mib", type=float, default=9.56,
                    help="per expert per card; GLM-5.2 int4, logged at boot")
    ap.add_argument("--cpu-ms-per-expert", type=float, default=0.074,
                    help="CPU time one expert costs; measured by CPU-skip")
    ap.add_argument("--cpu-slowdown", type=float, default=1.0,
                    help="multiply the CPU expert time, to model a slower host")
    ap.add_argument("--experts-per-layer", type=float, default=1.71,
                    help="distinct non-resident experts, after intersecting "
                         "the prediction with the residency mask")
    ap.add_argument("--layers", type=int, default=75, help="MoE layers")
    ap.add_argument("--gpu-ms-per-layer", type=float, default=0.19,
                    help="GPU kernel time per MoE layer; the pole prefetching "
                         "cannot go below. 37 ms of a 191 ms step, per the "
                         "occupancy profile, scaled to this box's 66.64 ms")
    ap.add_argument("--predictor-ms", type=float, default=5.03,
                    help="cost of the lookahead router itself, per step")
    ap.add_argument("--overlap-eff", type=float, default=0.90,
                    help="fraction of an in-window transfer that stays hidden")
    ap.add_argument("--baseline-ms", type=float, default=66.64,
                    help="step time with no predictor and no gather; used only "
                         "to derive this box's fixed CPU overhead")
    ap.add_argument("--sweep", action="store_true",
                    help="map the CPU-slowdown x link-speed plane")
    ap.add_argument("--out", default="bench/profile_out/prefetch_portability.json")
    a = ap.parse_args()

    link = a.link_gbs
    measured = False
    if link is None:
        torch.cuda.set_device(a.device)
        link = measure_link(a.device)
        measured = True

    # Back out this box's fixed per-layer CPU overhead: whatever is left of the
    # measured pole once the expert arithmetic is subtracted. That residue is
    # thread wake-up and queue synchronisation, so it stays put when the host's
    # arithmetic gets slower -- which is exactly why a slower machine gets MORE
    # out of prefetching, not less.
    cpu_fixed = a.baseline_ms / a.layers - a.experts_per_layer * a.cpu_ms_per_expert
    cpu_ms = a.cpu_ms_per_expert * a.cpu_slowdown
    r = evaluate(a.bytes_mib, link, cpu_ms, a.experts_per_layer, a.layers,
                 a.gpu_ms_per_layer, cpu_fixed, a.predictor_ms, a.overlap_eff)

    print(f"link {link:.1f} GiB/s" + (" (measured here)" if measured else " (given)"))
    print(f"{a.bytes_mib:.2f} MiB/expert, {a.experts_per_layer:.2f} experts/layer, "
          f"{a.layers} layers")
    print(f"CPU {cpu_ms*1000:.0f} us/expert"
          + (f"  ({a.cpu_slowdown:g}x slower than this box)" if a.cpu_slowdown != 1 else ""))
    print(f"CPU pole {r['cpu_layer_ms']:.2f} ms/layer = {cpu_fixed:.2f} fixed "
          f"submit/sync + {r['cpu_compute_ms']:.2f} expert compute "
          f"({r['cpu_compute_ms']/r['cpu_layer_ms']:.0%} removable)")
    print(f"GPU {a.gpu_ms_per_layer:.2f} ms/layer, baseline {r['baseline_ms']:.1f} ms/step")
    print()
    print(f"  R  unit economics   {r['R_unit_economics']:.2f}   "
          f"{'ship beats compute' if r['R_unit_economics'] >= 1 else 'compute beats ship'}")
    print(f"  F  fit in idle      {r['F_fit']:.2f}   "
          f"{'fits, transfer is nearly free' if r['F_fit'] <= 1 else 'SPILLS past the window'}")
    print()
    print(f"  transfer      {r['transfer_ms_per_layer']:.3f} ms/layer")
    print(f"  exposed      +{r['exposed_ms_per_step']:.2f} ms/step")
    print(f"  predictor    +{r['predictor_ms_per_step']:.2f} ms/step")
    print(f"  CPU saved    -{r['saving_ms_per_step']:.2f} ms/step")
    print(f"  net          {r['net_ms_per_step']:+.2f} ms/step  ->  "
          f"{r['step_ms']:.2f} ms  ({r['speedup']:.3f}x)")
    print()
    print("  " + ("WORTH SHIPPING on this machine" if r["speedup"] > 1.02 else
                  "not worth it here; see the sweep for where it turns"))

    out = {"link_gibs": link, "measured_link": measured, "point": r,
           "inputs": vars(a)}

    if a.sweep:
        print("\nspeedup vs CPU slowdown (rows) and link GiB/s (cols)")
        links = [12, 24, 48, 96, 400]
        print("        " + "".join(f"{l:>9}" for l in links))
        out["sweep"] = []
        for slow in (1, 2, 3, 4, 6, 8):
            cells = []
            for l in links:
                rr = evaluate(a.bytes_mib, l, a.cpu_ms_per_expert * slow,
                              a.experts_per_layer, a.layers, a.gpu_ms_per_layer,
                              cpu_fixed, a.predictor_ms, a.overlap_eff)
                cells.append(rr["speedup"])
                out["sweep"].append({"cpu_slowdown": slow, "link_gibs": l,
                                     "speedup": rr["speedup"],
                                     "net_ms": rr["net_ms_per_step"]})
            print(f"  {slow:>2}x   " + "".join(f"{c:>9.2f}" for c in cells))
        print("\n  columns: 12 = PCIe3 x16, 24 = Gen4 x16, 48 = Gen5 x16 (this box),")
        print("           96 = Gen6 / dual-link, 400 = NVLink-C2C class (GH200)")
        print("  rows: how much slower this machine's CPU experts are than this box's")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
