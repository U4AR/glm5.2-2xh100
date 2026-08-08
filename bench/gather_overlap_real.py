#!/usr/bin/env python3
"""What block count lets the REAL expert gather hide behind a layer of compute?

Runs bench/expert_stream_kernels.stream_experts -- the exact call the server
makes from pf_issue -- against pinned host buffers standing in for the kt
store, and measures two things per block count:

    cost     how long the gather takes on its own
    hidden   how much of that cost disappears when it is forked onto a side
             stream alongside a compute kernel, inside a CUDA graph

Why the pair matters. The old k_w2_weights was latency-bound (1 byte per
thread-iteration plus a 64-bit divide), so it hit its best bandwidth only by
flooding the GPU with stalled warps, and stalled warps still hold their SM
slots. That made the two columns pull against each other: any block count low
enough to leave room for compute was also low enough to make the transfer
crawl. Throttling it in the server made things worse, not better -- 94 -> 102
ms/step at KT_PREFETCH_BLOCKS=8 -- which is the signature of that trade.

With the vectorised kernel the transfer reaches link speed at 8 blocks, so the
two columns should stop fighting: there should be a block count that is both
fast and polite. If there is, that is the fix, and this prints its value.

The compute kernel is pure arithmetic with no memory traffic at all, so a
slowdown while the gather runs can only be contention for SMs.
"""
import argparse, json, os, statistics, sys
import torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BURN = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void k_burn(float* __restrict__ out, long iters, int dummy) {
  float a = threadIdx.x * 1e-3f, b = 1.000001f, c = 0.999999f;
  for (long i = 0; i < iters; ++i) { a = fmaf(a, b, c); a = fmaf(a, c, b); }
  if (dummy && a == 12345.678f) out[blockIdx.x] = a;
}
void burn(torch::Tensor out, long iters, int blocks) {
  k_burn<<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<float>(), iters, 0);
}
"""


def timed_graph(fn, dev, iters=20, warmup=5):
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
        e0.record(); g.replay(); e1.record()
        torch.cuda.synchronize(dev)
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--moe", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--n-tp", type=int, default=1)
    ap.add_argument("--n-exp", type=int, default=256)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--live-slots", type=int, default=2,
                    help="server averages 1.71 non-resident experts per layer")
    ap.add_argument("--blocks", default="4,8,16,32,64,128")
    ap.add_argument("--burn-blocks", type=int, default=132)
    ap.add_argument("--shadow-ms", type=float, default=0.70,
                    help="GPU-idle window the gather must hide in; the server "
                         "spends ~0.7 ms/layer waiting on the CPU experts")
    ap.add_argument("--out", default="bench/profile_out/gather_overlap_real.json")
    a = ap.parse_args()

    dev = a.device
    torch.cuda.set_device(dev)
    from expert_stream_kernels import kernels
    K = kernels()
    ext = load_inline(name="gather_overlap_burn",
                      cpp_sources="void burn(torch::Tensor, long, int);",
                      cuda_sources=BURN, functions=["burn"],
                      extra_cuda_cflags=["-O3"], verbose=False)

    tp_moe = a.moe // a.n_tp
    G = a.hidden // a.group          # groups along hidden, for w13 scales
    G2 = a.moe // a.group            # groups along moe,    for w2 scales
    tp_g = G2 // a.n_tp
    n_src = max(a.live_slots, 1)
    dcu = f"cuda:{dev}"

    def pin_u8(n):
        return torch.empty(n, dtype=torch.uint8, pin_memory=True)

    def pin_f32(n):
        return torch.empty(n, dtype=torch.float32, pin_memory=True)

    # Host store: regions 0 gate_w 1 gate_s 2 up_w 3 up_s 4 down_w 5 down_s
    src = [[[None] * a.n_exp for _ in range(a.n_tp)] for _ in range(6)]
    keep = []
    for tp in range(a.n_tp):
        base = []
        for k in range(n_src):
            gw, uw = pin_u8(tp_moe * a.hidden // 2), pin_u8(tp_moe * a.hidden // 2)
            gs, us = pin_f32(tp_moe * G), pin_f32(tp_moe * G)
            dw, ds = pin_u8(a.hidden * tp_moe // 2), pin_f32(a.hidden * tp_g)
            base.append((gw, gs, uw, us, dw, ds))
            keep.append(base[-1])
        for e in range(a.n_exp):
            for r in range(6):
                src[r][tp][e] = base[e % n_src][r]

    tbl = torch.zeros(6 * a.n_tp * a.n_exp, dtype=torch.int64)
    for r in range(6):
        for tp in range(a.n_tp):
            for e in range(a.n_exp):
                tbl[(r * a.n_tp + tp) * a.n_exp + e] = src[r][tp][e].data_ptr()
    tbl = tbl.to(dcu)

    sel = torch.full((a.slots,), -1, dtype=torch.int64, device=dcu)
    sel[:a.live_slots] = torch.arange(a.live_slots, device=dcu)

    w13 = torch.empty(a.slots, 2 * a.moe, a.hidden // 2, dtype=torch.uint8, device=dcu)
    w2 = torch.empty(a.slots, a.hidden, a.moe // 2, dtype=torch.uint8, device=dcu)
    w13_s = torch.empty(a.slots, 2 * a.moe * G, dtype=torch.bfloat16, device=dcu)
    w2_s = torch.empty(a.slots, a.hidden * G2, dtype=torch.bfloat16, device=dcu)
    burn_out = torch.zeros(4096, dtype=torch.float32, device=dcu)

    per_expert_mb = ((2 * tp_moe * a.hidden // 2 + a.hidden * tp_moe // 2) * a.n_tp
                     + (2 * tp_moe * G + a.hidden * tp_g) * 4 * a.n_tp) / 1048576.0
    moved_mb = per_expert_mb * a.live_slots

    def gather(b):
        K.stream_experts(tbl, sel, w13, w13_s, w2, w2_s,
                         a.n_tp, a.n_exp, a.moe, a.hidden, a.group, b)

    # Compute kernel calibrated to the GPU-idle shadow the gather must fit in.
    iters = 20000
    for _ in range(16):
        t = timed_graph(lambda: ext.burn(burn_out, iters, a.burn_blocks), dev)
        if abs(t - a.shadow_ms) / a.shadow_ms < 0.05:
            break
        iters = max(int(iters * (a.shadow_ms / max(t, 1e-6))), 100)
    t_burn = timed_graph(lambda: ext.burn(burn_out, iters, a.burn_blocks), dev)

    side = torch.cuda.Stream(device=dev)
    res = {"device": torch.cuda.get_device_name(dev),
           "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
           "per_expert_mib": per_expert_mb, "moved_mib": moved_mb,
           "live_slots": a.live_slots, "shadow_ms": t_burn, "rows": []}

    print(f"{res['device']}, {res['sms']} SMs")
    print(f"{per_expert_mb:.2f} MiB/expert/card, {a.live_slots} experts "
          f"= {moved_mb:.2f} MiB per layer")
    print(f"compute shadow {t_burn:.3f} ms (stands in for the CPU-expert wait)\n")
    hdr = (f"{'blocks':>7}{'gather ms':>11}{'GiB/s':>8}{'both ms':>10}"
           f"{'exposed':>9}{'hidden':>8}{'ms/step':>9}")
    print(hdr); print("-" * len(hdr))
    for b in [int(x) for x in a.blocks.split(",") if x.strip()]:
        t_g = timed_graph(lambda bb=b: gather(bb), dev)

        def both(bb=b):
            cur = torch.cuda.current_stream(dev)
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                gather(bb)
                ev = torch.cuda.Event(); ev.record(side)
            ext.burn(burn_out, iters, a.burn_blocks)
            cur.wait_event(ev)

        t_both = timed_graph(both, dev)
        exposed = t_both - t_burn
        hidden = 1.0 - exposed / t_g if t_g > 0 else 0.0
        res["rows"].append({"blocks": b, "gather_ms": t_g,
                            "gibs": (moved_mb / 1024.0) / (t_g / 1000.0),
                            "both_ms": t_both, "exposed_ms": exposed,
                            "hidden_frac": hidden,
                            "per_step_ms": exposed * 75})
        print(f"{b:>7}{t_g:>11.3f}{(moved_mb/1024.0)/(t_g/1000.0):>8.1f}"
              f"{t_both:>10.3f}{exposed:>9.3f}{hidden:>7.0%}{exposed*75:>9.1f}")

    print()
    print("exposed = what the gather adds to a layer once forked against compute")
    print("ms/step = exposed x 75 MoE layers; the CPU work it buys back is 9.5 ms,")
    print("so anything under that is a net win and anything over it is a loss.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
