#!/usr/bin/env python3
"""Is the expert gather bandwidth-bound, or is it a latency-bound byte loop?

Attempt six at the prefetcher root cause, and the first one aimed at a defect
I can point to in the kernel source rather than a mechanism I imagined. The
five refuted hypotheses are recorded in glm52-expert-prefetch-measured; the
short version is that link bandwidth, fence placement, host DRAM contention,
block count, and CUDA host nodes are all exonerated by measurement.

What the source actually says (bench/expert_stream_kernels.py):

    k_w13_weights   gate+up, 6 MiB/expert/card
                    copies uint4  -- 16 bytes per thread-iteration, no divide
    k_w2_weights    down,    3 MiB/expert/card
                    copies uint8_t -- ONE BYTE per thread-iteration, and each
                    iteration computes `i / src_row_bytes` on a 64-bit long

k_w2_weights moves HALF the bytes with EIGHT TIMES the loop trips, and pays an
emulated 64-bit integer division on every single byte. Two consequences, and
the second is the one that matters:

  1. It is slow in absolute terms.
  2. It reaches bandwidth only by brute-force occupancy. A 1-byte load from
     host memory across PCIe has a latency of order a microsecond, and this
     loop gives each thread 48 of them BACK TO BACK with nothing to interleave.
     The only way to keep the link busy is to have tens of thousands of warps
     in flight simultaneously -- which means the gather must occupy the entire
     GPU to go at any speed at all.

That would explain the two facts that have refused to fit together:

  * the gather does not overlap -- stalled warps still hold their SM slots, so
    a kernel that needs the whole machine to reach bandwidth locks the main
    stream's kernels out for its whole duration; and
  * throttling it made things WORSE (KT_PREFETCH_BLOCKS=8: 94->102 ms) --
    fewer blocks means fewer outstanding loads, and a latency-bound loop just
    takes proportionally longer.

A bandwidth-bound wide copy has neither property. So the test is direct:
reimplement the same transfer with 16-byte accesses and no division, and see
whether (a) it gets faster and (b) it starts overlapping with compute.

Shapes are the real ones: moe(per partition)=1024, hidden=6144, n_tp=1,
4 slots, int4-packed weights, matching the 9.56 MiB/expert/card the server
logs at boot.
"""
import argparse, json, os, statistics, sys
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

__device__ __forceinline__ const void* tbl(const unsigned long long* p, int region,
                                           int tp, int expert, int n_tp, int n_exp) {
  return reinterpret_cast<const void*>(p[((long)region * n_tp + tp) * n_exp + expert]);
}

// ------------------------------------------------------------------ CURRENT
// Verbatim from bench/expert_stream_kernels.py. One byte per iteration, one
// 64-bit division per byte.
__global__ void k_w2_cur(const unsigned long long* __restrict__ ptrs,
                         const long* __restrict__ sel,
                         uint8_t* __restrict__ dst,
                         int n_tp, int n_exp, int hidden,
                         long src_row_bytes, long dst_row_bytes,
                         long bytes_per_slot) {
  const int slot = blockIdx.z;
  const int tp = blockIdx.y;
  const int e = (int)sel[slot];
  if (e < 0) return;
  const uint8_t* src = (const uint8_t*)tbl(ptrs, 4, tp, e, n_tp, n_exp);
  uint8_t* out = dst + (long)slot * bytes_per_slot + (long)tp * src_row_bytes;
  const long total = (long)hidden * src_row_bytes;
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const long r = i / src_row_bytes;
    const long c = i - r * src_row_bytes;
    out[r * dst_row_bytes + c] = src[i];
  }
}

// ------------------------------------------------------------------- FIXED
// Same data movement, 16 bytes per thread-iteration and no division anywhere.
// The row index comes from blockIdx.y instead of from `i / src_row_bytes`, so
// the inner loop is a flat contiguous stride over one row. Rows are 512 B here
// and any int4 expert row is a multiple of 16 B, so the uint4 view is exact.
//
// gridDim.y carries (tp, row-chunk): y = tp * row_chunks + chunk, and each
// block walks a strided set of rows. That keeps the launch shape a plain dim3
// while giving every thread a long contiguous run to prefetch through.
__global__ void k_w2_vec(const unsigned long long* __restrict__ ptrs,
                         const long* __restrict__ sel,
                         uint8_t* __restrict__ dst,
                         int n_tp, int n_exp, int hidden,
                         long src_row_bytes, long dst_row_bytes,
                         long bytes_per_slot) {
  const int slot = blockIdx.z;
  const int tp = blockIdx.y;
  const int e = (int)sel[slot];
  if (e < 0) return;
  const uint4* src = (const uint4*)tbl(ptrs, 4, tp, e, n_tp, n_exp);
  uint4* out = (uint4*)(dst + (long)slot * bytes_per_slot + (long)tp * src_row_bytes);
  const long vec_per_row = src_row_bytes / 16;
  const long dst_vec_row = dst_row_bytes / 16;
  const long total_vec = (long)hidden * vec_per_row;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  // When the destination rows are contiguous with the source rows (n_tp == 1)
  // this degenerates to a flat copy; keep the general form, it costs one
  // multiply, and let the compiler hoist the rest.
  for (long i = tid; i < total_vec; i += stride) {
    const long r = i / vec_per_row;          // hoisted out of the byte loop:
    const long c = i - r * vec_per_row;      // 1 divide per 16 B, not per 1 B
    out[r * dst_vec_row + c] = src[i];
  }
}

// Contiguous special case: when src and dst row pitches agree there is no
// gather at all, just a straight run of bytes. This is what n_tp == 1 gives,
// which is what this deployment actually runs.
__global__ void k_w2_flat(const unsigned long long* __restrict__ ptrs,
                          const long* __restrict__ sel,
                          uint8_t* __restrict__ dst,
                          int n_tp, int n_exp, long n_vec, long bytes_per_slot) {
  const int slot = blockIdx.z;
  const int tp = blockIdx.y;
  const int e = (int)sel[slot];
  if (e < 0) return;
  const uint4* src = (const uint4*)tbl(ptrs, 4, tp, e, n_tp, n_exp);
  uint4* out = (uint4*)(dst + (long)slot * bytes_per_slot);
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
       i += (long)gridDim.x * blockDim.x) {
    out[i] = src[i];
  }
}

// Reference: the gate/up kernel's access pattern, already vectorised.
__global__ void k_w13(const unsigned long long* __restrict__ ptrs,
                      const long* __restrict__ sel, uint4* __restrict__ dst,
                      int n_tp, int n_exp, long vec_per_tp_block,
                      long vec_per_slot) {
  const int slot = blockIdx.z;
  const int blk = blockIdx.y;
  const int tp = blk % n_tp;
  const int is_up = blk / n_tp;
  const int e = (int)sel[slot];
  if (e < 0) return;
  const uint4* src = (const uint4*)tbl(ptrs, is_up ? 2 : 0, tp, e, n_tp, n_exp);
  uint4* out = dst + (long)slot * vec_per_slot
                   + (long)(is_up * n_tp + tp) * vec_per_tp_block;
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_per_tp_block;
       i += (long)gridDim.x * blockDim.x) {
    out[i] = src[i];
  }
}

__global__ void k_burn(float* __restrict__ out, long iters, int dummy) {
  float a = threadIdx.x * 1e-3f, b = 1.000001f, c = 0.999999f;
  for (long i = 0; i < iters; ++i) { a = fmaf(a, b, c); a = fmaf(a, c, b); }
  if (dummy && a == 12345.678f) out[blockIdx.x] = a;
}

void w2_cur(torch::Tensor ptrs, torch::Tensor sel, torch::Tensor dst,
            long n_tp, long n_exp, long hidden, long src_row, long dst_row,
            long blocks) {
  dim3 g(blocks, n_tp, sel.numel());
  k_w2_cur<<<g, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (const unsigned long long*)ptrs.data_ptr(), sel.data_ptr<long>(),
      (uint8_t*)dst.data_ptr(), n_tp, n_exp, hidden, src_row, dst_row,
      hidden * dst_row);
}

void w2_vec(torch::Tensor ptrs, torch::Tensor sel, torch::Tensor dst,
            long n_tp, long n_exp, long hidden, long src_row, long dst_row,
            long blocks) {
  dim3 g(blocks, n_tp, sel.numel());
  k_w2_vec<<<g, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (const unsigned long long*)ptrs.data_ptr(), sel.data_ptr<long>(),
      (uint8_t*)dst.data_ptr(), n_tp, n_exp, hidden, src_row, dst_row,
      hidden * dst_row);
}

void w2_flat(torch::Tensor ptrs, torch::Tensor sel, torch::Tensor dst,
             long n_tp, long n_exp, long hidden, long src_row, long dst_row,
             long blocks) {
  dim3 g(blocks, n_tp, sel.numel());
  k_w2_flat<<<g, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (const unsigned long long*)ptrs.data_ptr(), sel.data_ptr<long>(),
      (uint8_t*)dst.data_ptr(), n_tp, n_exp, hidden * src_row / 16,
      hidden * dst_row);
}

void w13(torch::Tensor ptrs, torch::Tensor sel, torch::Tensor dst,
         long n_tp, long n_exp, long vec_per_tp_block, long blocks) {
  dim3 g(blocks, 2 * n_tp, sel.numel());
  k_w13<<<g, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (const unsigned long long*)ptrs.data_ptr(), sel.data_ptr<long>(),
      (uint4*)dst.data_ptr(), n_tp, n_exp, vec_per_tp_block,
      vec_per_tp_block * 2 * n_tp);
}

void burn(torch::Tensor out, long iters, int blocks) {
  k_burn<<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<float>(), iters, 0);
}
"""

DECL = """
void w2_cur(torch::Tensor, torch::Tensor, torch::Tensor, long, long, long, long, long, long);
void w2_vec(torch::Tensor, torch::Tensor, torch::Tensor, long, long, long, long, long, long);
void w2_flat(torch::Tensor, torch::Tensor, torch::Tensor, long, long, long, long, long, long);
void w13(torch::Tensor, torch::Tensor, torch::Tensor, long, long, long, long);
void burn(torch::Tensor, long, int);
"""


def timed(fn, dev, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); fn(); e1.record()
        torch.cuda.synchronize(dev)
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


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
    ap.add_argument("--moe", type=int, default=1024, help="intermediate per GPU TP partition")
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--n-tp", type=int, default=1, help="CPU store TP partitions")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--n-exp", type=int, default=256)
    ap.add_argument("--live-slots", type=int, default=2,
                    help="slots actually holding an expert; server averages 1.71")
    ap.add_argument("--blocks", default="8,16,32,64,128,256")
    ap.add_argument("--burn-blocks", type=int, default=132)
    ap.add_argument("--verify", action="store_true",
                    help="only check that the variants move identical bytes")
    ap.add_argument("--out", default="bench/profile_out/gather_kernel.json")
    a = ap.parse_args()

    dev = a.device
    torch.cuda.set_device(dev)
    ext = load_inline(name="gather_kernel_probe", cpp_sources=DECL,
                      cuda_sources=SRC,
                      functions=["w2_cur", "w2_vec", "w2_flat", "w13", "burn"],
                      extra_cuda_cflags=["-O3"], verbose=False)

    tp_moe = a.moe // a.n_tp
    src_row = tp_moe // 2                       # int4 packed
    dst_row = a.moe // 2
    w2_bytes = a.hidden * src_row * a.n_tp
    vec_per_tp_block = (tp_moe * a.hidden // 2) // 16
    w13_bytes = vec_per_tp_block * 16 * 2 * a.n_tp

    # Host-side store: one pinned buffer per (region, tp, expert). Only a few
    # experts are needed to measure -- the pointer table is what the kernel
    # reads, so distinct experts just need distinct valid addresses.
    n_src = max(a.live_slots, 1)
    src_w2 = [torch.empty(a.hidden * src_row, dtype=torch.uint8, pin_memory=True)
              for _ in range(n_src * a.n_tp)]
    src_w13 = [torch.empty(vec_per_tp_block * 16, dtype=torch.uint8, pin_memory=True)
               for _ in range(n_src * a.n_tp * 2)]

    # table[region][tp][expert]; regions 0 gate_w 1 gate_s 2 up_w 3 up_s 4 down_w 5 down_s
    tbl = torch.zeros(6 * a.n_tp * a.n_exp, dtype=torch.int64)
    for tp in range(a.n_tp):
        for e in range(a.n_exp):
            k = e % n_src
            tbl[(0 * a.n_tp + tp) * a.n_exp + e] = src_w13[(k * a.n_tp + tp) * 2 + 0].data_ptr()
            tbl[(2 * a.n_tp + tp) * a.n_exp + e] = src_w13[(k * a.n_tp + tp) * 2 + 1].data_ptr()
            tbl[(4 * a.n_tp + tp) * a.n_exp + e] = src_w2[k * a.n_tp + tp].data_ptr()
    tbl = tbl.to(f"cuda:{dev}")

    sel = torch.full((a.slots,), -1, dtype=torch.int64, device=f"cuda:{dev}")
    sel[:a.live_slots] = torch.arange(a.live_slots, device=f"cuda:{dev}")

    dst_w2 = torch.empty(a.slots * a.hidden * dst_row, dtype=torch.uint8, device=f"cuda:{dev}")
    dst_w13 = torch.empty(a.slots * vec_per_tp_block * 16 * 2 * a.n_tp,
                          dtype=torch.uint8, device=f"cuda:{dev}")
    burn_out = torch.zeros(4096, dtype=torch.float32, device=f"cuda:{dev}")

    if a.verify:
        # Identical bytes is the whole point: this kernel writes expert weights
        # that cutlass then reads as truth. A faster gather that transposes a
        # row is a silent accuracy bug, not a speedup.
        for t in src_w2 + src_w13:
            t.copy_(torch.randint(0, 256, t.shape, dtype=torch.uint8))
        ref = None
        ok = True
        for nm, fn in (("w2_cur", ext.w2_cur), ("w2_vec", ext.w2_vec),
                       ("w2_flat", ext.w2_flat)):
            if nm == "w2_flat" and a.n_tp != 1:
                print(f"  {nm:<8} skipped (contiguous form assumes n_tp == 1)")
                continue
            for b in (8, 64, 256):
                dst_w2.fill_(0)
                fn(tbl, sel, dst_w2, a.n_tp, a.n_exp, a.hidden, src_row, dst_row, b)
                torch.cuda.synchronize(dev)
                got = dst_w2.clone()
                if ref is None:
                    ref = got
                elif not torch.equal(ref, got):
                    d = (ref != got).sum().item()
                    print(f"  {nm:<8} blocks={b:<4} MISMATCH on {d} bytes")
                    ok = False
                    continue
                print(f"  {nm:<8} blocks={b:<4} identical")
        nz = (ref != 0).sum().item()
        print(f"  n_tp={a.n_tp}: {nz} non-zero bytes written of {ref.numel()} "
              f"({a.live_slots}/{a.slots} slots live)")
        print("VERIFY OK" if ok and nz > 0 else "VERIFY FAILED")
        return 0 if (ok and nz > 0) else 1

    blocks_list = [int(x) for x in a.blocks.split(",") if x.strip()]
    live_mb = (w2_bytes * a.live_slots) / 1048576.0
    w13_mb = (w13_bytes * a.live_slots) / 1048576.0

    variants = {
        "w2_cur  (1B + 64b div)": lambda b: ext.w2_cur(tbl, sel, dst_w2, a.n_tp, a.n_exp,
                                                       a.hidden, src_row, dst_row, b),
        "w2_vec  (16B, div/16B)": lambda b: ext.w2_vec(tbl, sel, dst_w2, a.n_tp, a.n_exp,
                                                       a.hidden, src_row, dst_row, b),
        "w2_flat (16B, no div) ": lambda b: ext.w2_flat(tbl, sel, dst_w2, a.n_tp, a.n_exp,
                                                        a.hidden, src_row, dst_row, b),
        "w13     (16B, ref)    ": lambda b: ext.w13(tbl, sel, dst_w13, a.n_tp, a.n_exp,
                                                    vec_per_tp_block, b),
    }

    res = {"device": torch.cuda.get_device_name(dev),
           "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
           "moe": a.moe, "hidden": a.hidden, "n_tp": a.n_tp,
           "live_slots": a.live_slots, "w2_mb": live_mb, "w13_mb": w13_mb,
           "rows": []}

    print(f"{res['device']}, {res['sms']} SMs")
    print(f"moe/partition {a.moe}, hidden {a.hidden}, n_tp {a.n_tp}, "
          f"{a.live_slots} live slots of {a.slots}")
    print(f"  w2 moves {live_mb:.2f} MiB, w13 moves {w13_mb:.2f} MiB\n")
    hdr = f"{'kernel':<24}{'blocks':>7}{'ms':>9}{'GB/s':>8}"
    print(hdr); print("-" * len(hdr))
    for name, fn in variants.items():
        mb = w13_mb if name.startswith("w13") else live_mb
        for b in blocks_list:
            t = timed(lambda f=fn, bb=b: f(bb), dev)
            gbs = (mb / 1024.0) / (t / 1000.0)
            res["rows"].append({"kernel": name.strip(), "blocks": b,
                                "ms": t, "gbs": gbs, "mb": mb})
            print(f"{name:<24}{b:>7}{t:>9.3f}{gbs:>8.1f}")
        print()

    # Overlap: does each variant hide behind compute inside a captured graph?
    print("overlap against a pure-arithmetic kernel, in a CUDA graph")
    hdr2 = f"{'kernel':<24}{'blocks':>7}{'alone':>8}{'both':>8}{'serial':>8}{'hidden':>8}"
    print(hdr2); print("-" * len(hdr2))
    iters = 20000
    for _ in range(14):
        t = timed_graph(lambda: ext.burn(burn_out, iters, a.burn_blocks), dev)
        if abs(t - 0.35) / 0.35 < 0.08:
            break
        iters = max(int(iters * (0.35 / max(t, 1e-6))), 100)
    t_burn = timed_graph(lambda: ext.burn(burn_out, iters, a.burn_blocks), dev)
    res["t_burn_ms"] = t_burn
    res["overlap"] = []
    side = torch.cuda.Stream(device=dev)
    for name, fn in variants.items():
        for b in (64,):
            def both(f=fn, bb=b):
                cur = torch.cuda.current_stream(dev)
                side.wait_stream(cur)
                with torch.cuda.stream(side):
                    f(bb)
                    ev = torch.cuda.Event(); ev.record(side)
                ext.burn(burn_out, iters, a.burn_blocks)
                cur.wait_event(ev)
            t_alone = timed_graph(lambda f=fn, bb=b: f(bb), dev)
            t_both = timed_graph(both, dev)
            serial = t_alone + t_burn
            hidden = (serial - t_both) / min(t_alone, t_burn)
            res["overlap"].append({"kernel": name.strip(), "blocks": b,
                                   "alone_ms": t_alone, "both_ms": t_both,
                                   "serial_ms": serial, "hidden_frac": hidden})
            print(f"{name:<24}{b:>7}{t_alone:>8.3f}{t_both:>8.3f}"
                  f"{serial:>8.3f}{hidden:>7.0%}")
    print(f"\ncompute kernel alone {t_burn:.3f} ms")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
