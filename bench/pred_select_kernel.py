#!/usr/bin/env python3
"""One kernel to replace pf_issue's ~28 tiny ones.

The gather is fixed (full path 84.67 -> 72.97 ms). What stands between the
prefetcher and break-even on this box is 6.4 ms, and 5.03 of it is the
predictor -- which is not arithmetic. Per MoE layer, pf_issue currently issues
roughly this many CUDA kernels:

    mask gather, dtype cast, demand zero, index_add, >0, sum,
    topk(256 -> slots), >0, <=, and, full_like, where, copy,
    >=, zeros_like x2, clamp_min, zero, scatter, zero, scatter,
    scatter, sum, add x4

That is ~28 launches, times 75 layers, times every decode step: ~2100 extra
nodes in the captured graph. At the couple of microseconds a graph node costs
regardless of how little work it does, that is the 5 ms. None of it is FLOPs --
the whole computation is over 256 experts and at most 8 tokens, which is less
arithmetic than a single attention head does.

So the fix is fusion, not a better algorithm. (The obvious "better algorithm" --
predict from the previous step's routing instead of running a lookahead router
-- is already refuted by this project's own instrument: 27.1% whole-layer
coverage against the router's 75.2%, per EXPERT_PREFETCH_PLAN.md.)

This kernel does the entire selection in ONE launch, one block, 256 threads:

    1. zero the demand histogram
    2. accumulate demand over predicted ids, skipping GPU-resident experts
    3. count how many distinct experts are wanted
    4. pick the top `slots` by demand (serial argmax on one thread; 4 passes
       over 256 entries is ~1 us, against the ~28 LAUNCHES being deleted)
    5. apply the selective-skip rule and publish sel / landed / index / stats

Everything is static-shape and in-place, so it captures into the decode graph
and keeps working on replay, which is the constraint that shaped the original
Python version too.

Semantics are matched to the Python it replaces, deliberately including the
awkward parts: pf_index is NOT cleared between calls (stale entries are masked
by pf_landed), ties in demand go to the lower expert id, and the ROUTE /
CPUSKIP flags gate the two landed masks independently so the timing probes
still work.

Not wired in yet. `--verify` checks it against the Python path on random
inputs; `--bench` times both.
"""
import argparse, os, statistics, sys
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// One block. n_exp is 256 here and will not plausibly exceed a few thousand;
// the demand histogram lives in shared memory so the whole thing is one pass
// over the predicted ids plus a handful of block reductions.
extern "C" __global__ void k_pred_select(
    const long* __restrict__ pred,      // [n_pred] predicted expert ids
    const bool* __restrict__ resident,  // [n_exp] GPU-resident mask
    long* __restrict__ sel,             // [n_slot] out: expert id or -1
    bool* __restrict__ landed,          // [n_exp] out
    bool* __restrict__ landed_cpu,      // [n_exp] out
    int* __restrict__ index,            // [n_exp] out: slot per expert
    long* __restrict__ stats,           // [4] fetched, wanted, calls, skipped
    int n_pred, int n_exp, int n_slot,
    int selective, int route, int cpuskip) {
  extern __shared__ int s_dem[];
  const int tid = threadIdx.x;
  const int nth = blockDim.x;

  for (int i = tid; i < n_exp; i += nth) s_dem[i] = 0;
  __syncthreads();

  // Demand over the prediction, counting only what is NOT already resident.
  // Counting the raw predicted set overstates traffic ~3.7x (Stage D3).
  for (int i = tid; i < n_pred; i += nth) {
    const int e = (int)pred[i];
    if (e >= 0 && e < n_exp && !resident[e]) atomicAdd(&s_dem[e], 1);
  }
  __syncthreads();

  // Clear the output masks while the histogram is still being built; these are
  // independent of it and this is the only wide loop left.
  for (int i = tid; i < n_exp; i += nth) { landed[i] = false; landed_cpu[i] = false; }
  __syncthreads();   // thread 0 sets bits in these below; do not race the clear

  // The rest is 256 experts and 4 slots -- about a thousand serial steps, call
  // it a microsecond. A parallel argmax here would be a page of shuffle
  // reductions to save nothing measurable against the ~28 kernel LAUNCHES this
  // whole kernel exists to delete. Keep it obviously correct.
  if (tid == 0) {
    int n_want = 0;
    for (int i = 0; i < n_exp; ++i) n_want += (s_dem[i] > 0) ? 1 : 0;

    // Selective skip: a layer wanting more distinct non-resident experts than
    // there are slots can never be covered completely, and a partially covered
    // layer keeps its CPU call anyway -- so the bytes would buy nothing.
    const bool skip = selective && (n_want > n_slot);

    int fetched = 0;
    for (int k = 0; k < n_slot; ++k) {
      int bv = 0, bi = -1;
      for (int i = 0; i < n_exp; ++i) {      // ties to the lower id, as topk does
        if (s_dem[i] > bv) { bv = s_dem[i]; bi = i; }
      }
      if (bi >= 0) s_dem[bi] = 0;            // consume, so the next k differs
      const int pick = (bv > 0 && !skip) ? bi : -1;
      sel[k] = pick;
      if (pick >= 0) {
        ++fetched;
        if (route) landed[pick] = true;
        if (cpuskip) landed_cpu[pick] = true;
        index[pick] = k;                     // positional: slot k holds sel[k]
      }
    }
    // counters: [fetched, wanted, layers_attempted, layers_skipped]
    stats[0] += route ? fetched : 0;
    stats[1] += n_want;
    stats[2] += 1;
    stats[3] += (fetched == 0) ? 1 : 0;
  }
}

void pred_select(torch::Tensor pred, torch::Tensor resident, torch::Tensor sel,
                 torch::Tensor landed, torch::Tensor landed_cpu,
                 torch::Tensor index, torch::Tensor stats,
                 long selective, long route, long cpuskip) {
  const int n_exp = (int)resident.numel();
  const int n_slot = (int)sel.numel();
  const int n_pred = (int)pred.numel();
  TORCH_CHECK(pred.scalar_type() == torch::kLong, "pred must be int64");
  TORCH_CHECK(index.scalar_type() == torch::kInt, "index must be int32 -- "
              "logical_to_gpu_index is int32 and a silent promotion here "
              "corrupts cutlass's expert ids");
  k_pred_select<<<1, 256, n_exp * sizeof(int), c10::cuda::getCurrentCUDAStream()>>>(
      pred.data_ptr<long>(), resident.data_ptr<bool>(), sel.data_ptr<long>(),
      landed.data_ptr<bool>(), landed_cpu.data_ptr<bool>(),
      index.data_ptr<int>(), stats.data_ptr<long>(),
      n_pred, n_exp, n_slot, (int)selective, (int)route, (int)cpuskip);
}
"""

DECL = ("void pred_select(torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "long, long, long);")

_K = None


def kernels():
    global _K
    if _K is None:
        _K = load_inline(name="pred_select_kernel", cpp_sources=DECL,
                         cuda_sources=CUDA_SRC, functions=["pred_select"],
                         extra_cuda_cflags=["-O3"], verbose=False)
    return _K


def python_reference(pred, resident, n_slot, selective, route, cpuskip,
                     index_prev, stats_prev):
    """Exactly what pf_issue does today, for equivalence checking."""
    n_exp = resident.numel()
    flat = pred.reshape(-1)
    nonres = (~resident[flat]).to(torch.int32)
    demand = torch.zeros(n_exp, dtype=torch.int32, device=pred.device)
    demand.index_add_(0, flat, nonres)
    n_want = (demand > 0).sum()
    vals, ids = torch.topk(demand, n_slot)
    keep = vals > 0
    if selective:
        keep = keep & (n_want <= n_slot)
    sel = torch.where(keep, ids, torch.full_like(ids, -1)).to(torch.int64)

    landed_b = sel >= 0
    valid = landed_b if route else torch.zeros_like(landed_b)
    valid_cpu = landed_b if cpuskip else torch.zeros_like(landed_b)
    ids0 = sel.clamp_min(0)
    landed = torch.zeros(n_exp, dtype=torch.bool, device=pred.device)
    landed.scatter_(0, ids0, valid)
    landed_cpu = torch.zeros(n_exp, dtype=torch.bool, device=pred.device)
    landed_cpu.scatter_(0, ids0, valid_cpu)
    index = index_prev.clone()
    index.scatter_(0, ids0, torch.arange(n_slot, dtype=torch.int32,
                                         device=pred.device))
    stats = stats_prev.clone()
    stats[0] += valid.sum()
    stats[1] += n_want
    stats[2] += 1
    stats[3] += (~keep.any()).to(torch.int64)
    return sel, landed, landed_cpu, index, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-exp", type=int, default=256)
    ap.add_argument("--n-slot", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--p", type=int, default=2, help="predicted experts/token")
    ap.add_argument("--resident", type=int, default=100)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()

    dev = "cuda:0"
    torch.cuda.set_device(0)
    K = kernels()
    g = torch.Generator(device=dev)

    def fresh(seed):
        g.manual_seed(seed)
        resident = torch.zeros(a.n_exp, dtype=torch.bool, device=dev)
        perm = torch.randperm(a.n_exp, generator=g, device=dev)
        resident[perm[:a.resident]] = True
        pred = torch.randint(0, a.n_exp, (a.tokens, a.p), generator=g,
                             device=dev, dtype=torch.int64)
        return resident, pred

    if a.verify:
        bad = 0
        for t in range(a.trials):
            resident, pred = fresh(t)
            for selective in (0, 1):
                for route, cpuskip in ((1, 1), (0, 0), (1, 0)):
                    idx0 = torch.full((a.n_exp,), -1, dtype=torch.int32, device=dev)
                    st0 = torch.zeros(4, dtype=torch.int64, device=dev)
                    r = python_reference(pred, resident, a.n_slot, selective,
                                         route, cpuskip, idx0, st0)
                    sel = torch.full((a.n_slot,), -1, dtype=torch.int64, device=dev)
                    landed = torch.zeros(a.n_exp, dtype=torch.bool, device=dev)
                    landed_cpu = torch.zeros(a.n_exp, dtype=torch.bool, device=dev)
                    index = idx0.clone()
                    stats = st0.clone()
                    K.pred_select(pred, resident, sel, landed, landed_cpu,
                                  index, stats, selective, route, cpuskip)
                    torch.cuda.synchronize()
                    got = (sel, landed, landed_cpu, index, stats)
                    names = ("sel", "landed", "landed_cpu", "index", "stats")
                    # DO NOT compare sel or index field-by-field against the
                    # reference. Demand is a count over 4 tokens x 2 predictions,
                    # so nearly every wanted expert sits at demand 1 and ties are
                    # the normal case. The kernel breaks ties toward the lower
                    # expert id; CUDA topk does not document its order. Two
                    # different tie orders give a different sel PERMUTATION and
                    # hence a different expert->slot map, both internally
                    # consistent and both correct. Comparing them literally
                    # reports ~64% "failures" that are nothing of the kind --
                    # measured, that is exactly what the first version did.
                    #
                    # Check the INVARIANTS instead, which is what the consumer
                    # actually depends on.
                    tag = f"{selective}/{route}{cpuskip}"
                    ksel, klanded, klcpu, kindex, kstats = got
                    psel = r[0]
                    # 1. Same MULTISET of demand values -> same set of experts up
                    #    to ties. Compare the demand each side selected.
                    dem = torch.zeros(a.n_exp + 1, dtype=torch.int32, device=dev)
                    flat = pred.reshape(-1)
                    dem[:a.n_exp].index_add_(
                        0, flat, (~resident[flat]).to(torch.int32))
                    dk = sorted(dem[ksel.clamp_min(0)].tolist())
                    dp = sorted(dem[psel.clamp_min(0)].tolist())
                    if (ksel >= 0).sum() != (psel >= 0).sum() or dk != dp:
                        bad += 1
                        print(f"trial {t} selection/{tag}: kernel picked demand "
                              f"{dk}, reference {dp}")
                    # 2. slot k holds sel[k]: index[sel[k]] == k for every filled
                    #    slot, and landed is set exactly on those experts.
                    for k in range(a.n_slot):
                        e = int(ksel[k])
                        if e < 0:
                            continue
                        if int(kindex[e]) != k:
                            bad += 1
                            print(f"trial {t} index/{tag}: expert {e} in slot {k} "
                                  f"but index says {int(kindex[e])}")
                    want = torch.zeros(a.n_exp, dtype=torch.bool, device=dev)
                    want[ksel[ksel >= 0]] = True
                    if not torch.equal(klanded, want if route else
                                       torch.zeros_like(want)):
                        bad += 1
                        print(f"trial {t} landed/{tag} differs from sel")
                    if not torch.equal(klcpu, want if cpuskip else
                                       torch.zeros_like(want)):
                        bad += 1
                        print(f"trial {t} landed_cpu/{tag} differs from sel")
                    # 3. stats are counts, so ties cannot move them.
                    if not torch.equal(kstats, r[4]):
                        bad += 1
                        print(f"trial {t} stats/{tag}: {kstats.tolist()} vs "
                              f"{r[4].tolist()}")
        print("VERIFY OK" if bad == 0 else f"VERIFY FAILED ({bad} mismatches)")
        return 0 if bad == 0 else 1

    if a.bench:
        resident, pred = fresh(0)
        sel = torch.full((a.n_slot,), -1, dtype=torch.int64, device=dev)
        landed = torch.zeros(a.n_exp, dtype=torch.bool, device=dev)
        landed_cpu = torch.zeros(a.n_exp, dtype=torch.bool, device=dev)
        index = torch.full((a.n_exp,), -1, dtype=torch.int32, device=dev)
        stats = torch.zeros(4, dtype=torch.int64, device=dev)

        def fused():
            K.pred_select(pred, resident, sel, landed, landed_cpu, index,
                          stats, 1, 1, 1)

        # NOT python_reference(): that one allocates its outputs, and a harness
        # that charges the incumbent for 75 allocations it does not actually do
        # would manufacture the speedup this file exists to test. What follows
        # is pf_issue transcribed literally -- preallocated buffers, in-place
        # everywhere, same op sequence -- so the comparison is launch count
        # against launch count.
        demand = torch.zeros(a.n_exp, dtype=torch.int32, device=dev)
        slot_ids = torch.arange(a.n_slot, dtype=torch.int32, device=dev)
        n_slot_t = torch.tensor(a.n_slot, device=dev)

        def pyver():
            flat = pred.reshape(-1)
            nonres = (~resident[flat]).to(demand.dtype)
            demand.zero_()
            demand.index_add_(0, flat, nonres)
            n_want = (demand > 0).sum()
            vals, ids = torch.topk(demand, a.n_slot)
            keep = (vals > 0) & (n_want <= n_slot_t)
            sel.copy_(torch.where(keep, ids, torch.full_like(ids, -1)))
            valid = sel >= 0
            ids0 = sel.clamp_min(0)
            landed.zero_()
            landed.scatter_(0, ids0, valid)
            landed_cpu.zero_()
            landed_cpu.scatter_(0, ids0, valid)
            index.scatter_(0, ids0, slot_ids)
            stats[0] += valid.sum()
            stats[1] += n_want
            stats[2] += 1
            stats[3] += (~keep.any()).to(torch.int64)

        def timed(fn, n=50):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for _ in range(75):     # one decode step's worth of MoE layers
                    fn()
            for _ in range(5):
                gr.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(n):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); gr.replay(); e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1))
            return statistics.median(ts)

        tf, tp = timed(fused), timed(pyver)
        print(f"75 layers' worth of selection, captured in a graph:")
        print(f"  current Python path  {tp:7.3f} ms/step")
        print(f"  fused kernel         {tf:7.3f} ms/step")
        print(f"  saved                {tp - tf:7.3f} ms/step  ({tp/max(tf,1e-9):.1f}x)")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
