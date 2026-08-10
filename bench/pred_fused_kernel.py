#!/usr/bin/env python3
"""The whole lookahead predictor in ONE kernel launch per layer.

WHY. The predictor costs 5.47 ms/step, which is 73 us per layer, and none of it
is arithmetic (Stage H7). It is ~36 kernel launches per layer -- one gate GEMM,
the sigmoid/bias/topk scoring path, a top-P gather, then ~28 tiny selection ops
-- times 75 layers, about 2,700 extra graph nodes per decode step at ~2 us of
dispatch each. Every tensor involved is four tokens wide, so nothing runs long
enough to hide its own launch.

THE SHAPE OF THE FIX. Two pieces, and only the second is this file:

  1. The gate GEMM becomes FREE by folding into the network. At the "pre"
     prediction point layer L's own gate and the lookahead for L+1 are applied
     to the SAME hidden state, so if the 75 gate weights are stacked
     contiguously, layer L can take a 512-row slice and get both sets of logits
     from ONE F.linear. Zero extra launches, zero extra memory (a view, not a
     copy), and the extra 3.1 MB of weight read per layer is ~0.08 ms/step of
     HBM at 3 TB/s.

     This is why the fused path requires KT_PRED_POINT=pre. The "post" point
     feeds a different, renormalised input per target and cannot share the GEMM.
     Post measured 0.32 ms better (Stage H6); the fusion is worth ~2 ms. Take
     the fusion.

  2. Everything after the GEMM becomes ONE launch -- this file. Scoring,
     top-8, top-P, the residency intersect, the demand histogram, the slot
     pick, and publishing sel/landed/index/stats.

WHY ONE BLOCK IS ENOUGH HERE, AND WHY IT WOULD NOT BE FOR THE GEMM. After the
GEMM this kernel touches [T,256] logits, [256] bias and [256] mask -- a few KB.
A single block does that in microseconds. Fusing the GEMM in as well would mean
one block reading the 3.1 MB gate weight, which at one SM's share of HBM is
~100 us: worse than the 30 us being removed. That is exactly why the GEMM is
folded into the network's own gate instead of into this kernel.

MODEL SPECIFICS, from config.json -- checked, not assumed:
  scoring_func       sigmoid
  topk_method        noaux_tc   -> scores_for_choice = sigmoid(logit) + bias
  n_group / topk_group  1 / 1   -> the group stage is a no-op, plain topk
  norm_topk_prob     True       -> weights renormalised, which is MONOTONE
                                   within a token, so "top-P by weight among the
                                   top-K" == "top-P by sigmoid score among them"
                                   and the renormalise can be skipped entirely.

Semantics are matched to `_kt_pred_topP` + `pf_issue` deliberately, including
the awkward parts: ties go to the lower expert id, pf_index is not cleared
between calls, and the ROUTE / CPUSKIP flags gate the two landed masks
independently so the timing probes still work.

    --verify   check against a Python transcription of the shipped path
    --bench    time it against that transcription, 75 layers, in a CUDA graph
"""
import argparse
import statistics
import sys

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <climits>
#include <cuda_bf16.h>


// A block-wide argmax with LOWER-INDEX tie-breaking, matching torch.topk's
// behaviour on this path. Warp shuffles first, then one pass over the 8 warp
// partials.
//
// This replaces a serial scan on thread 0. That scan is what made the first
// version of this kernel cost 74 us per launch: 4 tokens x 8 rounds x 256
// comparisons, ~9k serial iterations, which is far worse than the ~36 launches
// the kernel exists to delete. Measured 1.3x; the parallel form is the point.
__device__ __forceinline__ void block_argmax(const float* __restrict__ s,
                                             int n, int tid, int nth,
                                             float* sv, int* si,
                                             float& out_v, int& out_i) {
  float v = -1e30f;
  int   idx = INT_MAX;                 // sentinel above every real index, so
  for (int i = tid; i < n; i += nth) { // "smaller index wins" needs no special
    if (s[i] > v) { v = s[i]; idx = i; }  // case
  }
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffff, v, off);
    const int   oi = __shfl_down_sync(0xffffffff, idx, off);
    if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
  }
  const int lane = tid & 31, warp = tid >> 5;
  if (lane == 0) { sv[warp] = v; si[warp] = idx; }
  __syncthreads();
  if (tid == 0) {
    const int nw = (nth + 31) >> 5;
    float bv = sv[0]; int bi = si[0];
    for (int w = 1; w < nw; ++w) {
      if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
    }
    sv[0] = bv; si[0] = bi;
  }
  __syncthreads();
  out_v = sv[0];
  out_i = si[0];
}

// One block. Shared memory holds the demand histogram plus one token's scores.
// n_exp is 256 here; the launch sizes shared memory from it so a larger router
// still works as long as it fits.
extern "C" __global__ void k_pred_fused(
    const void*  __restrict__ logits,   // [T, n_exp] lookahead router output,
                                        // float32 or bfloat16 (see is_bf16)
    const float* __restrict__ bias,     // [n_exp] e_score_correction_bias
    const bool*  __restrict__ resident, // [n_exp] GPU-residency mask
    long* __restrict__ sel,             // [n_slot] out: expert id or -1
    bool* __restrict__ landed,          // [n_exp] out
    bool* __restrict__ landed_cpu,      // [n_exp] out
    int*  __restrict__ index,           // [n_exp] out: landing slot per expert
    long* __restrict__ stats,           // [5] fetched, wanted, calls, skipped,
                                        //     reusable (see `reuse`)
    long* __restrict__ hold,            // [n_slot] in/out: which expert each slot
                                        // PHYSICALLY holds right now. Only the
                                        // gather writes a slot, and slots are
                                        // per-layer, so an entry stays true until
                                        // this layer fetches over it. -1 = unknown.
    int reuse,                          // 0 off, 1 measure only, 2 act
    int T, int n_exp, int n_slot, int topk, int topP,
    int slot_base,                      // ABSOLUTE index of landing slot 0 in
                                        // the layer's cutlass expert tensors.
                                        // `index` is merged with
                                        // logical_to_gpu_index, which is
                                        // absolute, so writing the relative slot
                                        // number here routes a landed expert to
                                        // GPU expert 0..n_slot-1 -- four real,
                                        // resident, and completely wrong experts.
    int ld,                             // row stride of `logits`, in elements
    int is_bf16,                        // the model runs bf16, so the fused
                                        // gate output is bf16 -- converting it
                                        // in Python would cost the launch this
                                        // kernel exists to remove
    int selective, int route, int cpuskip) {
  // ONE WARP PER TOKEN. The decode batch is at most 8 tokens and n_exp is 256,
  // so a warp holds a whole token's router row in registers -- 8 experts per
  // lane -- and the top-k selection becomes pure warp shuffles with NO
  // __syncthreads at all.
  //
  // This is the third shape of this kernel and the measurements drove every
  // step. Serial scan on thread 0: 74 us/launch (worse than the launches it
  // deletes). Block-wide parallel argmax: 33 us, dominated by 36 sequential
  // rounds x 2 block syncs. Warp-per-token: the 8 rounds happen once for all
  // tokens at once and cost no sync.
  extern __shared__ char smem[];
  int* s_dem = (int*)smem;                        // [n_exp] demand histogram
  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nth  = blockDim.x;

  for (int i = tid; i < n_exp; i += nth) s_dem[i] = 0;
  for (int i = tid; i < n_exp; i += nth) { landed[i] = false; landed_cpu[i] = false; }
  __syncthreads();

  // ---- per-token selection, one warp each -------------------------------
  // PER_LANE is a compile-time 8 (256/32). Dynamic indexing into a register
  // array would spill it to local memory, so every access is unrolled.
  const int PER_LANE = 8;
  if (warp < T && n_exp == 32 * PER_LANE) {
    const long roff = (long)warp * ld;
    float sc[8], ch[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = lane * 8 + j;
      const float x = is_bf16
          ? __bfloat162float(((const __nv_bfloat16*)logits)[roff + e])
          : ((const float*)logits)[roff + e];
      const float v = 1.0f / (1.0f + __expf(-x));        // sigmoid, in float
      sc[j] = v;
      ch[j] = v + bias[e];                               // noaux_tc scoring
    }
    // The router's top-`topk` by (score + bias), lower id winning ties.
    int   pick[64];
    float pick_s[64];
    const int K = topk < 64 ? topk : 64;
    for (int k = 0; k < K; ++k) {
      float bv = -1e30f; int bi = INT_MAX;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (ch[j] > bv) { bv = ch[j]; bi = lane * 8 + j; }
      }
      for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffff, bv, off);
        const int   oi = __shfl_down_sync(0xffffffff, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      bi = __shfl_sync(0xffffffff, bi, 0);
      // Consume it in whichever lane owns it, and broadcast its raw score.
      float s_of = -1e30f;
      if (bi != INT_MAX && bi / 8 == lane) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (bi % 8 == j) { ch[j] = -1e30f; s_of = sc[j]; }
        }
      }
      s_of = __shfl_sync(0xffffffff, s_of, bi == INT_MAX ? 0 : bi / 8);
      pick[k]   = (bi == INT_MAX) ? -1 : bi;
      pick_s[k] = s_of;
    }
    // Among those, the top-`topP` by RAW score -- which is the same ordering as
    // by renormalised weight, the renormalise being a positive per-token
    // scalar. K is 8, so one lane is the right place for this.
    if (lane == 0) {
      const int P = topP < K ? topP : K;
      for (int p = 0; p < P; ++p) {
        float bv = -1e30f; int bk = -1;
        for (int k = 0; k < K; ++k) {
          if (pick[k] >= 0 && pick_s[k] > bv) { bv = pick_s[k]; bk = k; }
        }
        if (bk < 0) break;
        const int e = pick[bk];
        pick[bk] = -1;                              // consume
        if (!resident[e]) atomicAdd(&s_dem[e], 1);  // NON-RESIDENT demand only
      }
    }
  }
  __syncthreads();

  // ---- slot selection, identical to pf_issue ----------------------------
  // n_slot is 4-16 and the histogram is 256 ints; one warp, same shuffle form.
  if (warp == 0) {
    int d[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) d[j] = (n_exp == 256) ? s_dem[lane * 8 + j] : 0;
    int nw = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) nw += (d[j] > 0) ? 1 : 0;
    for (int off = 16; off > 0; off >>= 1) nw += __shfl_down_sync(0xffffffff, nw, off);
    const int n_want = __shfl_sync(0xffffffff, nw, 0);
    // A layer wanting more distinct non-resident experts than there are slots
    // can never be covered completely, and a partially covered layer keeps its
    // CPU call anyway -- so the bytes would buy nothing.
    const bool skip = selective && (n_want > n_slot);
    // REUSE. A slot's bytes survive to the next call -- nothing but this layer's
    // own gather ever writes them -- so an expert still sitting in a slot and
    // wanted again costs zero bytes to "fetch". `s_dem` is untouched by the pick
    // loop below (only the register copy `d[]` is consumed), so it still carries
    // the original demand and can be probed by expert id.
    //
    // Counted BEFORE any picking, and in mode 1 only counted: the whole point is
    // to learn the hit rate before changing what the server does. Stage H15
    // prices this at ~5.1 ms per expert/call, so the counter decides whether the
    // path is worth building at all.
    int reusable = 0;
    unsigned keep_mask = 0;
    if (reuse && lane == 0) {
      int seen[16];
      int n_seen = 0;
      for (int k = 0; k < n_slot; ++k) {
        const int h = (int)hold[k];
        if (h < 0 || s_dem[h] <= 0) continue;
        // Two slots can transiently hold the SAME expert (mode 1 re-fetches an
        // expert a slot already had, into a different slot). Covering it twice
        // would burn a slot and race two writers on index[h], so the duplicate
        // is passed over and left for the fill pass to reclaim.
        bool dup = false;
        for (int s = 0; s < n_seen; ++s) if (seen[s] == h) { dup = true; break; }
        if (dup) continue;
        seen[n_seen++] = h;
        ++reusable;
        if (reuse >= 2) {
          keep_mask |= (1u << k);
          // The slot already holds this expert's bytes, so publish the routing
          // and fetch NOTHING: sel[k] = -1 makes all four gather kernels
          // early-out. Coverage identical, bytes zero.
          if (route)   landed[h]     = true;
          if (cpuskip) landed_cpu[h] = true;
          index[h] = slot_base + k;
          sel[k]   = -1;
        }
      }
    }
    reusable  = __shfl_sync(0xffffffff, reusable, 0);
    keep_mask = __shfl_sync(0xffffffff, keep_mask, 0);
    // Remove kept experts from the demand the fill pass sees, in whichever lane
    // owns them, so the same expert is never fetched into a second slot.
    if (keep_mask) {
      for (int k = 0; k < n_slot; ++k) {
        if (!(keep_mask & (1u << k))) continue;
        const int h = (int)hold[k];
        if (h >= 0 && h / 8 == lane) {
#pragma unroll
          for (int j = 0; j < 8; ++j) { if (h % 8 == j) d[j] = 0; }
        }
      }
    }
    int fetched = 0;
    for (int k = 0; k < n_slot; ++k) {
      // A kept slot is already published and must not be refilled.
      if (keep_mask & (1u << k)) continue;
      int bv = 0, bi = INT_MAX;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (d[j] > bv) { bv = d[j]; bi = lane * 8 + j; }
      }
      for (int off = 16; off > 0; off >>= 1) {
        const int ov = __shfl_down_sync(0xffffffff, bv, off);
        const int oi = __shfl_down_sync(0xffffffff, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      bv = __shfl_sync(0xffffffff, bv, 0);
      bi = __shfl_sync(0xffffffff, bi, 0);
      if (bi != INT_MAX && bi / 8 == lane) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { if (bi % 8 == j) d[j] = 0; }   // consume
      }
      const int pick = (bv > 0 && bi != INT_MAX && !skip) ? bi : -1;
      if (lane == 0) {
        sel[k] = pick;
        if (pick >= 0) {
          if (route)   landed[pick]     = true;
          if (cpuskip) landed_cpu[pick] = true;
          index[pick] = slot_base + k;      // positional: slot k holds sel[k],
                                            // and slots are the TRAILING experts
                                            // [num_gpu_experts, +n_slot) of the
                                            // same tensor cutlass indexes.
        }
      }
      if (pick >= 0) ++fetched;
    }
    if (lane == 0) {
      // A slot is only overwritten when something was actually fetched into it,
      // so `hold` tracks physical contents exactly. A slot that got no pick keeps
      // its previous expert -- and keeping that entry is what makes the NEXT call
      // able to reuse it.
      if (reuse) {
        for (int k = 0; k < n_slot; ++k) {
          if (sel[k] >= 0) hold[k] = sel[k];
        }
      }
      stats[0] += route ? fetched : 0;
      stats[1] += n_want;
      stats[2] += 1;
      stats[3] += (fetched == 0) ? 1 : 0;
      stats[4] += reusable;
    }
  }
}

void pred_fused(torch::Tensor logits, torch::Tensor bias, torch::Tensor resident,
                torch::Tensor sel, torch::Tensor landed, torch::Tensor landed_cpu,
                torch::Tensor index, torch::Tensor stats, torch::Tensor hold,
                long topk, long topP, long col_off, long slot_base,
                long selective, long route, long cpuskip, long reuse) {
  const int T     = (int)logits.size(0);
  const int ld    = (int)logits.stride(0);
  const int n_exp = (int)resident.numel();
  const int n_slot = (int)sel.numel();
  const bool bf16 = logits.scalar_type() == torch::kBFloat16;
  TORCH_CHECK(bf16 || logits.scalar_type() == torch::kFloat,
              "logits must be float32 or bfloat16");
  TORCH_CHECK(bias.scalar_type() == torch::kFloat, "bias must be float32");
  TORCH_CHECK(index.scalar_type() == torch::kInt, "index must be int32 -- "
              "logical_to_gpu_index is int32 and a silent promotion here "
              "corrupts cutlass's expert ids");
  TORCH_CHECK(stats.numel() >= 5, "stats needs 5 slots (reusable is [4])");
  TORCH_CHECK(hold.numel() == sel.numel(), "hold must be one entry per slot");
  TORCH_CHECK(hold.scalar_type() == torch::kLong, "hold must be int64");
  // The keep pass carries a 16-entry duplicate list and a 32-bit slot mask.
  TORCH_CHECK(sel.numel() <= 16, "reuse keep-pass assumes at most 16 slots");
  TORCH_CHECK(n_exp == 256, "warp-per-token form assumes 256 experts");
  TORCH_CHECK(T <= 8, "one warp per token, 8 warps in the block");
  TORCH_CHECK(topk <= 64, "topk > 64 needs a bigger `chosen` array");
  // demand + scores + choice, then 32 float and 32 int warp partials and a
  // 64-entry pick list.
  // Only the demand histogram lives in shared memory now; the scores stay in
  // registers, one token per warp.
  const size_t shmem = (size_t)n_exp * sizeof(int);
  TORCH_CHECK(logits.stride(1) == 1, "logits rows must be unit-stride");
  TORCH_CHECK(col_off + n_exp <= logits.size(1), "col_off out of range");
  const void* lp = bf16
      ? (const void*)(logits.data_ptr<at::BFloat16>() + col_off)
      : (const void*)(logits.data_ptr<float>() + col_off);
  k_pred_fused<<<1, 256, shmem, c10::cuda::getCurrentCUDAStream()>>>(
      lp, bias.data_ptr<float>(),
      resident.data_ptr<bool>(), sel.data_ptr<long>(),
      landed.data_ptr<bool>(), landed_cpu.data_ptr<bool>(),
      index.data_ptr<int>(), stats.data_ptr<long>(), hold.data_ptr<long>(),
      (int)reuse,
      T, n_exp, n_slot, (int)topk, (int)topP, (int)slot_base, ld, bf16 ? 1 : 0,
      (int)selective, (int)route, (int)cpuskip);
}
"""

DECL = ("void pred_fused(torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, "
        "long, long, long, long, long, long, long, long);")

_K = None


def kernels():
    global _K
    if _K is None:
        _K = load_inline(name="pred_fused_kernel", cpp_sources=DECL,
                         cuda_sources=CUDA_SRC, functions=["pred_fused"],
                         extra_cuda_cflags=["-O3"], verbose=False)
    return _K


def python_path(logits, bias, resident, n_slot, topk, topP,
                selective, route, cpuskip):
    """`_kt_pred_topP` followed by `pf_issue`, transcribed.

    This is the incumbent: the sigmoid/bias scoring, the router's top-k, the
    top-P gather, then the demand histogram and slot pick.
    """
    n_exp = resident.numel()
    scores = torch.sigmoid(logits.float())
    choice = scores + bias
    _, ids = torch.topk(choice, topk, dim=-1)
    w = torch.gather(scores, 1, ids)
    w = w / torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-9)   # norm_topk_prob
    top = torch.gather(ids, 1, torch.topk(w, topP, dim=-1).indices)

    flat = top.reshape(-1)
    demand = torch.zeros(n_exp, dtype=torch.int32, device=logits.device)
    demand.index_add_(0, flat, (~resident[flat]).to(torch.int32))
    n_want = (demand > 0).sum()
    vals, sids = torch.topk(demand, n_slot)
    keep = vals > 0
    if selective:
        keep = keep & (n_want <= n_slot)
    sel = torch.where(keep, sids, torch.full_like(sids, -1)).to(torch.int64)
    return sel, demand, int(n_want)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-exp", type=int, default=256)
    ap.add_argument("--n-slot", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--topP", type=int, default=2)
    ap.add_argument("--resident", type=int, default=100)
    ap.add_argument("--trials", type=int, default=200)
    # Non-zero on purpose. Slot 0 is the trailing expert `num_gpu_experts` of the
    # layer's cutlass tensors, never index 0, and the original kernel wrote the
    # RELATIVE slot number -- which routed every landed expert to GPU experts
    # 0..n_slot-1, four real and entirely wrong experts. A default of 0 here would
    # let that come back without a single test noticing.
    ap.add_argument("--slot-base", type=int, default=100)
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
        resident[torch.randperm(a.n_exp, generator=g, device=dev)[:a.resident]] = True
        # Router logits are small and centred; sigmoid of a wide normal would
        # saturate and make every score identical, hiding tie behaviour.
        logits = torch.randn(a.tokens, a.n_exp, generator=g, device=dev) * 0.5
        # The server feeds bf16 -- the model's dtype -- and a float32-only
        # kernel failed capture on the first boot. Alternate, so both are
        # covered every run.
        if seed % 2:
            logits = logits.to(torch.bfloat16)
        bias = torch.randn(a.n_exp, generator=g, device=dev) * 0.1
        return resident, logits, bias

    def run_kernel(logits, bias, resident, selective, route, cpuskip, idx0, st0,
                   hold=None, reuse=0):
        sel = torch.full((a.n_slot,), -1, dtype=torch.int64, device=dev)
        landed = torch.zeros(a.n_exp, dtype=torch.bool, device=dev)
        landed_cpu = torch.zeros(a.n_exp, dtype=torch.bool, device=dev)
        index, stats = idx0.clone(), st0.clone()
        if hold is None:
            hold = torch.full((a.n_slot,), -1, dtype=torch.int64, device=dev)
        K.pred_fused(logits, bias, resident, sel, landed, landed_cpu, index,
                     stats, hold, a.topk, a.topP, 0, a.slot_base,
                     selective, route, cpuskip, reuse)
        torch.cuda.synchronize()
        return sel, landed, landed_cpu, index, stats, hold

    if a.verify:
        bad = 0
        for t in range(a.trials):
            resident, logits, bias = fresh(t)
            for selective in (0, 1):
                for route, cpuskip in ((1, 1), (0, 0), (1, 0)):
                    idx0 = torch.full((a.n_exp,), -1, dtype=torch.int32, device=dev)
                    st0 = torch.zeros(5, dtype=torch.int64, device=dev)
                    psel, pdem, pwant = python_path(
                        logits, bias, resident, a.n_slot, a.topk, a.topP,
                        selective, route, cpuskip)
                    # A NON-EMPTY hold, seeded from the reference demand, so the
                    # reuse counter is exercised on every trial rather than only
                    # on the degenerate all -1 case. Two of the four slots are
                    # deliberately experts the reference DID want and two are
                    # arbitrary, so a kernel that counted everything or nothing
                    # would fail.
                    hold = torch.full((a.n_slot,), -1, dtype=torch.int64, device=dev)
                    wanted_ids = torch.nonzero(pdem > 0).flatten()
                    if wanted_ids.numel() >= 2:
                        hold[0] = wanted_ids[0]
                        hold[1] = wanted_ids[-1]
                    hold[2 % a.n_slot] = (t * 7) % a.n_exp
                    hold_in = hold.clone()
                    ksel, klanded, klcpu, kindex, kstats, khold = run_kernel(
                        logits, bias, resident, selective, route, cpuskip,
                        idx0, st0, hold=hold, reuse=1)
                    tag = f"{selective}/{route}{cpuskip}"
                    # Independent recomputation: count hold entries whose expert
                    # the REFERENCE path found demand for. Different code,
                    # different data structure, same claim.
                    #
                    # DISTINCT experts. An expert held in two slots at once --
                    # which mode 1 produces, by re-fetching into a free slot
                    # something an older slot still has -- is wanted once and can
                    # skip one fetch, not two. Counting it twice would both
                    # overstate the saving and imply two slots writing index[h].
                    # The first version of this line did not dedupe and failed
                    # trial 80, where the seeded arbitrary expert (80*7)%256=48
                    # collided with a wanted one. The KERNEL was right.
                    _seen, exp_reuse = set(), 0
                    for h in hold_in.tolist():
                        if h >= 0 and int(pdem[h]) > 0 and h not in _seen:
                            _seen.add(h)
                            exp_reuse += 1
                    if int(kstats[4]) != exp_reuse:
                        bad += 1
                        print(f"trial {t} reusable/{tag}: kernel {int(kstats[4])} "
                              f"vs reference {exp_reuse}")
                    # Mode 1 must not change WHAT is selected -- it only counts.
                    # And `hold` must end up describing the slots' real contents:
                    # overwritten where a fetch landed, preserved everywhere else.
                    for k in range(a.n_slot):
                        want_h = int(ksel[k]) if int(ksel[k]) >= 0 else int(hold_in[k])
                        if int(khold[k]) != want_h:
                            bad += 1
                            print(f"trial {t} hold/{tag}: slot {k} is "
                                  f"{int(khold[k])}, expected {want_h}")

                    # ---- mode 2: same hold, acting on it -------------------
                    # Reuse is only sound if it changes BYTES and not COVERAGE.
                    # Both are asserted here against a reference built from the
                    # incumbent's demand, not from the kernel's own bookkeeping.
                    hold2 = hold_in.clone()
                    s2, l2, lc2, i2, st2, h2 = run_kernel(
                        logits, bias, resident, selective, route, cpuskip,
                        idx0, st0, hold=hold2, reuse=2)
                    exp_kept, seen = {}, set()
                    for k in range(a.n_slot):
                        h = int(hold_in[k])
                        if h >= 0 and int(pdem[h]) > 0 and h not in seen:
                            seen.add(h)
                            exp_kept[k] = h
                    for k, h in exp_kept.items():
                        if int(s2[k]) != -1:
                            bad += 1
                            print(f"trial {t} keep/{tag}: slot {k} holds wanted "
                                  f"expert {h} but still fetches {int(s2[k])}")
                        if int(i2[h]) != a.slot_base + k:
                            bad += 1
                            print(f"trial {t} keepidx/{tag}: expert {h} -> "
                                  f"{int(i2[h])}, expected {a.slot_base + k}")
                        if route and not bool(l2[h]):
                            bad += 1
                            print(f"trial {t} keeplanded/{tag}: expert {h} unrouted")
                        if int(h2[k]) != h:
                            bad += 1
                            print(f"trial {t} keephold/{tag}: slot {k} lost {h}")
                    # An expert must never be kept AND fetched: that wastes a
                    # slot and races two writers on index[].
                    fetched2 = {int(x) for x in s2.tolist() if int(x) >= 0}
                    if fetched2 & set(exp_kept.values()):
                        bad += 1
                        print(f"trial {t} dup/{tag}: {fetched2 & set(exp_kept.values())} "
                              f"both kept and fetched")
                    # THE invariant. Freed slots may cover MORE, never less.
                    if route and int(l2.sum()) < int(klanded.sum()):
                        bad += 1
                        print(f"trial {t} coverage/{tag}: mode 2 covers "
                              f"{int(l2.sum())} < mode 1 {int(klanded.sum())}")
                    # Compare by the DEMAND each side selected, not by the ids:
                    # demand is a small integer count so ties are the normal
                    # case, and two tie orders give different-but-equally-correct
                    # id sets. Learned the hard way -- comparing ids literally
                    # reported ~64% false failures on the earlier kernel.
                    dk = sorted(pdem[ksel.clamp_min(0)].tolist())
                    dp = sorted(pdem[psel.clamp_min(0)].tolist())
                    if (ksel >= 0).sum() != (psel >= 0).sum() or dk != dp:
                        bad += 1
                        print(f"trial {t} selection/{tag}: kernel demand {dk}, "
                              f"reference {dp}")
                    if int(kstats[1]) != pwant:
                        bad += 1
                        print(f"trial {t} wanted/{tag}: {int(kstats[1])} vs {pwant}")
                    # Invariants the consumer depends on.
                    # ABSOLUTE, not relative. This assertion used to read
                    # `!= k`, which is exactly the defect it was supposed to
                    # catch: the kernel and its test shared one misunderstanding,
                    # so 150 trials passed while every landed expert in the
                    # server was being computed as GPU expert 0..3.
                    for k in range(a.n_slot):
                        e = int(ksel[k])
                        if e >= 0 and int(kindex[e]) != a.slot_base + k:
                            bad += 1
                            print(f"trial {t} index/{tag}: expert {e} in slot {k} "
                                  f"should map to {a.slot_base + k} "
                                  f"but index says {int(kindex[e])}")
                    want = torch.zeros(a.n_exp, dtype=torch.bool, device=dev)
                    if (ksel >= 0).any():
                        want[ksel[ksel >= 0]] = True
                    zero = torch.zeros_like(want)
                    if not torch.equal(klanded, want if route else zero):
                        bad += 1
                        print(f"trial {t} landed/{tag} differs from sel")
                    if not torch.equal(klcpu, want if cpuskip else zero):
                        bad += 1
                        print(f"trial {t} landed_cpu/{tag} differs from sel")
        print("VERIFY OK" if bad == 0 else f"VERIFY FAILED ({bad} mismatches)")
        return 0 if bad == 0 else 1

    if a.bench:
        resident, logits, bias = fresh(0)
        sel = torch.full((a.n_slot,), -1, dtype=torch.int64, device=dev)
        landed = torch.zeros(a.n_exp, dtype=torch.bool, device=dev)
        landed_cpu = torch.zeros(a.n_exp, dtype=torch.bool, device=dev)
        index = torch.full((a.n_exp,), -1, dtype=torch.int32, device=dev)
        stats = torch.zeros(5, dtype=torch.int64, device=dev)
        hold = torch.full((a.n_slot,), -1, dtype=torch.int64, device=dev)

        def fused():
            # reuse=1 (measure) is what the server will run first, so time THAT,
            # not a mode nothing uses.
            K.pred_fused(logits, bias, resident, sel, landed, landed_cpu,
                         index, stats, hold, a.topk, a.topP, 0, a.slot_base,
                         1, 1, 1, 1)

        # The incumbent, in place where pf_issue is in place, so the comparison
        # is launch count against launch count and not allocator against
        # allocator.
        demand = torch.zeros(a.n_exp, dtype=torch.int32, device=dev)
        # Matches pf_issue's `pf_slot_ids`, which is absolute.
        slot_ids = torch.arange(a.slot_base, a.slot_base + a.n_slot,
                                dtype=torch.int32, device=dev)
        n_slot_t = torch.tensor(a.n_slot, device=dev)

        def pyver():
            scores = torch.sigmoid(logits.float())
            choice = scores + bias
            _, ids = torch.topk(choice, a.topk, dim=-1)
            w = torch.gather(scores, 1, ids)
            w = w / torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-9)
            top = torch.gather(ids, 1, torch.topk(w, a.topP, dim=-1).indices)
            flat = top.reshape(-1)
            demand.zero_()
            demand.index_add_(0, flat, (~resident[flat]).to(torch.int32))
            n_want = (demand > 0).sum()
            vals, sids = torch.topk(demand, a.n_slot)
            keep = (vals > 0) & (n_want <= n_slot_t)
            sel.copy_(torch.where(keep, sids, torch.full_like(sids, -1)))
            valid = sel >= 0
            ids0 = sel.clamp_min(0)
            landed.zero_(); landed.scatter_(0, ids0, valid)
            landed_cpu.zero_(); landed_cpu.scatter_(0, ids0, valid)
            index.scatter_(0, ids0, slot_ids)
            stats[0] += valid.sum(); stats[1] += n_want; stats[2] += 1
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
                for _ in range(75):        # one decode step's worth of MoE layers
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
        print("75 layers of scoring + top-P + selection, captured in a graph:")
        print(f"  shipped path, in place  {tp:7.3f} ms/step")
        print(f"  one fused kernel        {tf:7.3f} ms/step")
        print(f"  saved                   {tp - tf:7.3f} ms/step  "
              f"({tp / max(tf, 1e-9):.1f}x)")
        print("\nNOTE: the gate GEMM is NOT in either column -- it is removed")
        print("separately by folding it into the layer's own gate. Add ~0.08 ms")
        print("of HBM for the wider weight read to the fused column.")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
