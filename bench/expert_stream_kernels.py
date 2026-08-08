"""Gather one or more experts out of kt's CPU store straight into cutlass W4A8 layout.

Stage B, but not the way the plan wrote it.  The plan proposed hoisting the
scale interleave to boot by keeping a second, pre-interleaved host copy of the
scales (~11 GB).  That is unnecessary: the gather is *already* a copy, so the
interleave costs nothing if it is folded into the destination indexing rather
than run as its own GPU pass.  That matters more than the plan assumed --
`phase3_mechanism.json` measures a standalone repack at 0.0375 ms/expert, and at
~2.3 experts x 62 active layer-calls per step that would be ~5.4 ms/step against
a ~58 ms target, i.e. a third of the projected win spent on a permutation.

Folding it in also means the fp32->bf16 scale conversion happens in the same
pass.  The cost of NOT keeping a bf16 host copy is that the link moves fp32
scales: +3.03% bytes per expert, measured in Stage A1.  That is the whole
tradeoff, and 3% of link duty is cheaper than 11 GB of RAM plus a second
store to keep coherent.

Source layout (kt CPU store, per TP partition, per expert):
    gate/up   [tp_moe, hidden/2]      packed int4, low nibble = even k
    down      [hidden, tp_moe/2]      packed int4
    *_scale   fp32, [rows, cols/group]

Destination layout (what cutlass_w4a8_moe consumes):
    w13   [2*moe, hidden/2]           gate rows then up rows, TP partitions
                                      concatenated along the intermediate dim
    w2    [hidden, moe/2]             TP partitions concatenated along K
    w13_s [G13/4, 2*moe*4] bf16       G13 = hidden/group
    w2_s  [G2/4,  hidden*4] bf16      G2  = moe/group
  where the scale interleave is dst[g, n*4 + a] = src[n, g*4 + a], matching
  sglang's `interleave_scales` (alignment 4).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// The store is one allocation per (tp, expert, region), so every kernel here
// reads through a device-resident pointer TABLE rather than a base+stride.  The
// table is filled once at boot; `sel` is what changes between graph replays,
// which is what lets one captured graph move different experts every step.
//   table layout: ptrs[region][tp][expert], region order
//                 0 gate_w, 1 gate_s, 2 up_w, 3 up_s, 4 down_w, 5 down_s
__device__ __forceinline__ const void* tbl(const unsigned long long* p, int region,
                                           int tp, int expert, int n_tp, int n_exp) {
  return reinterpret_cast<const void*>(p[((long)region * n_tp + tp) * n_exp + expert]);
}

// ---- w13 weights: four contiguous row-blocks per slot ----------------------
// dst rows [0,moe) = gate (tp-major), [moe,2*moe) = up (tp-major).
__global__ void k_w13_weights(const unsigned long long* __restrict__ ptrs,
                              const long* __restrict__ sel,
                              uint4* __restrict__ dst,
                              int n_tp, int n_exp, long vec_per_tp_block,
                              long vec_per_slot) {
  const int slot = blockIdx.z;
  const int blk = blockIdx.y;          // 0..2*n_tp-1
  const int tp = blk % n_tp;
  const int is_up = blk / n_tp;        // 0 = gate, 1 = up
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

// ---- w2 weights: row-strided, TP partitions concatenated along K -----------
// dst[r, tp*tp_moe/2 + c] = down[tp][r, c]
//
// Sixteen bytes per thread-iteration. The byte-granular version this replaces
// was LATENCY-bound rather than bandwidth-bound: one uint8 load per trip over
// PCIe, with an emulated 64-bit divide on top of every byte. It reached only
// 30 GB/s, and only at >= 64 blocks -- 4.1 GB/s at 8 -- so it could move
// anything at all only by filling the machine with stalled warps. That is why
// the gather never overlapped with the layer it was supposed to hide behind,
// and why throttling its grid made the server SLOWER instead of politer
// (KT_PREFETCH_BLOCKS=8: 94 -> 102 ms/step). The uint4 form holds 46 GB/s,
// which is link speed here, from 8 blocks upward.
//
// Rows of an int4-packed expert are always a multiple of 16 B, so the vector
// view is exact; checked byte-identical against the old kernel at 8/64/256
// blocks for n_tp 1 and 2 by `bench/gather_kernel_probe.py --verify`.
__global__ void k_w2_weights(const unsigned long long* __restrict__ ptrs,
                             const long* __restrict__ sel,
                             uint8_t* __restrict__ dst,
                             int n_tp, int n_exp, int hidden,
                             int vec_per_row, int dst_vec_row,
                             long bytes_per_slot) {
  const int slot = blockIdx.z;
  const int tp = blockIdx.y;
  const int e = (int)sel[slot];
  if (e < 0) return;
  const uint4* src = (const uint4*)tbl(ptrs, 4, tp, e, n_tp, n_exp);
  uint4* out = (uint4*)(dst + (long)slot * bytes_per_slot
                            + (long)tp * vec_per_row * 16);
  const int total = hidden * vec_per_row;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += gridDim.x * blockDim.x) {
    const int r = i / vec_per_row;      // one divide per 16 B, not per byte
    const int c = i - r * vec_per_row;
    out[(long)r * dst_vec_row + c] = src[i];
  }
}

// ---- w13 scales: fp32 -> bf16 and interleave in the same pass ---------------
// logical src [2*moe, G], assembled from (gate|up) x tp blocks of tp_moe rows.
// dst[g, n*4 + a] = src[n, g*4 + a]
__global__ void k_w13_scales(const unsigned long long* __restrict__ ptrs,
                             const long* __restrict__ sel,
                             __nv_bfloat16* __restrict__ dst,
                             int n_tp, int n_exp, int moe, int G, int tp_moe,
                             long elems_per_slot) {
  const int slot = blockIdx.z;
  const int blk = blockIdx.y;
  const int tp = blk % n_tp;
  const int is_up = blk / n_tp;
  const int e = (int)sel[slot];
  if (e < 0) return;
  const float* src = (const float*)tbl(ptrs, is_up ? 3 : 1, tp, e, n_tp, n_exp);
  __nv_bfloat16* out = dst + (long)slot * elems_per_slot;
  const long total = (long)tp_moe * G;
  const int N4 = 2 * moe * 4;
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const int r = (int)(i / G);          // row inside this tp block
    const int g = (int)(i - (long)r * G);
    const int n = is_up * moe + tp * tp_moe + r;   // logical row
    out[(long)(g >> 2) * N4 + (long)n * 4 + (g & 3)] = __float2bfloat16(src[i]);
  }
}

// ---- w2 scales: same, but TP partitions concatenate along the GROUP axis ----
// logical src [hidden, G2], src[r, tp*tp_g + c] = down_s[tp][r, c]
__global__ void k_w2_scales(const unsigned long long* __restrict__ ptrs,
                            const long* __restrict__ sel,
                            __nv_bfloat16* __restrict__ dst,
                            int n_tp, int n_exp, int hidden, int G2, int tp_g,
                            long elems_per_slot) {
  const int slot = blockIdx.z;
  const int tp = blockIdx.y;
  const int e = (int)sel[slot];
  if (e < 0) return;
  const float* src = (const float*)tbl(ptrs, 5, tp, e, n_tp, n_exp);
  __nv_bfloat16* out = dst + (long)slot * elems_per_slot;
  const long total = (long)hidden * tp_g;
  const int N4 = hidden * 4;
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const int r = (int)(i / tp_g);
    const int c = (int)(i - (long)r * tp_g);
    const int g = tp * tp_g + c;
    out[(long)(g >> 2) * N4 + (long)r * 4 + (g & 3)] = __float2bfloat16(src[i]);
  }
}

void stream_experts(torch::Tensor ptrs, torch::Tensor sel,
                    torch::Tensor w13, torch::Tensor w13_s,
                    torch::Tensor w2, torch::Tensor w2_s,
                    long n_tp, long n_exp, long moe, long hidden, long group,
                    long blocks) {
  TORCH_CHECK(ptrs.is_cuda() && sel.is_cuda(), "pointer table and sel must be device tensors");
  TORCH_CHECK(w13.is_cuda() && w2.is_cuda(), "destinations must be device tensors");
  const long n_slot = sel.numel();
  auto stream = c10::cuda::getCurrentCUDAStream();

  const long tp_moe = moe / n_tp;
  const long vec_per_tp_block = (tp_moe * hidden / 2) / (long)sizeof(uint4);
  const long vec_per_slot = vec_per_tp_block * 2 * n_tp;
  {
    dim3 grid(blocks, 2 * n_tp, n_slot);
    k_w13_weights<<<grid, 256, 0, stream>>>(
        (const unsigned long long*)ptrs.data_ptr(), sel.data_ptr<long>(),
        (uint4*)w13.data_ptr(), n_tp, n_exp, vec_per_tp_block, vec_per_slot);
  }
  {
    const long src_row_bytes = tp_moe / 2;
    const long dst_row_bytes = moe / 2;
    TORCH_CHECK(src_row_bytes % 16 == 0 && dst_row_bytes % 16 == 0,
                "w2 row pitch must be a multiple of 16 B for the vectorised "
                "gather; got src ", src_row_bytes, " dst ", dst_row_bytes);
    dim3 grid(blocks, n_tp, n_slot);
    k_w2_weights<<<grid, 256, 0, stream>>>(
        (const unsigned long long*)ptrs.data_ptr(), sel.data_ptr<long>(),
        (uint8_t*)w2.data_ptr(), n_tp, n_exp, (int)hidden,
        (int)(src_row_bytes / 16), (int)(dst_row_bytes / 16),
        hidden * dst_row_bytes);
  }
  {
    const long G = hidden / group;
    dim3 grid(blocks, 2 * n_tp, n_slot);
    k_w13_scales<<<grid, 256, 0, stream>>>(
        (const unsigned long long*)ptrs.data_ptr(), sel.data_ptr<long>(),
        (__nv_bfloat16*)w13_s.data_ptr(), n_tp, n_exp, (int)moe, (int)G,
        (int)tp_moe, 2 * moe * G);
  }
  {
    const long G2 = moe / group;
    const long tp_g = G2 / n_tp;
    dim3 grid(blocks, n_tp, n_slot);
    k_w2_scales<<<grid, 256, 0, stream>>>(
        (const unsigned long long*)ptrs.data_ptr(), sel.data_ptr<long>(),
        (__nv_bfloat16*)w2_s.data_ptr(), n_tp, n_exp, (int)hidden, (int)G2,
        (int)tp_g, hidden * G2);
  }
}
"""

CPP_DECL = (
    "#include <torch/extension.h>\n"
    "void stream_experts(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
    " torch::Tensor, torch::Tensor, long, long, long, long, long, long);\n"
)

_MOD = None


def kernels():
    global _MOD
    if _MOD is None:
        _MOD = load_inline(
            name="expert_stream_kernels", cpp_sources=CPP_DECL,
            cuda_sources=CUDA_SRC, functions=["stream_experts"],
            with_cuda=True, verbose=False,
        )
    return _MOD


# ---- host page pinning ------------------------------------------------------
# cudaHostRegister locks kt's store in place, so the gather reads the SAME bytes
# the CPU kernel uses. No duplicate copy, no extra RAM.
_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
_cudart.cudaHostRegister.restype = ctypes.c_int
_cudart.cudaHostGetDevicePointer.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
_cudart.cudaHostGetDevicePointer.restype = ctypes.c_int

CUDA_HOST_REGISTER_MAPPED = 0x02
_ALREADY_REGISTERED = 712
_PAGE = 4096


def host_register(addr: int, nbytes: int) -> int:
    """Pin the pages spanning [addr, addr+nbytes) and return the device address."""
    start = addr & ~(_PAGE - 1)
    end = (addr + nbytes + _PAGE - 1) & ~(_PAGE - 1)
    rc = _cudart.cudaHostRegister(ctypes.c_void_p(start), end - start,
                                  CUDA_HOST_REGISTER_MAPPED)
    if rc not in (0, _ALREADY_REGISTERED):
        raise RuntimeError(f"cudaHostRegister rc={rc} addr=0x{addr:x} bytes={nbytes}")
    dev = ctypes.c_void_p()
    rc2 = _cudart.cudaHostGetDevicePointer(ctypes.byref(dev), ctypes.c_void_p(addr), 0)
    if rc2 != 0:
        raise RuntimeError(f"cudaHostGetDevicePointer rc={rc2}")
    return int(dev.value)


def build_pointer_table_shm(desc: dict, n_exp: int, own_tp: list[int],
                            need_mask=None):
    """Map this layer's store from shared memory and pin it, on ANY rank.

    Route 3. The store lives in a named POSIX segment (KT_STORE_SHM=1), so a
    rank that never built it can still map and gather from it -- which is what
    lets both cards' PCIe links carry the traffic. One link cannot: a full-width
    expert is ~0.46 ms and 2.31 of them across 62 active layers is ~66 ms
    against a ~60 ms step.

    `own_tp` selects the CPU TP partitions this GPU rank is responsible for, so
    each card gathers exactly its own shard of the intermediate dimension.

    Registration is done ONCE over each whole segment rather than per expert:
    Stage A1 measured 1.22 GB/s doing it per region and 10.08 GB/s in bulk, and
    the difference is pure per-call overhead.
    """
    import mmap as _mmap

    names = desc["names"]
    sizes = desc["sizes"]
    offs = desc["offsets"]
    region_bytes = list(desc["bytes"])
    if not names or not names[0]:
        raise RuntimeError(
            "expert store is not in shared memory; boot with KT_STORE_SHM=1 "
            "(without it only rank 0 can gather and one link cannot carry it)")

    table = torch.zeros(6, len(own_tp), n_exp, dtype=torch.int64)
    keep = []
    for slot, tp in enumerate(own_tp):
        # RDWR, not RDONLY: ctypes.from_buffer (how we take the address) refuses
        # a read-only buffer, and cudaHostRegister wants a writable mapping too.
        # The gather only ever reads these pages.
        fd = os.open(f"/dev/shm{names[tp]}", os.O_RDWR)
        try:
            buf = _mmap.mmap(fd, sizes[tp], flags=_mmap.MAP_SHARED,
                             prot=_mmap.PROT_READ | _mmap.PROT_WRITE)
        finally:
            os.close(fd)
        keep.append(buf)
        base = _mmap_address(buf)
        for e in range(n_exp):
            row = offs[tp][e]
            if not row:
                continue
            # Register ONE span per expert, not the whole segment. Registering
            # the segment faults in every page -- including the GPU-tier experts
            # that are never staged -- which materialises the full declared size
            # of the store and blows past /dev/shm (SIGBUS). The six regions are
            # consecutive bump allocations, so one span covers them all.
            if need_mask is not None and not bool(need_mask[e]):
                continue
            # Do NOT assume the six regions are in ascending address order.
            # They are for the shm bump allocator, but the private-heap store
            # uses three independent aligned_alloc calls whose order is up to
            # malloc -- which underflows this span and asks CUDA to register a
            # negative length.
            span_lo = min(row)
            span_hi = max(o + region_bytes[i] for i, o in enumerate(row))
            dev_lo = host_register(base + span_lo, span_hi - span_lo)
            for r in range(6):
                table[r, slot, e] = dev_lo + (row[r] - span_lo)
    return table.cuda(), region_bytes, len(own_tp), keep


def _mmap_address(buf) -> int:
    """Virtual address of an mmap object's first byte."""
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def build_pointer_table(moe_obj, n_exp: int, register: bool = True):
    """Pin the whole CPU store for one layer and return its device pointer table.

    Returns (table[6, n_tp, n_exp] int64 CUDA tensor, region_bytes list, n_tp).
    A GPU-tier expert has no CPU-side allocation and gets a 0 entry; `sel` must
    never name one (the caller intersects with the residency mask first).
    """
    ptrs = moe_obj.expert_store_ptrs()
    sizes = moe_obj.expert_store_bytes()
    n_tp = len(ptrs)
    table = torch.zeros(6, n_tp, n_exp, dtype=torch.int64)
    for tp in range(n_tp):
        for e in range(n_exp):
            row = ptrs[tp][e]
            if not row:
                continue
            for r in range(6):
                table[r, tp, e] = host_register(row[r], sizes[r]) if register else row[r]
    return table.cuda(), list(sizes), n_tp
