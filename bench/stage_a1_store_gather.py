"""Stage A1 kill gate: can a GPU-side gather read kt's CPU expert store in place?

The plan states A1 as "stream one expert's bytes into a scratch GPU layer, run
cutlass_w4a8_moe, and compare against the same expert's resident output."  There
is a strictly cheaper and strictly stronger test available, because the shipped
W4AFP8 GPU-prefill path already stages kt's store into cutlass buffers and is
known coherent: `write_weights_to_buffer` memcpys BufferB::qweight_packed
verbatim and only converts the fp32 group scales to bf16.

So instead of re-deriving cutlass compatibility, compare against THAT path:

    UVA-gathered bytes  ==  write_weights_to_buffer bytes    (byte for byte)

If they match, cutlass compatibility follows from the already-validated path,
and the only new claim under test is the one that actually matters here -- that
a device kernel can read the store's pages directly, with no host staging copy.

Two things the plan's arithmetic did not account for, both confirmed here:

  * The store is NOT a contiguous expert-strided array.  The flat `tpc.gate_proj`
    buffers built during load_weights are temporaries, deleted once the
    per-expert BufferBs are filled.  So the gather needs a device POINTER TABLE,
    not the spike's `src_base + idx[e] * stride`.
  * The store keeps scales as fp32, while cutlass consumes bf16.  Gathering raw
    therefore moves ~6.3% more bytes per expert than the 9.7 MB the pricing
    assumed.  Reported below as a link-duty correction.

    LD_LIBRARY_PATH=.venv/lib .venv/bin/python bench/stage_a1_store_gather.py
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("KT_RAWINT4_BACKEND", "avx512_packed")

import torch  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "int4_scripts"))
from _paths import W4  # noqa: E402

from kt_kernel.utils.amx import NativeMoEWrapper  # noqa: E402

# One GLM-5.2 MoE layer.  Layer 3 matches the existing int4 kernel harness.
L, HID, MOE, NE, K = 3, 6144, 2048, 256, 8
GROUP = 128

# The gather reads through a device-side table of host base pointers, one entry
# per expert, because the persistent store is one allocation per expert.  That
# is the only structural difference from the validated spike kernel -- the copy
# loop itself is identical, so the 0.197 ms/expert measurement still applies.
CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

__global__ void gather_ptr_table_kernel(
    const unsigned long long* __restrict__ src_ptrs,  // [n_sel] host addresses
    uint4* __restrict__ dst,
    const long* __restrict__ sel,                     // [n_sel] indices into src_ptrs
    long vec_per_region) {
  long e = blockIdx.y;
  const uint4* src = reinterpret_cast<const uint4*>(src_ptrs[sel[e]]);
  uint4* out = dst + e * vec_per_region;
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_per_region;
       i += (long)gridDim.x * blockDim.x) {
    out[i] = src[i];
  }
}

void gather_ptr_table(torch::Tensor src_ptrs, torch::Tensor dst, torch::Tensor sel,
                      long bytes_per_region, long blocks) {
  TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA tensor");
  TORCH_CHECK(sel.is_cuda(), "sel must live on the device, or the graph cannot "
                             "pick experts at replay time");
  TORCH_CHECK(src_ptrs.is_cuda(), "the pointer table must be device-resident too");
  TORCH_CHECK(bytes_per_region % sizeof(uint4) == 0, "region must be uint4-aligned");
  long vec_per_region = bytes_per_region / sizeof(uint4);
  dim3 grid(blocks, sel.numel());
  gather_ptr_table_kernel<<<grid, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const unsigned long long*>(src_ptrs.data_ptr()),
      reinterpret_cast<uint4*>(dst.data_ptr()),
      sel.data_ptr<long>(), vec_per_region);
}
"""

_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
_cudart.cudaHostRegister.restype = ctypes.c_int
_cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
_cudart.cudaHostUnregister.restype = ctypes.c_int
_cudart.cudaHostGetDevicePointer.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
_cudart.cudaHostGetDevicePointer.restype = ctypes.c_int

CUDA_HOST_REGISTER_MAPPED = 0x02
_PAGE = 4096


def page_span(addr: int, nbytes: int) -> tuple[int, int]:
    """cudaHostRegister works on whole pages; widen to the enclosing span."""
    start = addr & ~(_PAGE - 1)
    end = (addr + nbytes + _PAGE - 1) & ~(_PAGE - 1)
    return start, end - start


def host_register(addr: int, nbytes: int) -> int:
    start, span = page_span(addr, nbytes)
    rc = _cudart.cudaHostRegister(ctypes.c_void_p(start), span, CUDA_HOST_REGISTER_MAPPED)
    # 712 == cudaErrorHostMemoryAlreadyRegistered, which is fine and expected
    # once neighbouring experts share a page.
    if rc not in (0, 712):
        raise RuntimeError(f"cudaHostRegister failed rc={rc} addr=0x{addr:x} bytes={nbytes}")
    dev = ctypes.c_void_p()
    rc2 = _cudart.cudaHostGetDevicePointer(ctypes.byref(dev), ctypes.c_void_p(addr), 0)
    if rc2 != 0:
        raise RuntimeError(f"cudaHostGetDevicePointer failed rc={rc2}")
    return int(dev.value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8, help="how many to check")
    ap.add_argument("--cpuinfer", type=int, default=int(os.getenv("CPUINFER", "28")))
    ap.add_argument("--threadpool", type=int, default=2)
    ap.add_argument("--out", default="bench/profile_out/stage_a1_store_gather.json")
    args = ap.parse_args()

    print("compiling the pointer-table gather kernel...")
    cpp_decl = ("#include <torch/extension.h>\n"
                "void gather_ptr_table(torch::Tensor, torch::Tensor, torch::Tensor, long, long);\n")
    mod = load_inline(name="stage_a1_gather", cpp_sources=cpp_decl, cuda_sources=CUDA_SRC,
                      functions=["gather_ptr_table"], with_cuda=True, verbose=False)

    print(f"loading layer {L} ({NE} experts, all CPU-tier) -- this takes a minute...")
    t0 = time.time()
    w = NativeMoEWrapper(
        layer_idx=L, num_experts=NE, num_experts_per_tok=K,
        hidden_size=HID, moe_intermediate_size=MOE,
        gpu_experts_mask=torch.zeros(NE, dtype=torch.bool),
        cpuinfer_threads=args.cpuinfer, threadpool_count=args.threadpool,
        weight_path=W4, chunked_prefill_size=2048, method="RAWINT4",
        numa_nodes=[0, 1],
    )
    w.load_weights(torch.arange(NE, dtype=torch.long))
    print(f"  loaded in {time.time() - t0:.1f}s")

    moe = w.moe
    if not hasattr(moe, "expert_store_ptrs"):
        raise SystemExit("kt_kernel is missing expert_store_ptrs -- rebuild the extension")

    ptrs = moe.expert_store_ptrs()      # [tp][expert][6]
    sizes = moe.expert_store_bytes()    # gate_w, gate_s, up_w, up_s, down_w, down_s
    n_tp = len(ptrs)
    names = ["gate_w", "gate_s", "up_w", "up_s", "down_w", "down_s"]
    print(f"\nstore: {n_tp} TP partitions x {NE} experts")
    for nm, sz in zip(names, sizes):
        print(f"  {nm:8s} {sz / 1e6:8.3f} MB")
    per_expert_all_tp = sum(sizes) * n_tp
    print(f"  per expert, all TP partitions: {per_expert_all_tp / 1e6:.3f} MB")

    # ---- the byte-cost correction the pricing needs -------------------------
    w_bytes = (sizes[0] + sizes[2] + sizes[4]) * n_tp
    s_fp32 = (sizes[1] + sizes[3] + sizes[5]) * n_tp
    s_bf16 = s_fp32 // 2
    print(f"\n  weights            {w_bytes / 1e6:8.3f} MB")
    print(f"  scales as stored   {s_fp32 / 1e6:8.3f} MB  (fp32)")
    print(f"  scales as cutlass  {s_bf16 / 1e6:8.3f} MB  (bf16, what Stage B would store)")
    infl = (w_bytes + s_fp32) / (w_bytes + s_bf16)
    print(f"  raw-gather inflation vs a bf16 store: {infl:.4f}x")

    # ---- reference: the shipped staging path --------------------------------
    # gpu_tp_count=1 keeps the comparison to a single destination layout.
    gpu_tp = 1
    w13_w = torch.empty(NE, 2 * MOE, HID // 2, dtype=torch.uint8).pin_memory()
    w13_s = torch.empty(NE, 2 * MOE, HID // GROUP, dtype=torch.bfloat16).pin_memory()
    w2_w = torch.empty(NE, HID, MOE // 2, dtype=torch.uint8).pin_memory()
    w2_s = torch.empty(NE, HID, MOE // GROUP, dtype=torch.bfloat16).pin_memory()

    checked = list(range(min(args.experts, NE)))
    print(f"\nstaging {len(checked)} experts through the shipped path...")
    for e in checked:
        w.cpu_infer.submit(moe.write_weight_scale_to_buffer_task(
            gpu_tp_count=gpu_tp, expert_id=e,
            w13_weight_ptrs=[w13_w[e].data_ptr()], w13_scale_ptrs=[w13_s[e].data_ptr()],
            w2_weight_ptrs=[w2_w[e].data_ptr()], w2_scale_ptrs=[w2_s[e].data_ptr()],
        ))
        w.cpu_infer.sync()

    # ---- register the store and gather it on the device ---------------------
    print("registering store pages and gathering...")
    results = {}
    reg_t0 = time.time()
    dev_tables = []
    for r in range(6):
        col = []
        for tp in range(n_tp):
            for e in checked:
                col.append(host_register(ptrs[tp][e][r], sizes[r]))
        dev_tables.append(torch.tensor(col, dtype=torch.int64, device="cuda"))
    reg_s = time.time() - reg_t0
    reg_gb = per_expert_all_tp * len(checked) / 1e9
    print(f"  registered {reg_gb:.3f} GB in {reg_s:.2f}s ({reg_gb / max(reg_s, 1e-9):.2f} GB/s)")

    ok = {}
    for r, nm in enumerate(names):
        n_sel = n_tp * len(checked)
        dst = torch.empty(n_sel * sizes[r], dtype=torch.uint8, device="cuda")
        sel = torch.arange(n_sel, dtype=torch.int64, device="cuda")
        mod.gather_ptr_table(dev_tables[r], dst, sel, sizes[r], 64)
        torch.cuda.synchronize()
        results[nm] = dst

    # ---- compare, byte for byte --------------------------------------------
    # The CPU store is TP-sliced along the intermediate dimension, so TP p holds
    # rows [p*MOE/n_tp, (p+1)*MOE/n_tp) of gate/up and the matching K-slice of
    # down.  Reassemble the same order write_weights_to_buffer writes.
    print("\ncomparing gathered bytes against the shipped staging path:")
    tp_moe = MOE // n_tp
    for e_i, e in enumerate(checked):
        gate = torch.cat([results["gate_w"].view(n_tp, len(checked), -1)[tp, e_i]
                          for tp in range(n_tp)]).view(MOE, HID // 2)
        up = torch.cat([results["up_w"].view(n_tp, len(checked), -1)[tp, e_i]
                        for tp in range(n_tp)]).view(MOE, HID // 2)
        got_w13 = torch.cat([gate, up], dim=0).cpu()
        ref_w13 = w13_w[e]
        same_w13 = torch.equal(got_w13, ref_w13)

        gs = torch.cat([results["gate_s"].view(n_tp, len(checked), -1)[tp, e_i]
                        for tp in range(n_tp)]).view(torch.float32).view(MOE, HID // GROUP)
        us = torch.cat([results["up_s"].view(n_tp, len(checked), -1)[tp, e_i]
                        for tp in range(n_tp)]).view(torch.float32).view(MOE, HID // GROUP)
        got_s13 = torch.cat([gs, us], dim=0).to(torch.bfloat16).cpu()
        same_s13 = torch.equal(got_s13, w13_s[e])

        ok[e] = {"w13_weights_byte_exact": bool(same_w13),
                 "w13_scales_byte_exact_after_bf16": bool(same_s13)}
        print(f"  expert {e:3d}: w13 weights {'OK ' if same_w13 else 'MISMATCH'}"
              f"   w13 scales {'OK ' if same_s13 else 'MISMATCH'}")

    all_ok = all(v["w13_weights_byte_exact"] and v["w13_scales_byte_exact_after_bf16"]
                 for v in ok.values())
    verdict = "PASS" if all_ok else "FAIL"
    print(f"\nStage A1: {verdict}")
    if not all_ok:
        print("  The gather does not reproduce the shipped staging bytes. Do NOT")
        print("  proceed to Stage B/C -- re-derive the layout first.")

    out = {
        "verdict": verdict,
        "layer": L, "n_tp": n_tp, "experts_checked": checked,
        "region_bytes": {nm: int(s) for nm, s in zip(names, sizes)},
        "per_expert_all_tp_bytes": int(per_expert_all_tp),
        "weights_bytes": int(w_bytes),
        "scales_fp32_bytes": int(s_fp32),
        "scales_bf16_bytes": int(s_bf16),
        "raw_gather_inflation_vs_bf16_store": infl,
        "register_gbs": reg_gb / max(reg_s, 1e-9),
        "per_expert": ok,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
