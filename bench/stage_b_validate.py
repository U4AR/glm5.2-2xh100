"""Stage B gate: the fused gather must reproduce the shipped staging path exactly.

Reference is `write_weight_scale_to_buffer` (kt's own, already-coherent GPU-prefill
staging) followed by sglang's `interleave_scales` -- i.e. precisely the bytes
cutlass_w4a8_moe consumes today. If the fused kernel matches those bit-for-bit,
the streamed path is equivalent to the resident path by construction, and no
separate numerical experiment is needed.

Checks all four destinations, because Stage A1 only checked w13:
  w13 weights, w2 weights, w13 scales (interleaved bf16), w2 scales.

    LD_LIBRARY_PATH=.venv/lib .venv/bin/python bench/stage_b_validate.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("KT_RAWINT4_BACKEND", "avx512_packed")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "int4_scripts"))
from _paths import W4  # noqa: E402
from expert_stream_kernels import build_pointer_table, kernels  # noqa: E402

from kt_kernel.utils.amx import NativeMoEWrapper  # noqa: E402
from sglang.srt.layers.quantization.w4afp8 import interleave_scales  # noqa: E402

L, HID, MOE, NE, K = 3, 6144, 2048, 256, 8
GROUP = 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--cpuinfer", type=int, default=int(os.getenv("CPUINFER", "28")))
    ap.add_argument("--threadpool", type=int, default=2)
    ap.add_argument("--iters", type=int, default=200, help="timing iterations")
    ap.add_argument("--out", default="bench/profile_out/stage_b_fused_gather.json")
    args = ap.parse_args()

    mod = kernels()

    print(f"loading layer {L} ({NE} experts, all CPU-tier)...")
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

    print("pinning the store and building the pointer table...")
    t0 = time.time()
    table, sizes, n_tp = build_pointer_table(moe, NE)
    reg_s = time.time() - t0
    store_gb = sum(sizes) * n_tp * NE / 1e9
    print(f"  pinned {store_gb:.3f} GB (one layer) in {reg_s:.2f}s "
          f"-> {store_gb / max(reg_s, 1e-9):.2f} GB/s")

    checked = list(range(min(args.experts, NE)))
    n_slot = len(checked)

    # ---- reference: shipped staging + sglang's interleave -------------------
    ref_w13 = torch.empty(n_slot, 2 * MOE, HID // 2, dtype=torch.uint8).pin_memory()
    ref_w13_s = torch.empty(n_slot, 2 * MOE, HID // GROUP, dtype=torch.bfloat16).pin_memory()
    ref_w2 = torch.empty(n_slot, HID, MOE // 2, dtype=torch.uint8).pin_memory()
    ref_w2_s = torch.empty(n_slot, HID, MOE // GROUP, dtype=torch.bfloat16).pin_memory()
    print(f"staging {n_slot} experts through the shipped path...")
    for i, e in enumerate(checked):
        w.cpu_infer.submit(moe.write_weight_scale_to_buffer_task(
            gpu_tp_count=1, expert_id=e,
            w13_weight_ptrs=[ref_w13[i].data_ptr()], w13_scale_ptrs=[ref_w13_s[i].data_ptr()],
            w2_weight_ptrs=[ref_w2[i].data_ptr()], w2_scale_ptrs=[ref_w2_s[i].data_ptr()],
        ))
        w.cpu_infer.sync()
    ref_w13_si = interleave_scales(ref_w13_s.cuda())
    ref_w2_si = interleave_scales(ref_w2_s.cuda())

    # ---- the fused gather ---------------------------------------------------
    sel = torch.tensor(checked, dtype=torch.int64, device="cuda")
    got_w13 = torch.zeros(n_slot, 2 * MOE, HID // 2, dtype=torch.uint8, device="cuda")
    got_w2 = torch.zeros(n_slot, HID, MOE // 2, dtype=torch.uint8, device="cuda")
    got_w13_s = torch.zeros_like(ref_w13_si)
    got_w2_s = torch.zeros_like(ref_w2_si)

    mod.stream_experts(table, sel, got_w13, got_w13_s, got_w2, got_w2_s,
                       n_tp, NE, MOE, HID, GROUP, 64)
    torch.cuda.synchronize()

    checks = {
        "w13_weights": torch.equal(got_w13.cpu(), ref_w13),
        "w2_weights": torch.equal(got_w2.cpu(), ref_w2),
        "w13_scales_interleaved": torch.equal(got_w13_s, ref_w13_si),
        "w2_scales_interleaved": torch.equal(got_w2_s, ref_w2_si),
    }
    print("\nfused gather vs shipped staging + interleave_scales:")
    for k, v in checks.items():
        print(f"  {k:24s} {'OK' if v else 'MISMATCH'}")
        if not v:
            a = {"w13_weights": got_w13.cpu(), "w2_weights": got_w2.cpu(),
                 "w13_scales_interleaved": got_w13_s,
                 "w2_scales_interleaved": got_w2_s}[k].float()
            b = {"w13_weights": ref_w13, "w2_weights": ref_w2,
                 "w13_scales_interleaved": ref_w13_si,
                 "w2_scales_interleaved": ref_w2_si}[k].float()
            bad = (a != b)
            print(f"      {int(bad.sum())}/{bad.numel()} elements differ")

    all_ok = all(checks.values())

    # ---- what it costs ------------------------------------------------------
    ms = None
    if all_ok:
        for _ in range(10):
            mod.stream_experts(table, sel, got_w13, got_w13_s, got_w2, got_w2_s,
                               n_tp, NE, MOE, HID, GROUP, 64)
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        for _ in range(args.iters):
            mod.stream_experts(table, sel, got_w13, got_w13_s, got_w2, got_w2_s,
                               n_tp, NE, MOE, HID, GROUP, 64)
        ev1.record()
        torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1) / args.iters
        per_expert = ms / n_slot
        moved_gb = sum(sizes) * n_tp * n_slot / 1e9
        print(f"\n{n_slot} experts in {ms:.3f} ms  ->  {per_expert:.4f} ms/expert, "
              f"{moved_gb / (ms / 1e3):.1f} GB/s")
        print(f"  spike's separate-repack baseline was 0.197 gather + 0.0375 repack "
              f"= 0.2345 ms/expert")

    print(f"\nStage B: {'PASS' if all_ok else 'FAIL'}")
    out = {
        "verdict": "PASS" if all_ok else "FAIL",
        "checks": {k: bool(v) for k, v in checks.items()},
        "n_tp": n_tp, "experts": checked,
        "ms_per_call": ms, "ms_per_expert": (ms / n_slot) if ms else None,
        "layer_store_gb": store_gb, "pin_gbs": store_gb / max(reg_s, 1e-9),
        "note": "interleave folded into the gather; no second host scale copy",
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
