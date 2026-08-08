"""Root-cause probe: does the gather write outside its landing slots?

Stage B validated the kernel at the OFFLINE geometry (both CPU TP partitions
into one full-width destination, n_tp=2 / moe=2048). The server runs a different
one: each rank gathers ONE CPU partition into a per-partition destination,
n_tp=1 / moe=1024. The bisect showed the gather corrupts state even when nothing
is routed to a landing slot, so the suspicion is an out-of-bounds write at the
live geometry.

This reproduces the live geometry exactly and surrounds every destination with a
sentinel guard region. It answers three questions separately:

  1. do the gathered bytes match the reference (correctness)?
  2. does anything outside slots [0, SLOTS) change (bounds)?
  3. does the region BEFORE the slots -- i.e. the resident experts -- change?

    LD_LIBRARY_PATH=.venv/lib .venv/bin/python bench/stage_c_bounds.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("KT_RAWINT4_BACKEND", "avx512_packed")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "int4_scripts"))
from _paths import W4  # noqa: E402
from expert_stream_kernels import build_pointer_table, kernels  # noqa: E402

from kt_kernel.utils.amx import NativeMoEWrapper  # noqa: E402

L, HID, MOE_FULL, NE, K = 3, 6144, 2048, 256, 8
GROUP = 128
N_CPU_TP = 2          # kt threadpool_count, as deployed
GPU_TP = 2            # sglang --tp-size
MOE = MOE_FULL // GPU_TP   # intermediate_size_per_partition == 1024
RESIDENT = 100        # GPU_EXPERTS per layer, as in the failing boot
SLOTS = 4
SENTINEL = 0xA5


def main() -> int:
    mod = kernels()
    print(f"live geometry: n_tp=1, moe={MOE}, hidden={HID}, "
          f"resident={RESIDENT}, slots={SLOTS}")

    w = NativeMoEWrapper(
        layer_idx=L, num_experts=NE, num_experts_per_tok=K,
        hidden_size=HID, moe_intermediate_size=MOE_FULL,
        gpu_experts_mask=torch.zeros(NE, dtype=torch.bool),
        cpuinfer_threads=int(os.getenv("CPUINFER", "28")),
        threadpool_count=N_CPU_TP, weight_path=W4,
        chunked_prefill_size=2048, method="RAWINT4", numa_nodes=[0, 1],
    )
    w.load_weights(torch.arange(NE, dtype=torch.long))
    moe = w.moe

    table_all, sizes, n_tp_all = build_pointer_table(moe, NE)
    # Emulate rank 0: it owns CPU partition 0 only.
    table = table_all[:, :1, :].contiguous()
    print(f"region bytes: {[int(s) for s in sizes]}")

    total = RESIDENT + SLOTS
    # Full-size destinations, exactly as the server allocates them.
    w13 = torch.full((total, 2 * MOE, HID // 2), SENTINEL, dtype=torch.uint8, device="cuda")
    w2 = torch.full((total, HID, MOE // 2), SENTINEL, dtype=torch.uint8, device="cuda")
    w13_s = torch.full((total, (HID // GROUP) // 4, 2 * MOE * 4), float(SENTINEL),
                       dtype=torch.bfloat16, device="cuda")
    w2_s = torch.full((total, (MOE // GROUP) // 4, HID * 4), float(SENTINEL),
                      dtype=torch.bfloat16, device="cuda")

    before = {"w13": w13.clone(), "w2": w2.clone(),
              "w13_s": w13_s.clone(), "w2_s": w2_s.clone()}

    sel = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda")
    b = RESIDENT
    mod.stream_experts(table, sel, w13[b:], w13_s[b:], w2[b:], w2_s[b:],
                       1, NE, MOE, HID, GROUP, 64)
    torch.cuda.synchronize()

    print("\nregion                      changed?   expected")
    bad = False
    for name, cur, prev in (("w13", w13, before["w13"]), ("w2", w2, before["w2"]),
                            ("w13_s", w13_s, before["w13_s"]),
                            ("w2_s", w2_s, before["w2_s"])):
        resident_changed = not torch.equal(cur[:b], prev[:b])
        slots_changed = not torch.equal(cur[b:], prev[b:])
        print(f"  {name:6s} resident[0:{b}]        "
              f"{'CHANGED' if resident_changed else 'intact ':10s} intact")
        print(f"  {name:6s} slots[{b}:{total}]        "
              f"{'written' if slots_changed else 'UNTOUCHED':10s} written")
        if resident_changed:
            n = int((cur[:b] != prev[:b]).sum())
            first = int((cur[:b] != prev[:b]).nonzero()[0][0])
            print(f"      -> {n} bytes differ, first at resident expert {first}")
            bad = True
        if not slots_changed:
            bad = True

    # How far past the slot base did it write? Compare against the flat extent
    # the kernel should own.
    flat = w13.view(-1)
    fb = before["w13"].view(-1)
    diff = (flat != fb).nonzero().flatten()
    if diff.numel():
        lo, hi = int(diff[0]), int(diff[-1])
        per = 2 * MOE * (HID // 2)
        print(f"\n  w13 written byte range [{lo}, {hi}]  "
              f"= experts [{lo // per}, {hi // per}]")
        print(f"  slot region is experts [{b}, {total}) "
              f"= bytes [{b * per}, {total * per})")
        if lo < b * per:
            print(f"  *** OUT OF BOUNDS: wrote {b * per - lo} bytes BEFORE the "
                  f"slot base, into resident experts ***")

    print(f"\nverdict: {'FAIL' if bad else 'PASS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
