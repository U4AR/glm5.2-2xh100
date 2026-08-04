#!/usr/bin/env python3
"""Does the split policy behave sensibly on machines that are not this one?

The whole value of a calibration file is that the same code reaches a different
answer on different hardware. That claim is worth testing rather than asserting,
because the failure mode is silent: a policy that quietly always says "stream 4"
looks fine here and is wrong everywhere else.

Each case below is a hypothetical machine expressed only through measured-style
constants. Run with:

    .venv/bin/python bench/test_hybrid_policy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_policy import Constants, break_even_cpu_ms_per_layer, choose_k, predict_step

MB = 1 / 1024.0  # GB per MB

# This box, from bench/profile_out/hybrid_profile.H100-VM1.json.
H100_SAFE8 = dict(
    cpu_ms_per_layer=1.264, units_per_layer=18.12, distinct_per_layer=12.38,
    bytes_per_expert_gb=9.7 * MB, link_gbs=52.4, contention_ms_per_gb=1.21,
    gpu_ms_per_expert=0.038,
)

CASES = [
    (
        "this box, safe8 (2xH100, PCIe Gen5)",
        Constants(**H100_SAFE8),
        lambda k: k > 0,
        "a fat CPU pole and a usable link: stream several",
    ),
    (
        "this box, safe2",
        Constants(**{**H100_SAFE8, "cpu_ms_per_layer": 0.168,
                     "units_per_layer": 3.60, "distinct_per_layer": 2.50}),
        lambda k: k == 0,
        "CPU pole smaller than one transfer: stream nothing",
    ),
    (
        "GH200-class link (900 GB/s C2C), same model+tier",
        Constants(**{**H100_SAFE8, "link_gbs": 900.0,
                     "contention_ms_per_gb": 0.10}),
        lambda k: k >= 8,
        "transfers ~17x cheaper: stream nearly everything",
    ),
    (
        "GH200-class link at the low tier that this box refuses",
        Constants(**{**H100_SAFE8, "cpu_ms_per_layer": 0.168,
                     "units_per_layer": 3.60, "distinct_per_layer": 2.50,
                     "link_gbs": 900.0, "contention_ms_per_gb": 0.10}),
        lambda k: k > 0,
        "same tier, different machine, opposite answer",
    ),
    (
        "PCIe Gen3 x8 (~6 GB/s), safe8",
        Constants(**{**H100_SAFE8, "link_gbs": 6.0}),
        lambda k: k == 0,
        "link too slow to beat the CPU: stream nothing",
    ),
    (
        "small VRAM box: few resident experts, so a huge CPU pole",
        Constants(**{**H100_SAFE8, "cpu_ms_per_layer": 4.0,
                     "units_per_layer": 28.0, "distinct_per_layer": 20.0}),
        lambda k: k > 4,
        "more CPU work to displace: stream more than this box does",
    ),
    (
        "no MTP (1 token per step, so no reuse)",
        Constants(**{**H100_SAFE8, "units_per_layer": 4.75,
                     "distinct_per_layer": 4.75}),
        lambda k: k >= 0,
        "reuse 1.0: each transfer buys only one token of CPU work",
    ),
    (
        "tiny experts (a smaller model)",
        Constants(**{**H100_SAFE8, "bytes_per_expert_gb": 1.0 * MB}),
        lambda k: k >= 8,
        "cheap transfers: stream nearly everything",
    ),
]


def main() -> int:
    print(f"{'machine / tier':<52}{'k':>4}{'D':>6}{'break-even':>12}"
          f"{'has':>8}{'speedup':>9}  bound")
    failures = []
    for name, c, expect, why in CASES:
        best = choose_k(c)
        # Give each hypothetical machine a step time consistent with its own
        # CPU pole: this box's 53.6 ms non-expert floor plus 75 layers of pole.
        step_ms = 53.6 + c.cpu_ms_per_layer * 75
        p = predict_step(c, 75, step_ms)
        if not p["inputs_consistent"]:
            failures.append(f"{name}: inconsistent step_ms")
        be = break_even_cpu_ms_per_layer(c)
        ok = expect(best.k)
        flag = "" if ok else "   <-- UNEXPECTED"
        print(f"{name:<52}{best.k:>4}{c.distinct_per_layer:>6.1f}{be:>12.3f}"
              f"{c.cpu_ms_per_layer:>8.3f}{p['speedup']:>8.3f}x  {p['bound_by']}{flag}")
        print(f"{'':<52}{why}")
        if not ok:
            failures.append(name)

    # Monotonicity: a faster link must never make the policy stream less.
    ks = []
    for bw in (6, 12, 25, 52, 120, 400, 900):
        ks.append(choose_k(Constants(**{**H100_SAFE8, "link_gbs": float(bw)})).k)
    if ks != sorted(ks):
        failures.append(f"k is not monotonic in link bandwidth: {ks}")
    print(f"\nk vs link GB/s (6,12,25,52,120,400,900): {ks}  "
          f"{'monotonic' if ks == sorted(ks) else 'NOT MONOTONIC'}")

    # A bigger CPU pole must never make the policy stream less either.
    ks2 = []
    for pole in (0.1, 0.3, 0.6, 1.26, 2.5, 5.0):
        ks2.append(choose_k(Constants(**{**H100_SAFE8, "cpu_ms_per_layer": pole})).k)
    if ks2 != sorted(ks2):
        failures.append(f"k is not monotonic in CPU pole: {ks2}")
    print(f"k vs CPU ms/layer (0.1..5.0):            {ks2}  "
          f"{'monotonic' if ks2 == sorted(ks2) else 'NOT MONOTONIC'}")

    # A missing/zero-bandwidth profile must fall back to all-CPU, never to a
    # divide-by-zero or an optimistic guess.
    dead = choose_k(Constants(**{**H100_SAFE8, "link_gbs": 0.0}))
    if dead.k != 0:
        failures.append("zero-bandwidth profile did not fall back to k=0")
    print(f"zero-bandwidth fallback:                 k={dead.k} "
          f"{'(falls back to all-CPU)' if dead.k == 0 else '(WRONG)'}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall policy cases behaved as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
