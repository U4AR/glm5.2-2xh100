#!/usr/bin/env python3
"""Phase 2 go/no-go: how many DISTINCT experts does a layer need per step?

The streaming split lives or dies on one ratio. The CPU pays per expert *per
token*: eight routed experts times four verify-batch tokens is 32 pieces of
work. PCIe pays per *distinct* expert per step: if those 32 collapse onto 12
different experts, one transfer serves 2.7 tokens and the transfer bill is 2.7x
cheaper than the naive count. If every token wants different experts, streaming
pays full freight and the payoff collapses to the bottom of Phase 1's band.

D is a property of the router, not of this machine, so measuring it once here
tells every other machine what to expect -- which is why the calibration file
stores it alongside the hardware constants.

Input is a `KT_DUMP_TOPK=1` routing dump: records of (layer_idx, ids, weights)
plus a sibling `<tag>.masks.pt` giving the GPU-resident mask per layer. The dump
is taken during PREFILL, which is not CUDA-graph captured, so the tokens are
consecutive real tokens rather than draft tokens -- the closest available proxy
for a verify batch, and stated as such rather than assumed away.

    .venv/bin/python bench/measure_distinct_experts.py \
        --dump bench/profile_out/routing/phase2D.pt --window 4
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def analyse(dump_path: Path, window: int, tiers: list[int]) -> dict:
    records = torch.load(dump_path, map_location="cpu", weights_only=False)
    masks_path = dump_path.with_suffix(".masks.pt")
    if not masks_path.is_file():
        raise SystemExit(
            f"missing {masks_path} -- the dump was taken without the residency "
            "mask, so it cannot say which experts would hit the CPU"
        )
    masks = torch.load(masks_path, map_location="cpu", weights_only=False)

    # Group by layer; a chunked prefill emits several records per layer.
    by_layer: dict[int, list[torch.Tensor]] = {}
    for rec in records:
        layer, ids = rec[0], rec[1]
        if ids.dim() != 2 or ids.shape[0] < window:
            continue
        by_layer.setdefault(int(layer), []).append(ids.long())

    out: dict[int, dict] = {}
    for tier in tiers:
        per_layer_D, per_layer_U, per_layer_reuse = [], [], []
        for layer, chunks in sorted(by_layer.items()):
            mask = masks.get(layer)
            if mask is None:
                continue
            resident = mask.bool()
            Ds, Us = [], []
            for ids in chunks:
                kept = ids[:, :tier]                     # genuine top-`tier`
                if kept.numel() == 0:
                    continue
                # A routed expert costs CPU time exactly when it is not resident.
                cpu_hit = ~resident[kept]                # (T, tier) bool
                T = kept.shape[0]
                for start in range(0, T - window + 1, window):
                    w_ids = kept[start:start + window]
                    w_hit = cpu_hit[start:start + window]
                    sel = w_ids[w_hit]
                    Us.append(int(w_hit.sum()))          # expert-token units
                    Ds.append(int(torch.unique(sel).numel()))  # distinct experts
            if not Ds:
                continue
            mD, mU = statistics.mean(Ds), statistics.mean(Us)
            per_layer_D.append(mD)
            per_layer_U.append(mU)
            per_layer_reuse.append(mU / mD if mD else 0.0)

        if not per_layer_D:
            continue
        D = statistics.mean(per_layer_D)
        U = statistics.mean(per_layer_U)
        out[tier] = {
            "D_mean": D,
            "U_mean": U,
            "reuse": U / D if D else 0.0,
            "D_min_layer": min(per_layer_D),
            "D_max_layer": max(per_layer_D),
            "reuse_min_layer": min(per_layer_reuse),
            "reuse_max_layer": max(per_layer_reuse),
            "layers": len(per_layer_D),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="bench/profile_out/routing/phase2D.pt")
    ap.add_argument("--window", type=int, default=4,
                    help="tokens per step (MTP draft tokens)")
    ap.add_argument("--tiers", default="2,4,6,8")
    ap.add_argument("--out", default="bench/profile_out/phase2_distinct_experts.json")
    args = ap.parse_args()

    tiers = [int(t) for t in args.tiers.split(",") if t.strip()]
    res = analyse(Path(args.dump), args.window, tiers)
    if not res:
        raise SystemExit("no usable records in the dump")

    print(f"window = {args.window} tokens/step\n")
    print(f"{'tier':>5} {'U (expert-tokens)':>18} {'D (distinct)':>13} "
          f"{'reuse U/D':>10} {'D range over layers':>21}")
    for tier, r in sorted(res.items()):
        print(f"{tier:>5} {r['U_mean']:>18.2f} {r['D_mean']:>13.2f} "
              f"{r['reuse']:>10.2f} {r['D_min_layer']:>9.1f} - {r['D_max_layer']:<9.1f}")

    print("\nreuse is the multiplier streaming gets for free: one transfer, "
          "`reuse` tokens of CPU work removed.")
    Path(args.out).write_text(json.dumps(
        {"window": args.window, "dump": args.dump,
         "per_tier": {str(k): v for k, v in res.items()}}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
