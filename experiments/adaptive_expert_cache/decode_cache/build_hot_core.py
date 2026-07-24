#!/usr/bin/env python3
"""Build a portable, N-agnostic hot-core expert ranking from real routing
captures, and (optionally) materialize an exact GPU-placement mask for a given
per-layer budget N.

Ranking file (committed to the repo): hot_core_ranking.pt
  torch tensor int16 [78, 256]: for each model layer, the 256 expert IDs sorted
  by genuine-top-2 usage DESCENDING. Layer rows that never route (dense layers)
  are filled with 0..255 in order and ignored at load (runtime forces them GPU).
  System-independent: a new box slices row[:, :N] for its own GPU_EXPERTS=N.

Usage:
  build            # (re)build ranking from ../..'s routing captures
  mask N OUT.pt    # write a [78,256] bool oracle mask with N True per MoE layer
"""
import sys, glob, os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RANK_PT = os.path.join(HERE, "hot_core_ranking.pt")
CAP = os.path.join(HERE, "..", "..", "expert_footprint_top2_vs_top8", "runs")
NL, NE = 78, 256
FIRST_DENSE, MOE_FREQ = 3, 1  # GLM-5.2: layers 0..2 dense, 3..80 MoE


def build():
    counts = torch.zeros(NL, NE, dtype=torch.float64)
    routed = torch.zeros(NL, dtype=torch.bool)
    for f in sorted(glob.glob(f"{CAP}/*.pt")):
        for layer, ids, w in torch.load(f, map_location="cpu", weights_only=False):
            top2 = w.argsort(dim=-1, descending=True)[:, :2]
            sel = torch.gather(ids.long(), 1, top2).reshape(-1)
            counts[layer].index_add_(0, sel, torch.ones(sel.numel(), dtype=torch.float64))
            routed[layer] = True
    ranking = torch.zeros(NL, NE, dtype=torch.int16)
    for l in range(NL):
        if routed[l]:
            ranking[l] = torch.argsort(counts[l], descending=True).to(torch.int16)
        else:
            ranking[l] = torch.arange(NE, dtype=torch.int16)
    torch.save(ranking, RANK_PT)
    ev = int(counts.sum())
    print(f"wrote {RANK_PT}  routed_layers={int(routed.sum())}  top2_events={ev}")
    # report coverage of the top-N slice for a few N
    for N in (24, 48, 64, 96, 128):
        cov = 0.0; nl = 0
        for l in range(NL):
            if routed[l]:
                tot = counts[l].sum()
                cov += counts[l][ranking[l][:N].long()].sum().item() / tot.item(); nl += 1
        print(f"  N={N:3d}  mean top-2 coverage of hottest-N = {cov/nl:.3f}")


def mask(N, out):
    ranking = torch.load(RANK_PT, map_location="cpu", weights_only=True).long()
    m = torch.zeros(NL, NE, dtype=torch.bool)
    for l in range(NL):
        is_moe = (l >= FIRST_DENSE) and (l % MOE_FREQ == 0)
        if is_moe:
            m[l, ranking[l][:N]] = True
        # dense layers left False; runtime forces them True
    # sanity: exactly N per MoE layer
    assert int(m[FIRST_DENSE].sum()) == N, int(m[FIRST_DENSE].sum())
    torch.save(m, out)
    print(f"wrote {out}  N={N}/layer  moe_layers={int((m.sum(1)==N).sum())}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "mask":
        mask(int(sys.argv[2]), sys.argv[3])
    else:
        build()
