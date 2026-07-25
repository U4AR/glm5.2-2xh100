#!/usr/bin/env python3
"""Build a portable, N-agnostic hot-core expert ranking from real routing
captures, and (optionally) materialize an exact GPU-placement mask for a given
per-layer budget N.

Ranking file (committed to the repo): hot_core_ranking.pt
  torch tensor int16 [78, 256]: for each model layer, the 256 expert IDs sorted
  by genuine-top-2 usage DESCENDING. Layer rows that never route (dense layers)
  are filled with 0..255 in order and ignored at load (runtime forces them GPU).
  System-independent: a new box slices row[:, :N] for its own GPU_EXPERTS=N.

Prior file (committed to the repo): hot_core_prior.pt
  A versioned dictionary containing normalized genuine-top-2 usage scores
  [78, 256]. Runtime seeds its live adaptive counters from these scores, so a
  warm placement is not immediately discarded by counters starting at zero.

Usage:
  build            # (re)build ranking from ../..'s routing captures
  adopt COUNTS.pt  # replace artifacts from a complete runtime counter dump
  mask N OUT.pt    # write a [78,256] bool oracle mask with N True per MoE layer
"""
import sys, glob, os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RANK_PT = os.path.join(HERE, "hot_core_ranking.pt")
PRIOR_PT = os.path.join(HERE, "hot_core_prior.pt")
CAP = os.path.join(HERE, "..", "..", "expert_footprint_top2_vs_top8", "runs")
NL, NE = 78, 256
FIRST_DENSE, MOE_FREQ = 3, 1  # GLM-5.2: layers 0..2 dense, 3..80 MoE


def write_artifacts(counts, source):
    routed = torch.zeros(NL, dtype=torch.bool)
    routed[counts.sum(dim=1) > 0] = True
    expected = torch.zeros(NL, dtype=torch.bool)
    expected[FIRST_DENSE:] = True
    if not torch.equal(routed, expected):
        missing = torch.where(expected & ~routed)[0].tolist()
        raise ValueError(f"incomplete routed-layer counts; missing layers {missing}")
    ranking = torch.zeros(NL, NE, dtype=torch.int16)
    for l in range(NL):
        if routed[l]:
            ranking[l] = torch.argsort(counts[l], descending=True).to(torch.int16)
        else:
            ranking[l] = torch.arange(NE, dtype=torch.int16)
    torch.save(ranking, RANK_PT)
    ev = int(counts.sum())
    row_totals = counts.sum(dim=1, keepdim=True)
    scores = torch.where(row_totals > 0, counts / row_totals.clamp_min(1.0), counts)
    torch.save(
        {
            "version": 1,
            "scores": scores.to(torch.float32),
            "events_per_layer": row_totals.squeeze(1).to(torch.float32),
            "total_events": ev,
            "source": source,
        },
        PRIOR_PT,
    )
    print(f"wrote {RANK_PT}  routed_layers={int(routed.sum())}  top2_events={ev}")
    print(f"wrote {PRIOR_PT}  normalized_scores={tuple(scores.shape)}")
    # report coverage of the top-N slice for a few N
    for N in (24, 48, 64, 96, 128):
        cov = 0.0; nl = 0
        for l in range(NL):
            if routed[l]:
                tot = counts[l].sum()
                cov += counts[l][ranking[l][:N].long()].sum().item() / tot.item(); nl += 1
        print(f"  N={N:3d}  mean top-2 coverage of hottest-N = {cov/nl:.3f}")


def build():
    counts = torch.zeros(NL, NE, dtype=torch.float64)
    for f in sorted(glob.glob(f"{CAP}/*.pt")):
        for layer, ids, w in torch.load(f, map_location="cpu", weights_only=False):
            top2 = w.argsort(dim=-1, descending=True)[:, :2]
            sel = torch.gather(ids.long(), 1, top2).reshape(-1)
            counts[layer].index_add_(0, sel, torch.ones(sel.numel(), dtype=torch.float64))
    write_artifacts(counts, "diverse-routing-captures")


def adopt(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    snapshots = payload.get("counts") if isinstance(payload, dict) else payload
    counts = torch.zeros(NL, NE, dtype=torch.float64)
    if isinstance(snapshots, dict):
        for layer, values in snapshots.items():
            counts[int(layer)] = values.to(torch.float64)
    elif isinstance(snapshots, torch.Tensor) and tuple(snapshots.shape) == (NL, NE):
        counts.copy_(snapshots.to(torch.float64))
    else:
        raise ValueError("expected a layer->counts dictionary or [78,256] tensor")
    write_artifacts(counts, f"runtime-counter-dump:{os.path.basename(path)}")


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
    elif len(sys.argv) >= 2 and sys.argv[1] == "adopt":
        adopt(sys.argv[2])
    else:
        build()
