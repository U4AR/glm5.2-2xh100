#!/usr/bin/env python3
"""Build oracle GPU-placement masks from the captured true top-8 traces.

For each routed layer, pick the K=96 most-frequently-selected experts (over all 8
top-k slots) as the GPU-resident set. Output boolean masks [78, 256] with exactly
96 True per routed layer (dense layers left False; the runtime forces them True).

Writes:
  runs/oracle_masks/global.pt          (hottest across all 5 tasks)
  runs/oracle_masks/<task>.pt          (per-question)
  runs/oracle_masks/coverage.json      (top-8 GPU-hit coverage summary)
"""
from __future__ import annotations
import json, torch
from pathlib import Path
from capture import DEFAULT_TASKS

NL, NE, K = 78, 256, 96
HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "oracle_masks"


def load_counts(tasks):
    c = torch.zeros(NL, NE, dtype=torch.float64)
    for t in tasks:
        for layer_idx, ids, _w in torch.load(HERE / "runs" / f"exp3_{t}.pt", weights_only=False):
            f = ids.reshape(-1).long()
            c[layer_idx].index_add_(0, f, torch.ones_like(f, dtype=torch.float64))
    return c


def mask_from_counts(counts):
    m = torch.zeros(NL, NE, dtype=torch.bool)
    for l in range(NL):
        if counts[l].sum() > 0:
            m[l, torch.topk(counts[l], K).indices] = True
    return m


def coverage(mask, tasks):
    tot = hit = 0
    for t in tasks:
        for layer_idx, ids, _w in torch.load(HERE / "runs" / f"exp3_{t}.pt", weights_only=False):
            mm = mask[layer_idx].bool(); sel = ids.reshape(-1).long()
            hit += int(mm[sel].sum()); tot += sel.numel()
    return round(hit / tot * 100, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"uniform_baseline_pct": round(K / NE * 100, 1), "K_per_layer": K}

    gmask = mask_from_counts(load_counts(DEFAULT_TASKS))
    assert int(gmask.sum(1)[3]) == K, gmask.sum(1)[3]
    torch.save(gmask, OUT / "global.pt")
    summary["global"] = {
        "on_all_tasks_pct": coverage(gmask, DEFAULT_TASKS),
        "per_task_pct": {t: coverage(gmask, [t]) for t in DEFAULT_TASKS},
    }

    summary["per_question"] = {}
    for t in DEFAULT_TASKS:
        m = mask_from_counts(load_counts([t]))
        torch.save(m, OUT / f"{t}.pt")
        summary["per_question"][t] = coverage(m, [t])

    (OUT / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
