#!/usr/bin/env python3
"""Focused checks for persisted adaptive-cache warm-start scores."""

import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PRIOR = HERE / "hot_core_prior.pt"
RANKING = HERE / "hot_core_ranking.pt"
os.environ.setdefault("KT_ADAPTIVE_PRIOR_PT", str(PRIOR))

from sglang.srt.layers.moe import kt_ep_wrapper as adaptive  # noqa: E402


def main() -> None:
    payload = torch.load(PRIOR, map_location="cpu", weights_only=True)
    assert payload["version"] == 1
    assert payload["total_events"] == 441000
    scores = payload["scores"]
    events_per_layer = payload["events_per_layer"]
    ranking = torch.load(RANKING, map_location="cpu", weights_only=True).long()
    assert tuple(scores.shape) == (78, 256)
    assert tuple(events_per_layer.shape) == (78,)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()
    routed = scores.sum(1) > 0
    assert int(routed.sum()) == 75
    assert torch.allclose(scores[routed].sum(1), torch.ones(75), atol=1e-5)
    assert int(events_per_layer.sum().item()) == payload["total_events"]

    loaded = adaptive._kt_load_adaptive_prior()
    assert loaded is not None
    for n in (1, 8, 16, 24, 48, 64, 96, 104, 128, 256):
        for layer in range(3, 78):
            resident = torch.zeros(256, dtype=torch.bool)
            resident[ranking[layer, :n]] = True
            selected = adaptive._kt_adaptive_select(
                loaded[layer].to(torch.float64) * 64.0, resident, n
            )
            selected_mask = torch.zeros_like(resident)
            selected_mask[selected] = True
            # The saved placement is exactly optimal under its matching prior
            # and therefore must not churn before any live evidence arrives.
            assert torch.equal(selected_mask, resident), (n, layer)
    print(
        "hot-core prior: normalized, 75 routed layers, "
        "zero initial swaps across N=1..256"
    )


if __name__ == "__main__":
    main()
