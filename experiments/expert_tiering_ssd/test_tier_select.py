#!/usr/bin/env python3
"""Offline checks for the dynamic tier-movement policy (_kt_tier_select).

The policy is the part that can silently corrupt the model, so the invariants
it must hold are checked directly rather than inferred from a server run:

  * tiers stay DISJOINT and exactly sized -- an expert in both GPU and RAM
    wastes a slot; one in neither has no weights anywhere;
  * the GPU tier only ever draws from experts that are CURRENTLY CPU-resident,
    because a GPU restage streams its source out of the kt CPU store;
  * covered demand mass NEVER decreases -- every swap requires the challenger
    to beat the incumbent, so a tick that loses coverage is a policy bug. (A
    demoted GPU expert is NOT entitled to a RAM slot: it competes for one, and
    if it is colder than the weakest RAM incumbent, SSD is the right answer.)
  * repeated application on stable counts QUIESCES. Ratio hysteresis alone
    never does: on the tail, counts of 5 vs 2 cross any margin every sweep
    while moving no coverage, and the restage churn then costs more than the
    swaps recover.

Run: .venv/bin/python experiments/expert_tiering_ssd/test_tier_select.py
"""
import os
import sys

os.environ.setdefault("KT_RAM_EXPERTS", "32")
import torch  # noqa: E402

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from sglang.srt.layers.moe.kt_ep_wrapper import _kt_tier_select  # noqa: E402

E, N_GPU, N_RAM = 256, 96, 32


def make_state(seed=0):
    g = torch.Generator().manual_seed(seed)
    # Zipf-ish demand, like real MoE routing.
    counts = torch.zeros(E, dtype=torch.float64)
    perm = torch.randperm(E, generator=g)
    counts[perm] = torch.tensor(
        [1000.0 / (i + 1) for i in range(E)], dtype=torch.float64
    )
    gpu = torch.zeros(E, dtype=torch.bool)
    ram = torch.zeros(E, dtype=torch.bool)
    idx = torch.randperm(E, generator=g)
    gpu[idx[:N_GPU]] = True
    ram[idx[N_GPU : N_GPU + N_RAM]] = True
    return counts, gpu, ram


def check(counts, gpu, ram, new_gpu, new_ram, tag):
    assert int(new_gpu.sum()) == N_GPU, f"[{tag}] GPU tier is {int(new_gpu.sum())}"
    assert int(new_ram.sum()) == N_RAM, f"[{tag}] RAM tier is {int(new_ram.sum())}"
    assert not bool((new_gpu & new_ram).any()), f"[{tag}] tiers overlap"
    # GPU may only take from the previous CPU-resident set.
    entered_gpu = new_gpu & ~gpu
    assert bool((entered_gpu <= (ram | gpu)).all()), (
        f"[{tag}] GPU promoted an expert that was not CPU-resident -- its "
        "restage would stream garbage out of the kt CPU store"
    )
    # Covered mass is monotone: every swap requires the challenger to win.
    before = float(counts[gpu | ram].sum())
    after = float(counts[new_gpu | new_ram].sum())
    assert after >= before - 1e-6, (
        f"[{tag}] covered demand mass DROPPED {before:.1f} -> {after:.1f}; "
        "a tick must never make coverage worse"
    )


counts, gpu, ram = make_state(0)
new_gpu, new_ram = _kt_tier_select(counts, gpu, ram, N_GPU, N_RAM)
check(counts, gpu, ram, new_gpu, new_ram, "first")
moved_gpu = int((new_gpu & ~gpu).sum())
moved_ram = int((new_ram & ~ram & ~gpu).sum())
print(f"ok  first tick: {moved_gpu} into GPU, {moved_ram} newly staged into RAM")

# Budget: KT_TIER_MAX_PROMOTE caps the GPU swaps per visit. The RAM tier can
# additionally absorb GPU demotions and backfill to its exact size, so its
# movement is not capped by the same number.
cap = int(os.environ.get("KT_TIER_MAX_PROMOTE", "2"))
assert moved_gpu <= cap, f"GPU moved {moved_gpu} > cap {cap}"
print(f"ok  GPU movement respects KT_TIER_MAX_PROMOTE={cap}")

# Quiescence on stable counts.
g, r = gpu.clone(), ram.clone()
history = []
for tick in range(200):
    ng, nr = _kt_tier_select(counts, g, r, N_GPU, N_RAM)
    check(counts, g, r, ng, nr, f"tick{tick}")
    changed = int((ng != g).sum()) + int((nr != r).sum())
    history.append(changed)
    g, r = ng, nr
    if changed == 0:
        break
assert history[-1] == 0, (
    f"never quiesced in 200 ticks; last 10 change counts: {history[-10:]}"
)
print(f"ok  quiesced after {len(history)} ticks (stable demand -> no churn)")

# The converged state should be the true two-cut of the ranking.
order = torch.argsort(counts, descending=True)
want_gpu = set(order[:N_GPU].tolist())
want_ram = set(order[N_GPU : N_GPU + N_RAM].tolist())
got_gpu, got_ram = set(torch.where(g)[0].tolist()), set(torch.where(r)[0].tolist())
print(
    f"ok  converged coverage: GPU {len(got_gpu & want_gpu)}/{N_GPU} of the true top-{N_GPU}, "
    f"RAM {len(got_ram & want_ram)}/{N_RAM} of the next {N_RAM}"
)

# A demand SHIFT must be tracked: rotate the ranking and confirm the tiers move
# toward the new hot set instead of freezing.
counts2 = counts.roll(97)
before = float(counts2[g].sum())
for _ in range(400):
    ng, nr = _kt_tier_select(counts2, g, r, N_GPU, N_RAM)
    if int((ng != g).sum()) + int((nr != r).sum()) == 0:
        g, r = ng, nr
        break
    g, r = ng, nr
after = float(counts2[g].sum())
assert after > before, f"GPU coverage did not improve after a demand shift ({before:.0f} -> {after:.0f})"
print(f"ok  tracks a demand shift: GPU-covered mass {before:.0f} -> {after:.0f}")

print("\nall tier-movement invariants hold")
