#!/usr/bin/env python3
"""The energy swap gate must quiesce on a settled demand distribution.

Without a gate the executor ran pinned at its move cap for an entire 450-tick
run -- 1800 promotions, 65 GiB of expert traffic, and every layer waiting ~75
steps for service because the budget was going to churn rather than to
genuinely hot experts. This checks the gate stops that while still admitting a
real surge.
"""
import os, sys, torch
sys.path.insert(0, ".venv/lib/python3.12/site-packages")
os.environ.setdefault("KT_RAM_EXPERTS", "32")
from sglang.srt.layers.moe.kt_ep_wrapper import _kt_energy_worth_moving

E = 256
g = torch.Generator().manual_seed(0)
# Zipf-ish settled energies, like a converged layer.
e = torch.tensor([1.0 / (i + 1) for i in range(E)])[torch.randperm(E, generator=g)]

srt = torch.sort(e, descending=True).values
gpu_bar, ram_bar = srt[95], srt[127]
# Neighbouring ranks in the tail: the pairs that caused the thrash.
tail = torch.sort(e, descending=True).values[128:]
churn = sum(
    1 for a, b in zip(tail[:-1], tail[1:])
    if _kt_energy_worth_moving(e, a, b, "KT_ENERGY")
)
print(f"adjacent tail pairs admitted by the gate: {churn} of {len(tail)-1}")
assert churn == 0, f"gate still admits {churn} near-tied tail swaps -> thrash"
print("ok  1. settled tail produces NO swaps (the thrash is gated out)")

# A real surge must still get through: transient energy lifts a cold expert
# well above the RAM floor.
cold = float(e.min())
surge = float(ram_bar) * 4.0
assert _kt_energy_worth_moving(e, torch.tensor(surge), torch.tensor(cold), "KT_ENERGY"), \
    "gate blocks a genuine surge"
print(f"ok  2. a genuine surge ({surge:.4f} vs incumbent {cold:.4f}) is admitted")

# A marginal difference must not be.
inc = float(ram_bar)
assert not _kt_energy_worth_moving(e, torch.tensor(inc * 1.1), torch.tensor(inc), "KT_ENERGY"), \
    "gate admits a 10% difference; that is the churn regime"
print("ok  3. a 10% edge is rejected (below the 1.5x ratio margin)")
print("\nenergy swap gate behaves as required")
