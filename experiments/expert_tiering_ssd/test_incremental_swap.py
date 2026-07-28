#!/usr/bin/env python3
"""Offline checks for the stable-slot incremental GPU swap.

A tier swap of 2 experts used to cost ~250 ms because GPU slots are numbered by
POSITION in the sorted resident list: changing the set by one renumbers most of
it, and every renumbered expert's weights then have to be re-copied. The
incremental path instead hands each arriving expert the slot its departing
partner just vacated, so the work is proportional to what changed.

That makes the slot bookkeeping load-bearing. `logical_to_gpu_index` and
`gpu_index_to_logical` are inverse maps consulted by the routing kernel; if
they ever disagree, tokens are silently routed to the wrong expert's weights
and nothing raises. So the invariants are checked directly here rather than
inferred from a server run -- the same reason test_tier_select.py exists.

Run: .venv/bin/python experiments/expert_tiering_ssd/test_incremental_swap.py
"""
import os
import sys

os.environ.setdefault("KT_RAM_EXPERTS", "32")
import torch  # noqa: E402

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod  # noqa: E402

E = 32          # logical experts
N = 8           # GPU-resident slots
NAMES = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")


class FakeLayer:
    """Decode-resident layer: one row per SLOT."""
    def __init__(self):
        for n in NAMES:
            setattr(self, n, torch.zeros(N, 4))


class FakeScratch:
    """Scratch layer: one row per LOGICAL expert, tagged so a mis-copy shows."""
    def __init__(self):
        for n in NAMES:
            t = torch.zeros(E, 4)
            for e in range(E):
                t[e] = float(e)
            setattr(self, n, t)


class FakeCtx:
    is_w4afp8_quant = True

    def __init__(self):
        self.gpu_layer = FakeScratch()


class FakeMethod:
    """Only the attributes _apply_expert_selection_incremental touches."""
    tp_rank = 1          # skip the kt wrapper mask update
    wrapper = None

    def __init__(self, resident):
        self.gpu_experts_mask = torch.zeros(E, dtype=torch.bool)
        self.gpu_experts_mask[resident] = True
        self.logical_to_gpu_index = torch.full((E,), -1, dtype=torch.int32)
        self.gpu_index_to_logical = torch.zeros(N, dtype=torch.int32)
        for slot, e in enumerate(resident):
            self.logical_to_gpu_index[e] = slot
            self.gpu_index_to_logical[slot] = e
        self.gpu_experts_mask_cuda = self.gpu_experts_mask.clone()
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.clone()
        self.layer = FakeLayer()
        self.built_with = None

    def _build_full_context(self, layer, only_experts=None):
        # Record what was asked for: streaming only the arrivals is the point.
        self.built_with = None if only_experts is None else sorted(only_experts)
        return FakeCtx()

    def kt_cpu_skip_mask(self):
        return self.gpu_experts_mask


def call(m, selected):
    return KTEPWrapperMethod._apply_expert_selection_incremental(
        m, m.layer, torch.tensor(sorted(selected), dtype=torch.int64)
    )


def check_consistent(m, where):
    resident = torch.where(m.gpu_experts_mask)[0].tolist()
    assert len(resident) == N, f"{where}: {len(resident)} resident, expected {N}"
    slots = [int(m.logical_to_gpu_index[e]) for e in resident]
    assert sorted(slots) == list(range(N)), f"{where}: slots not a permutation: {sorted(slots)}"
    for e in resident:
        s = int(m.logical_to_gpu_index[e])
        assert int(m.gpu_index_to_logical[s]) == e, (
            f"{where}: forward/reverse map disagree at expert {e} slot {s} "
            f"(reverse says {int(m.gpu_index_to_logical[s])})"
        )
    for e in range(E):
        if not bool(m.gpu_experts_mask[e]):
            assert int(m.logical_to_gpu_index[e]) == -1, (
                f"{where}: non-resident expert {e} still holds slot "
                f"{int(m.logical_to_gpu_index[e])}"
            )
    assert torch.equal(m.gpu_experts_mask, m.gpu_experts_mask_cuda), f"{where}: mask mirror stale"
    assert torch.equal(m.logical_to_gpu_index, m.logical_to_gpu_index_cuda), (
        f"{where}: index mirror stale"
    )


resident0 = list(range(N))          # experts 0..7 in slots 0..7

# 1. A 2-for-2 swap keeps every map consistent.
m = FakeMethod(resident0)
new = [e for e in resident0 if e not in (2, 5)] + [20, 21]
assert call(m, new) is True, "incremental swap declined a valid 2-for-2 swap"
check_consistent(m, "after 2-for-2")
print("ok  1. 2-for-2 swap leaves logical/slot maps mutually consistent")

# 2. It streamed ONLY the arrivals -- the whole reason this path exists.
assert m.built_with == [20, 21], f"streamed {m.built_with}, expected [20, 21]"
print(f"ok  2. streamed only the arrivals ({m.built_with}), not all {E}")

# 3. Untouched experts kept their original slots (no renumbering).
for e in resident0:
    if e in (2, 5):
        continue
    assert int(m.logical_to_gpu_index[e]) == e, (
        f"expert {e} moved from slot {e} to {int(m.logical_to_gpu_index[e])}"
    )
print("ok  3. untouched experts keep their slots (nothing renumbered)")

# 4. Arrivals landed in the vacated slots, with the DEPARTING expert's slot and
#    the ARRIVING expert's weights -- the pairing that makes this correct.
for a, d in ((20, 2), (21, 5)):
    slot = int(m.logical_to_gpu_index[a])
    assert slot == d, f"expert {a} took slot {slot}, expected vacated slot {d}"
    for n in NAMES:
        got = float(getattr(m.layer, n)[slot][0])
        assert got == float(a), f"{n}[{slot}] holds expert {got}, expected {a}"
print("ok  4. arrivals occupy the vacated slots and carry their own weights")

# 5. Declines what it cannot do, WITHOUT mutating anything.
m2 = FakeMethod(resident0)
before = (m2.gpu_experts_mask.clone(), m2.logical_to_gpu_index.clone())
assert call(m2, [e for e in resident0 if e != 3]) is False, "accepted a size change"
assert torch.equal(m2.gpu_experts_mask, before[0]), "mutated state before declining"
assert torch.equal(m2.logical_to_gpu_index, before[1]), "mutated indices before declining"
print("ok  5. declines a set-size change and leaves state untouched")

# 6. Declines an oversized swap so the full restage can take it.
os.environ["KT_TIER_INCREMENTAL_MAX"] = "2"
m3 = FakeMethod(resident0)
big = [e for e in resident0 if e not in (0, 1, 2, 3)] + [24, 25, 26, 27]
assert call(m3, big) is False, "accepted a swap larger than KT_TIER_INCREMENTAL_MAX"
print("ok  6. declines an oversized swap (falls back to full restage)")

# 7. A no-op selection is accepted and changes nothing.
m4 = FakeMethod(resident0)
assert call(m4, resident0) is True, "declined a no-op"
check_consistent(m4, "after no-op")
assert m4.built_with is None, "built a context for a no-op"
print("ok  7. no-op selection short-circuits without building a context")

# 8. Repeated swaps stay consistent -- slot reuse must not drift over time.
m5 = FakeMethod(resident0)
cur = set(resident0)
nxt = 16
for i in range(10):
    out = sorted(cur)[0]
    cur = (cur - {out}) | {nxt}
    assert call(m5, sorted(cur)) is True, f"declined swap {i}"
    check_consistent(m5, f"after repeated swap {i}")
    nxt += 1
print("ok  8. 10 successive swaps keep the maps consistent (no slot drift)")

print("\nincremental swap bookkeeping holds")
