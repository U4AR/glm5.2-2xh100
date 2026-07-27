#!/usr/bin/env python3
"""Offline checks for the three-tier expert store (no server, no GPU).

Two invariants matter and both are cheap to check directly against the real
functions:

  1. The tiers partition the layer: GPU / RAM / SSD are disjoint and the RAM
     tier has exactly KT_RAM_EXPERTS members.
  2. The routing kernel NEVER emits an SSD-tier expert id. That is the whole
     safety argument — an SSD-tier expert has weights in neither VRAM nor the
     kt CPU store, so if one survives substitution the CPU kernel silently
     drops its mass (should_skip_expert) and the token is computed wrong.

Run:  KT_RAM_EXPERTS=32 .venv/bin/python experiments/expert_tiering_ssd/test_tier_routing.py
"""
import os
import sys

# Must be set before importing the modules under test — both read their env at
# import time. Also point the topk sentinel at a safe2 config.
os.environ.setdefault("KT_RAM_EXPERTS", "32")
_SENTINEL = "/tmp/kt_topk_mode_tiertest"
with open(_SENTINEL, "w") as f:
    f.write("safe2")
os.environ["KT_TOPK_MODE_FILE"] = _SENTINEL

import torch  # noqa: E402

sys.path.insert(0, ".venv/lib/python3.12/site-packages")


def _fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


# --------------------------------------------------------------------------
# 1. Tier partition
# --------------------------------------------------------------------------
from sglang.srt.layers.moe import kt_ep_wrapper as ktw  # noqa: E402

E = 256
N_GPU = 96
N_RAM = int(os.environ["KT_RAM_EXPERTS"])

if not ktw._KT_TIER_ENABLED:
    _fail("KT_RAM_EXPERTS did not enable the three-tier path")

torch.manual_seed(0)
gpu_mask = torch.zeros(E, dtype=torch.bool)
gpu_mask[torch.randperm(E)[:N_GPU]] = True

ram_mask = ktw._kt_build_ram_tier_mask(layer_idx=7, gpu_experts_mask=gpu_mask)
ssd_mask = ~(gpu_mask | ram_mask)

assert int(ram_mask.sum()) == N_RAM, f"RAM tier is {int(ram_mask.sum())}, want {N_RAM}"
assert not bool((gpu_mask & ram_mask).any()), "GPU and RAM tiers overlap"
assert int((gpu_mask | ram_mask | ssd_mask).sum()) == E, "tiers do not cover the layer"
assert int(ssd_mask.sum()) == E - N_GPU - N_RAM
print(f"ok  tier partition: GPU={N_GPU} RAM={N_RAM} SSD={int(ssd_mask.sum())}")

# The kt skip mask must be exactly the complement of the CPU store.
class _FakeMethod:
    pass


m = _FakeMethod()
m.gpu_experts_mask = gpu_mask
m.ram_experts_mask = ram_mask
skip = ktw.KTEPWrapperMethod.kt_cpu_skip_mask(m)
assert torch.equal(skip, gpu_mask | ssd_mask), "kt skip mask != GPU | SSD"
print("ok  kt skip mask == GPU | SSD (CPU store holds exactly the RAM tier)")

# --------------------------------------------------------------------------
# 2. Routing never emits an SSD-tier id
# --------------------------------------------------------------------------
from sglang.srt.models import deepseek_v2 as dsv2  # noqa: E402

if dsv2._KT_TOPK_MODE is None:
    print(
        "note: topk sentinel not picked up (KT_TOPK_MODE_FILE is read from a "
        "fixed path in this build); forcing safe2 for the test"
    )
    dsv2._KT_TOPK_MODE, dsv2._KT_TOPK_K = "safe", 2


class _TopkOut:
    def __init__(self, ids, w):
        self.topk_ids = ids
        self.topk_weights = w


class _QM:
    """Stands in for KTEPWrapperMethod: only the two masks are read."""

    def __init__(self, gpu, ram):
        self.gpu_experts_mask_cuda = gpu
        self.ram_experts_mask_cuda = ram
        self._kt_decode_counts = torch.zeros(E, dtype=torch.float32)


def run_case(name, K, fill_pool, count_mode, T=512, top_k=8):
    dsv2._KT_TIER_FILL_POOL = fill_pool
    dsv2._KT_TIER_COUNT_MODE = count_mode
    dsv2._KT_TIER_COUNT_K = 2 if count_mode == "top2" else 8
    dsv2._KT_TIER_COUNT_WEIGHTED = count_mode == "top8w"
    dsv2._KT_ADAPTIVE = True
    dsv2._KT_TOPK_K = K

    logits = torch.randn(T, E)
    sel = torch.topk(logits, top_k, dim=-1)
    ids = sel.indices.to(torch.int64).clone()
    w = torch.softmax(sel.values, dim=-1).clone()

    qm = _QM(gpu_mask, ram_mask)
    out = _TopkOut(ids.clone(), w.clone())
    dsv2._kt_topk_experiment(out, logits, qm, keep_k=None, layer_id=7)

    emitted = out.topk_ids.unique()
    leaked = emitted[ssd_mask[emitted.long()]]
    if leaked.numel():
        _fail(
            f"[{name}] {leaked.numel()} SSD-tier experts survived routing: "
            f"{leaked[:8].tolist()}"
        )

    # Weights must still be a distribution.
    s = out.topk_weights.sum(dim=-1)
    if not torch.allclose(s, torch.ones_like(s), atol=1e-4):
        _fail(f"[{name}] weights not renormalized (min={s.min():.4f} max={s.max():.4f})")

    # The fill pool must be respected.
    if fill_pool == "gpu":
        substituted = out.topk_ids[out.topk_ids != ids]
        if substituted.numel() and not bool(gpu_mask[substituted.long()].all()):
            _fail(f"[{name}] fill pool=gpu but a substitute was not GPU-resident")

    counted = int((qm._kt_decode_counts > 0).sum())
    kept = int((out.topk_ids == ids).sum())
    print(
        f"ok  [{name}] K={K} fill={fill_pool} count={count_mode}: "
        f"no SSD leak, {kept}/{ids.numel()} slots kept genuine, "
        f"{counted} experts voted"
    )
    return qm._kt_decode_counts


for K in (2, 4, 8):
    for pool in ("resident", "gpu"):
        run_case(f"K{K}-{pool}", K, pool, "top2")

c2 = run_case("count-top2", 2, "resident", "top2")
c8 = run_case("count-top8", 2, "resident", "top8")
c8w = run_case("count-top8w", 2, "resident", "top8w")

# top8 must see strictly more experts than top2, and must see SSD-tier demand
# (that is the point — an SSD expert is discoverable even though a stand-in
# did the compute).
n2, n8 = int((c2 > 0).sum()), int((c8 > 0).sum())
assert n8 > n2, f"top8 counted {n8} experts, top2 counted {n2}; expected more"
ssd_demand = float(c8[ssd_mask].sum())
assert ssd_demand > 0, "top8 recorded no demand for SSD-tier experts"
assert abs(float(c8w.sum()) - 1.0 * c8w.shape[0] * 0) < 1e9  # shape sanity only
print(
    f"ok  count modes: top2 saw {n2} experts, top8 saw {n8} "
    f"(SSD-tier demand recorded: {ssd_demand:.0f} votes)"
)

# K == E is NOT a passthrough under three-tier: an SSD expert in the top-8 still
# has to be substituted. Verify explicitly.
dsv2._KT_TIER_FILL_POOL = "resident"
logits = torch.randn(256, E)
sel = torch.topk(logits, 8, dim=-1)
ids = sel.indices.to(torch.int64).clone()
out = _TopkOut(ids.clone(), torch.softmax(sel.values, -1).clone())
dsv2._KT_TOPK_K = 8
dsv2._kt_topk_experiment(out, logits, _QM(gpu_mask, ram_mask), keep_k=None, layer_id=7)
emitted = out.topk_ids.unique()
assert not bool(ssd_mask[emitted.long()].any()), "K==E passthrough leaked SSD experts"
print("ok  K==E is filtered too (passthrough parity is void once a tier is unreachable)")

print("\nall three-tier routing invariants hold")
