#!/usr/bin/env python3
"""Check that the expert-energy model behaves the way the design claims.

The claims under test, in the order they matter:

  1. IMMEDIATE PROMOTION: one firing of a cold (SSD-tier) expert lifts it above
     the weakest RAM incumbent on the very next decision, so it is eligible
     without waiting for a slow average to build.
  2. ~4-TOKEN HOLD AT THE GPU BAR, THEN A SLIDE: if it is not re-used it drops
     under the GPU bar after ~4 steps but stays above the RAM bar for longer,
     so it slides GPU -> RAM -> SSD. This falls out of ONE geometric decay
     crossing TWO thresholds; there is no separate timer per tier.
  2b. STICKY DECAY: an expert still drawing router mass -- even a trickle from
     outside the top-8 -- descends more slowly than one drawing nothing.
  3. RARITY BONUS: the same router mass lifts a rarely-used expert further than
     a saturated one, because idf multiplies only the transient term.
  4. GRADUATION: repeated firings build the persistent term, so a genuinely
     recurring expert stops depending on the transient and stays promoted.
  5. POPULARITY IS NOT PUNISHED: a constantly-used expert must keep the highest
     energy of all. If idf were applied to the persistent term instead, the
     hottest experts would be demoted -- the failure mode this ordering avoids.

Runs the REAL update (deepseek_v2._kt_topk_experiment) against fake buffers,
not a re-implementation, so a change to the model is caught here.

Run: KT_ENERGY=1 .venv/bin/python experiments/expert_tiering_ssd/test_energy_model.py
"""
import os
import sys

os.environ["KT_ENERGY"] = "1"
os.environ.setdefault("KT_RAM_EXPERTS", "32")

import torch  # noqa: E402

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from sglang.srt.models import deepseek_v2 as dsv2  # noqa: E402
from sglang.srt.layers.moe.kt_ep_wrapper import _kt_energy_of  # noqa: E402

if dsv2._KT_TOPK_MODE is None:
    dsv2._KT_TOPK_MODE, dsv2._KT_TOPK_K = "safe", 2
dsv2._KT_ADAPTIVE = False

E, TOP_K, N_GPU, N_RAM = 256, 8, 96, 32
LF, LS = dsv2._KT_ENERGY_LF, dsv2._KT_ENERGY_LS
print(
    f"model: lambda_fast={LF}..{dsv2._KT_ENERGY_LF_MAX} (sticky k={dsv2._KT_ENERGY_STICKY_K}) "
    f"lambda_slow={LS} idf_c={dsv2._KT_ENERGY_IDF_C} "
    f"hold={os.environ.get('KT_ENERGY_HOLD_STEPS', '4')} steps"
)


class QM:
    def __init__(self):
        self.gpu_experts_mask_cuda = torch.zeros(E, dtype=torch.bool)
        self.gpu_experts_mask_cuda[:96] = True
        self.ram_experts_mask_cuda = torch.zeros(E, dtype=torch.bool)
        self.ram_experts_mask_cuda[96:128] = True
        self._kt_energy_fast = torch.zeros(E, dtype=torch.float32)
        self._kt_energy_slow = torch.zeros(E, dtype=torch.float32)
        self._kt_decode_counts = torch.zeros(E, dtype=torch.float32)


class TopkOut:
    def __init__(self, ids, w):
        self.topk_ids, self.topk_weights = ids, w


def step(qm, hot_expert, hot_logit=3.9, base=None):
    """One decode step of a single token whose router puts `hot_logit` on
    `hot_expert` and a fixed background elsewhere. The default logit gives the
    fired expert ~0.3 of the router mass -- a realistic top-2 share. Firing it
    with the WHOLE distribution instead makes a single hit look like ten and
    the expert graduates on its persistent term alone, hiding the transient."""
    logits = base.clone() if base is not None else torch.full((1, E), -2.0)
    if hot_expert is not None:
        logits[0, hot_expert] = hot_logit
    sel = torch.topk(logits, TOP_K, dim=-1)
    out = TopkOut(sel.indices.to(torch.int64).clone(), torch.softmax(sel.values, -1).clone())
    dsv2._kt_topk_experiment(out, logits, qm, keep_k=None, layer_id=0)


def energy(qm):
    return _kt_energy_of(qm, num_gpu=N_GPU)


def bars(qm):
    """The GPU and RAM admission bars: the N_GPU-th and (N_GPU+N_RAM)-th
    largest energies in the layer."""
    e = energy(qm)
    s = torch.sort(e, descending=True).values
    return float(s[N_GPU - 1]), float(s[N_GPU + N_RAM - 1])


# Realistic background: a Zipf-ish router over ~150 experts, so the GPU and RAM
# bars sit where they would on a real layer. (An artificially empty tail makes
# every bar ~0 and any surge looks like it holds forever.)
base = torch.full((1, E), -12.0)
base[0, :150] = torch.log(torch.tensor([1.0 / (i + 1) for i in range(150)])) + 3.0

qm = QM()
for _ in range(600):
    step(qm, None, base=base)
gpu_bar, ram_bar = bars(qm)
COLD = 200  # far out in the tail: SSD tier
print(f"warm-up: GPU bar={gpu_bar:.6f}  RAM bar={ram_bar:.6f}  cold expert {COLD}={float(energy(qm)[COLD]):.3e}")

# 1. Immediate promotion, in one step, past the GPU bar.
before = float(energy(qm)[COLD])
step(qm, COLD, base=base)
after = float(energy(qm)[COLD])
assert after > gpu_bar, f"one firing did not clear the GPU bar: {after:.6f} vs {gpu_bar:.6f}"
print(f"ok  1. immediate promotion: {before:.3e} -> {after:.6f} clears the GPU bar "
      f"{gpu_bar:.6f} in ONE step")

# 2. Hold above the GPU bar, then slide down through the RAM bar.
gpu_hold = 1
ram_hold = 1
for t in range(1, 60):
    step(qm, None, base=base)
    e = float(energy(qm)[COLD])
    if e > gpu_bar:
        gpu_hold = t + 1
    if e > ram_bar:
        ram_hold = t + 1
    else:
        break
print(f"ok  2. held {gpu_hold} steps above the GPU bar, {ram_hold} above the RAM bar "
      f"-> slides GPU -> RAM -> SSD on one decay")
assert 2 <= gpu_hold <= 8, f"GPU hold was {gpu_hold} steps; asked for ~4"
assert ram_hold > gpu_hold, "expert did not linger in RAM after leaving GPU"

# 2b. Sticky decay: a trickle of router mass slows the descent.
def descent_steps(trickle_logit):
    q = QM()
    for _ in range(600):
        step(q, None, base=base)
    gb, _ = bars(q)
    step(q, COLD, base=base)
    b = base.clone()
    if trickle_logit is not None:
        b[0, COLD] = trickle_logit  # small, never in the top-8
    n = 1
    for _ in range(200):
        step(q, None, base=b)
        if float(energy(q)[COLD]) > gb:
            n += 1
        else:
            break
    return n


dry = descent_steps(None)
wet = descent_steps(0.5)  # ~1.5% of the router mass: present, never top-8
assert wet > dry, f"trickle did not slow the descent ({wet} vs {dry} steps)"
print(f"ok  2b. sticky decay: {dry} steps with no input, {wet} with a sub-top-8 trickle "
      f"({wet / dry:.1f}x longer)")

# 3. Rarity bonus: same mass, different history.
qm2 = QM()
for _ in range(400):
    step(qm2, None, base=base)
RARE, COMMON = 201, 5  # 5 sits in the established hot core
# Zero the transient for the expert under test first, so what is read back IS
# the bump. Subtracting `before * lambda` cannot work now that lambda is
# per-expert and sticky: the common expert retains more of its old transient,
# which swamped the comparison and made the bump look larger for the WRONG one.
qm2._kt_energy_fast[RARE] = 0.0
step(qm2, RARE, base=base)
step_rare = float(qm2._kt_energy_fast[RARE])

qm3 = QM()
for _ in range(600):
    step(qm3, None, base=base)
qm3._kt_energy_fast[COMMON] = 0.0
step(qm3, COMMON, base=base)
step_common = float(qm3._kt_energy_fast[COMMON])
assert step_rare > step_common, (
    f"rare expert bump {step_rare:.5f} not larger than common {step_common:.5f}"
)
print(f"ok  3. rarity bonus: identical router mass bumps the RARE expert "
      f"{step_rare / max(step_common, 1e-9):.1f}x harder than the COMMON one")

# 4. Graduation: repeated use builds the persistent term.
qm4 = QM()
for _ in range(400):
    step(qm4, None, base=base)
for _ in range(60):
    step(qm4, COLD, base=base)
slow_rate = float(qm4._kt_energy_slow[COLD]) * (1.0 - LS)
for _ in range(20):  # long enough for the transient to be gone
    step(qm4, None, base=base)
resid = float(energy(qm4)[COLD])
_, ram_bar4 = bars(qm4)
assert resid > ram_bar4, (
    f"a repeatedly-used expert did not stay promoted once its transient decayed "
    f"({resid:.6f} vs RAM bar {ram_bar4:.6f})"
)
print(f"ok  4. graduation: after 60 firings it holds {resid:.6f} > RAM bar {ram_bar4:.6f} "
      f"on the PERSISTENT term alone (rate {slow_rate:.5f})")

# 5. Popularity is not punished.
e = energy(qm4)
top = int(torch.argmax(e))
assert top < 96, f"highest-energy expert is {top}, outside the established hot core"
print(f"ok  5. popularity intact: highest-energy expert is {top} (in the hot core), "
      f"not demoted by the rarity bonus")

# Full router mass really is counted: an expert that is NEVER in the top-8 must
# still accumulate something, which is the early-warning signal.
never_routed = int(torch.argmin(energy(qm)))
assert float(qm._kt_energy_slow[never_routed]) > 0.0, "unrouted experts accumulated nothing"
print(f"ok  6. tail counted: never-routed expert {never_routed} still accumulates "
      f"{float(qm._kt_energy_slow[never_routed]):.3e}")

print("\nall energy-model claims hold")
