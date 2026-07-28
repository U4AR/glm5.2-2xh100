#!/usr/bin/env python3
"""Do the tier gates fail to bind because the demand signal is too noisy?

ANSWER: NO. This script was written to confirm that hypothesis and refuted it.
Kept because the refutation is the useful part.

The hypothesis was reasonable. Movement on the server is saturated at 100% of
budget on every visit forever (gpu_swaps/visit 2.00 of 2, ram_promote/visit
4.00 of 4, flat across the whole run), and `_kt_tier_select`'s own docstring
warns that "ratio alone never quiesces on the tail". The RAM cut's absolute
floor is set 50x weaker than the GPU cut's, which looked like the hole.

Two independent results say otherwise:

1. Below: under Poisson-sampled demand at a realistic vote rate, the SHIPPED
   gate largely quiesces (tail ram ~0.66 of a budget of 4) -- it does not
   reproduce the saturation seen on the server at all. Raising the floor moves
   it to ~0.05 while leaving reachable coverage within 0.001. The gate is not
   the mechanism.

2. The server's own promotion ROI accounting: 87-89% of promoted experts are
   CALLED before the next decision, and only 0.2-1.1% are demoted again. Noise
   churn would promote experts nothing asks for. These promotions are wanted.

What the saturation actually means, then, is that the working set is larger
than the RAM tier: there is always another genuinely-warm expert outside it, so
the selector always has something legitimate to promote and never runs out.
That is a capacity limit, and no gate can fix a capacity limit. It also
explains the separate finding that quadrupling the tick rate did not improve
quality -- latency was never the binding constraint.

Run: .venv/bin/python experiments/expert_tiering_ssd/tier_gate_sim.py
"""
import os
import sys

os.environ.setdefault("KT_RAM_EXPERTS", "72")
import torch  # noqa: E402

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from sglang.srt.layers.moe import kt_ep_wrapper as K  # noqa: E402

PRIOR = "experiments/adaptive_expert_cache/decode_cache/hot_core_prior.pt"
N_GPU, N_RAM, N_EXPERTS = 104, 72, 256
TICKS = 80
VOTES_PER_TICK = 256      # ~32 steps x 4 MTP positions x top-2
PRIOR_MASS = 64.0
DECAY = 0.98


def simulate(rate, gates, gen, ticks=TICKS):
    """One layer, stationary true demand, Poisson-sampled observations.

    Returns (tail_gpu_moves, tail_ram_moves, reachable_coverage_of_TRUE_demand).
    Coverage is scored against the TRUE rate, not the noisy sample -- chasing
    the sample is the failure being measured, so the sample cannot also be the
    scorer.
    """
    for k, v in gates.items():
        os.environ[k] = str(v)

    p = (rate / rate.sum()).double()
    order = torch.argsort(rate, descending=True)
    gpu = torch.zeros(N_EXPERTS, dtype=torch.bool)
    ram = torch.zeros(N_EXPERTS, dtype=torch.bool)
    gpu[order[:N_GPU]] = True
    ram[order[N_GPU:N_GPU + N_RAM]] = True

    acc = p * PRIOR_MASS          # the seeded prior, as the server does
    moves = []
    for _ in range(ticks):
        draw = torch.poisson(p * VOTES_PER_TICK, generator=gen).double()
        acc = acc * DECAY + draw
        new_gpu, new_ram = K._kt_tier_select(acc, gpu, ram, N_GPU, N_RAM)
        moves.append((int((new_gpu & ~gpu).sum()), int((new_ram & ~ram).sum())))
        gpu, ram = new_gpu, new_ram

    tail = moves[-20:]
    cov = float(rate[gpu | ram].sum() / rate.sum())
    return (sum(m[0] for m in tail) / len(tail),
            sum(m[1] for m in tail) / len(tail),
            cov)


prior = torch.load(PRIOR, map_location="cpu")
scores = prior["scores"]
LAYERS = (10, 30, 50, 70)

SHIPPED = {
    "KT_ADAPTIVE_MARGIN": 1.3, "KT_ADAPTIVE_MIN_GAIN": 0.001,
    "KT_TIER_RAM_MARGIN": 1.05, "KT_TIER_RAM_MIN_GAIN": 0.00002,
}

# Candidates. The shipped RAM gate is a 1.05 ratio plus a floor of 0.002% of
# layer mass -- on ~300 accumulated votes that floor is 0.006 of a vote, i.e.
# no floor at all. Each candidate raises the ABSOLUTE bar, since that is the
# term that can outrun sampling noise.
CANDIDATES = [
    ("shipped", dict(SHIPPED)),
    ("ram_min_gain 0.0005", {**SHIPPED, "KT_TIER_RAM_MIN_GAIN": 0.0005}),
    ("ram_min_gain 0.002", {**SHIPPED, "KT_TIER_RAM_MIN_GAIN": 0.002}),
    ("ram_min_gain 0.005", {**SHIPPED, "KT_TIER_RAM_MIN_GAIN": 0.005}),
    ("ram_margin 1.5 + gain 0.002",
     {**SHIPPED, "KT_TIER_RAM_MARGIN": 1.5, "KT_TIER_RAM_MIN_GAIN": 0.002}),
    ("ram_margin 2.0 + gain 0.005",
     {**SHIPPED, "KT_TIER_RAM_MARGIN": 2.0, "KT_TIER_RAM_MIN_GAIN": 0.005}),
]

print(f"Poisson-sampled demand, {VOTES_PER_TICK} votes/tick, {TICKS} ticks, "
      f"warm-started at the correct answer.")
print("Tail movement is measured over the last 20 ticks; on stationary demand")
print("from a correct start, ANY tail movement is churn. Budgets: gpu=2 ram=4.")
print("Coverage is of TRUE demand (reachable = GPU + RAM tiers).\n")
print(f"{'gate':30s} {'tail_gpu':>9s} {'tail_ram':>9s} {'reachable':>10s}")

# The static warm start, scored the same way: the bar any movement must clear.
base = []
for layer in LAYERS:
    rate = scores[layer].double()
    order = torch.argsort(rate, descending=True)
    m = torch.zeros(N_EXPERTS, dtype=torch.bool)
    m[order[:N_GPU + N_RAM]] = True
    base.append(float(rate[m].sum() / rate.sum()))
print(f"{'STATIC warm start (no ticks)':30s} {0.0:9.2f} {0.0:9.2f} "
      f"{sum(base)/len(base):10.4f}")

for name, gates in CANDIDATES:
    g = r = c = 0.0
    gen = torch.Generator().manual_seed(1234)
    for layer in LAYERS:
        a, b, cov = simulate(scores[layer].double(), gates, gen)
        g += a; r += b; c += cov
    n = len(LAYERS)
    print(f"{name:30s} {g/n:9.2f} {r/n:9.2f} {c/n:10.4f}")

print("\nThe question each row answers: does this gate stop churning, and does")
print("it still reach at least as much true demand as never moving at all?")
