#!/usr/bin/env python3
"""Does the tier selector ever STOP moving experts?

Measured on the server, every configuration moves exactly its budget on every
visit, from the first quarter of the run to the last:

    RAM=72 dynamic   gpu_swaps/visit 2.00 2.00 2.00 2.00   (budget 2)
    RAM=32 nogpu     ram_promote/visit 4.00 4.00 3.94 3.91 (budget 4)
    RAM=32 fast      ram_promote/visit 1.00 1.00 1.00 1.00 (budget 1)

Saturated at 100% of budget, forever, while unreachable demand improves by
under a percentage point. An adaptive store is supposed to find the working set
and settle; this one churns at maximum rate indefinitely, which is why the
per-visit cost never amortises and why an expert's promotion says nothing about
whether it was needed.

This drives the REAL selector offline with the REAL demand prior, so the answer
does not depend on a server run or on log parsing.

What it establishes is narrow but load-bearing: the selector is CORRECT on a
clean signal (exact counts, warm start -> zero movement, as it should be) and
churns at full budget once the counts are perturbed. That rules out a plain
logic bug in the pair rule.

It does NOT establish that noise is what drives the server's churn -- see
tier_gate_sim.py, which modelled the noise realistically and failed to
reproduce the saturation, and the server's ROI accounting, which finds 87-89%
of promoted experts are actually called. The evidence points instead at a
working set larger than the tier. Read this file as "the gate logic works",
not as "the gate is the problem".

Run: .venv/bin/python experiments/expert_tiering_ssd/test_tier_quiesce.py
"""
import os
import sys

os.environ.setdefault("KT_RAM_EXPERTS", "72")
import torch  # noqa: E402

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from sglang.srt.layers.moe.kt_ep_wrapper import _kt_tier_select  # noqa: E402

PRIOR = "experiments/adaptive_expert_cache/decode_cache/hot_core_prior.pt"
N_GPU, N_RAM, N_EXPERTS = 104, 72, 256
DECAY = float(os.environ.get("KT_ADAPTIVE_DECAY", "0.98"))
TICKS = 60


def run(layer, counts, label, seed_from_ranking=True, noise=0.0, gen=None):
    """Tick the selector repeatedly on a STATIONARY demand distribution. If the
    demand never changes, a correct selector must converge and then stop."""
    order = torch.argsort(counts, descending=True)
    gpu = torch.zeros(N_EXPERTS, dtype=torch.bool)
    ram = torch.zeros(N_EXPERTS, dtype=torch.bool)
    if seed_from_ranking:
        # Warm start: the shipped config seeds both tiers from the committed
        # ranking, so the selector begins at (near) the right answer.
        gpu[order[:N_GPU]] = True
        ram[order[N_GPU:N_GPU + N_RAM]] = True
    else:
        gpu[torch.arange(N_GPU)] = True
        ram[torch.arange(N_GPU, N_GPU + N_RAM)] = True

    acc = counts.clone()
    moves = []
    for t in range(TICKS):
        c = acc.clone()
        if noise:
            # Finite-sample jitter: a tick sees a few hundred tokens, not the
            # asymptotic distribution. This is what the tail counts really look
            # like -- and the tail is where the gate has to hold.
            c = torch.clamp(c + torch.randn(N_EXPERTS, generator=gen) * noise * c.mean(), min=0.0)
        new_gpu, new_ram = _kt_tier_select(c, gpu, ram, N_GPU, N_RAM)
        g_sw = int((new_gpu & ~gpu).sum())
        r_sw = int((new_ram & ~ram).sum())
        moves.append((g_sw, r_sw))
        gpu, ram = new_gpu, new_ram
        acc.mul_(DECAY)

    first = moves[:10]
    last = moves[-10:]
    fg = sum(m[0] for m in first) / len(first)
    fr = sum(m[1] for m in first) / len(first)
    lg = sum(m[0] for m in last) / len(last)
    lr = sum(m[1] for m in last) / len(last)
    cov = float(counts[gpu | ram].sum() / counts.sum())
    print(f"  {label:34s} first10 gpu={fg:.2f} ram={fr:.2f}   "
          f"last10 gpu={lg:.2f} ram={lr:.2f}   reachable={cov:.4f}")
    return lg, lr


prior = torch.load(PRIOR, map_location="cpu")
scores = prior["scores"]

print(f"stationary demand, {TICKS} ticks, budgets gpu=2 ram=4 "
      f"(margins {os.environ.get('KT_ADAPTIVE_MARGIN','1.3')} / "
      f"{os.environ.get('KT_TIER_RAM_MARGIN','1.05')})\n")

gen = torch.Generator().manual_seed(0)
tail_g = tail_r = 0.0
n = 0
for layer in (10, 30, 50, 70):
    counts = scores[layer].double()
    run(layer, counts, f"layer {layer} exact, warm start")
    lg, lr = run(layer, counts, f"layer {layer} +10% jitter, warm start",
                 noise=0.10, gen=gen)
    tail_g += lg
    tail_r += lr
    n += 1

print()
print("Interpretation: with demand held STATIONARY and the tiers warm-started")
print("at the right answer, any movement in the last 10 ticks is pure churn --")
print("there is nothing left to discover. Budgets are gpu=2, ram=4, so those")
print("are the saturation values.")
print(f"\nmean tail movement: gpu={tail_g/n:.2f}/2  ram={tail_r/n:.2f}/4")
if tail_r / n > 2.0 or tail_g / n > 1.0:
    print("\nVERDICT: the gates do NOT quiesce on stationary demand.")
else:
    print("\nVERDICT: the gates quiesce.")
