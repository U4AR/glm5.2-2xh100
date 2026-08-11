"""Does slot eviction take the WEAKEST resident, or just the lowest slot index?

Runs the real kernel and checks its choice against a policy oracle written from
the SPEC ("evict the minimum hold_score, ties to the lower index"), not against
the kernel's own arithmetic. A test that re-implements the code's reasoning
proves only self-consistency -- that has already produced two false passes in
this project (a gather checked against an older gather, and a verify that
asserted the kernel's own wrong slot convention).

The discriminating case is the one the old code got wrong: a slot holding a
proven-useful expert sitting at a LOW index, and a useless expert at a HIGH
index. Positional eviction takes the low index -- the useful one. Ranked
eviction takes the high index.

    .venv/bin/python bench/slot_rank_test.py
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pred_fused_kernel import kernels                     # noqa: E402

N_EXP, T, TOPK, SLOT_BASE = 256, 4, 8, 100


def make_logits(wanted, device):
    """Router logits that put `wanted` experts firmly in the top-8."""
    lg = torch.full((T, N_EXP), -8.0, device=device, dtype=torch.float32)
    for e in wanted:
        lg[:, e] = 9.0
    # Filler so the top-8 is well defined without touching the wanted set.
    filler = [e for e in range(N_EXP) if e not in wanted][:TOPK]
    for i, e in enumerate(filler):
        lg[:, e] = 1.0 - 0.01 * i
    return lg


def run(hold, score, wanted, resident_ids, n_slot, device):
    resident = torch.zeros(N_EXP, dtype=torch.bool, device=device)
    resident[list(resident_ids)] = True
    sel = torch.full((n_slot,), -1, dtype=torch.int64, device=device)
    landed = torch.zeros(N_EXP + 1, dtype=torch.bool, device=device)
    landed_cpu = torch.zeros(N_EXP + 1, dtype=torch.bool, device=device)
    index = torch.zeros(N_EXP + 1, dtype=torch.int32, device=device)
    stats = torch.zeros(5, dtype=torch.int64, device=device)
    h = torch.tensor(hold, dtype=torch.int64, device=device)
    hs = torch.tensor(score, dtype=torch.int32, device=device)
    bias = torch.zeros(N_EXP, dtype=torch.float32, device=device)
    kernels().pred_fused(
        make_logits(wanted, device), bias, resident,
        sel, landed[:N_EXP], landed_cpu[:N_EXP], index[:N_EXP], stats, h, hs,
        TOPK, TOPK, 0, SLOT_BASE, 0, 1, 1, 2)
    torch.cuda.synchronize()
    return sel.tolist(), h.tolist(), hs.tolist()


def oracle_victim(hold, score, keep):
    """Which slot the SPEC says takes the first arrival."""
    cand = [k for k in range(len(hold)) if k not in keep]
    return min(cand, key=lambda k: (score[k], k)) if cand else None


def main():
    if not torch.cuda.is_available():
        print("no CUDA; this test needs the real kernel")
        return 1
    dev = "cuda"
    fails = 0

    # --- the discriminating case -------------------------------------------
    # Slot 0 holds expert 10 and has proven useful (score 40).
    # Slot 3 holds expert 13 and has proven useless (score 0).
    # A brand-new expert 200 is wanted. Nothing currently held is wanted, so
    # every slot is a candidate and the policies disagree.
    # Held experts must be NON-resident (>=100): a resident expert is never
    # fetched into a slot, so its demand is zero and the keep pass rightly
    # ignores it. Using resident ids here tested nothing and looked like a
    # kernel bug.
    hold = [110, 111, 112, 113]
    score = [40, 30, 20, 0]
    sel, h2, hs2 = run(hold, score, wanted=[200], resident_ids=range(100),
                       n_slot=4, device=dev)
    victim = [k for k in range(4) if sel[k] == 200]
    victim = victim[0] if victim else None
    want = oracle_victim(hold, score, keep=set())
    print(f"discriminating case: arrival landed in slot {victim}, "
          f"spec says {want} (positional would say 0)")
    if victim != want:
        print("  FAIL"); fails += 1
    if victim == 0:
        print("  FAIL: still positional -- the useful expert was evicted"); fails += 1

    # --- a held-and-wanted expert must never be evicted ---------------------
    # Expert 11 is held in slot 1 AND wanted. It must be kept (sel[1] = -1,
    # i.e. no bytes moved) no matter how low its score is.
    hold = [110, 111, 112, 113]
    score = [50, 0, 50, 50]
    sel, h2, hs2 = run(hold, score, wanted=[111, 201], resident_ids=range(100),
                       n_slot=4, device=dev)
    if sel[1] != -1 or h2[1] != 111:
        print(f"  FAIL: wanted-and-held expert evicted; sel={sel} hold={h2}")
        fails += 1
    else:
        print(f"held-and-wanted kept despite score 0: sel={sel}")
    if hs2[1] <= 0:
        print(f"  FAIL: reuse did not raise the kept slot's score ({hs2[1]})")
        fails += 1

    # --- ties reproduce the old positional order ---------------------------
    # All scores equal (the state at boot) must behave exactly as before, or
    # every previously measured row silently changes meaning.
    sel, _, _ = run([110, 111, 112, 113], [7, 7, 7, 7], wanted=[202],
                    resident_ids=range(100), n_slot=4, device=dev)
    if sel[0] != 202:
        print(f"  FAIL: ties must go to the lowest slot; sel={sel}"); fails += 1
    else:
        print("equal scores -> lowest slot, identical to the old behaviour")

    # --- aging: an unused slot must decay toward evictability --------------
    hold, score = [110, 111, 112, 113], [3, 30, 30, 30]
    for _ in range(4):
        _, hold, score = run(hold, score, wanted=[203], resident_ids=range(100),
                             n_slot=4, device=dev)
    print(f"after 4 calls of aging: scores {score}")

    print("OK" if not fails else f"{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
