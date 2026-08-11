"""Independent check of the prefetch publish step (TODO item 10).

Deliberately does NOT import kt_ep_wrapper. The oracle here is the SPEC --
"after publishing, landed[e] is true exactly for the experts in a filled slot,
and index[e] is that slot's absolute id" -- computed with a plain Python loop.
A test that re-ran the shipped tensor expression against itself would have
agreed with the bug; that has already happened twice in this stage
(gather_kernel_probe checked a gather against an older gather).

    .venv/bin/python bench/pf_publish_test.py
"""
import torch

N_EXP = 256


def spec(sel, slot_base):
    """What publishing MUST produce, written as a loop over slots."""
    landed = [False] * N_EXP
    index = [0] * N_EXP
    for k, e in enumerate(sel):
        if e >= 0:
            landed[e] = True
            index[e] = slot_base + k
    return landed, index


def old(sel_t, slot_base, n_slots):
    landed = torch.zeros(N_EXP, dtype=torch.bool)
    index = torch.zeros(N_EXP, dtype=torch.int32)
    slot_ids = torch.arange(slot_base, slot_base + n_slots, dtype=torch.int32)
    valid = sel_t >= 0
    ids0 = sel_t.clamp_min(0)
    landed.scatter_(0, ids0, valid)
    index.scatter_(0, ids0, slot_ids)
    return landed, index


def new(sel_t, slot_base, n_slots):
    full_landed = torch.zeros(N_EXP + 1, dtype=torch.bool)
    full_index = torch.zeros(N_EXP + 1, dtype=torch.int32)
    slot_ids = torch.arange(slot_base, slot_base + n_slots, dtype=torch.int32)
    pad = torch.full((n_slots,), N_EXP, dtype=torch.int64)
    valid = sel_t >= 0
    ids0 = torch.where(sel_t >= 0, sel_t, pad)
    full_landed.scatter_(0, ids0, valid)
    full_index.scatter_(0, ids0, slot_ids)
    return full_landed[:N_EXP], full_index[:N_EXP]


def check(impl, sel, slot_base, n_slots):
    """True if impl matches the spec. index is only compared where landed."""
    want_l, want_i = spec(sel, slot_base)
    got_l, got_i = impl(torch.tensor(sel, dtype=torch.int64), slot_base, n_slots)
    for e in range(N_EXP):
        if bool(got_l[e]) != want_l[e]:
            return False
        if want_l[e] and int(got_i[e]) != want_i[e]:
            return False
    return True


def main():
    slot_base = 100

    # The named case from item 10: expert 0 fetched, the other slots empty.
    named = [0, -1, -1, -1]
    old_named = check(old, named, slot_base, 4)
    new_named = check(new, named, slot_base, 4)
    print(f"sel={named}   old matches spec: {old_named}   new: {new_named}")
    assert not old_named, "expected the OLD form to fail here; if it passes the test is wrong"
    assert new_named

    # Random sweep, so the fix is not judged on its motivating case alone.
    torch.manual_seed(0)
    n_old_fail = n_new_fail = 0
    trials = 4000
    for _ in range(trials):
        n_slots = int(torch.randint(1, 9, (1,)))
        n_fill = int(torch.randint(0, n_slots + 1, (1,)))
        picks = torch.randperm(N_EXP)[:n_fill].tolist()      # distinct, as the selector guarantees
        sel = picks + [-1] * (n_slots - n_fill)
        if not check(old, sel, slot_base, n_slots):
            n_old_fail += 1
        if not check(new, sel, slot_base, n_slots):
            n_new_fail += 1
    print(f"random {trials} trials:  old fails {n_old_fail} "
          f"({100*n_old_fail/trials:.2f}%)   new fails {n_new_fail}")
    assert n_new_fail == 0
    # Every old failure must involve expert 0 in a partly-empty slot set; if the
    # old form failed for some OTHER reason the diagnosis in item 10 is wrong.
    print("OK")


if __name__ == "__main__":
    main()
