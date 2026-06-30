"""Offline unit test for the per-token keep_k routing kernel logic.

Reproduces _kt_topk_experiment (sub mode) standalone on CPU and checks:
  - K==E reproduces the ORIGINAL routing exactly (baseline parity).
  - a fixed K matches the legacy "keep top-K, substitute rest" behavior.
  - mixed per-token K rows are each handled independently.
"""
import torch

torch.manual_seed(0)
NEG = torch.finfo(torch.float32).min


def kernel_sub(w, ids, scores, gpu_mask, K_vec, E):
    """Vectorized per-token version (mirrors deepseek_v2._kt_topk_experiment)."""
    T = w.shape[0]
    order = torch.argsort(w, dim=-1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(E).expand(T, E))
    keep_mask = rank < K_vec[:, None]

    allow = gpu_mask.bool().view(1, -1).expand(T, -1)
    s = torch.where(allow, scores, NEG).clone()
    s.scatter_(1, ids.long(), NEG)
    fill_s, fill_ids = torch.topk(s, E, dim=-1)
    fill_w = torch.sigmoid(fill_s)
    drop_idx = torch.clamp(rank - K_vec[:, None], min=0)
    sel_fill_ids = torch.gather(fill_ids, 1, drop_idx)
    sel_fill_w = torch.gather(fill_w, 1, drop_idx)
    new_ids = torch.where(keep_mask, ids, sel_fill_ids.to(ids.dtype))
    new_w = torch.where(keep_mask, w, sel_fill_w.to(w.dtype))
    new_w = new_w / torch.clamp(new_w.sum(dim=-1, keepdim=True), min=1e-9)
    return new_ids, new_w


def legacy_sub_one_row(w_row, ids_row, scores_row, gpu_mask, K, E):
    """Reference: the OLD code path for a single row at fixed K."""
    order = torch.argsort(w_row, descending=True)
    keep_pos = order[:K]
    kept_w = w_row[keep_pos]
    kept_ids = ids_row[keep_pos]
    nfill = E - K
    allow = gpu_mask.bool()
    s = torch.where(allow, scores_row, NEG).clone()
    s[ids_row.long()] = NEG
    fill_s, fill_ids = torch.topk(s, nfill)
    fill_w = torch.sigmoid(fill_s)
    new_ids = torch.cat([kept_ids, fill_ids])
    new_w = torch.cat([kept_w, fill_w])
    new_w = new_w / torch.clamp(new_w.sum(), min=1e-9)
    # sort by id for order-independent comparison
    si = torch.argsort(new_ids)
    return new_ids[si], new_w[si]


def as_set(ids, w):
    si = torch.argsort(ids)
    return ids[si], w[si]


N_EXPERTS, E = 256, 8
T = 6
# random routed selection per token
ids = torch.stack([torch.randperm(N_EXPERTS)[:E] for _ in range(T)])
w = torch.rand(T, E)
w = w / w.sum(-1, keepdim=True)
scores = torch.randn(T, N_EXPERTS)
gpu_mask = torch.zeros(N_EXPERTS)
gpu_mask[:104] = 1  # 104 GPU-resident experts

# --- Test 1: K==E is identity (baseline parity) ---
Kfull = torch.full((T,), E)
nid, nw = kernel_sub(w, ids, scores, gpu_mask, Kfull, E)
assert torch.equal(nid, ids), "K==E changed ids!"
assert torch.allclose(nw, w, atol=1e-6), "K==E changed weights!"
print("PASS test1: K==E is exact baseline identity")

# --- Test 2: fixed K matches legacy per-row reference ---
for K in (0, 2, 4):
    Kv = torch.full((T,), K)
    nid, nw = kernel_sub(w, ids, scores, gpu_mask, Kv, E)
    for t in range(T):
        rid, rw = legacy_sub_one_row(w[t], ids[t], scores[t], gpu_mask, K, E)
        kid, kw = as_set(nid[t], nw[t])
        assert torch.equal(kid, rid), f"K={K} row{t} ids mismatch\n{kid}\n{rid}"
        assert torch.allclose(kw, rw, atol=1e-5), f"K={K} row{t} w mismatch"
    print(f"PASS test2: fixed K={K} matches legacy reference")

# --- Test 3: mixed per-token K, each row independent ---
Kmix = torch.tensor([8, 4, 2, 0, 2, 4])
nid, nw = kernel_sub(w, ids, scores, gpu_mask, Kmix, E)
for t in range(T):
    K = int(Kmix[t])
    if K == E:
        kid, kw = as_set(nid[t], nw[t])
        rid, rw = as_set(ids[t], w[t])
    else:
        rid, rw = legacy_sub_one_row(w[t], ids[t], scores[t], gpu_mask, K, E)
        kid, kw = as_set(nid[t], nw[t])
    assert torch.equal(kid, rid), f"mixed row{t} K={K} ids mismatch"
    assert torch.allclose(kw, rw, atol=1e-5), f"mixed row{t} K={K} w mismatch"
# all fills/keeps must be valid distinct experts, weights sum to 1
assert torch.allclose(nw.sum(-1), torch.ones(T), atol=1e-5)
for t in range(T):
    assert len(set(nid[t].tolist())) == E, f"row{t} has duplicate experts!"
print("PASS test3: mixed per-token K rows each correct, no dup experts, weights normalized")

print("\nALL KERNEL TESTS PASSED")
