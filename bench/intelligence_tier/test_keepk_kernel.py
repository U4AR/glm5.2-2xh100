"""Offline unit test for the per-token keep_k routing kernel logic.

Reproduces _kt_topk_experiment (sub mode) standalone on CPU and checks:
  - K==E reproduces the ORIGINAL routing exactly (baseline parity).
  - a fixed K matches the legacy "keep top-K, substitute rest" behavior.
  - mixed per-token K rows are each handled independently.
  - encoded adaptive K chooses from per-row router mass and speed level.
"""
import torch

torch.manual_seed(0)
NEG = torch.finfo(torch.float32).min
ADAPT_BASE = -1000
ADAPT_DEFAULT_SPEED = 50


def adaptive_keep_k(w, E, speed_level=None):
    """Mirror the adaptive router-mass rule in deepseek_v2.py."""
    wf = w.float()
    ws, _ = torch.sort(wf, dim=-1, descending=True)
    denom = torch.clamp(ws.sum(dim=-1), min=1e-9)
    mass2 = ws[:, : min(2, E)].sum(dim=-1) / denom
    mass4 = ws[:, : min(4, E)].sum(dim=-1) / denom
    if speed_level is None:
        speed_raw = torch.full_like(mass2, float(ADAPT_DEFAULT_SPEED))
    else:
        speed_raw = speed_level.to(dtype=torch.float32)
    speed_raw = torch.clamp(speed_raw, 0.0, 100.0)
    speed = speed_raw / 100.0
    top2_threshold = 0.60 - 0.20 * speed
    top4_threshold = 0.80 - 0.20 * speed
    k0_ramp = torch.clamp((speed_raw - 75.0) / 25.0, 0.0, 1.0)
    k0_threshold = 0.92 - 0.42 * k0_ramp
    k2 = torch.full((w.shape[0],), min(2, E), dtype=torch.long)
    k0 = torch.zeros_like(k2)
    k4 = torch.full_like(k2, min(4, E))
    k8 = torch.full_like(k2, E)
    nonzero_k = torch.where(
        mass2 >= top2_threshold,
        k2,
        torch.where(mass4 >= top4_threshold, k4, k8),
    )
    adaptive_k = torch.where(mass2 >= k0_threshold, k0, nonzero_k)
    return torch.where(
        speed_raw <= 0.0,
        k8,
        torch.where(speed_raw >= 100.0, k0, adaptive_k),
    )


def kernel_sub(w, ids, scores, gpu_mask, K_vec, E, default_k=2):
    """Vectorized per-token version (mirrors deepseek_v2._kt_topk_experiment)."""
    T = w.shape[0]
    raw_K_vec = K_vec.to(dtype=torch.long)
    adaptive_default_mask = raw_K_vec == -2
    adaptive_level_mask = raw_K_vec <= ADAPT_BASE
    adaptive_mask = adaptive_default_mask | adaptive_level_mask
    K_vec = torch.where(
        (raw_K_vec < 0) | (raw_K_vec > E),
        torch.full_like(raw_K_vec, min(default_k, E)),
        raw_K_vec,
    )
    speed_level = torch.where(
        adaptive_level_mask,
        torch.clamp((-raw_K_vec) + ADAPT_BASE, 0, 100),
        torch.full_like(raw_K_vec, ADAPT_DEFAULT_SPEED),
    )
    K_vec = torch.where(adaptive_mask, adaptive_keep_k(w, E, speed_level), K_vec)

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

# --- Test 4: adaptive K sentinel chooses K=2/K=4/K=8 from router mass ---
adapt_w = torch.tensor([
    [0.35, 0.20, 0.09, 0.08, 0.07, 0.07, 0.07, 0.07],  # top2=.55 -> K=2
    [0.25, 0.20, 0.16, 0.14, 0.08, 0.07, 0.06, 0.04],  # top4=.75 -> K=4
    [0.18, 0.16, 0.14, 0.12, 0.11, 0.10, 0.10, 0.09],  # diffuse -> K=8
])
adapt_ids = ids[:3]
adapt_scores = scores[:3]
expected_K = torch.tensor([2, 4, 8])
got_K = adaptive_keep_k(adapt_w, E)
assert torch.equal(got_K, expected_K), f"adaptive K mismatch: {got_K} != {expected_K}"
nid, nw = kernel_sub(adapt_w, adapt_ids, adapt_scores, gpu_mask, torch.full((3,), -2), E)
for t, K in enumerate(expected_K.tolist()):
    if K == E:
        kid, kw = as_set(nid[t], nw[t])
        rid, rw = as_set(adapt_ids[t], adapt_w[t])
    else:
        rid, rw = legacy_sub_one_row(adapt_w[t], adapt_ids[t], adapt_scores[t], gpu_mask, K, E)
        kid, kw = as_set(nid[t], nw[t])
    assert torch.equal(kid, rid), f"adaptive row{t} K={K} ids mismatch"
    assert torch.allclose(kw, rw, atol=1e-5), f"adaptive row{t} K={K} w mismatch"
assert torch.allclose(nw.sum(-1), torch.ones(3), atol=1e-5)
print("PASS test4: adaptive default chooses K=2/K=4/K=8 from router mass")

# --- Test 5: mixed fixed/adaptive rows coexist in one vectorized batch ---
Kmix_adapt = torch.tensor([8, ADAPT_BASE - 50, 2, ADAPT_BASE - 50, 4, ADAPT_BASE - 50])
mix_w = w.clone()
mix_w[1] = adapt_w[0]
mix_w[3] = adapt_w[1]
mix_w[5] = adapt_w[2]
nid, nw = kernel_sub(mix_w, ids, scores, gpu_mask, Kmix_adapt, E)
resolved = torch.tensor([8, 2, 2, 4, 4, 8])
for t, K in enumerate(resolved.tolist()):
    if K == E:
        kid, kw = as_set(nid[t], nw[t])
        rid, rw = as_set(ids[t], mix_w[t])
    else:
        rid, rw = legacy_sub_one_row(mix_w[t], ids[t], scores[t], gpu_mask, K, E)
        kid, kw = as_set(nid[t], nw[t])
    assert torch.equal(kid, rid), f"mixed adaptive row{t} K={K} ids mismatch"
    assert torch.allclose(kw, rw, atol=1e-5), f"mixed adaptive row{t} K={K} w mismatch"
assert torch.allclose(nw.sum(-1), torch.ones(T), atol=1e-5)
for t in range(T):
    assert len(set(nid[t].tolist())) == E, f"mixed adaptive row{t} has duplicate experts!"
print("PASS test5: mixed fixed/adaptive K rows are valid and normalized")

# --- Test 6: adaptive speed level controls quality/speed threshold bias ---
borderline_w = torch.tensor([
    [0.24, 0.21, 0.11, 0.09, 0.08, 0.08, 0.10, 0.09],  # top2=.45, top4=.65
])
speed0 = torch.tensor([0])
speed50 = torch.tensor([50])
speed100 = torch.tensor([100])
assert torch.equal(adaptive_keep_k(borderline_w, E, speed0), torch.tensor([8]))
assert torch.equal(adaptive_keep_k(borderline_w, E, speed50), torch.tensor([8]))
assert torch.equal(adaptive_keep_k(borderline_w, E, speed100), torch.tensor([0]))
for level, expected in ((0, 8), (50, 8), (100, 0)):
    nid, nw = kernel_sub(
        borderline_w,
        ids[:1],
        scores[:1],
        gpu_mask,
        torch.tensor([ADAPT_BASE - level]),
        E,
    )
    if expected == E:
        kid, kw = as_set(nid[0], nw[0])
        rid, rw = as_set(ids[0], borderline_w[0])
    else:
        rid, rw = legacy_sub_one_row(borderline_w[0], ids[0], scores[0], gpu_mask, expected, E)
        kid, kw = as_set(nid[0], nw[0])
    assert torch.equal(kid, rid), f"adapt{level} ids mismatch"
    assert torch.allclose(kw, rw, atol=1e-5), f"adapt{level} weights mismatch"
print("PASS test6: adaptive speed level reaches K=0 at adapt100")

# --- Test 7: adapt0 is exact full K=8 even for confident router rows ---
confident_w = torch.tensor([
    [0.42, 0.25, 0.10, 0.08, 0.05, 0.04, 0.03, 0.03],  # top2=.67
])
assert torch.equal(adaptive_keep_k(confident_w, E, speed0), torch.tensor([8]))
nid, nw = kernel_sub(
    confident_w,
    ids[:1],
    scores[:1],
    gpu_mask,
    torch.tensor([ADAPT_BASE]),
    E,
)
assert torch.equal(nid[0], ids[0]), "adapt0 changed ids despite K=8 endpoint"
assert torch.allclose(nw[0], confident_w[0], atol=1e-6), "adapt0 changed weights"
print("PASS test7: adapt0 is exact K=8/full-routing endpoint")

# --- Test 8: high speed levels introduce confident-row K=0 gradually ---
high_speed_w = torch.tensor([
    [0.39, 0.31, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03],  # top2=.70
])
assert torch.equal(adaptive_keep_k(high_speed_w, E, torch.tensor([85])), torch.tensor([2]))
assert torch.equal(adaptive_keep_k(high_speed_w, E, torch.tensor([90])), torch.tensor([0]))
assert torch.equal(adaptive_keep_k(high_speed_w, E, torch.tensor([95])), torch.tensor([0]))
print("PASS test8: high speed adaptive levels can choose K=0 before adapt100")

print("\nALL KERNEL TESTS PASSED")
