"""Analyze a KT_DUMP_TOPK routing trace: list of (layer_idx, int32 tensor [T, top_k]).
Computes, per decode step (T==1 rows), expert working-set + temporal reuse, which
bound how much a VRAM expert cache / static hot-placement would help.
"""
import sys, collections
import torch

path = sys.argv[1]
trace = torch.load(path, map_location="cpu", weights_only=False)
print(f"loaded {len(trace)} (layer,topk) records from {path}")

# Group selections by layer. Each record's tensor is [T, top_k]; decode steps have T==1.
by_layer = collections.defaultdict(list)   # layer -> list of selected-id sets (per step)
by_layer_top2 = collections.defaultdict(list)
for lid, t in trace:
    t = t.view(-1, t.shape[-1])
    for row in t:                       # one row per token/step
        ids = row.tolist()
        by_layer[lid].append(ids)
        by_layer_top2[lid].append(ids[:2])  # router order ~ descending weight

layers = sorted(by_layer)
top_k = len(by_layer[layers[0]][0])
print(f"layers traced: {len(layers)}  top_k={top_k}")

# Aggregate across layers
ws_sizes, reuse_rates, top2_reuse = [], [], []
n_steps_total = 0
for lid in layers:
    steps = by_layer[lid]
    n = len(steps)
    if n < 2:
        continue
    n_steps_total = max(n_steps_total, n)
    distinct = set()
    reuse_hits = reuse_tot = 0
    t2_hits = t2_tot = 0
    prev = None
    for i, ids in enumerate(steps):
        s = set(ids)
        distinct |= s
        if prev is not None:
            reuse_hits += len(s & prev); reuse_tot += len(s)
        prev = s
    # top-2 temporal reuse
    prev2 = None
    for ids2 in by_layer_top2[lid]:
        s2 = set(ids2)
        if prev2 is not None:
            t2_hits += len(s2 & prev2); t2_tot += len(s2)
        prev2 = s2
    ws_sizes.append(len(distinct))
    reuse_rates.append(reuse_hits / max(reuse_tot, 1))
    top2_reuse.append(t2_hits / max(t2_tot, 1))

import statistics as st
print(f"\ndecode steps/layer ~ {n_steps_total}")
print(f"working-set size per layer (distinct experts over the whole decode):")
print(f"   min={min(ws_sizes)} median={int(st.median(ws_sizes))} max={max(ws_sizes)} (of 256)")
print(f"top-8 temporal reuse (frac of a step's experts also used at prev step):")
print(f"   mean={st.mean(reuse_rates):.3f}  median={st.median(reuse_rates):.3f}")
print(f"top-2 temporal reuse (frac of the 2 dominant experts repeated next step):")
print(f"   mean={st.mean(top2_reuse):.3f}  median={st.median(top2_reuse):.3f}")

# LRU cache-miss model: for a per-layer VRAM cache of size C, simulate misses
for C in (32, 64, 104, 128):
    miss = tot = 0
    for lid in layers:
        cache = collections.OrderedDict()
        for ids in by_layer[lid]:
            for e in ids:
                tot += 1
                if e in cache:
                    cache.move_to_end(e)
                else:
                    miss += 1
                    cache[e] = 1
                    if len(cache) > C:
                        cache.popitem(last=False)
    print(f"LRU cache C={C:3d}/layer: miss rate {miss/tot:.3f}  "
          f"(-> {miss/tot*top_k:.2f} transfers/layer/token avg)")
