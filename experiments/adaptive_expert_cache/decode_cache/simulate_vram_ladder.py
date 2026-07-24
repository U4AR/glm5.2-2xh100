#!/usr/bin/env python3
"""Simulate adaptive-cache speed vs VRAM budget (experts/layer N) from REAL
routing captures (experiments/expert_footprint_top2_vs_top8/runs/*.pt: per-token
top-8 ids+weights per MoE layer, several coding/math/git tasks).

Model (principled, batch=1 safe2 routing):
  - Only the GENUINE top-2 experts per token per layer can miss (ranks 2-7 are
    substituted with residents). CPU work per step is proportional to the number
    of missed top-2 activations = 2 * (1 - coverage(N)) per layer.
  - 1/tok_s = t0 + t_miss * (1 - coverage)  -- linear in miss rate.
  Calibrated on same-boot measurements (GPU_EXPERTS=96, MTP-d3, coherent):
    uniform  N=96  coverage 0.500 -> 28.2 tok/s
    adaptive N=96  coverage 0.897 -> 39.0 tok/s
  Check: predicts 43.3 at coverage 1.0; measured top0 (zero CPU + handshake
  skipped) = ~52 -> the remaining gap is the per-layer submit/sync handshake,
  consistent.

VRAM: expert bytes = N * 78 layers * 9.45 MiB/card (int4 W4AFP8, TP2-sharded).
Non-expert base (dense weights shard + KV pool + CUDA graphs + NSA/MLA bufs)
~= 14 GiB/card at the 82k-token KV config.
"""
import glob, json, os
import torch

RUNS = "/data/models/RunGLM/experiments/expert_footprint_top2_vs_top8/runs"
OUT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)

NUM_EXPERTS, NUM_LAYERS = 256, 78
MIB_PER_EXPERT_CARD = 9.45          # int4, per card (TP2 shard)
BASE_GIB_CARD = 14.0                # non-expert VRAM per card
# calibration (same boot, measured)
CAL = [(0.500, 28.2), (0.897, 39.0)]
(c1, s1), (c2, s2) = CAL
t_miss = (1 / s1 - 1 / s2) / ((1 - c1) - (1 - c2))
t0 = 1 / s2 - t_miss * (1 - c2)
TOP0_MEASURED = 52.0  # handshake also skipped -> above model's cov=1 point

# ---- aggregate genuine top-2 counts per layer from all captures ----
counts = torch.zeros(NUM_LAYERS + 1, NUM_EXPERTS, dtype=torch.float64)  # moe layers 3..80 -> index by layer-3
layer_ids = set()
files = sorted(glob.glob(f"{RUNS}/*.pt"))
tok_total = 0
for f in files:
    recs = torch.load(f, map_location="cpu")
    for layer, ids, w in recs:
        top2 = w.argsort(dim=-1, descending=True)[:, :2]
        sel = torch.gather(ids.long(), 1, top2).reshape(-1)
        li = layer - 3
        counts[li].index_add_(0, sel, torch.ones(sel.numel(), dtype=torch.float64))
        layer_ids.add(layer)
        if layer == 3:
            tok_total += ids.shape[0]
L = len(layer_ids)
counts = counts[:L]
print(f"files={len(files)} moe_layers={L} tokens~{tok_total} total_top2_events={counts.sum():.0f}")

sorted_share = torch.sort(counts, dim=1, descending=True).values
sorted_share = sorted_share / sorted_share.sum(dim=1, keepdim=True).clamp(min=1)
cum = sorted_share.cumsum(dim=1)                     # [L,256] coverage of hottest-N
cov_mean = cum.mean(0); cov_min = cum.min(0).values; cov_max = cum.max(0).values

def tps(cov): return 1.0 / (t0 + t_miss * (1.0 - cov))

Ns = [16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256]
rows = []
for N in Ns:
    cov = cov_mean[N - 1].item()
    vram = N * NUM_LAYERS * MIB_PER_EXPERT_CARD / 1024 + BASE_GIB_CARD
    rows.append((N, cov, vram, tps(cov)))
    print(f"N={N:4d}  cov={cov:.3f}  VRAM/card={vram:6.1f} GiB  pred={tps(cov):5.1f} tok/s")

# concentration stats
for q in (0.50, 0.80, 0.90, 0.95, 0.99):
    need = (cum >= q).float().argmax(dim=1) + 1
    print(f"experts/layer for {q:.0%} of top-2 traffic: mean={need.float().mean():.0f} "
          f"min={need.min().item()} max={need.max().item()}")

json.dump({"rows": rows, "t0": t0, "t_miss": t_miss}, open(f"{OUT}/vram_ladder.json", "w"), indent=1)

# ---------------- figures ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, NUM_EXPERTS + 1)

# Fig 1: usage distribution
fig, ax = plt.subplots(figsize=(8, 5))
for li, lab in [(10, "layer 13"), (40, "layer 43"), (70, "layer 73")]:
    ax.plot(x, sorted_share[li].numpy(), lw=1, alpha=0.6, label=lab)
ax.plot(x, sorted_share.mean(0).numpy(), "k", lw=2.2, label="mean over 78 layers")
ax.set_yscale("log"); ax.set_xlabel("expert rank within layer (sorted by usage)")
ax.set_ylabel("share of genuine top-2 firings")
ax.set_title("GLM-5.2 expert usage is Zipf-like: a small head carries most traffic")
ax.axvline(96, color="tab:red", ls="--", lw=1); ax.text(98, 2e-2, "N=96 (current)", color="tab:red")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(f"{FIGS}/fig1_usage_distribution.png", dpi=140)

# Fig 2: coverage vs N
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, cov_mean.numpy(), "k", lw=2, label="mean coverage")
ax.fill_between(x, cov_min.numpy(), cov_max.numpy(), alpha=0.2, label="min–max across layers")
for q in (0.8, 0.9, 0.95):
    n = int((cov_mean >= q).float().argmax()) + 1
    ax.plot([n], [q], "o", color="tab:orange")
    ax.annotate(f"{q:.0%} @ N={n}", (n, q), textcoords="offset points", xytext=(8, -12))
ax.axvline(96, color="tab:red", ls="--", lw=1)
ax.set_xlabel("resident experts per layer (N)"); ax.set_ylabel("top-2 traffic coverage")
ax.set_title("Hottest-N coverage of genuine top-2 routing"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIGS}/fig2_coverage_vs_N.png", dpi=140)

# Fig 3: predicted tok/s vs VRAM
Nfull = np.arange(8, 257)
covf = cov_mean.numpy()[Nfull - 1]
vramf = Nfull * NUM_LAYERS * MIB_PER_EXPERT_CARD / 1024 + BASE_GIB_CARD
tpsf = 1.0 / (t0 + t_miss * (1 - covf))
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(vramf, tpsf, lw=2.2, color="tab:blue", label="model (oracle-converged cache)")
meas = [(96, 39.0, "adaptive @96 (measured 38–40)"), (96, 28.2, "uniform @96 (measured)")]
ax.plot([96 * 78 * 9.45 / 1024 + 14], [39.0], "o", color="tab:green", ms=9)
ax.annotate("adaptive@96: 38–40 meas.", (96 * 78 * 9.45 / 1024 + 14, 39), xytext=(-160, 10), textcoords="offset points", color="tab:green")
ax.plot([96 * 78 * 9.45 / 1024 + 14], [28.2], "s", color="tab:gray", ms=8)
ax.annotate("uniform@96: 28.2 meas.", (96 * 78 * 9.45 / 1024 + 14, 28.2), xytext=(-160, -14), textcoords="offset points", color="tab:gray")
ax.axhline(TOP0_MEASURED, color="tab:red", ls=":", lw=1.5)
ax.text(20, TOP0_MEASURED + 0.6, "top0 all-substituted ceiling ~52 (handshake skipped)", color="tab:red", fontsize=9)
for N in (24, 48, 96, 160, 256):
    v = N * 78 * 9.45 / 1024 + 14
    t = 1.0 / (t0 + t_miss * (1 - cov_mean[N - 1].item()))
    ax.annotate(f"N={N}", (v, t), textcoords="offset points", xytext=(4, -14), fontsize=9)
    ax.plot([v], [t], ".", color="tab:blue")
ax.set_xlabel("VRAM per card (GiB): experts + ~14 GiB base"); ax.set_ylabel("decode tok/s (MTP-d3, coherent)")
ax.set_title("Predicted speed vs VRAM budget — 2×H100 hybrid, adaptive cache converged")
ax.legend(loc="lower right"); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(f"{FIGS}/fig3_tps_vs_vram.png", dpi=140)

# Fig 4: tok/s vs N + coverage twin
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(Nfull, tpsf, lw=2.2, color="tab:blue")
ax.set_xlabel("resident experts per layer (N)"); ax.set_ylabel("decode tok/s", color="tab:blue")
ax2 = ax.twinx(); ax2.plot(Nfull, covf, color="tab:orange", lw=1.5, ls="--")
ax2.set_ylabel("top-2 coverage", color="tab:orange")
ax.set_title("Diminishing returns: speed & coverage vs experts/layer")
ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(f"{FIGS}/fig4_tps_vs_N.png", dpi=140)
print("figs saved to", FIGS)
