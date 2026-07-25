#!/usr/bin/env python3
"""Low-VRAM figure: measured VRAM footprint + tok/s, plus the small-GPU haircut.

Two panels:
  (left)  measured VRAM/card vs held-out tok/s on 2xH100 (footprint-only sweep)
  (right) the SAME points re-priced by HBM bandwidth onto real 2x24GiB cards
          (the compute caveat: low-VRAM runs still used full H100 compute).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")

# measured 2026-07-25 (warm-start, MAX_TOTAL_TOKENS=4096, MEM_FRACTION=0.60)
# N: (VRAM GiB/card, coverage, held-out tok/s on 2xH100)
DATA = {
    4:  (19.9, 0.16, 19.4),
    8:  (23.1, 0.26, 27.1),
    16: (28.7, 0.43, 30.3),
    32: (40.1, 0.61, 32.3),
    96: (54.0, 0.89, 38.7),   # from the main ladder (MEM_FRACTION 0.85)
}
Ns = sorted(DATA)
vram = [DATA[n][0] for n in Ns]
tps = [DATA[n][2] for n in Ns]

H100_BW = 3350.0
F_GPU = 0.70
CARDS = {"2xH100 (measured)": 3350.0, "2x RTX 4090": 1008.0, "2x A10": 600.0, "2x L4": 300.0}

def reprice(t, bw):
    step = 1.0 / t
    return 1.0 / (step * F_GPU * (H100_BW / bw) + step * (1 - F_GPU))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

axL.plot(vram, tps, "o-", color="tab:blue", lw=2, ms=8)
for n in Ns:
    axL.annotate(f"N={n}\n({DATA[n][1]:.2f} cov)", (DATA[n][0], DATA[n][2]),
                 textcoords="offset points", xytext=(6, -18), fontsize=8)
axL.axvspan(0, 17.5, color="tab:red", alpha=0.08)
axL.axvline(17.5, color="tab:red", ls="--", lw=1)
axL.text(17.8, 21, "≈17.5 GiB/card base floor\n(dense trunk, TP2 — N→0 can't\ngo below; 12 GiB unreachable)",
         color="tab:red", fontsize=8, va="center")
axL.set_xlabel("VRAM per card (GiB, measured)"); axL.set_ylabel("held-out decode tok/s")
axL.set_title("Footprint sweep on 2×H100 (compute UNthrottled)")
axL.set_xlim(12, 58); axL.grid(alpha=0.3)

for name, bw in CARDS.items():
    y = [reprice(DATA[n][2], bw) for n in Ns]
    style = "o-" if "H100" in name else "s--"
    axR.plot(vram, y, style, lw=1.8, ms=6, label=name)
axR.axvline(24, color="gray", ls=":", lw=1); axR.text(24.3, 5, "24 GiB\ncard limit", fontsize=8, color="gray")
axR.set_xlabel("VRAM per card (GiB)"); axR.set_ylabel("projected decode tok/s")
axR.set_title("Same points re-priced by HBM bandwidth\n(first-order caveat: BW-only, UPPER bound)")
axR.set_xlim(12, 58); axR.grid(alpha=0.3); axR.legend(fontsize=8)

fig.suptitle("Low-VRAM regime — adaptive hybrid cache (GLM-5.2, TP2)", fontsize=13)
fig.tight_layout()
out = os.path.join(FIGS, "fig5_low_vram.png")
fig.savefig(out, dpi=140)
print("saved", out)
