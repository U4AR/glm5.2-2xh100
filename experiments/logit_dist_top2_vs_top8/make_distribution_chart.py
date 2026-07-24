#!/usr/bin/env python3
"""Render an overlaid SVG histogram for top2/top8 sampled-token logprobs."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "latest"
RECORDS = RUN / "records.jsonl"
OUT = RUN / "sampled_logprob_distribution.svg"


def load() -> dict[int, list[float]]:
    vals = {2: [], 8: []}
    for line in RECORDS.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        lp = rec.get("logprob")
        tier = rec.get("tier")
        if tier in vals and isinstance(lp, (int, float)):
            vals[tier].append(lp)
    return vals


def hist(xs: list[float], lo: float, hi: float, bins: int) -> list[int]:
    width = (hi - lo) / bins
    out = [0] * bins
    for x in xs:
        i = min(bins - 1, max(0, int((x - lo) / width)))
        out[i] += 1
    return out


def esc(s: object) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> int:
    vals = load()
    all_vals = vals[2] + vals[8]
    bins = 18
    lo = math.floor((min(all_vals) - 0.02) * 10) / 10
    hi = 0.02
    h2 = hist(vals[2], lo, hi, bins)
    h8 = hist(vals[8], lo, hi, bins)
    max_count = max(h2 + h8)

    w, h = 1120, 680
    ml, mr, mt, mb = 90, 40, 92, 92
    pw, ph = w - ml - mr, h - mt - mb
    bw = pw / bins

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}.small{font-size:14px}.axis{font-size:13px;fill:#475569}.title{font-size:28px;font-weight:700}.sub{font-size:15px;fill:#526173}.legend{font-size:15px;font-weight:650}</style>',
        '<text x="90" y="44" class="title">Sampled Output-Token Logprob Distribution</text>',
        '<text x="90" y="72" class="sub">GLM5.2-top2 vs GLM5.2-top8, 96 paired prompts, first visible token, temperature 0</text>',
    ]

    # Grid and y-axis labels.
    for tick in range(0, max_count + 1, max(1, math.ceil(max_count / 5))):
        y = mt + ph - (tick / max_count) * ph
        parts.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="#dbe3ee" stroke-width="1"/>')
        parts.append(f'<text x="{ml-14}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick}</text>')

    # Bars: draw top8 first, then top2 over it with opacity.
    for i, c in enumerate(h8):
        x = ml + i * bw + 5
        bh = (c / max_count) * ph
        y = mt + ph - bh
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-10:.1f}" height="{bh:.1f}" fill="#2563eb" opacity="0.55" rx="3"><title>top8 bin {i}: {c}</title></rect>')
    for i, c in enumerate(h2):
        x = ml + i * bw + 14
        bh = (c / max_count) * ph
        y = mt + ph - bh
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-28:.1f}" height="{bh:.1f}" fill="#f97316" opacity="0.62" rx="3"><title>top2 bin {i}: {c}</title></rect>')

    # Axes.
    parts.append(f'<line x1="{ml}" x2="{w-mr}" y1="{mt+ph}" y2="{mt+ph}" stroke="#334155" stroke-width="1.4"/>')
    parts.append(f'<line x1="{ml}" x2="{ml}" y1="{mt}" y2="{mt+ph}" stroke="#334155" stroke-width="1.4"/>')

    # X labels every third bin.
    step = (hi - lo) / bins
    for i in range(0, bins + 1, 3):
        x = ml + i * bw
        label = lo + i * step
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{mt+ph}" y2="{mt+ph+6}" stroke="#334155"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt+ph+26}" text-anchor="middle" class="axis">{label:.2f}</text>')

    # Axis titles and legend.
    parts.append(f'<text x="{ml+pw/2:.1f}" y="{h-24}" text-anchor="middle" class="small">sampled token logprob, closer to 0 means higher confidence</text>')
    parts.append(f'<text x="24" y="{mt+ph/2:.1f}" transform="rotate(-90 24 {mt+ph/2:.1f})" text-anchor="middle" class="small">count of prompts</text>')
    parts.append('<rect x="790" y="34" width="18" height="18" fill="#2563eb" opacity="0.55" rx="3"/>')
    parts.append('<text x="816" y="49" class="legend">top8</text>')
    parts.append('<rect x="884" y="34" width="18" height="18" fill="#f97316" opacity="0.62" rx="3"/>')
    parts.append('<text x="910" y="49" class="legend">top2</text>')

    note = "Note: server top_logprobs payload exposed one packed candidate, so this chart uses sampled-token logprobs."
    parts.append(f'<text x="90" y="{h-52}" class="axis">{esc(note)}</text>')
    parts.append("</svg>")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
