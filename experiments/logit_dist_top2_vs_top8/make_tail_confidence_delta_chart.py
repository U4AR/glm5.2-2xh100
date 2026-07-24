#!/usr/bin/env python3
"""Render confidence deltas for low/high baseline-logit token samples.

The x-axis contains selected token samples from the bottom and top tails by the
top8 sampled-token logprob.  The y-axis is the confidence change:
P_top2(sampled token) - P_top8(sampled token).
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "latest"
RECORDS = RUN / "records.jsonl"
OUT = RUN / "tail_confidence_delta.svg"
TAIL_N = 20


def esc(s: object) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def load_pairs() -> list[dict]:
    by_prompt: dict[int, dict[int, dict]] = {}
    for line in RECORDS.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        by_prompt.setdefault(rec["prompt_id"], {})[rec["tier"]] = rec

    pairs = []
    for prompt_id, tiers in sorted(by_prompt.items()):
        if 2 not in tiers or 8 not in tiers:
            continue
        top2 = tiers[2]
        top8 = tiers[8]
        p2 = top2.get("prob")
        p8 = top8.get("prob")
        lp8 = top8.get("logprob")
        if not all(isinstance(x, (int, float)) for x in (p2, p8, lp8)):
            continue
        pairs.append({
            "prompt_id": prompt_id,
            "prompt": top8["prompt"],
            "top2_token": top2.get("token"),
            "top8_token": top8.get("token"),
            "top8_logprob": lp8,
            "top2_prob": p2,
            "top8_prob": p8,
            "delta": p2 - p8,
        })
    return pairs


def main() -> int:
    pairs = load_pairs()
    ordered = sorted(pairs, key=lambda p: p["top8_logprob"])
    low = ordered[:TAIL_N]
    high = ordered[-TAIL_N:]
    selected = low + high

    w, h = 1280, 720
    ml, mr, mt, mb = 88, 42, 92, 150
    pw, ph = w - ml - mr, h - mt - mb
    ymin, ymax = -0.75, 0.35

    def x_at(i: int) -> float:
        return ml + (i + 0.5) * pw / len(selected)

    def y_at(v: float) -> float:
        return mt + (ymax - v) / (ymax - ymin) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}.title{font-size:28px;font-weight:700}.sub{font-size:15px;fill:#526173}.axis{font-size:13px;fill:#475569}.small{font-size:14px}.legend{font-size:15px;font-weight:650}</style>',
        '<text x="88" y="44" class="title">Confidence Change on Low vs High Baseline-Logit Tokens</text>',
        f'<text x="88" y="72" class="sub">Bottom {TAIL_N} and top {TAIL_N} token samples by top8 logprob; y = P(top2 sampled token) - P(top8 sampled token)</text>',
    ]

    # Background bands.
    split_x = ml + TAIL_N * pw / len(selected)
    parts.append(f'<rect x="{ml}" y="{mt}" width="{split_x-ml:.1f}" height="{ph}" fill="#fee2e2" opacity="0.34"/>')
    parts.append(f'<rect x="{split_x:.1f}" y="{mt}" width="{w-mr-split_x:.1f}" height="{ph}" fill="#dcfce7" opacity="0.34"/>')
    parts.append(f'<text x="{ml+12}" y="{mt+24}" class="legend" fill="#991b1b">lowest top8 logprob tail</text>')
    parts.append(f'<text x="{split_x+12:.1f}" y="{mt+24}" class="legend" fill="#166534">highest top8 logprob tail</text>')

    # Grid and y labels.
    for tick in (-0.75, -0.50, -0.25, 0.0, 0.25):
        y = y_at(tick)
        stroke = "#0f172a" if tick == 0 else "#dbe3ee"
        sw = "1.5" if tick == 0 else "1"
        parts.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="{sw}"/>')
        parts.append(f'<text x="{ml-14}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick:+.2f}</text>')

    # Bars.
    bw = pw / len(selected) * 0.74
    zero = y_at(0.0)
    for i, p in enumerate(selected):
        x = x_at(i) - bw / 2
        yv = y_at(p["delta"])
        y = min(zero, yv)
        bh = abs(zero - yv)
        color = "#f97316" if p["delta"] < 0 else "#16a34a"
        title = (
            f"id={p['prompt_id']} prompt={p['prompt']} "
            f"top2={p['top2_token']} p2={p['top2_prob']:.4f}; "
            f"top8={p['top8_token']} p8={p['top8_prob']:.4f}; delta={p['delta']:+.4f}"
        )
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.82" rx="3"><title>{esc(title)}</title></rect>')

    # Axes.
    parts.append(f'<line x1="{ml}" x2="{w-mr}" y1="{mt+ph}" y2="{mt+ph}" stroke="#334155" stroke-width="1.4"/>')
    parts.append(f'<line x1="{ml}" x2="{ml}" y1="{mt}" y2="{mt+ph}" stroke="#334155" stroke-width="1.4"/>')
    parts.append(f'<line x1="{split_x:.1f}" x2="{split_x:.1f}" y1="{mt}" y2="{mt+ph}" stroke="#64748b" stroke-dasharray="6 6"/>')

    # X-axis selected prompt labels.
    for i, p in enumerate(selected):
        x = x_at(i)
        label = f"{p['prompt_id']}"
        parts.append(f'<text x="{x:.1f}" y="{mt+ph+22}" text-anchor="middle" class="axis" transform="rotate(-55 {x:.1f} {mt+ph+22})">{label}</text>')

    # Legend.
    parts.append('<rect x="826" y="34" width="18" height="18" fill="#16a34a" opacity="0.82" rx="3"/>')
    parts.append('<text x="852" y="49" class="legend">top2 more confident</text>')
    parts.append('<rect x="1030" y="34" width="18" height="18" fill="#f97316" opacity="0.82" rx="3"/>')
    parts.append('<text x="1056" y="49" class="legend">top2 less confident</text>')

    parts.append(f'<text x="{ml+pw/2:.1f}" y="{h-42}" text-anchor="middle" class="small">selected token sample id, sorted by top8 logprob within each tail</text>')
    parts.append(f'<text x="24" y="{mt+ph/2:.1f}" transform="rotate(-90 24 {mt+ph/2:.1f})" text-anchor="middle" class="small">confidence delta: P_top2 - P_top8</text>')
    parts.append(f'<text x="88" y="{h-18}" class="axis">Token labels and prompt text are in SVG hover tooltips. Negative bars mean top2 assigned lower confidence than top8.</text>')
    parts.append("</svg>")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
