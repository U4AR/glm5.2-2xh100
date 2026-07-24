#!/usr/bin/env python3
"""Render self-contained SVG charts from expert footprint summary JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

from _consts import MODEL_NAME, N_ROUTED_EXPERTS, PER_EXPERT_MIB


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
FIGS = RUNS / "figs"
BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#059669"
GRID = "#dbe3ee"
INK = "#172033"


def esc(s: object) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_open(w: int, h: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}.small{font-size:14px}.axis{font-size:13px;fill:#475569}.title{font-size:26px;font-weight:700}.sub{font-size:15px;fill:#526173}.legend{font-size:15px;font-weight:650}</style>',
        f'<text x="84" y="44" class="title">{esc(title)}</text>',
        f'<text x="84" y="72" class="sub">{esc(subtitle)}</text>',
    ]


def nice_ticks(max_y: float, n: int = 5) -> list[float]:
    if max_y <= 0:
        return [0, 1]
    raw = max_y / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 5, 10), key=lambda m: abs(m * mag - raw)) * mag
    end = math.ceil(max_y / step) * step
    ticks = []
    v = 0.0
    while v <= end + step / 2:
        ticks.append(v)
        v += step
    return ticks


def draw_axes(parts: list[str], ml: int, mt: int, pw: int, ph: int, max_y: float) -> float:
    ticks = nice_ticks(max_y)
    y_max = max(ticks[-1], 1.0)
    for tick in ticks:
        y = mt + ph - (tick / y_max) * ph
        label = f"{tick:.0f}" if tick >= 10 else f"{tick:g}"
        parts.append(f'<line x1="{ml}" x2="{ml+pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{ml-12}" y="{y+4:.1f}" text-anchor="end" class="axis">{label}</text>')
    parts.append(f'<line x1="{ml}" x2="{ml+pw}" y1="{mt+ph}" y2="{mt+ph}" stroke="#334155" stroke-width="1.4"/>')
    parts.append(f'<line x1="{ml}" x2="{ml}" y1="{mt}" y2="{mt+ph}" stroke="#334155" stroke-width="1.4"/>')
    return y_max


def line_points(values: list[float], ml: int, mt: int, pw: int, ph: int, y_max: float) -> str:
    if not values:
        return ""
    denom = max(len(values) - 1, 1)
    pts = []
    for i, val in enumerate(values):
        x = ml + (i / denom) * pw
        y = mt + ph - (val / y_max) * ph
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def write_usage_curve(summary: dict, out: Path) -> None:
    vals2 = sorted(summary["top2"]["expert_token_counts"], reverse=True)
    vals8 = sorted(summary["top8"]["expert_token_counts"], reverse=True)
    w, h = 1120, 680
    ml, mr, mt, mb = 88, 42, 102, 92
    pw, ph = w - ml - mr, h - mt - mb
    parts = svg_open(
        w,
        h,
        f"Expert Usage Distribution: {summary['tag']}",
        f"{MODEL_NAME}; counts aggregate selected experts across analyzed routed layers",
    )
    y_max = draw_axes(parts, ml, mt, pw, ph, max(vals2 + vals8))
    for x_tick in (1, 64, 128, 192, 256):
        x = ml + ((x_tick - 1) / (N_ROUTED_EXPERTS - 1)) * pw
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{mt+ph}" y2="{mt+ph+6}" stroke="#334155"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt+ph+26}" text-anchor="middle" class="axis">{x_tick}</text>')
    parts.append(f'<polyline fill="none" stroke="{BLUE}" stroke-width="3" points="{line_points(vals8, ml, mt, pw, ph, y_max)}"><title>top8 sorted per-expert token counts</title></polyline>')
    parts.append(f'<polyline fill="none" stroke="{ORANGE}" stroke-width="3" points="{line_points(vals2, ml, mt, pw, ph, y_max)}"><title>top2 sorted per-expert token counts</title></polyline>')
    parts.append(f'<text x="{ml+pw/2:.1f}" y="{h-24}" text-anchor="middle" class="small">expert rank after sorting by token count</text>')
    parts.append(f'<text x="24" y="{mt+ph/2:.1f}" transform="rotate(-90 24 {mt+ph/2:.1f})" text-anchor="middle" class="small">selected-token count</text>')
    legend(parts, w - 230, 34)
    parts.append(f'<text x="84" y="{h-52}" class="axis">Top-2 is derived from the two highest router weights within each captured top-8 row.</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def write_layer_counts(summary: dict, out: Path) -> None:
    layers = [p["layer"] for p in summary["top8"]["per_layer"]]
    vals2 = [p["distinct_count"] for p in summary["top2"]["per_layer"]]
    vals8 = [p["distinct_count"] for p in summary["top8"]["per_layer"]]
    w, h = 1120, 680
    ml, mr, mt, mb = 88, 42, 102, 92
    pw, ph = w - ml - mr, h - mt - mb
    parts = svg_open(
        w,
        h,
        f"Per-Layer Distinct Experts: {summary['tag']}",
        f"{MODEL_NAME}; one point per routed MoE layer, per expert is {PER_EXPERT_MIB:.1f} MiB",
    )
    y_max = draw_axes(parts, ml, mt, pw, ph, max(vals2 + vals8 + [N_ROUTED_EXPERTS]))
    for i, layer in enumerate(layers):
        if i % 10 and i != len(layers) - 1:
            continue
        x = ml + (i / max(len(layers) - 1, 1)) * pw
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{mt+ph}" y2="{mt+ph+6}" stroke="#334155"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt+ph+26}" text-anchor="middle" class="axis">{layer}</text>')
    parts.append(f'<polyline fill="none" stroke="{BLUE}" stroke-width="3" points="{line_points(vals8, ml, mt, pw, ph, y_max)}"><title>top8 distinct experts per layer</title></polyline>')
    parts.append(f'<polyline fill="none" stroke="{ORANGE}" stroke-width="3" points="{line_points(vals2, ml, mt, pw, ph, y_max)}"><title>top2 distinct experts per layer</title></polyline>')
    for color, key in ((BLUE, "top8"), (ORANGE, "top2")):
        p90 = summary[key]["p90_count"]
        y = mt + ph - (p90 / y_max) * ph
        parts.append(f'<line x1="{ml}" x2="{ml+pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="7 7" opacity="0.75"><title>{key} p90 distinct count {p90:.1f}</title></line>')
    parts.append(f'<text x="{ml+pw/2:.1f}" y="{h-24}" text-anchor="middle" class="small">layer id</text>')
    parts.append(f'<text x="24" y="{mt+ph/2:.1f}" transform="rotate(-90 24 {mt+ph/2:.1f})" text-anchor="middle" class="small">distinct experts touched</text>')
    legend(parts, w - 230, 34)
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def legend(parts: list[str], x: int, y: int) -> None:
    parts.append(f'<rect x="{x}" y="{y}" width="18" height="18" fill="{BLUE}" opacity="0.78" rx="3"/>')
    parts.append(f'<text x="{x+26}" y="{y+15}" class="legend">top8</text>')
    parts.append(f'<rect x="{x+92}" y="{y}" width="18" height="18" fill="{ORANGE}" opacity="0.78" rx="3"/>')
    parts.append(f'<text x="{x+118}" y="{y+15}" class="legend">top2</text>')


def write_summary_bars(summaries: list[dict], out: Path) -> None:
    w = max(1120, 190 + 120 * len(summaries))
    h = 700
    ml, mr, mt, mb = 96, 42, 104, 150
    pw, ph = w - ml - mr, h - mt - mb
    max_y = max((s["top8"]["total_footprint_mib"] for s in summaries), default=1)
    parts = svg_open(
        w,
        h,
        "Total Expert Footprint",
        f"{MODEL_NAME}; sum of per-layer distinct expert footprints",
    )
    y_max = draw_axes(parts, ml, mt, pw, ph, max_y)
    group_w = pw / max(len(summaries), 1)
    bar_w = min(34, group_w / 4)
    for i, s in enumerate(summaries):
        cx = ml + i * group_w + group_w / 2
        for j, (key, color) in enumerate((("top8", BLUE), ("top2", ORANGE))):
            val = s[key]["total_footprint_mib"]
            bh = (val / y_max) * ph
            x = cx + (j - 1) * (bar_w + 4)
            y = mt + ph - bh
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.78" rx="3"><title>{esc(s["tag"])} {key}: {val:.1f} MiB</title></rect>')
        label = esc(s["tag"].replace("exp3_", "").replace("exp2_", ""))
        parts.append(f'<text x="{cx:.1f}" y="{mt+ph+26}" text-anchor="end" transform="rotate(-32 {cx:.1f} {mt+ph+26})" class="axis">{label}</text>')
    parts.append(f'<text x="{ml+pw/2:.1f}" y="{h-24}" text-anchor="middle" class="small">trace</text>')
    parts.append(f'<text x="24" y="{mt+ph/2:.1f}" transform="rotate(-90 24 {mt+ph/2:.1f})" text-anchor="middle" class="small">total footprint, MiB</text>')
    legend(parts, w - 230, 34)
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def write_combined_usage(summaries: list[dict], out: Path) -> None:
    combined = {
        "tag": "combined",
        "top2": {"expert_token_counts": [0] * N_ROUTED_EXPERTS},
        "top8": {"expert_token_counts": [0] * N_ROUTED_EXPERTS},
    }
    for s in summaries:
        for key in ("top2", "top8"):
            combined[key]["expert_token_counts"] = [
                a + b
                for a, b in zip(
                    combined[key]["expert_token_counts"],
                    s[key]["expert_token_counts"],
                )
            ]
    write_usage_curve(combined, out)


def load_summaries(paths: Iterable[Path]) -> list[dict]:
    summaries = []
    for path in paths:
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="*", help="summary JSON files; defaults to runs/*.summary.json")
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--out", type=Path, default=FIGS)
    args = parser.parse_args()

    paths = [Path(p) for p in args.summaries] if args.summaries else sorted(args.runs.glob("*.summary.json"))
    if not paths:
        raise SystemExit(f"no summary JSON files found in {args.runs}")
    summaries = load_summaries(paths)
    args.out.mkdir(parents=True, exist_ok=True)

    for s in summaries:
        write_usage_curve(s, args.out / f"{s['tag']}_usage_distribution.svg")
        write_layer_counts(s, args.out / f"{s['tag']}_per_layer_distinct.svg")
    write_summary_bars(summaries, args.out / "summary_total_footprint.svg")
    write_combined_usage(summaries, args.out / "combined_usage_distribution.svg")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
