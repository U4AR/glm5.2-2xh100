#!/usr/bin/env python3
"""Aggregate KT_DUMP_TOPK traces into top-2/top-8 expert footprint summaries."""

from __future__ import annotations

import argparse
import collections
import json
import statistics as stats
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _consts import (
    MODEL_NAME,
    N_ROUTED_EXPERTS,
    NUM_EXPERTS_PER_TOKEN,
    PER_EXPERT_BYTES,
    PER_EXPERT_MIB,
    ROUTED_LAYER_IDS,
    TOP2_K,
)


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def valid_expert(eid: int) -> bool:
    return 0 <= eid < N_ROUTED_EXPERTS


def load_trace(path: Path) -> list[Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def iter_sorted_rows(trace: list[Any]):
    for rec in trace:
        if len(rec) == 3:
            layer_idx, ids, weights = rec
        elif len(rec) == 2:
            layer_idx, ids = rec
            weights = None
        else:
            continue

        ids_t = torch.as_tensor(ids)
        ids_t = ids_t.view(-1, ids_t.shape[-1]).to(torch.long)
        if weights is None:
            cols = ids_t.shape[-1]
            weights_t = torch.arange(cols, 0, -1, dtype=torch.float32).expand_as(ids_t)
        else:
            weights_t = torch.as_tensor(weights).view(ids_t.shape).to(torch.float32)
        order = torch.argsort(weights_t, dim=-1, descending=True)
        sorted_ids = torch.gather(ids_t, 1, order)
        for row in sorted_ids:
            yield int(layer_idx), row.tolist()


def layer_population(captured_layers: set[int], include_all: bool) -> list[int]:
    if include_all:
        return sorted(captured_layers)
    routed = set(ROUTED_LAYER_IDS)
    if captured_layers & routed:
        return list(ROUTED_LAYER_IDS)
    return sorted(captured_layers)


def summarize_trace(path: Path, include_all_layers: bool = False) -> dict[str, Any]:
    trace = load_trace(path)
    layer_sets = {
        "top2": collections.defaultdict(set),
        "top8": collections.defaultdict(set),
    }
    layer_counts = {
        "top2": collections.defaultdict(collections.Counter),
        "top8": collections.defaultdict(collections.Counter),
    }
    expert_counts = {
        "top2": collections.Counter(),
        "top8": collections.Counter(),
    }
    captured_layers: set[int] = set()
    token_layer_rows = 0

    for layer_idx, row in iter_sorted_rows(trace):
        captured_layers.add(layer_idx)
        token_layer_rows += 1
        top8 = [eid for eid in row[:NUM_EXPERTS_PER_TOKEN] if valid_expert(eid)]
        top2 = [eid for eid in row[:TOP2_K] if valid_expert(eid)]
        for key, ids in (("top2", top2), ("top8", top8)):
            layer_sets[key][layer_idx].update(ids)
            layer_counts[key][layer_idx].update(ids)
            expert_counts[key].update(ids)

    layers = layer_population(captured_layers, include_all_layers)
    out: dict[str, Any] = {
        "tag": path.stem,
        "path": str(path),
        "model": MODEL_NAME,
        "per_expert_bytes": PER_EXPERT_BYTES,
        "per_expert_mib": PER_EXPERT_MIB,
        "records": len(trace),
        "captured_layers": sorted(captured_layers),
        "layers_analyzed": layers,
        "token_layer_rows": token_layer_rows,
        "percentile_method": "numpy.percentile linear",
    }

    for key in ("top2", "top8"):
        per_layer = []
        counts = []
        for layer in layers:
            distinct_count = len(layer_sets[key].get(layer, set()))
            counts.append(distinct_count)
            per_layer.append(
                {
                    "layer": layer,
                    "distinct_count": distinct_count,
                    "footprint_mib": distinct_count * PER_EXPERT_MIB,
                }
            )
        footprint = [count * PER_EXPERT_MIB for count in counts]
        out[key] = {
            "total_distinct_count": int(sum(counts)),
            "total_footprint_mib": float(sum(footprint)),
            "p90_count": float(np.percentile(counts, 90)) if counts else 0.0,
            "p90_footprint_mib": float(np.percentile(footprint, 90)) if footprint else 0.0,
            "min_count": int(min(counts)) if counts else 0,
            "median_count": float(stats.median(counts)) if counts else 0.0,
            "max_count": int(max(counts)) if counts else 0,
            "per_layer": per_layer,
            "expert_token_counts": [
                int(expert_counts[key].get(eid, 0)) for eid in range(N_ROUTED_EXPERTS)
            ],
            "p90_expert_load": float(
                np.percentile(
                    [expert_counts[key].get(eid, 0) for eid in range(N_ROUTED_EXPERTS)],
                    90,
                )
            ),
        }

    t2_total = max(out["top2"]["total_footprint_mib"], 1e-9)
    t2_p90 = max(out["top2"]["p90_footprint_mib"], 1e-9)
    out["ratios"] = {
        "total_footprint_top8_over_top2": out["top8"]["total_footprint_mib"] / t2_total,
        "p90_footprint_top8_over_top2": out["top8"]["p90_footprint_mib"] / t2_p90,
    }
    return out


def fmt_mib(x: float) -> str:
    return f"{x:,.1f}"


def render_results(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Expert Footprint: Top-2 vs Top-8",
        "",
        f"Model: {MODEL_NAME}. Per routed expert: {PER_EXPERT_MIB:.1f} MiB int4 weights.",
        "Top-2 means the two highest-weight experts from each captured top-8 router row.",
        "p90 uses `numpy.percentile(..., 90)` over the analyzed routed layers.",
        "",
        "| trace | layers | top2 total MiB | top8 total MiB | top8/top2 | top2 p90 MiB | top8 p90 MiB | p90 ratio | top2 count | top8 count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {tag} | {layers} | {t2_total} | {t8_total} | {ratio:.2f} | "
            "{t2_p90} | {t8_p90} | {p90_ratio:.2f} | {t2_count:,} | {t8_count:,} |".format(
                tag=s["tag"],
                layers=len(s["layers_analyzed"]),
                t2_total=fmt_mib(s["top2"]["total_footprint_mib"]),
                t8_total=fmt_mib(s["top8"]["total_footprint_mib"]),
                ratio=s["ratios"]["total_footprint_top8_over_top2"],
                t2_p90=fmt_mib(s["top2"]["p90_footprint_mib"]),
                t8_p90=fmt_mib(s["top8"]["p90_footprint_mib"]),
                p90_ratio=s["ratios"]["p90_footprint_top8_over_top2"],
                t2_count=s["top2"]["total_distinct_count"],
                t8_count=s["top8"]["total_distinct_count"],
            )
        )
    lines.extend(
        [
            "",
            "Footnote: footprint ignores W4AFP8 scale metadata; that is below 1% of expert weight bytes for this headline comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def trace_paths(args: argparse.Namespace) -> list[Path]:
    if args.traces:
        return [Path(p) for p in args.traces]
    return sorted(
        p
        for p in args.runs.glob("*.pt")
        if not p.stem.endswith("_FLUSH") and not p.name.startswith(".")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="*", help="trace .pt files; defaults to runs/*.pt")
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument(
        "--include-all-layers",
        action="store_true",
        help="analyze captured layer ids exactly instead of GLM routed layers 3..77",
    )
    args = parser.parse_args()

    paths = trace_paths(args)
    if not paths:
        raise SystemExit(f"no trace files found in {args.runs}")

    summaries = []
    args.runs.mkdir(parents=True, exist_ok=True)
    for path in paths:
        summary = summarize_trace(path, include_all_layers=args.include_all_layers)
        out = args.runs / f"{path.stem}.summary.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(
            f"{path.name}: top2={summary['top2']['total_footprint_mib']:.1f} MiB "
            f"top8={summary['top8']['total_footprint_mib']:.1f} MiB "
            f"ratio={summary['ratios']['total_footprint_top8_over_top2']:.2f}"
        )

    results = render_results(summaries)
    results_path = args.runs / "results.md"
    results_path.write_text(results, encoding="utf-8")
    print(results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
