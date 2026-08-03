#!/usr/bin/env python3
"""Measure the CPU expert path's cost curve: is it a + b*n, or just b*n?

The whole streaming-hybrid design turns on this. If the CPU path costs
`a + b*n` with a large fixed `a` per layer, then shaving experts off the CPU
buys little and the only real win is removing the CPU from a layer entirely
(a corner). If it is proportional (`a` ~ 0), an interior split is worth its
full share and the balance formula is the whole game.

`a` was previously INFERRED from the oracle placement ceiling, never measured.
This measures it directly.

Method: the per-request tier suffix (`<model>-topN`, N=0..8) changes how many
genuine experts survive routing, LIVE, in one CUDA graph, no restart. Fewer
kept experts means fewer land on the CPU. Sweeping N and reading ms/step gives
the curve; the N=0 point is the intercept measured rather than extrapolated --
the CPU submit/sync still fires, but with every expert masked off, so whatever
time remains is fixed cost.

One SSE chunk is one forward STEP (with MTP a step emits accept_length tokens),
so ms/step is the metric: it does not move when accept length does.

    .venv/bin/python bench/cpu_fixed_cost.py --tiers 0,1,2,4,6,8 --runs 3

Tiers are interleaved across rounds, not run back to back, so a drifting box
shows up as spread within a tier instead of a fake trend across tiers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from urllib import request

PROMPT = (
    "Write a detailed technical essay about how modern CPUs and GPUs differ in "
    "their approach to parallelism, memory hierarchy, and scheduling."
)


def decode_run(url: str, model: str, tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = request.Request(f"{url}/v1/chat/completions", data=body,
                          headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, usage, text = [], {}, []
    with request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = (obj.get("choices") or [{}])[0].get("delta") or {}
            piece = ch.get("content") or ch.get("reasoning_content") or ""
            if piece:
                stamps.append(time.perf_counter())
                text.append(piece)
    if len(stamps) < 5:
        raise RuntimeError(f"too few steps ({len(stamps)}) to measure")
    gaps = [(stamps[i] - stamps[i - 1]) * 1000 for i in range(1, len(stamps))]
    n_steps = len(stamps)
    n_tok = usage.get("completion_tokens") or n_steps
    decode_s = stamps[-1] - stamps[0]
    return {
        "ttft_s": stamps[0] - t0,
        "n_steps": n_steps,
        "n_tokens": n_tok,
        "accept_length": n_tok / n_steps,
        "ms_per_step_median": statistics.median(gaps),
        "ms_per_step_mean": statistics.mean(gaps),
        "steps_per_s": (n_steps - 1) / decode_s if decode_s else 0.0,
        "tok_per_s": (n_tok - n_tok / n_steps) / decode_s if decode_s else 0.0,
        "sample": "".join(text)[:120],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="GLM5.2")
    ap.add_argument("--tiers", default="0,1,2,4,6,8")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--out", default="bench/profile_out/cpu_fixed_cost.json")
    args = ap.parse_args()

    tiers = [int(t) for t in args.tiers.split(",") if t.strip() != ""]
    print("warmup...")
    decode_run(args.url, args.model, 48)

    per_tier: dict[int, list[dict]] = {t: [] for t in tiers}
    for rnd in range(args.runs):
        for t in tiers:
            model = f"{args.model}-top{t}"
            try:
                r = decode_run(args.url, model, args.tokens)
            except Exception as e:  # a tier that the server rejects is data too
                print(f"  round {rnd + 1} top{t}: FAILED {e}")
                continue
            per_tier[t].append(r)
            print(f"  round {rnd + 1} top{t}: {r['ms_per_step_median']:7.2f} ms/step  "
                  f"accept {r['accept_length']:.2f}  {r['tok_per_s']:6.2f} tok/s  "
                  f"{r['steps_per_s']:5.2f} steps/s")

    print(f"\n{'tier':>5} {'ms/step':>9} {'spread%':>8} {'steps/s':>8} "
          f"{'accept':>7} {'tok/s':>7}")
    summary = {}
    for t in tiers:
        runs = per_tier[t]
        if not runs:
            continue
        vals = sorted(r["ms_per_step_median"] for r in runs)
        med = statistics.median(vals)
        spread = (vals[-1] - vals[0]) / med * 100 if med else 0.0
        summary[t] = {
            "ms_per_step": med,
            "spread_pct": spread,
            "steps_per_s": statistics.median([r["steps_per_s"] for r in runs]),
            "accept_length": statistics.median([r["accept_length"] for r in runs]),
            "tok_per_s": statistics.median([r["tok_per_s"] for r in runs]),
            "n_runs": len(runs),
        }
        s = summary[t]
        print(f"{t:>5} {med:9.2f} {spread:8.1f} {s['steps_per_s']:8.2f} "
              f"{s['accept_length']:7.2f} {s['tok_per_s']:7.2f}")

    # Least-squares fit of ms/step against tier, and the directly measured
    # intercept. If the fitted intercept and the top0 point agree, the affine
    # model is real; if top0 sits far below the fit, the cost is not affine and
    # the fixed-cost story is wrong.
    pts = [(t, summary[t]["ms_per_step"]) for t in tiers if t in summary and t > 0]
    if len(pts) >= 2:
        n = len(pts)
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] * p[0] for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        denom = n * sxx - sx * sx
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        print(f"\nfit over tiers>0:  ms/step = {a:.2f} + {b:.2f} * K")
        print(f"  slope  {b:.2f} ms per kept expert per step "
              f"({b / 75:.3f} ms per expert per layer)")
        print(f"  fitted intercept {a:.2f} ms/step")
        if 0 in summary:
            m0 = summary[0]["ms_per_step"]
            print(f"  MEASURED top0    {m0:.2f} ms/step  "
                  f"(CPU submit/sync still fires, every expert masked)")
            print(f"  fixed share of the top8 step: "
                  f"{m0 / summary[8]['ms_per_step'] * 100:.1f}%"
                  if 8 in summary else "")
        summary["fit"] = {"intercept_ms": a, "slope_ms_per_tier": b}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"ts": time.time(), "args": vars(args),
         "per_tier": {str(k): v for k, v in per_tier.items()},
         "summary": {str(k): v for k, v in summary.items()}}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
