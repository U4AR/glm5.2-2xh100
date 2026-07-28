#!/usr/bin/env python3
"""Inter-token latency distribution, and what tier movement does to its tail.

Throughput hides stalls. A 300 ms tier move inside a 300-token generation costs
about 3% of the mean and is invisible next to boot-to-boot variation, but to
someone reading the output it is a visible hitch -- and there are 75 layers
taking turns, so the hitches keep coming. Mean tok/s is the wrong instrument
for a cost that arrives in lumps.

It also replaces a broken statistic. "Blocked % of wall time" was computed by
summing the `took=` field and dividing by the span between the first and last
tier log line -- a span that includes prefill, idle and the gaps between
benchmarks, none of which can contain a tick. The denominator was therefore too
large and the number too small, which is the most likely reason a "4.5% blocked"
config that should have run at ~38.8 tok/s measured 33.45. Here the denominator
is exactly the decode phase, because every timestamp comes from a token
actually arriving.

Usage:
  token_latency.py [n_tokens] [n_runs]
Environment:
  TIER_BENCH_BASE   server base URL (default http://127.0.0.1:8000)
  TIER_BENCH_MODEL  model name to request
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("TIER_BENCH_BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("TIER_BENCH_MODEL", "GLM5.2")

PROMPT = (
    "Write a long, detailed technical essay about how CPUs and GPUs differ in "
    "architecture and in the workloads they suit. Cover memory hierarchy, "
    "parallelism, and scheduling."
)


def stream_gaps(max_tokens):
    """Return (gaps, ttft, total) where gaps are seconds between successive
    content chunks. Chunks, not tokens: with speculative decoding several
    tokens can land in one chunk, which is exactly the granularity a reader
    perceives."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t_start = time.perf_counter()
    prev = None
    ttft = None
    gaps = []
    for raw in urllib.request.urlopen(req, timeout=1800):
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        got = False
        for ch in obj.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content") or d.get("reasoning_content"):
                got = True
        if not got:
            continue
        now = time.perf_counter()
        if prev is None:
            ttft = now - t_start
        else:
            gaps.append(now - prev)
        prev = now
    return gaps, ttft, (prev - t_start) if prev else 0.0


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[i]


def main():
    n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    all_gaps = []
    for i in range(n_runs):
        gaps, ttft, total = stream_gaps(n_tokens)
        all_gaps.extend(gaps)
        print(f"  run{i}: {len(gaps)+1} chunks  ttft {ttft*1000:.0f}ms  "
              f"decode {total:.2f}s")

    s = sorted(all_gaps)
    med = pct(s, 0.50)
    # A "hitch" is a gap far longer than this generation's own normal gap. The
    # threshold is relative to the median rather than absolute so it stays
    # meaningful across configs that differ in baseline speed -- otherwise a
    # slower config would be scored as having more hitches purely for being
    # slower.
    thresh = max(4.0 * med, 0.050)
    hitches = [g for g in s if g > thresh]
    print(f"\n  chunks {len(s)}")
    print(f"  gap p50 {med*1000:.1f}ms  p90 {pct(s,0.90)*1000:.1f}ms  "
          f"p99 {pct(s,0.99)*1000:.1f}ms  max {s[-1]*1000:.1f}ms")
    print(f"  hitches (>{thresh*1000:.0f}ms) {len(hitches)} "
          f"= {100.0*len(hitches)/max(1,len(s)):.2f}% of chunks, "
          f"{sum(hitches):.2f}s total = "
          f"{100.0*sum(hitches)/max(1e-9,sum(s)):.1f}% of decode time")
    print(f"  decode time in hitches vs steady: {sum(hitches):.2f}s / {sum(s):.2f}s")
    # This is the honest version of "blocked %": the numerator is measured from
    # arrival times, and the denominator is decode time and nothing else.
    print(f"\nSTALL_FRACTION {100.0*sum(hitches)/max(1e-9,sum(s)):.2f}")


if __name__ == "__main__":
    main()
