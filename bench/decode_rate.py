#!/usr/bin/env python3
"""Decode-only tok/s, the measuring stick for the streaming-hybrid experiment.

decode_bench.sh reports e2e tok/s with prefill amortized in, which moves with
TTFT and with the prompt. This reports the number the expert path actually
governs -- tokens after the first, divided by the time after the first -- plus
the MTP accept length, because a change that trades accept length for step rate
is not a speedup and the two must be read together.

    .venv/bin/python bench/decode_rate.py --runs 5 --tokens 400
    .venv/bin/python bench/decode_rate.py --label baseline-e60-sub2 --out bench/profile_out/rates

Reports the median run. Repeats matter: a single run at low reachability lies
(see 68b1a7c), so anything below --runs 3 is refused.
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


def one_run(url: str, model: str, tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    t_first = None
    n_chunks = 0
    usage = None
    with request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                if t_first is None:
                    t_first = time.perf_counter()
                n_chunks += 1
    t_end = time.perf_counter()

    if t_first is None:
        raise RuntimeError("no content streamed back")
    completion = (usage or {}).get("completion_tokens") or n_chunks
    decode_s = t_end - t_first
    # Chunks are the step signal: one streamed chunk is one accepted token, so
    # steps = chunks / accept_length. We cannot see accept length directly from
    # the OpenAI surface, so report the per-token rate and leave accept to the
    # server log; ms_per_token is what the expert path moves.
    return {
        "ttft_s": t_first - t0,
        "decode_s": decode_s,
        "completion_tokens": completion,
        "chunks": n_chunks,
        "tok_per_s": (completion - 1) / decode_s if decode_s > 0 else 0.0,
        "ms_per_token": decode_s * 1000.0 / max(completion - 1, 1),
        "e2e_tok_per_s": completion / (t_end - t0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="GLM5.2")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.runs < 3:
        ap.error("--runs must be >= 3; one sample at low reachability is not a measurement")

    for _ in range(args.warmup):
        one_run(args.url, args.model, 48)

    runs = []
    for i in range(args.runs):
        r = one_run(args.url, args.model, args.tokens)
        runs.append(r)
        print(f"  run {i + 1}: {r['tok_per_s']:6.2f} tok/s decode-only   "
              f"{r['ms_per_token']:6.1f} ms/token   ttft {r['ttft_s']:.2f}s   "
              f"{r['e2e_tok_per_s']:6.2f} e2e")

    rates = sorted(r["tok_per_s"] for r in runs)
    median = statistics.median(rates)
    spread = (rates[-1] - rates[0]) / median * 100 if median else 0.0
    print(f"\n{args.label}: median {median:.2f} tok/s decode-only "
          f"(min {rates[0]:.2f}, max {rates[-1]:.2f}, spread {spread:.1f}%)")
    if spread > 10:
        print("  WARNING: >10% spread across runs -- the box is not settled, "
              "do not compare this against another config")

    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{args.label}.json"
        path.write_text(json.dumps(
            {"label": args.label, "ts": time.time(), "args": vars(args),
             "runs": runs, "median_tok_per_s": median, "spread_pct": spread},
            indent=1))
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
