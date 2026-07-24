#!/usr/bin/env python3
"""Oracle expert-placement CEILING measurement.

We do not need to physically install a per-question oracle GPU placement to know
its speed ceiling. On the live server the per-request tier `GLM5.2-topN`
controls how many TRUE experts each token keeps; the rest of the 8 slots are
substituted with the best GPU-RESIDENT experts:

  top8 : keep all 8 true -> non-resident trues compute on the CPU expert path
         (this is the current placement, correct output)  == BASELINE
  top0 : keep 0 true -> all 8 slots are GPU-resident experts, ZERO CPU calls
         (compute profile == "8 experts, all resident")   == PLACEMENT CEILING

A perfect ORACLE that keeps the true top-8 but makes them all GPU-resident has
the SAME compute profile as top0 (8 resident experts/token, no CPU), so top0's
decode speed is exactly the maximum speed any placement policy can reach. top4
and top2 are intermediate points.

Streaming is used so we separate TTFT (prefill) from pure DECODE tok/s.
"""
from __future__ import annotations

import argparse, json, statistics, time, urllib.request
from pathlib import Path

from capture import DEFAULT_BASE, DEFAULT_TASKS, discover_tb_root, read_task_instruction


def stream_chat(base, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_chunks += 1
    total = time.perf_counter() - t0
    comp = int(usage.get("completion_tokens") or 0)
    prompt_tok = int(usage.get("prompt_tokens") or 0)
    decode_s = max(total - (ttft or 0), 1e-6)
    return {
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp,
        "ttft_s": round(ttft or 0, 3),
        "total_s": round(total, 3),
        "decode_s": round(decode_s, 3),
        "decode_tok_s": round(comp / decode_s, 2) if comp else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--tiers", nargs="*", default=["top8", "top4", "top2", "top0"])
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--served", default="GLM5.2")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "runs" / "oracle_ceiling.json")
    args = ap.parse_args()

    tb_root = discover_tb_root()
    prompts = {t: read_task_instruction(tb_root, t) for t in args.tasks}

    # warmup so caches/graphs are hot and first-run cost doesn't skew a tier
    print("warmup...")
    stream_chat(args.base, f"{args.served}-top2", "hello", 8, args.timeout)

    results = {}
    for tier in args.tiers:
        model = f"{args.served}-{tier}"
        rows = []
        for task in args.tasks:
            r = stream_chat(args.base, model, prompts[task], args.max_tokens, args.timeout)
            r["task"] = task
            rows.append(r)
            print(f"{tier:6s} {task:34s} decode={r['decode_tok_s']:6.2f} tok/s  "
                  f"ttft={r['ttft_s']:.2f}s comp={r['completion_tokens']} total={r['total_s']:.2f}s")
        decode_rates = [x["decode_tok_s"] for x in rows if x["completion_tokens"] > 4]
        comp = sum(x["completion_tokens"] for x in rows)
        dsum = sum(x["decode_s"] for x in rows)
        results[tier] = {
            "rows": rows,
            "mean_decode_tok_s": round(statistics.mean(decode_rates), 2) if decode_rates else 0,
            "agg_decode_tok_s": round(comp / dsum, 2) if dsum else 0,
            "total_completion_tokens": comp,
        }
        print(f"==> {tier}: mean_decode={results[tier]['mean_decode_tok_s']} "
              f"agg_decode={results[tier]['agg_decode_tok_s']} tok/s\n")

    base = results.get("top8", {}).get("agg_decode_tok_s") or 1
    print("=== PLACEMENT CEILING SUMMARY (decode tok/s, aggregate) ===")
    for tier in args.tiers:
        agg = results[tier]["agg_decode_tok_s"]
        print(f"  {tier:6s} {agg:7.2f} tok/s   {agg/base:.2f}x vs top8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
