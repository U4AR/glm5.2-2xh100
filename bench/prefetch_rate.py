"""Measure decode rate with and without the expert prefetch, on one boot.

The prefetch can be turned off per-request only by rebooting, so this script
measures whatever the running server has configured and writes a labelled row.
Run it twice -- once with KT_PREFETCH_SLOTS=0, once with it set -- and diff.

Reports tok/s and ms/step at the shipped tier. NEVER uses sub2: the tier is
selected through the OpenAI model field as <base>-topN with safe routing, which
keeps the genuine top-K and substitutes only the tail.

    .venv/bin/python bench/prefetch_rate.py --label prefetch-4slot
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


def run_once(url: str, model: str, tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = request.Request(f"{url}/v1/chat/completions", data=body,
                          headers={"Content-Type": "application/json"})
    stamps, usage, text = [], {}, []
    t0 = time.perf_counter()
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
            d = (obj.get("choices") or [{}])[0].get("delta") or {}
            piece = d.get("content") or d.get("reasoning_content")
            if piece:
                stamps.append(time.perf_counter())
                text.append(piece)
    if len(stamps) < 5:
        raise RuntimeError("too few decode steps to measure")
    gaps = [(stamps[i] - stamps[i - 1]) * 1000 for i in range(1, len(stamps))]
    n_tok = (usage or {}).get("completion_tokens") or len(stamps)
    ms = statistics.median(gaps)
    return {
        "ms_per_step": ms,
        "steps": len(stamps),
        "completion_tokens": n_tok,
        "accept": n_tok / len(stamps),
        "tok_per_s": (n_tok / len(stamps)) / (ms / 1000.0),
        "wall_s": time.perf_counter() - t0,
        "sample": "".join(text)[:400],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="GLM5.2")
    ap.add_argument("--tier", default="2", help="safe tier N (never sub2)")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="bench/profile_out/prefetch_rate.json")
    args = ap.parse_args()

    model = f"{args.model}-top{args.tier}"
    print(f"warming up ({model})...")
    run_once(args.url, model, 40)

    rows = []
    for i in range(args.runs):
        r = run_once(args.url, model, args.tokens)
        rows.append(r)
        print(f"  run {i + 1}: {r['ms_per_step']:7.2f} ms/step  "
              f"accept {r['accept']:.3f}  {r['tok_per_s']:6.2f} tok/s")

    # Accept is `completion_tokens / forward_steps`, so at --tokens 200 it moves
    # in rungs of ~1/70 = 0.04 and nothing finer than that is resolvable. Record
    # the raw counts so a later reader can tell a real shift from a single step
    # landing on the other side of a rung (TODO item 8).
    med = {
        "label": args.label,
        "tier": args.tier,
        "ms_per_step": statistics.median([r["ms_per_step"] for r in rows]),
        "accept": statistics.median([r["accept"] for r in rows]),
        "tok_per_s": statistics.median([r["tok_per_s"] for r in rows]),
        "steps": [r["steps"] for r in rows],
        "completion_tokens": [r["completion_tokens"] for r in rows],
        "runs": len(rows),
        "ts": time.time(),
        "sample": rows[0]["sample"],
    }
    print(f"\n{args.label}: {med['ms_per_step']:.2f} ms/step, "
          f"accept {med['accept']:.3f} ({med['completion_tokens']} tok / "
          f"{med['steps']} steps), {med['tok_per_s']:.2f} tok/s")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    hist = json.loads(p.read_text()) if p.is_file() else []
    hist.append(med)
    p.write_text(json.dumps(hist, indent=1))
    print(f"appended to {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
