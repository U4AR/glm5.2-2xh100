#!/usr/bin/env python3
"""Load the document into the model context, then measure asking it things.

Imports the prompt straight from doc_ui, so the bench and the UI send a
byte-identical prefix and therefore share one radix-cache entry. Running this
first is what makes the UI's first question fast.

  python3 bench/doc_context_bench.py            # warm, then the default questions
  python3 bench/doc_context_bench.py --warm-only
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from doc_ui import SYSTEM_PROMPT  # noqa: E402  (path set above)

BASE = os.environ.get("MODEL_BASE", "http://127.0.0.1:8000")

QUESTIONS = [
    "In one sentence: what is this document and who issued it?",
    "What are the average emergency response times the system is designed for, "
    "in urban and in rural areas?",
    "How many Police Response Vehicles (PRVs) are deployed, and what equipment "
    "is fitted in them?",
    "What is the bid security / EMD amount a bidder must submit?",
    "List the penalties or liquidated damages for missing the response-time SLA.",
]


def stream(messages, max_tokens, tier=None):
    """One streamed turn. Returns (text, ttft_s, decode_tok_s, usage)."""
    model = "GLM5.2" + (f"-top{tier}" if tier else "")
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})

    t0 = time.time()
    first = None
    text, reasoning, usage = "", "", None
    with urllib.request.urlopen(req, timeout=None) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            try:
                j = json.loads(data)
            except ValueError:
                continue
            if j.get("usage"):
                usage = j["usage"]
            delta = (j.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("reasoning_content"):
                if first is None:
                    first = time.time()
                reasoning += delta["reasoning_content"]
            if delta.get("content"):
                if first is None:
                    first = time.time()
                text += delta["content"]

    ttft = (first or time.time()) - t0
    gen = usage["completion_tokens"] if usage else 0
    rate = gen / max(time.time() - first, 1e-6) if first and gen else 0.0
    # A stream cut short (server restart, abort) never delivers the usage block;
    # report what we have rather than dying on a subscript.
    if usage is None:
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
    return (text or "[reasoning only] " + reasoning), ttft, rate, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-only", action="store_true")
    ap.add_argument("--tier", default=None, help="expert tier suffix, e.g. 4 or 2")
    ap.add_argument("--max-tokens", type=int, default=700)
    args = ap.parse_args()

    print(f"document prefix: {len(SYSTEM_PROMPT):,} chars")
    print("warming (first call pays the full document prefill)…", flush=True)

    t0 = time.time()
    _, ttft, rate, usage = stream(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "Reply with OK."}],
        max_tokens=8, tier=args.tier)
    print(f"  prefill: {ttft:8.1f}s   prompt_tokens={usage['prompt_tokens']:,}"
          f"   total {time.time()-t0:.1f}s")

    if args.warm_only:
        return

    for q in QUESTIONS:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}]
        text, ttft, rate, usage = stream(msgs, args.max_tokens, args.tier)
        print(f"\nQ: {q}")
        print(f"   TTFT {ttft:.1f}s · {rate:.1f} tok/s decode · "
              f"{usage['prompt_tokens']:,} prompt tok")
        print(f"   {text.strip()[:600]}")


if __name__ == "__main__":
    main()
