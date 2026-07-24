#!/usr/bin/env python3
"""Warm-repeat oracle test.

The live server runs with --kt-enable-dynamic-expert-update (the adaptive cache).
Question: if we run the SAME question repeatedly at top8 (correct output, so the
updater learns the TRUE top-8 router ids), does the resident set converge toward
this question's true experts and pull decode tok/s up from the 19.5 baseline
toward the 72.7 all-resident ceiling?

We force a FRESH prefill each repeat (unique nonce) so the adaptive updater fires
every pass instead of being skipped by the prefix cache.
"""
from __future__ import annotations
import json, time, urllib.request, sys
from capture import DEFAULT_BASE, discover_tb_root, read_task_instruction

BASE = DEFAULT_BASE
TASK = sys.argv[1] if len(sys.argv) > 1 else "largest-eigenval"
TIER = sys.argv[2] if len(sys.argv) > 2 else "top8"
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
MAXTOK = 200


def stream(model, prompt):
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": MAXTOK, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; usage = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            o = json.loads(d)
            if o.get("usage"):
                usage = o["usage"]
            ch = o.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                if (delta.get("content") or delta.get("reasoning_content")) and ttft is None:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    comp = int(usage.get("completion_tokens") or 0)
    dec = max(total - (ttft or 0), 1e-6)
    return comp, ttft or 0, round(comp / dec, 2)


def main():
    prompt0 = read_task_instruction(discover_tb_root(), TASK)
    print(f"task={TASK} tier={TIER} repeats={REPEATS}\n")
    for i in range(REPEATS):
        # unique nonce -> fresh prefill each pass so the adaptive updater runs
        p = f"[run {i} {time.time():.6f}]\n{prompt0}"
        comp, ttft, dec = stream(f"GLM5.2-{TIER}", p)
        print(f"  pass {i}: decode={dec:6.2f} tok/s  ttft={ttft:.2f}s comp={comp}")


if __name__ == "__main__":
    main()
