#!/usr/bin/env python3
"""Adaptive-cache convergence driver. Runs passes of long decode traffic and
reports per-pass decode tok/s from usage.completion_tokens (NOT chunk counts —
under MTP one chunk carries accept_len tokens). Coverage comes from the server
log's [kt-adaptive] lines, scraped separately."""
import json, sys, time, urllib.request

BASE = "http://127.0.0.1:8000"
PASSES = int(sys.argv[1]) if len(sys.argv) > 1 else 8
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 500
TOPIC = sys.argv[3] if len(sys.argv) > 3 else "llm"

PROMPTS = {
    "llm": ("Write a thorough, coherent essay (do NOT stop early, keep going) explaining "
            "how modern LLM inference engines work: tokenization, KV cache, paged attention, "
            "continuous batching, MoE routing, speculative decoding, and quantization. "
            "Cover each topic in depth with a single consistent narrative voice."),
    "bio": ("Write a thorough, coherent essay (do NOT stop early, keep going) explaining how "
            "the human immune system works: innate immunity, adaptive immunity, antibodies, "
            "T cells, vaccines, and autoimmune disease. Cover each in depth."),
}

def run(prompt, gen):
    body = json.dumps({"model": "GLM5.2-top2",
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": gen, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    tf = tl = None; chunks = 0; comp = None
    for raw in urllib.request.urlopen(req, timeout=3600):
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        p = line[5:].strip()
        if p == "[DONE]":
            break
        try:
            o = json.loads(p)
        except Exception:
            continue
        u = o.get("usage")
        if u:
            comp = u.get("completion_tokens")
        ch = o.get("choices", [])
        if ch:
            d = ch[0].get("delta", {})
            if d.get("reasoning_content") or d.get("content"):
                now = time.time()
                if tf is None:
                    tf = now
                tl = now; chunks += 1
    dt = (tl - tf) if (tf and tl) else 0.0
    tps = comp / dt if (comp and dt > 0) else 0.0
    al = comp / chunks if (comp and chunks) else 0.0
    return comp, dt, tps, al

prompt = PROMPTS[TOPIC]
print(f"topic={TOPIC} passes={PASSES} gen={GEN}")
for p in range(PASSES):
    comp, dt, tps, al = run(prompt, GEN)
    print(f"pass {p}: tokens={comp} decode_s={dt:.1f} TOK/S={tps:.1f} accept~{al:.2f}", flush=True)
