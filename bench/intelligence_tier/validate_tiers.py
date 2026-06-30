"""Validate per-request expert-tier selection via the OpenAI model name.

Checks: (1) each '<base>-topN' tier returns coherent text and measures decode
tok/s; (2) '-top8' == bare-name baseline routing (greedy/temp0 output match);
(3) mixed concurrent tiers in flight (batched) all stay coherent.
"""
import json
import sys
import time
import threading
import urllib.request

BASE = "http://127.0.0.1:8000"


def models():
    r = urllib.request.urlopen(BASE + "/v1/models", timeout=10)
    return json.load(r)["data"][0]["id"]


def chat(model, prompt, max_tokens=200, temp=0.0):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=300)
    j = json.load(r)
    dt = time.time() - t0
    msg = j["choices"][0]["message"]
    txt = msg.get("content") or ""
    rc = msg.get("reasoning_content") or ""
    ntok = j.get("usage", {}).get("completion_tokens", 0)
    return txt, rc, ntok, dt


SERVED = models()
print(f"served model id: {SERVED}\n")
PROMPT = "Write a short coherent paragraph about why the sky is blue. Then list the first 8 prime numbers."

# --- per-tier coherence + speed ---
results = {}
for N in (8, 4, 2, 0):
    model = f"{SERVED}-top{N}"
    txt, rc, ntok, dt = chat(model, PROMPT, max_tokens=200)
    toks = ntok / dt if dt else 0
    full = (rc + " " + txt).strip()
    coherent = ("blue" in full.lower() or "scatter" in full.lower()) and len(full) > 40
    results[N] = (txt, toks, coherent)
    print(f"-top{N}: {ntok} tok, {toks:.1f} tok/s, coherent={coherent}")
    print(f"   ANSWER: {txt[:160]!r}\n")

# --- baseline parity: top8 vs bare name, greedy ---
b1, _, _, _ = chat(SERVED, "Count from 1 to 10.", max_tokens=60, temp=0.0)
b2, _, _, _ = chat(f"{SERVED}-top8", "Count from 1 to 10.", max_tokens=60, temp=0.0)
print(f"baseline parity (bare vs -top8, greedy): match={b1.strip()==b2.strip()}")
if b1.strip() != b2.strip():
    print(f"   bare : {b1[:120]!r}\n   top8 : {b2[:120]!r}")

# --- mixed concurrent tiers (batched in flight) ---
print("\nmixed concurrent batch (top8 + top2 + top4 + top0 simultaneously):")
out = {}
def worker(N):
    txt, rc, ntok, dt = chat(f"{SERVED}-top{N}", PROMPT, max_tokens=150)
    out[N] = (len((rc+txt).strip()) > 40, ntok/dt if dt else 0)
ths = [threading.Thread(target=worker, args=(N,)) for N in (8, 4, 2, 0)]
t0 = time.time()
for t in ths: t.start()
for t in ths: t.join()
agg = time.time() - t0
for N in (8, 4, 2, 0):
    coh, sp = out.get(N, (False, 0))
    print(f"   -top{N}: coherent={coh}, {sp:.1f} tok/s (in mixed batch)")
print(f"   mixed wall: {agg:.1f}s")

print("\nDONE")
