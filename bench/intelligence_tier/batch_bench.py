"""Batch-scaling benchmark for the top-2 tier.

Fires B identical concurrent decode requests and reports per-stream and
aggregate decode tok/s. B=1 is the single-stream number; B>1 is batched.
"""
import json, sys, time, threading, urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "GLM5.2-top2"
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 256
BATCHES = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["1", "2", "4", "8"])]
PROMPT = ("Write a detailed, coherent multi-paragraph explanation of how a "
          "transformer neural network works, covering attention, feed-forward "
          "layers, and training. Be thorough.")


def one(out, idx):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
                       "temperature": 0.7, "max_tokens": GEN,
                       "stream": False}).encode()
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=600)
    j = json.load(r)
    dt = time.time() - t0
    u = j.get("usage", {})
    ntok = u.get("completion_tokens", 0)
    out[idx] = (ntok, dt)


def warmup():
    one({}, 0)


print(f"model={MODEL}  gen_tokens={GEN}\n")
warmup()
print(f"{'B':>3} {'agg tok/s':>10} {'per-stream':>11} {'wall s':>7} {'tot tok':>8}")
for B in BATCHES:
    out = {}
    ths = [threading.Thread(target=one, args=(out, i)) for i in range(B)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    tot = sum(out[i][0] for i in range(B))
    agg = tot / wall
    per = agg / B
    print(f"{B:>3} {agg:>10.1f} {per:>11.1f} {wall:>7.1f} {tot:>8}")
print("\n(agg tok/s = total generated / wall; per-stream = agg / B)")
