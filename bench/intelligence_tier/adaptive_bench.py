"""Compare fixed expert tiers against adaptive router-mass speed levels.

Requires the OpenAI-compatible server on localhost:8000. Reports median decode
tok/s and simple repetition/coherence checks for fixed and adaptive tiers.
"""
import json
import statistics
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
SERVED = sys.argv[1] if len(sys.argv) > 1 else "GLM5.2"
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 256
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
TIERS = (
    sys.argv[4].split(",")
    if len(sys.argv) > 4
    else ["top8", "top4", "top2", "adapt0", "adapt50", "adapt100"]
)

PROMPTS = [
    "Write a concise explanation of why the sky is blue, then list the first 8 prime numbers.",
    "Implement a Python function that returns the largest eigenvalue of a 2x2 matrix and explain the formula.",
    "Describe how an inference batching scheduler should trade latency against throughput in five paragraphs.",
]


def chat(model, prompt):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": GEN,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=600)
    j = json.load(r)
    dt = time.time() - t0
    msg = j["choices"][0]["message"]
    txt = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
    ntok = j.get("usage", {}).get("completion_tokens", 0)
    return txt.strip(), ntok, dt


def has_repetition(text):
    words = text.lower().split()
    if len(words) < 24:
        return False
    chunks = [" ".join(words[i : i + 8]) for i in range(0, len(words) - 7, 8)]
    return len(chunks) - len(set(chunks)) >= 2


print(f"served={SERVED} gen_tokens={GEN} repeats={REPEATS}\n")
print(f"{'model':<18} {'median tok/s':>12} {'min tok/s':>10} {'coherent':>9} {'repeat':>8}")

for tier in TIERS:
    model = f"{SERVED}-{tier}"
    speeds = []
    coherent = 0
    repeated = 0
    total = 0
    for _ in range(REPEATS):
        for prompt in PROMPTS:
            text, ntok, dt = chat(model, prompt)
            total += 1
            speeds.append(ntok / dt if dt else 0.0)
            coherent += int(len(text) > 80)
            repeated += int(has_repetition(text))
    print(
        f"{model:<18} {statistics.median(speeds):>12.1f} "
        f"{min(speeds):>10.1f} {coherent:>4}/{total:<4} {repeated:>4}/{total:<3}"
    )
