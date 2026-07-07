"""Adaptive expert-cache convergence benchmark.

Sends a FIXED set of questions to the running server for N passes and records
the per-request DECODE tok/s (isolated from prefill via streaming: rate =
(tokens-1)/(t_last_token - t_first_token)). The point is to watch whether the
adaptive expert cache (--kt-enable-dynamic-expert-update) makes repeated
questions get FASTER over passes as hot experts migrate onto the GPU, and how
close that gets to the top0 (CPU-eliminated) ceiling.

Usage:
  python cache_convergence_bench.py [served] [tier] [passes] [gen_tokens]

  served     base model name (default GLM5.2)
  tier       expert tier suffix (default top8 = pure placement, no substitution)
  passes     number of times to loop the whole question set (default 6)
  gen_tokens max_tokens per request (default 200)

Reference points (run separately or via --refs):
  tier top0  = substitution ceiling (CPU eliminated)  ~= fast bound
  tier top8  = full top-8 experts, placement-only      = what the cache moves
"""
import json
import statistics
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
SERVED = sys.argv[1] if len(sys.argv) > 1 else "GLM5.2"
TIER = sys.argv[2] if len(sys.argv) > 2 else "top8"
PASSES = int(sys.argv[3]) if len(sys.argv) > 3 else 6
GEN = int(sys.argv[4]) if len(sys.argv) > 4 else 200

# Prompts are deliberately long-ish (>64 tok) so a low --kt-gpu-prefill-token-
# threshold fires the re-placement path on every request, and deterministic
# (temp 0) so the SAME experts are exercised every pass -> the cache can converge.
PROMPTS = [
    "You are given a rotated sorted array of distinct integers. Explain in detail "
    "the binary-search algorithm to find a target value in O(log n) time, then "
    "write a clean, well-commented Python implementation and walk through it on the "
    "example array [4,5,6,7,0,1,2] searching for 0.",
    "Explain, from first principles and in several paragraphs, how a modern "
    "mixture-of-experts transformer routes tokens to experts, why top-k routing is "
    "used, and what load-balancing auxiliary losses are for. Be concrete and "
    "quantitative where you can.",
    "Derive the closed-form solution for ordinary least-squares linear regression "
    "starting from the sum-of-squared-errors objective. Show every step of the "
    "matrix calculus, state the normal equations, and note the invertibility "
    "condition on X^T X.",
    "Describe how you would design a latency-optimized inference server for a large "
    "language model: batching strategy, KV-cache management, speculative decoding, "
    "and how CPU and GPU work can be overlapped. Give five concrete design rules.",
]


def chat_stream(model, prompt, gen):
    """Return (decode_tok_s, ttft_s, n_tokens, text). Rate excludes prefill."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": gen,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    t_first = None
    t_last = None
    n = 0
    chunks = []
    r = urllib.request.urlopen(req, timeout=1200)
    for raw in r:
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
        delta = obj.get("choices", [{}])[0].get("delta", {})
        piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
        if piece:
            now = time.time()
            if t_first is None:
                t_first = now
            t_last = now
            n += 1
            chunks.append(piece)
    ttft = (t_first - t0) if t_first else 0.0
    decode_span = (t_last - t_first) if (t_first and t_last and n > 1) else 0.0
    rate = (n - 1) / decode_span if decode_span > 0 else 0.0
    return rate, ttft, n, "".join(chunks)


def run_tier(tier, passes, gen):
    model = f"{SERVED}-{tier}"
    print(f"\n=== tier {tier}  ({model})  passes={passes} gen={gen} ===")
    print(f"{'pass':>4} {'median tok/s':>13} {'mean tok/s':>11} {'min':>7} {'max':>7} {'ttft_s':>7}")
    per_pass = []
    for p in range(passes):
        rates = []
        ttfts = []
        for prompt in PROMPTS:
            rate, ttft, n, txt = chat_stream(model, prompt, gen)
            rates.append(rate)
            ttfts.append(ttft)
        med = statistics.median(rates)
        mean = statistics.mean(rates)
        per_pass.append(med)
        print(f"{p:>4} {med:>13.2f} {mean:>11.2f} {min(rates):>7.2f} "
              f"{max(rates):>7.2f} {statistics.median(ttfts):>7.2f}")
    if len(per_pass) >= 2:
        delta = (per_pass[-1] / per_pass[0] - 1.0) * 100.0
        print(f"  convergence: pass0={per_pass[0]:.2f} -> "
              f"pass{passes-1}={per_pass[-1]:.2f} tok/s ({delta:+.1f}%)")
    return per_pass


if __name__ == "__main__":
    print(f"served={SERVED} base_tier={TIER} passes={PASSES} gen={GEN}")
    result = {"tier": run_tier(TIER, PASSES, GEN)}
    out = f"/data/models/RunGLM/experiments/adaptive_expert_cache/conv_{TIER}.json"
    try:
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {out}")
    except Exception as e:
        print(f"(could not write {out}: {e})")
