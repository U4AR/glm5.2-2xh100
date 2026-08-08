#!/usr/bin/env python3
"""Is the decode path reproducible at temperature 0?

The accept-length regression under prefetch (2.857 -> 2.667) was read as a
numerics effect: landed experts are computed by the cutlass W4AFP8 kernel
instead of the CPU packed-int4 one, and W4A8 vs W4AFP8 round differently.

But that explanation predicts the WRONG SIGNATURE. Different-but-fixed numerics
are still a function: same prompt, greedy sampling, same answer every run, same
step count every run. The baseline does exactly that -- [70, 70, 70]. Prefetch
gave [73, 77, 75], three different answers to the same question.

So this asks the only question that separates the two: run one prompt N times
at temperature 0 and see whether the completions are identical. Numerics drift
shifts every run the same way; a race shifts them differently. It also reports
WHERE the first divergence is, because a divergence at token 3 and a divergence
at token 150 have different causes -- the first implicates the prefetch itself,
the second implicates something that only starts once the KV cache is deep.
"""
import argparse, hashlib, json, sys, time
from collections import Counter
import urllib.request

PROMPT = ("Explain, step by step, how a mixture-of-experts transformer routes a "
          "token through its experts, and why the router is trained with an "
          "auxiliary loss.")


def gen(url, model, tokens, prompt):
    """One greedy completion. Non-streaming: we want the text and the step
    accounting, not first-token latency."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    txt = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    u = d.get("usage", {})
    return {
        "text": txt,
        "completion_tokens": u.get("completion_tokens"),
        "steps": (u.get("spec_verify_ct") or u.get("forward_steps")),
        "wall": dt,
    }


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            lo = max(0, i - 40)
            return i, repr(a[lo:i + 40]), repr(b[lo:i + 40])
    if len(a) != len(b):
        return n, "<end>", repr(b[n:n + 40]) if len(b) > n else repr(a[n:n + 40])
    return None, "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="GLM5.2")
    ap.add_argument("--tier", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="")
    # Keep run 0's text so rows can be compared ACROSS boots. Where two configs
    # first disagree is the whole diagnosis: rounding-level differences track
    # each other for many tokens and then drift, while a structural error --
    # an expert dropped, double-counted, or routed to the wrong slot -- shows up
    # in the first few tokens.
    ap.add_argument("--save-text", default="")
    a = ap.parse_args()

    model = f"{a.model}-top{a.tier}" if a.tier is not None else a.model
    outs = []
    for i in range(a.runs):
        try:
            r = gen(a.url, model, a.tokens, PROMPT)
        except Exception as e:
            body = getattr(e, "read", lambda: b"")()
            print(f"  run {i}: FAILED {e} {body[:300]!r}")
            return 1
        h = hashlib.sha1(r["text"].encode()).hexdigest()[:10]
        outs.append(r | {"hash": h})
        print(f"  run {i}: {h}  tok={r['completion_tokens']}  "
              f"steps={r['steps']}  {r['wall']:.1f}s  len={len(r['text'])}")

    hashes = [o["hash"] for o in outs]
    c = Counter(hashes)
    distinct = len(c)
    print(f"\n{a.label}: {distinct} distinct completion(s) out of {a.runs}"
          f"   {dict(c)}")
    if distinct == 1:
        print("  DETERMINISTIC")
    else:
        print("  NON-DETERMINISTIC at temperature 0")
        base = outs[0]["text"]
        for o in outs[1:]:
            if o["hash"] == outs[0]["hash"]:
                continue
            i, sa, sb = first_divergence(base, o["text"])
            print(f"  first divergence vs run0 at char {i}")
            print(f"    run0: {sa}")
            print(f"    this: {sb}")
            break
    steps = [o["steps"] for o in outs]
    toks = [o["completion_tokens"] for o in outs]
    print(f"  steps={steps}  tokens={toks}")

    if a.save_text:
        import os
        os.makedirs(a.save_text, exist_ok=True)
        p = os.path.join(a.save_text, f"{a.label}.txt")
        with open(p, "w") as f:
            f.write(outs[0]["text"])
        # Compare against every row already saved, so the ladder reports its own
        # cross-row divergences without a second pass.
        for fn in sorted(os.listdir(a.save_text)):
            if not fn.endswith(".txt") or fn == f"{a.label}.txt":
                continue
            other = open(os.path.join(a.save_text, fn)).read()
            i, _, _ = first_divergence(outs[0]["text"], other)
            where = "IDENTICAL" if i is None else f"char {i}"
            print(f"  vs {fn[:-4]:<16} {where}")

    if a.out:
        try:
            with open(a.out) as f:
                db = json.load(f)
        except Exception:
            db = {}
        db[a.label] = {"hashes": hashes, "distinct": distinct,
                       "steps": steps, "tokens": toks,
                       "lens": [len(o["text"]) for o in outs]}
        with open(a.out, "w") as f:
            json.dump(db, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
