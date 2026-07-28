#!/usr/bin/env python3
"""Accuracy check for a three-tier expert-store configuration.

Two signals, because neither alone is enough:

1. ABSOLUTE — exact-match on a short-answer set. Interpretable ("it still knows
   things") but coarse and noisy at this sample size; use it to catch a config
   that has actually broken, not to rank configs a few points apart.

2. RELATIVE — greedy-continuation agreement against a saved reference run
   (normally the full-RAM config, where every expert is reachable and the model
   is exactly the two-tier build). This is the sensitive one: at temperature 0
   the same model must emit the same tokens, so any divergence is caused by the
   substitution. Reported as both "fraction of prompts identical" and "mean
   agreed prefix length", the latter degrading smoothly instead of falling off
   a cliff on the first differing token.

Usage:
  accuracy_eval.py reference <out.json>     # run and save as the reference
  accuracy_eval.py compare <ref.json> <label>
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_lock import exclusive_bench  # noqa: E402

BASE = os.environ.get("TIER_BENCH_BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("TIER_BENCH_MODEL", "GLM5.2-top2")

# Short-answer set. Answers are matched case-insensitively as substrings of the
# FINAL content (reasoning is excluded), so phrasing is free but the fact is not.
QA = [
    ("What is 17 * 23? Reply with only the number.", ["391"]),
    ("What is 144 divided by 12? Reply with only the number.", ["12"]),
    ("What is the capital city of Australia? Reply with only the city name.", ["canberra"]),
    ("What is the chemical symbol for gold? Reply with only the symbol.", ["au"]),
    ("How many sides does a hexagon have? Reply with only the number.", ["6", "six"]),
    ("Who wrote the play 'Hamlet'? Reply with only the name.", ["shakespeare"]),
    ("What is the largest planet in our solar system? Reply with only the name.", ["jupiter"]),
    ("In what year did the Berlin Wall fall? Reply with only the year.", ["1989"]),
    ("What is the square root of 169? Reply with only the number.", ["13"]),
    ("What gas do plants absorb from the atmosphere for photosynthesis? Reply with only the gas name.",
     ["carbon dioxide", "co2", "co₂"]),
    ("What is 2 to the power of 10? Reply with only the number.", ["1024"]),
    ("What is the freezing point of water in Fahrenheit? Reply with only the number.", ["32"]),
    ("Which ocean is the largest by area? Reply with only the name.", ["pacific"]),
    ("What does CPU stand for? Reply with only the expansion.", ["central processing unit"]),
    ("In Python, what does len([1,2,3]) return? Reply with only the number.", ["3"]),
    ("What is the time complexity of binary search on a sorted array of n elements? "
     "Reply with only the big-O expression.", ["o(log n)", "log n"]),
]

# Open-ended prompts for greedy agreement. Deterministic, moderate length.
GEN = [
    "Explain in one paragraph what a KV cache is in transformer inference.",
    "Explain in one paragraph how mixture-of-experts routing works.",
    "Write a short Python function that reverses a linked list. Code only.",
    "Summarize the causes of the French Revolution in one paragraph.",
    "Explain in one paragraph why quicksort is usually faster than mergesort in practice.",
    "Describe in one paragraph what happens during protein synthesis.",
]


def call(prompt, max_tokens):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    content, reasoning = [], []
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
        for ch in o.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content"):
                content.append(d["content"])
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
    return "".join(content), "".join(reasoning)


def run_all(qa_tokens=1024, gen_tokens=200):
    """qa_tokens covers the reasoning trace as well as the answer — this is a
    thinking model. Producing no `content` inside the budget is scored as a
    FAILURE, not skipped: on questions this trivial an empty answer means the
    model burned thousands of tokens looping in its reasoning, which is a real
    degradation mode of aggressive top-K substitution and exactly what a tier
    sweep needs to detect. It is tracked separately from a wrong answer so the
    two can be told apart."""
    out = {"qa": [], "gen": []}
    hits = 0
    looped = 0
    for i, (q, answers) in enumerate(QA):
        content, reasoning = call(q, qa_tokens)
        hay = content.lower()
        no_answer = not content.strip()
        ok = (not no_answer) and any(a in hay for a in answers)
        hits += int(ok)
        looped += int(no_answer)
        out["qa"].append(
            {"q": q, "content": content, "ok": ok, "answers": answers,
             "no_answer": no_answer, "reasoning_chars": len(reasoning)}
        )
        verdict = "LOOP" if no_answer else ("PASS" if ok else "FAIL")
        print(f"  qa {i:2d} {verdict:5s} r={len(reasoning):5d}  {content.strip()[:60]!r}")
    out["qa_accuracy"] = hits / len(QA)
    out["qa_loop_rate"] = looped / len(QA)
    out["qa_mean_reasoning_chars"] = sum(
        x["reasoning_chars"] for x in out["qa"]
    ) / len(QA)
    for i, p in enumerate(GEN):
        content, reasoning = call(p, gen_tokens)
        out["gen"].append({"p": p, "text": reasoning + content})
        print(f"  gen {i:2d} {len(reasoning + content):5d} chars")
    return out


def prefix_agreement(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    denom = max(len(a), len(b), 1)
    return i / denom


def main():
    mode = sys.argv[1]
    if mode == "reference":
        out_path = sys.argv[2]
        res = run_all()
        with open(out_path, "w") as f:
            json.dump(res, f)
        print(f"\nqa_accuracy = {res['qa_accuracy']:.3f}  (reference saved to {out_path})")
        return

    if mode == "diff":
        # Offline comparison of two saved runs, so every config can be measured
        # once (fast boot) and compared later against whichever run is chosen as
        # the reference.
        with open(sys.argv[2]) as f:
            ref = json.load(f)
        with open(sys.argv[3]) as f:
            res = json.load(f)
        label = sys.argv[4] if len(sys.argv) > 4 else sys.argv[3]
    else:
        ref_path, label = sys.argv[2], sys.argv[3]
        with open(ref_path) as f:
            ref = json.load(f)
        res = run_all()

    identical = 0
    fracs = []
    for r, c in zip(ref["gen"], res["gen"]):
        fr = prefix_agreement(r["text"], c["text"])
        fracs.append(fr)
        identical += int(r["text"] == c["text"])
    qa_agree = sum(
        int(r["ok"] == c["ok"]) for r, c in zip(ref["qa"], res["qa"])
    ) / len(ref["qa"])

    summary = {
        "label": label,
        "gpu_experts": int(os.environ.get("GPU_EXPERTS", "-1")),
        "ram_experts": int(os.environ.get("KT_RAM_EXPERTS", "-1")),
        "fill_pool": os.environ.get("KT_TIER_FILL_POOL", "resident"),
        "count_mode": os.environ.get("KT_TIER_COUNT_MODE", "top2"),
        "qa_accuracy": round(res["qa_accuracy"], 4),
        "qa_loop_rate": round(res.get("qa_loop_rate", 0.0), 4),
        "qa_accuracy_reference": round(ref["qa_accuracy"], 4),
        "qa_verdict_agreement": round(qa_agree, 4),
        "gen_identical": f"{identical}/{len(fracs)}",
        "gen_prefix_agreement": round(sum(fracs) / len(fracs), 4),
        "gen_prefix_agreement_each": [round(x, 3) for x in fracs],
    }
    print("\n" + json.dumps(summary, indent=2))
    out = os.environ.get(
        "TIER_ACC_OUT", "experiments/expert_tiering_ssd/accuracy.jsonl"
    )
    with open(out, "a") as f:
        f.write(json.dumps(summary) + "\n")

    # Persist the RAW run too, not just the summary against whichever anchor
    # happened to be current. Each of these costs a server boot, and an anchor
    # can turn out to be the wrong control later -- the SSD=0 reference used on
    # 07-28 differed from the tier rows in two variables (SSD tier AND adaptive
    # state), so its prefix-agreement numbers bound the damage rather than
    # attributing it. With the raw run on disk, `diff` re-anchors any pair
    # offline instead of re-measuring the model.
    if mode != "diff":
        raw_dir = os.environ.get(
            "TIER_RAW_DIR", "experiments/expert_tiering_ssd/runs"
        )
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, f"raw_{label}.json")
        with open(raw_path, "w") as f:
            json.dump(res, f)
        print(f"raw run saved to {raw_path}")


if __name__ == "__main__":
    with exclusive_bench("accuracy_eval"):
        main()
