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
#
# LEGACY_QA is the original 16 items, kept verbatim and in order so that runs
# saved before the set was extended still compare item-for-item. Sixteen items
# turned out to be too coarse for what it was being asked to do: one item is
# worth 0.0625, so every interesting comparison on the board ("0.75 vs 0.75",
# "0.9375 vs 0.875") was a one- or two-item difference with no way to tell it
# from sampling noise. The binomial 95% CI on 12/16 is roughly +/-0.21 -- wider
# than the entire quality range being ranked. EXTRA_QA takes the set to 66,
# which brings that to about +/-0.10: still coarse, but no longer wider than
# the effect. Legacy accuracy is reported separately for continuity.
LEGACY_QA = [
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

# Fifty more of the same kind, spread across arithmetic, physical science,
# biology, geography, history, literature and CS so that a degradation
# concentrated in one region of the expert space still shows up. Answers stay
# short and unambiguous; where a fact is genuinely disputed (longest river,
# most populous country) every defensible answer is accepted, because the
# signal being measured is degradation, not trivia knowledge.
EXTRA_QA = [
    ("What is 25 * 4? Reply with only the number.", ["100"]),
    ("What is 7 factorial? Reply with only the number.", ["5040"]),
    ("What is 15% of 200? Reply with only the number.", ["30"]),
    ("What is the sum of the interior angles of a triangle in degrees? "
     "Reply with only the number.", ["180"]),
    ("What is the next prime number after 13? Reply with only the number.", ["17"]),
    ("What is 1000 minus 379? Reply with only the number.", ["621"]),
    ("What is the cube root of 27? Reply with only the number.", ["3"]),
    ("How many bits are in a byte? Reply with only the number.", ["8", "eight"]),
    ("What is the atomic number of carbon? Reply with only the number.", ["6", "six"]),
    ("How many chromosomes are in a normal human somatic cell? "
     "Reply with only the number.", ["46"]),
    ("Approximately how many kilometers per second does light travel in a vacuum? "
     "Reply with only the number.", ["300,000", "300000", "299,792", "299792", "3x10", "3 x 10"]),
    ("Which organelle is known as the powerhouse of the cell? "
     "Reply with only the name.", ["mitochondri"]),
    ("What is the hardest naturally occurring mineral? Reply with only the name.", ["diamond"]),
    ("How many planets are in our solar system? Reply with only the number.", ["8", "eight"]),
    ("What is the pH of pure water at 25 degrees Celsius? Reply with only the number.", ["7"]),
    ("Which blood type is the universal donor? Reply with only the type.", ["o negative", "o-"]),
    ("What is the boiling point of water at sea level in Celsius? "
     "Reply with only the number.", ["100"]),
    ("Which planet is known as the Red Planet? Reply with only the name.", ["mars"]),
    ("What force keeps the planets in orbit around the Sun? "
     "Reply with only the name of the force.", ["gravit"]),
    ("What is the most abundant gas in Earth's atmosphere? "
     "Reply with only the gas name.", ["nitrogen"]),
    ("What is the capital of Japan? Reply with only the city name.", ["tokyo"]),
    ("What is the longest river in the world? Reply with only the name.", ["nile", "amazon"]),
    ("What is the tallest mountain above sea level? Reply with only the name.", ["everest"]),
    ("On which continent is the Sahara Desert? Reply with only the continent.", ["africa"]),
    ("What is the capital of Canada? Reply with only the city name.", ["ottawa"]),
    ("How many continents are there? Reply with only the number.", ["7", "seven"]),
    ("Which country has the largest population? Reply with only the country name.",
     ["india", "china"]),
    ("What is the smallest country in the world by area? Reply with only the name.",
     ["vatican"]),
    ("In what year did World War II end? Reply with only the year.", ["1945"]),
    ("Who was the first person to walk on the Moon? Reply with only the name.", ["armstrong"]),
    ("In what year did the Titanic sink? Reply with only the year.", ["1912"]),
    ("Who was the first President of the United States? Reply with only the name.",
     ["washington"]),
    ("In what year did the French Revolution begin? Reply with only the year.", ["1789"]),
    ("Julius Caesar was a leader of which ancient civilization? "
     "Reply with only the name.", ["roman", "rome"]),
    ("Who wrote 'Pride and Prejudice'? Reply with only the name.", ["austen"]),
    ("Who wrote 'One Hundred Years of Solitude'? Reply with only the name.",
     ["marquez", "márquez"]),
    ("How many letters are in the English alphabet? Reply with only the number.", ["26"]),
    ("What is the plural of the animal 'mouse'? Reply with only the word.", ["mice"]),
    ("In which language was the New Testament originally written? "
     "Reply with only the language.", ["greek"]),
    ("Who wrote 'On the Origin of Species'? Reply with only the name.", ["darwin"]),
    ("What does HTTP stand for? Reply with only the expansion.",
     ["hypertext transfer protocol"]),
    ("What does RAM stand for? Reply with only the expansion.", ["random access memory"]),
    ("In Python, what does 7 // 2 evaluate to? Reply with only the number.", ["3"]),
    ("What is the decimal value of the binary number 1011? Reply with only the number.",
     ["11"]),
    ("What is the worst-case time complexity of quicksort? "
     "Reply with only the big-O expression.", ["o(n^2)", "o(n²)", "n^2", "n²", "n*n"]),
    ("Which data structure operates on a last-in, first-out basis? "
     "Reply with only the name.", ["stack"]),
    ("In Python, what type does the expression 3 / 2 return? "
     "Reply with only the type name.", ["float"]),
    ("What does SQL stand for? Reply with only the expansion.",
     ["structured query language"]),
    ("How many bytes are in a kibibyte? Reply with only the number.", ["1024"]),
    ("What does GPU stand for? Reply with only the expansion.",
     ["graphics processing unit"]),
]

QA = LEGACY_QA + EXTRA_QA
LEGACY_QUESTIONS = frozenset(q for q, _ in LEGACY_QA)

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


def wilson_halfwidth(hits, n, z=1.96):
    """Half-width of the Wilson 95% interval on the accuracy.

    Printed alongside every accuracy so a reader can see immediately whether a
    gap is resolvable at this sample size. Wilson rather than the normal
    approximation because accuracies here sit near 1.0, where the normal
    interval is badly wrong (and can exceed 1).
    """
    if n <= 0:
        return 0.0
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return (hi - lo) / 2.0


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
    # The original 16, scored on their own, so numbers recorded before the set
    # was extended stay comparable instead of silently changing meaning.
    leg = [x for x in out["qa"] if x["q"] in LEGACY_QUESTIONS]
    if leg:
        out["qa_accuracy_legacy16"] = sum(int(x["ok"]) for x in leg) / len(leg)
        out["qa_loop_rate_legacy16"] = sum(int(x["no_answer"]) for x in leg) / len(leg)
    out["qa_n"] = len(QA)
    out["qa_ci95"] = round(wilson_halfwidth(hits, len(QA)), 4)
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
    # Align on the QUESTION, not on list position. A reference saved before the
    # set was extended has 16 entries against today's 66, and zip() would
    # quietly score the first 16 pairs and divide by the reference length --
    # reporting a number that looks like full agreement but covers a quarter of
    # the set. Pair by text and say how many actually matched.
    ref_by_q = {r["q"]: r for r in ref["qa"]}
    pairs = [(ref_by_q[c["q"]], c) for c in res["qa"] if c["q"] in ref_by_q]
    qa_agree = (
        sum(int(r["ok"] == c["ok"]) for r, c in pairs) / len(pairs) if pairs else 0.0
    )
    if len(pairs) != len(res["qa"]):
        print(
            f"\nNOTE: reference covers {len(pairs)} of {len(res['qa'])} questions; "
            "verdict agreement is over the overlap only."
        )

    summary = {
        "label": label,
        "gpu_experts": int(os.environ.get("GPU_EXPERTS", "-1")),
        "ram_experts": int(os.environ.get("KT_RAM_EXPERTS", "-1")),
        "fill_pool": os.environ.get("KT_TIER_FILL_POOL", "resident"),
        "count_mode": os.environ.get("KT_TIER_COUNT_MODE", "top2"),
        "qa_n": res.get("qa_n", len(res["qa"])),
        "qa_accuracy": round(res["qa_accuracy"], 4),
        "qa_ci95": res.get("qa_ci95"),
        "qa_loop_rate": round(res.get("qa_loop_rate", 0.0), 4),
        "qa_accuracy_legacy16": (
            round(res["qa_accuracy_legacy16"], 4)
            if "qa_accuracy_legacy16" in res else None
        ),
        "qa_accuracy_reference": round(ref["qa_accuracy"], 4),
        "qa_verdict_agreement": round(qa_agree, 4),
        "qa_verdict_agreement_n": len(pairs),
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
