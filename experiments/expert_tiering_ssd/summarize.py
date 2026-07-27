#!/usr/bin/env python3
"""Collate the three-tier ladder into one table.

Speed/RSS come from results.jsonl (tier_bench), accuracy from runs/*.json
(accuracy_eval). Accuracy is recomputed here from the per-item raw records
rather than trusting each file's stored aggregate, so runs measured under
different scoring revisions stay comparable.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF_RAM = int(os.environ.get("TIER_REF_RAM", "160"))


def prefix_agreement(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b), 1)


def load_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(HERE, "runs", "*.json"))):
        name = os.path.basename(path)[:-5]
        m = re.match(r"g(\d+)_r(\d+)_(\w+?)(?:_(top\d w?|top2|top8|top8w))?$", name)
        with open(path) as f:
            d = json.load(f)
        gpu = int(m.group(1)) if m else -1
        ram = int(m.group(2)) if m else -1
        fill = m.group(3) if m else "?"
        count = (m.group(4) or "top2") if m else "?"
        qa = d["qa"]
        d["_gpu"], d["_ram"], d["_fill"], d["_count"] = gpu, ram, fill, count
        d["_acc"] = sum(1 for x in qa if x["ok"]) / len(qa)
        d["_loop"] = sum(
            1 for x in qa if x.get("no_answer", x.get("truncated", False))
        ) / len(qa)
        d["_reason"] = sum(x["reasoning_chars"] for x in qa) / len(qa)
        d["_name"] = name
        runs[name] = d
    return runs


def load_speed():
    speed = {}
    p = os.path.join(HERE, "results.jsonl")
    if not os.path.exists(p):
        return speed
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            speed[r["label"]] = r  # later entries win
    return speed


def main():
    runs = load_runs()
    speed = load_speed()
    if not runs:
        print("no runs yet")
        return

    ref = next((d for d in runs.values() if d["_ram"] == REF_RAM), None)

    rows = []
    for name, d in runs.items():
        sp = speed.get(name) or speed.get(f"{name}_top2") or {}
        # tier_bench labels and accuracy filenames can differ by the count-mode
        # suffix; fall back to matching on the tier sizes.
        if not sp:
            for lbl, r in speed.items():
                if r.get("ram_experts") == d["_ram"] and r.get("gpu_experts") == d["_gpu"] \
                        and r.get("fill_pool") == d["_fill"]:
                    sp = r
                    break
        agree = ident = None
        if ref is not None and d is not ref:
            fr = [
                prefix_agreement(a["text"], b["text"])
                for a, b in zip(ref["gen"], d["gen"])
            ]
            agree = sum(fr) / len(fr)
            ident = sum(1 for a, b in zip(ref["gen"], d["gen"]) if a["text"] == b["text"])
        rows.append(
            {
                "name": name,
                "gpu": d["_gpu"],
                "ram": d["_ram"],
                "ssd": 256 - d["_gpu"] - d["_ram"],
                "fill": d["_fill"],
                "count": d["_count"],
                "tok_s": sp.get("tok_s"),
                "rss": sp.get("tp0_rss_gb"),
                "acc": d["_acc"],
                "loop": d["_loop"],
                "reason": d["_reason"],
                "agree": agree,
                "ident": ident,
                "n_gen": len(d["gen"]),
            }
        )

    rows.sort(key=lambda r: (-r["ram"], r["fill"], r["count"]))
    print(
        f"{'config':<28} {'GPU':>4} {'RAM':>4} {'SSD':>4} {'tok/s':>7} {'RSS GB':>7} "
        f"{'QA acc':>7} {'loop':>6} {'reas ch':>8} {'agree':>7} {'ident':>6}"
    )
    print("-" * 104)
    for r in rows:
        print(
            f"{r['name']:<28} {r['gpu']:>4} {r['ram']:>4} {r['ssd']:>4} "
            f"{(f'{r['tok_s']:.2f}' if r['tok_s'] else '-'):>7} "
            f"{(f'{r['rss']:.1f}' if r['rss'] else '-'):>7} "
            f"{r['acc']*100:>6.1f}% {r['loop']*100:>5.1f}% {r['reason']:>8.0f} "
            f"{(f'{r['agree']*100:.1f}%' if r['agree'] is not None else 'ref'):>7} "
            f"{(f'{r['ident']}/{r['n_gen']}' if r['ident'] is not None else '-'):>6}"
        )
    print(
        f"\nreference for agreement: RAM={REF_RAM} "
        f"({'found' if ref else 'MISSING — agreement column is blank'})"
    )
    print(
        "agree = mean greedy-continuation prefix agreement vs the reference; "
        "loop = share of short-answer prompts that produced no answer at all."
    )


if __name__ == "__main__":
    main()
