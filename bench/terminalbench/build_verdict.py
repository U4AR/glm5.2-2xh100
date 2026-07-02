#!/usr/bin/env python3
"""Aggregate harbor per-task results into the TRUE verdict table vs original labels."""
import json, glob, os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "runs_full")

# original labels
labels = {}
for line in open(os.path.join(BASE, "task_labels.txt")):
    name, lab = line.split()
    labels[name] = lab

rows = []
for name, lab in sorted(labels.items()):
    # newest result.json under runs_full/<name>/*/result.json
    cands = sorted(glob.glob(os.path.join(OUT, name, "*", "result.json")))
    reward, status = None, "MISSING"
    if cands:
        d = json.load(open(cands[-1]))
        s = d["stats"]
        if s["n_completed_trials"] and not s["n_errored_trials"]:
            e = s["evals"]; k = list(e)[0]
            reward = e[k]["metrics"][0]["mean"]
            status = "ok"
        elif s["n_errored_trials"]:
            e = s["evals"]; k = list(e)[0]
            status = "ERROR:" + ",".join(e[k].get("exception_stats", {}).keys())
    ours = None if reward is None else ("passed" if reward >= 0.5 else "failed")
    agree = (ours == lab) if ours else None
    rows.append({"task": name, "orig": lab, "reward": reward,
                 "ours": ours, "agree": agree, "status": status})

# print table
print(f"{'task':40s} {'orig':7s} {'reward':7s} {'ours':7s} agree")
print("-" * 75)
for r in rows:
    rw = "-" if r["reward"] is None else f"{r['reward']:.2f}"
    ours = r["ours"] or r["status"]
    ag = "" if r["agree"] is None else ("YES" if r["agree"] else "no")
    print(f"{r['task']:40s} {r['orig']:7s} {rw:7s} {ours:7s} {ag}")

done = [r for r in rows if r["ours"]]
op = [r for r in rows if r["orig"] == "passed"]
of = [r for r in rows if r["orig"] == "failed"]
op_pass = [r for r in op if r["ours"] == "passed"]
of_fail = [r for r in of if r["ours"] == "failed"]
print("\n=== SUMMARY ===")
print(f"completed: {len(done)}/{len(rows)}")
print(f"orig-PASSED reproduced as pass: {len(op_pass)}/{len([r for r in op if r['ours']])} scored "
      f"(of {len(op)} total)")
print(f"orig-FAILED reproduced as fail: {len(of_fail)}/{len([r for r in of if r['ours']])} scored "
      f"(of {len(of)} total)")
agree = [r for r in done if r["agree"]]
print(f"label agreement: {len(agree)}/{len(done)}")
json.dump(rows, open(os.path.join(BASE, "verdict.json"), "w"), indent=2)
