#!/usr/bin/env python3
"""Summarise DeepSWE job results per task, next to Datacurve's graded GLM-5.2 rate.

    python bench/deepswe/report.py top2
    python bench/deepswe/report.py top2 top8     # side by side, border verdict

The reference column is the pass rate of `glm-5-2::effort=high` over its 4 graded
rollouts, read from the trials artifact if present. It is a rate, not a verdict:
a task Datacurve scored 2/4 tells you the model is at its limit there, so a single
local rollout passing or failing is one draw from that coin, not proof of a tier
gap. Read the columns together, and prefer -k > 1 if you want to separate them.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "jobs")

# glm-5-2 effort=high, DeepSWE v1.1, 4 rollouts each (deepswe.datacurve.ai)
REFERENCE = {
    "wazero-multi-module-snapshots": 0.25,
    "httpx-deterministic-cookie-store": 0.50,
    "onedump-dump-encryption-pipeline": 0.50,
    "ts-pattern-match-each": 0.50,
    "psd-tools-blend-range-api": 0.75,
}


def outcome(tier, task):
    """(verdict, reward, seconds) for one job, or None if it was never run."""
    d = os.path.join(JOBS, f"{tier}_{task}")
    res = os.path.join(d, "result.json")
    if not os.path.isfile(res):
        return None
    job = json.load(open(res))
    # The per-trial result carries the reward; the job result only aggregates.
    for sub in sorted(os.listdir(d)):
        tres = os.path.join(d, sub, "result.json")
        if not os.path.isfile(tres):
            continue
        t = json.load(open(tres))
        # The verifier is the ONLY source of a verdict. A trial that never reached
        # the verifier (agent crashed, server died, timeout) has reward None, and
        # scoring that as "fail" silently turns infrastructure faults into model
        # failures -- which is exactly what this script did on its first run.
        verifier = t.get("verifier_result") or {}
        reward = verifier.get("reward")
        exc = (t.get("exception_info") or {}).get("exception_type")
        secs = (t.get("agent_execution") or {}).get("duration_seconds")
        if reward is None:
            return ("ERROR", None, secs, exc or "no verifier result")
        return ("pass" if reward else "fail", reward, secs, "")
    stats = job.get("stats", {})
    if stats.get("n_errored_trials"):
        return ("ERROR", None, None, "trial errored")
    return None


def main():
    tiers = sys.argv[1:] or ["top2"]
    width = max(len(t) for t in REFERENCE) + 2
    head = f"{'task':{width}s} {'graded':>7s}"
    for t in tiers:
        head += f" {t:>10s}"
    print(head)
    print("-" * len(head))

    rows = []
    for task, ref in REFERENCE.items():
        line = f"{task:{width}s} {ref*4:>4.0f}/4 "
        cells = []
        for tier in tiers:
            o = outcome(tier, task)
            cells.append(o)
            line += f" {(o[0] if o else '-'):>10s}"
        rows.append((task, ref, cells))
        print(line)

    for i, tier in enumerate(tiers):
        got = [c[i] for _, _, c in rows if c[i]]
        p = sum(1 for c in got if c[0] == "pass")
        graded = [c for c in got if c[0] != "ERROR"]
        errs = [c for c in got if c[0] == "ERROR"]
        print(f"\n{tier}: {p}/{len(graded)} resolved of {len(graded)} graded")
        for c in errs:
            print(f"  ERRORED (not counted): {c[3]}")

    if len(tiers) == 2:
        fast, strong = tiers
        border = [t for t, _, c in rows
                  if c[0] and c[1] and c[0][0] == "fail" and c[1][0] == "pass"]
        print(f"\nborder ({strong} passes, {fast} fails): {border or 'none'}")


if __name__ == "__main__":
    main()
