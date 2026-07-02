#!/usr/bin/env python3
"""Turn a GLM-5.2 LiveBench run into a per-question pass/fail label set.

This mirrors the Terminal-Bench flow (task_labels.txt): we take GLM-5.2's OWN
scores on the LiveBench reasoning suite as the reference, binarise each question
to passed/failed, and freeze that as the label set. Future runs (a different
expert tier, a config change, a regression check) can then be compared against
these labels instead of only against the raw LiveBench ground truth.

  # 1) produce the reference run (writes livebench_results.json):
  ./bench/run_benchmark.sh                       # GLM5.2-top2, all 200 Qs

  # 2) freeze GLM-5.2's outcome as labels:
  python bench/livebench/make_labels.py livebench_results.json
        -> bench/livebench/livebench_labels.txt   (id  task  label  score)

  # 3) later, compare any new run against those frozen labels:
  python bench/livebench/make_labels.py new_run.json --compare

A question is "passed" when its LiveBench score >= --threshold. zebra_puzzle is
partial-credit (fraction of slots correct), so the default threshold of 1.0 means
"every slot right"; spatial and web_of_lies_v2 are already all-or-nothing.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABELS = os.path.join(HERE, "livebench_labels.txt")


def load_results(path):
    rs = json.load(open(path))
    # keep only graded rows (skip transport errors so a flaky call can't mislabel)
    return [r for r in rs if "err" not in r and r.get("score") is not None]


def label_of(score, threshold):
    return "passed" if score >= threshold else "failed"


def write_labels(results, threshold, out):
    lines = []
    for r in results:
        lines.append(f"{r['id']:12s} {r['task']:16s} "
                     f"{label_of(r['score'], threshold):6s} {r['score']:.3f}")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    npass = sum(1 for r in results if r["score"] >= threshold)
    nfail = len(results) - npass
    print(f"wrote {len(results)} labels -> {out}")
    print(f"  passed={npass}  failed={nfail}  (threshold={threshold})")
    # per-task split
    tasks = sorted({r["task"] for r in results})
    for t in tasks:
        tr = [r for r in results if r["task"] == t]
        tp = sum(1 for r in tr if r["score"] >= threshold)
        print(f"  {t:16s} {tp:3d}/{len(tr):<3d} passed")


def read_labels(path):
    labels = {}
    for line in open(path):
        parts = line.split()
        if len(parts) < 3:
            continue
        qid, task, lab = parts[0], parts[1], parts[2]
        labels[qid] = (task, lab)
    return labels


def compare(results, threshold, labels_path):
    if not os.path.exists(labels_path):
        sys.exit(f"no label set at {labels_path} — build one first (drop --compare)")
    labels = read_labels(labels_path)
    print(f"{'id':12s} {'task':16s} {'label':7s} {'now':7s} {'score':>6s}  agree")
    print("-" * 60)
    agree = seen = 0
    regress = []
    for r in results:
        qid = r["id"]
        if qid not in labels:
            continue
        seen += 1
        _, lab = labels[qid]
        now = label_of(r["score"], threshold)
        ok = (now == lab)
        agree += ok
        if lab == "passed" and now == "failed":
            regress.append(qid)
        print(f"{qid:12s} {r['task']:16s} {lab:7s} {now:7s} "
              f"{r['score']:6.3f}  {'YES' if ok else 'no'}")
    print("-" * 60)
    print(f"agreement: {agree}/{seen}")
    if regress:
        print(f"REGRESSIONS (was passed, now failed): {len(regress)} -> "
              f"{', '.join(regress[:10])}{'...' if len(regress) > 10 else ''}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="a livebench run JSON (from run_livebench.py)")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="score >= threshold counts as passed (default 1.0)")
    ap.add_argument("--out", default=DEFAULT_LABELS,
                    help=f"label file to write (default {DEFAULT_LABELS})")
    ap.add_argument("--compare", action="store_true",
                    help="compare this run against the existing label set "
                         "instead of writing labels")
    a = ap.parse_args()

    results = load_results(a.results)
    if not results:
        sys.exit(f"no graded rows in {a.results}")
    if a.compare:
        compare(results, a.threshold, a.out)
    else:
        write_labels(results, a.threshold, a.out)


if __name__ == "__main__":
    main()
